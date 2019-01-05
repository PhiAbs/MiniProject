%% initialize VO Pipeline with Kitty Dataset
clear all; close all; clc;

%% define Path and load parameters
path = 'datasets/kitti00/kitti';
% need to set kitti_path to folder containing "00" and "poses"
assert(exist('path', 'var') ~= 0);
ground_truth = load([path '/poses/00.txt']);
ground_truth = ground_truth(:, [end-8 end]);
last_frame = 4540;
% K = load([path '/00/K.txt']);
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
cameraParams = cameraParameters('IntrinsicMatrix',K');

% camera angles
anglex = K(1,3)/K(1,1);
angley = K(2,3)/K(2,2);

% paths to functions
addpath('code')

% parameters
bidirect_thresh = 3; % TODO  0.3: good
maxDistance_essential = 0.01;  % 0.1 is too big for parking!! 0.01 might work as well
maxNumTrials_Essential = 20000;
minQuality_Harris = 0.1;  %TODO  0.1: good
p3p_pixel_thresh = 1;  % TODO 1: good. 5: not so good
p3p_num_iter = 5000;
reprojection_thresh = 3;  %15: good. 10000: not so good for kitti, good for parking
triangAngleThres = 0.01;
nonmax_suppression_radius = 10;
harris_rejection_radius = 15; %TODO 10: good for kitti


%% Bootstrapping

img0 = uint8(single(imread([path '/00/image_0/' sprintf('%06d.png',0)])));
img1 = uint8(single(imread([path '/00/image_0/' sprintf('%06d.png',1)])));
img2 = uint8(single(imread([path '/00/image_0/' sprintf('%06d.png',2)])));

points = detectHarrisFeatures(img0,'MinQuality',0.1);
kps = points.Location;  %keypoint_start

pointTracker = vision.PointTracker('MaxBidirectionalError',0.3);
initialize(pointTracker, kps, img0);
setPoints(pointTracker,kps);
[kpl,keep] = pointTracker(img1);   % keypoints_latest
kps = kps(keep,:);
kpl = kpl(keep,:);

% track a second time
setPoints(pointTracker,kpl);
[kpl,keep] = pointTracker(img2);   % keypoints_latest
kps = kps(keep,:);
kpl = kpl(keep,:);

% plot to validate
% figure(1)
% subplot(2,1,1)
% showMatchedFeatures(img0,img1,kps,kpl)
% subplot(2,1,2)
% stem3(kps(:,1),kps(:,2),confidence,'fill'); view(-25,30);

%% estimate Fundamental Matrix and initial pose
[E, keep] = estimateEssentialMatrix(kps, ...
    kpl, cameraParams, 'MaxNumTrials', ...
    maxNumTrials_Essential, 'MaxDistance', maxDistance_essential);
kps = kps(keep,:);
kpl = kpl(keep,:);
% subplot(2,1,2)
% showMatchedFeatures(img0,img2,kps,kpl)

[R_WC,t_WC] = relativeCameraPose(E,cameraParams,kps,kpl);
% transform to our system
R_WC = R_WC';
t_WC = t_WC';
T_WC = [R_WC,t_WC];

T_CW = [R_WC',-R_WC'*t_WC];

%% triangulate Points
% M = cameraMatrix(cameraParams,R_W_C', t_W_C'); % this would be the other
% convention for triangulate...
M = (K*T_CW)';
[Xnew, reprojectionErrors] = triangulate( kps, kpl, (K*[eye(3) zeros(3,1)])', M);

c_inf_homo = T_WC(1:3,1:3)*inv(K)*[kpl ones(length(kpl),1)]';
f_inf_homo = inv(K)*[kps ones(length(kps),1)]';
triangulationAngles = acos(sum(c_inf_homo.*f_inf_homo,1)./...
(vecnorm(c_inf_homo,2).*vecnorm(f_inf_homo,2)));

% P = linearTriangulation([kps ones(length(kps),1)]',[kpl ones(length(kpl),1)]',(K*[eye(3) zeros(3,1)]),M');
% subplot(2,1,1)
% plot3(S.X(:,1),S.X(:,2),S.X(:,3),'*'); view(0,90);
% xlabel('x')
% ylabel('y')
% zlabel('z')
% grid on;

Xnew_cam = T_CW*[Xnew ones(size(Xnew,1),1)]';
keep = all((abs(Xnew_cam(1:2, :))<[anglex; angley].*Xnew_cam(3, :))'...
    & reprojectionErrors < reprojection_thresh...
    & triangulationAngles' > triangAngleThres,2);

%% continious

S.X = Xnew(keep,:);
S.P = kpl(keep,:);
% S.C = kpl(~keep,:);
% S.F = kps(~keep,:);
S.C = kpl;
S.F = kps;
S.findP = find(keep);
S.keepX = find(keep(S.findP));
T0 = [eye(3) zeros(3,1)];
S.T = T0(:)*ones(1,size(S.C,1));
S.Frames = 0*ones(1,size(S.C,1));

% extract new features in 2nd image
points = detectHarrisFeatures(img2,'MinQuality', minQuality_Harris);
kpl=checkIfKeypointIsNew(points.Location', S.P', harris_rejection_radius);
kpl=kpl';
S.C = [S.C; kpl];
S.T = [S.T T_WC(:)*ones(1,size(kpl,1))];
S.F = [S.F; kpl];
S.Frames = 2*ones(1,size(kpl,1));
img = img2;
clear img0 img1 img2 reprojection_error F T_CW;
img_prev = img;

% sizes = [1 size(S.P,1) size(S.C,1) 0 ... 
%      nnz(~(all((abs(S.X(:,1:2)-t_WC(1:2)')<[anglex angley].*(S.X(:,3)-t_WC(3))),2)))...
%         nnz(~(reprojectionErrors<reprojection_thres))];
% plotall(img_prev,S.X,S.P,Xnew,Xnew,...
%     reprojectionErrors < reprojection_thresh,...
%     triangulationAngles' > triangAngleThres,...
%     all((abs(Xnew_cam(1:2, :))<[anglex; angley].*Xnew_cam(3, :))',2),...
%     [zeros(3,1) t_WC])

% initialize KLT trackers for continuous mode
trackP = vision.PointTracker('MaxBidirectionalError', bidirect_thresh);
initialize(trackP, S.P, img_prev);
trackC = vision.PointTracker('MaxBidirectionalError', bidirect_thresh);
initialize(trackC, S.C, img_prev);

for i=3:last_frame
    
    % get new image
    img_prev = img;
    img = uint8(single(imread([path '/00/image_0/' sprintf('%06d.png',i)])));
    
    % track features into new image
%     [points, keepP] = trackP(img);
%     S.P = points(keepP,:);
%     S.X = S.X(keepP,:);
%     [pointsP, keepP] = trackP(img);
%     P = pointsP(keepP,:);
%     X = S.X(keepP,:);
    [points, keepC] = trackC(img);
    S.keepX = find(keepC(S.findP));
    S.findP = S.findP(keepC(S.findP));
    S.P = points(S.findP,:); % order is crucial, first S.P then S.C
    S.X = S.X(S.keepX,:);   
    S.C = points(keepC,:);
    S.T = S.T(:,keepC);
    S.F = S.F(keepC,:);
    keepCidx = find(keepC);
%     S.findP = find(ismember(keepCidx,S.findP(ismember(S.findP,keepCidx))));
    S.findP = find(ismember(S.findP,keepCidx));
    
    [R_WC, t_WC, keepP] = estimateWorldCameraPose(S.P,S.X,cameraParams,...
        'MaxNumTrials', p3p_num_iter, 'MaxReprojectionError', p3p_pixel_thresh);
    R_WC = R_WC';
    t_WC = t_WC';
    T_WC = [R_WC, t_WC];
    T_CW = [R_WC',-R_WC'*t_WC];
    S.X = S.X(keepP,:);
    S.P = S.P(keepP,:);
    S.findP = S.findP(keepP);
    S.keepX = S.keepX(keepP);

    % triangulate S.C and S.F
    Xnew = []; reprojectionErrors = []; triangulationAngles = [];
    M = (K*T_CW)';
    NonLms = find(~ismember([1:size(S.C,1)],S.findP));
    for j=NonLms 
        T_WCold = reshape(S.T(:,j),3,4);
        T_ColdW = [T_WCold(:,1:3)',-T_WCold(:,1:3)'*T_WCold(:,4)];
        Mold = (K*T_ColdW)';
        [new, reprojectionError] = triangulate( S.F(j,:), S.C(j,:), Mold, M);
        Xnew = [Xnew; new];
        reprojectionErrors = [reprojectionErrors; reprojectionError];
        
        c_inf_homo = T_ColdW(1:3,1:3)*T_WC(1:3,1:3)*inv(K)*[S.C(j,:) 1]';
        f_inf_homo = inv(K)*[S.F(j,:) 1]';
    
        triangulationAngle = acos(sum(c_inf_homo.*f_inf_homo,1)./...
        (vecnorm(c_inf_homo,2).*vecnorm(f_inf_homo,2)));
        triangulationAngles = [triangulationAngles; triangulationAngle];
    end
    
    Xnew_cam = T_CW*[Xnew ones(size(Xnew,1),1)]';
    keep = all((abs(Xnew_cam(1:2, :))<[anglex; angley].*Xnew_cam(3, :))' ...
                    & (Xnew_cam(3,:) > 0)' ...
                    & reprojectionErrors < reprojection_thresh ...
                    & triangulationAngles > triangAngleThres,2);

    % plot for debugging and tuning
    plotall(img, S.X, S.P, Xnew, S.C(NonLms,:), reprojectionErrors < reprojection_thresh, ...
        triangulationAngles > triangAngleThres,...
        all((abs(Xnew_cam(1:2, :))<[anglex; angley].*Xnew_cam(3, :))',2),...
        t_WC)%, sizes)
    
    S.findP = [S.findP; NonLms(keep)'];
    S.keepX = [S.keepX; ones(nnz(keep),1)];
    
    Xnew = Xnew(keep,:);
    S.X = [S.X; Xnew];
    S.P = [S.P; S.C(NonLms(keep),:)];
%     S.C = S.C(~keep,:);
%     S.T = S.T(:,~keep);
%     S.F = S.F(~keep,:);
    % plot3(Xnew(:,1),Xnew(:,2),Xnew(:,3),'*'); view(0,90);
    
    % extract new Keypints
    points = detectHarrisFeatures(img,'MinQuality', minQuality_Harris);
    kpl = points.Location;  %keypoint_latest
    
    % make sure that the same keypoint does not get tracked more than once
    kp_new_sorted_out = checkIfKeypointIsNew(kpl', ...
          [(S.C)'], harris_rejection_radius);
%         [(S.P)', (S.C)'], harris_rejection_radius);      
    S.C = [S.C; kp_new_sorted_out'];
    S.F = [S.F; kp_new_sorted_out'];
    S.T = [S.T, T_WC(:) * ones(1,size(kp_new_sorted_out, 2))];
    
%     setPoints(trackP, S.P);
    setPoints(trackC, S.C);
    
end



%% Functions

function plotall(image,X,P,Xnew,C,keepReprojection,keepAngle,keepBehind,t_WC,sizes)

%     figure('name','11','units','normalized','outerposition',[0.1 0.1 0.85 0.8]);
    figure(11)
    set(gcf,'units','normalized','outerposition',[0.1 0.1 0.85 0.8]);
    % plot upper image with 3D correspondences
    subplot(2,3,1:3)
    hold off;
    imshow(image);
    hold on;
    plot([0;C(~keepReprojection,1)],[0;C(~keepReprojection,2)],'bs','linewidth',1.5)
    plot([0;C(~keepAngle,1)],[0;C(~keepAngle,2)],'cd','linewidth',1.5)
    plot([0;C(~keepBehind,1)],[0;C(~keepBehind,2)],'b+','linewidth',1.5)
    plot([0;C(keepAngle&keepReprojection&keepBehind,1)],...
        [0;C(keepAngle&keepReprojection&keepBehind,2)],...
        'yo','linewidth',1.5)
    plot(P(:,1),P(:,2),'r*','linewidth',1.5)
    legend('ReprojectionError is too big',...
        'Angle too small',...
        'Triangulated behind camera',...
        'newly triangulated',...
        'old 3D correspondences')
    title('image with 3D correspondences and new kps')
    
    
%     % bar diagram with sizes of different matrices
%     subplot(2,3,4)
%     bar(sizes(:,1),sizes(:,2:end))
%     legend('#-landmarks','#-tracked keypoints','#-newly triangulated',...
%         '#-outside viewing angle','#-error too big')
%     title('Array sizes over frames')
    
    
    % plot lower left as 2D point cloud
    subplot(2,3,5)
    hold on;
    plot(Xnew(keepAngle&keepReprojection&keepBehind,1),Xnew(keepAngle&keepReprojection&keepBehind,3),...
        'bx','linewidth',1.5);
    plot(X(:,1),X(:,3),'rx','linewidth',1);
    legend('newly triangulated','old landmarks')
    title('2D Pointcloud')
    xlabel('x')
    ylabel('z')
    
    
    % plot Camera Positions
    subplot(2,3,6)
    hold on;
    plot(t_WC(1,:),t_WC(3,:),'rx','linewidth',1)
    title('Camera Position')
    xlabel('x')
    ylabel('z')
    axis equal

end


%%
function kp_new = checkIfKeypointIsNew(kp_new, kp_tracked, threshold)
% checkIfKeypointIsNew checks if a newly found keypoint (Harris) already
% existed as a keypoint which was tracked into the image from an earlier stage.
% Discard a point if it is within a certain radius around an existing
% keypoint
% input: 
% kn_new: newly found keypoints in Image, size 2xN
% kp_tracked: keypoints tracked from earlier image, size 2xM
% threshold: If a new point is closer to an existing keypoint than this
% threshold, the point gets discarded
% output: 
% kp_new: sorted out keypoints, size 2xK

for i = 1:length(kp_tracked)
    distance = sqrt(sum((kp_new - kp_tracked(:,i)).^2));
    kp_new = kp_new(:,distance > threshold);
end
end