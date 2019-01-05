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
maxDistance_essential = 0.1;  % 0.1 is too big for parking!! 0.01 might work as well
maxNumTrials_Essential = 20000;
minQuality_Harris = 0.01;  %TODO  0.1: good
p3p_pixel_thresh = 1;  % TODO 1: good. 5: not so good
p3p_num_iter = 10000;
reprojection_thresh = 1;  %15: good. 10000: not so good for kitti, good for parking
reprojection_thresh_p3p = 2;
triangAngleThres = 0.01;
nonmax_suppression_radius = 10;
harris_rejection_radius = 10; %TODO 10: good for kitti
BA_iter = 2; 
num_BA_frames = 20;
max_iter_BA = 100;
num_fixed_frames_BA = 1;
absoluteTolerance_BA = 0.001;
enable_BA = true;
enable_plot = false;


%% Bootstrapping

img0 = uint8(single(imread([path '/00/image_0/' sprintf('%06d.png',0)])));
img1 = uint8(single(imread([path '/00/image_0/' sprintf('%06d.png',1)])));
img2 = uint8(single(imread([path '/00/image_0/' sprintf('%06d.png',2)])));

points = detectHarrisFeatures(img0,'MinQuality',minQuality_Harris);
points = nonmaxSuppression(points, nonmax_suppression_radius);
kps = points.Location;  %keypoint_start

pointTracker = vision.PointTracker('MaxBidirectionalError',bidirect_thresh);
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

Xnew_cam = T_CW*[Xnew ones(size(Xnew,1),1)]';
keep = all((abs(Xnew_cam(1:2, :))<[anglex; angley].*Xnew_cam(3, :))'...
    & reprojectionErrors < reprojection_thresh...
    & triangulationAngles' > triangAngleThres,2);

%% setup for continuous

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
points = nonmaxSuppression(points, nonmax_suppression_radius);
kpl=checkIfKeypointIsNew(points.Location', S.P', harris_rejection_radius);
kpl=kpl';
S.C = [S.C; kpl];
S.T = [S.T T_WC(:)*ones(1,size(kpl,1))];
S.F = [S.F; kpl];
S.Frames = [S.Frames, 2 * ones(1,size(S.C,1))]; % TODO hardcoded last bootstrap frame

% store data for bundle adjustment
cameraPoses_all = table;
cameraPoses_all.ViewId(1) = uint32(2); % TODO: hardcoded number, is that the correct one? we start counting at 0
cameraPoses_all.Orientation{1} = T_WC(1:3, 1:3);
cameraPoses_all.Location{1} = T_WC(1:3, end)';
S.P_BA(:, num_BA_frames, 1) = S.P(:,1);
S.P_BA(:, num_BA_frames, 2) = S.P(:,2)';
S.X_BA = S.X; 
keep_P_BA = ones(size(S.P, 1), 1);
S.C_trace_tracker(:, num_BA_frames, 1) = S.C(:,1);
S.C_trace_tracker(:, num_BA_frames, 2) = S.C(:,2);

img = img2;
clear img0 img1 img2 reprojection_error F T_CW;
img_prev = img;


sizes = [0 0 0 0; 2 size(S.P,2) nnz(~keep) nnz(keep)];  
t_WC_BA = [0 0 0; T_WC(1:3,4)'];
X_hist = S.X;

plotall(img_prev, S.X, S.P, S.X, S.C([keep;true(length(kpl),1)],:), ...
    reprojectionErrors < reprojection_thresh, ...
    triangulationAngles' > triangAngleThres,...
    all((abs(Xnew_cam(1:2, :))<[anglex; angley].*Xnew_cam(3, :))',2),...
    T_WC, t_WC_BA', sizes, anglex)

pause(5)

% initialize KLT trackers for continuous mode
trackP = vision.PointTracker('MaxBidirectionalError', bidirect_thresh);
initialize(trackP, S.P, img_prev);
trackC = vision.PointTracker('MaxBidirectionalError', bidirect_thresh);
initialize(trackC, S.C, img_prev);

cam_x = [];
cam_z = [];

%% start continuous

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
    keepP = keepC(S.findP);
    S.findP = S.findP(keepC(S.findP));
    S.P = points(S.findP,:); % order is crucial, first S.P then S.C
    S.X = S.X(S.keepX,:);   
    S.C = points(keepC,:);
    S.T = S.T(:,keepC);
    S.F = S.F(keepC,:);
    S.C_trace_tracker(:, 1, :) = [];
    S.C_trace_tracker(~keepC, :, :) = [];
    S.C_trace_tracker(:, end+1, :) = points(keepC,:);
    S.Frames = S.Frames(keepC);
    
    % update data for bootstrap: update only the points that can still be
    % tracked!
    keep_Index = find(keep_P_BA);
    keep_Index = keep_Index(keepP);
    keep_P_BA = false(size(keep_P_BA));
    keep_P_BA(keep_Index) = 1;
    S.P_BA(:, 1, :) = [];
    S.P_BA(keep_P_BA, end+1, 1) = S.P(:,1);
    S.P_BA(keep_P_BA, end, 2) = S.P(:,2);
    keepCidx = find(keepC);
%     S.findP = find(ismember(keepCidx,S.findP(ismember(S.findP,keepCidx))));
    S.findP = find(ismember(S.findP,keepCidx));
    
    [R_WC, t_WC, keepP] = estimateWorldCameraPose(S.P,S.X,cameraParams,...
        'MaxNumTrials', p3p_num_iter, 'MaxReprojectionError', p3p_pixel_thresh);
    R_WC = R_WC';
    t_WC = t_WC';
    
        % we reproject the points on our own and define the threshold for the
    % points that should get discarded
    R_C_W = R_WC';
    t_C_W = -R_C_W*t_WC;
    % careful: transpose rotation matrix due to other convention
    projected_points = worldToImage(cameraParams,R_C_W',t_C_W',S.X);
    keepP = abs(sqrt(sum((projected_points - S.P).^2, 2))) ...
        < reprojection_thresh_p3p;
    
    T_WC = [R_WC, t_WC];
    T_CW = [R_WC',-R_WC'*t_WC];
    S.X = S.X(keepP,:);
    S.P = S.P(keepP,:);
    
    % update data for bootstrap: update only the points that can still be
    % tracked!
    keep_Index = find(keep_P_BA);
    keep_Index = keep_Index(keepP);
    keep_P_BA = false(size(keep_P_BA));
    keep_P_BA(keep_Index) = 1;
    S.P_BA(~keep_P_BA, end, :) = 0;
    S.findP = S.findP(keepP);
    S.keepX = S.keepX(keepP);

    % triangulate S.C and S.F
    Xnew = []; 
    reprojectionErrors = []; 
    triangulationAngles = [];
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
    sizes=[sizes; i size(S.P,1) size(S.C(NonLms,:),1) nnz(keep)];    
    t_WC_BA = [t_WC_BA; cameraPoses_all.Location{end}];
    X_hist = [X_hist; S.X];
    if size(sizes,1)>5
        sizes(1,:)=[];
    end
    
    plotall(img, X_hist, S.P, S.X, S.C(NonLms,:), reprojectionErrors < reprojection_thresh, ...
        triangulationAngles > triangAngleThres,...
        all((abs(Xnew_cam(1:2, :))<[anglex; angley].*Xnew_cam(3, :))',2),...
        T_WC, t_WC_BA', sizes, anglex)
    
    S.findP = [S.findP; NonLms(keep)'];
    S.keepX = [S.keepX; ones(nnz(keep),1)];
    
    Xnew = Xnew(keep,:);
    S.X = [S.X; Xnew];
    S.P = [S.P; S.C(NonLms(keep),:)];
    % S.P = [S.P; S.C(keep,:)];
%     S.C = S.C(~keep,:);
%     S.T = S.T(:,~keep);
%     S.F = S.F(~keep,:);
%     S.Frames = S.Frames(~keep); 
    % update data for bootstrap
    keep_P_BA = [keep_P_BA; ones(size(Xnew, 1), 1)];
    S.P_BA(end+1:end+size(Xnew, 1), :, :) = S.C_trace_tracker(NonLms(keep), :, :);
    S.X_BA = [S.X_BA; Xnew];
%     S.C_trace_tracker = S.C_trace_tracker(~keep, :, :);
    
    % extract new Keypints
    points = detectHarrisFeatures(img,'MinQuality', minQuality_Harris);
    points = nonmaxSuppression(points, nonmax_suppression_radius);
    kpl = points.Location;  %keypoint_latest
    
    % make sure that the same keypoint does not get tracked more than once
    kp_new_sorted_out = checkIfKeypointIsNew(kpl', ...
          [(S.C)'], harris_rejection_radius);
%         [(S.P)', (S.C)'], harris_rejection_radius);      
    S.C = [S.C; kp_new_sorted_out'];
    S.C_trace_tracker(end+1:end+size(kp_new_sorted_out, 2), end, :) = kp_new_sorted_out';
    S.F = [S.F; kp_new_sorted_out'];
    S.T = [S.T, T_WC(:) * ones(1,size(kp_new_sorted_out, 2))];
    S.Frames = [S.Frames, i*ones(1, size(kp_new_sorted_out, 2))]; 
    setPoints(trackP, S.P);
    setPoints(trackC, S.C);
    
            % add newest camera pose to all camera poses
    cameraPoses_new = table;
    cameraPoses_new.ViewId(1) = uint32(i);
    cameraPoses_new.Orientation{1} = T_WC(1:3, 1:3);
    cameraPoses_new.Location{1} = T_WC(1:3, end)';
    cameraPoses_all = [cameraPoses_all; cameraPoses_new];
    
    % bundle adjustment
    if BA_iter == num_BA_frames && enable_BA
        % delete all rows in the BA matrices which only contain zeros or
        % only one valid point (they cannot be used for BA

        valid_points = S.P_BA(:,:,1) > 0;
        untracked_landmark_idx = find(sum(valid_points, 2) == 0);
        S.P_BA(untracked_landmark_idx, :, :) = [];
        S.X_BA(untracked_landmark_idx, :) = [];
        keep_P_BA(untracked_landmark_idx) = [];
        
        [S, keep_P_BA, T_WC, cameraPoses_all] = ...
            bundle_adjustment(S, cameraPoses_all, num_BA_frames, ...
            keep_P_BA, K, max_iter_BA, num_fixed_frames_BA, absoluteTolerance_BA);
        
    else
        BA_iter = BA_iter + 1;
    end
    
    if enable_plot
        plotBundleAdjustment(cameraPoses_all);
    end
    
    cam_x = [cam_x, T_WC(1,4)];
    cam_z = [cam_z, T_WC(3,4)];
    
    i
    
end



%% Functions

function plotall(image,Xhist,P,X,C,keepReprojection,keepAngle,keepBehind,...
    T, t_BA, sizes, anglex)

    cam = T*[-anglex*5 0 anglex*5; 0 0 0; 5 0 5; 1 1 1];

    % figure('name','11','units','normalized','outerposition',[0.1 0.1 0.85 0.8]);
    figure(11)
    set(gcf,'units','normalized','outerposition',[0.1 0.1 0.85 0.8]);
    % plot upper image with 3D correspondences
   
    subplot(5,8,[1:6,9:14])
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
    legend({'ReprojectionError is too big',...
        'Angle too small',...
        'outside viewing angle',...
        'newly triangulated',...
        'old 3D landmarks'},'FontSize',8)
    title('image with 3D correspondences and new kps')
    
    % bar diagram with sizes of different matrices
    subplot(5,8,[31 32 39 40])
    bar(sizes(:,1),sizes(:,2:end))
    legend({'#-landmarks','#-tracked keypoints','#-newly triangulated'},'FontSize',6)
    title('Array sizes over frames')
    
    % plot lower left as 2D point cloud
    subplot(5,8,[17:19,25:27,33:35])
    plot(Xhist(:,1),Xhist(:,3),'bx',X(:,1),X(:,3),'rx','linewidth',1);
    legend({'old landmarks','currently visible'},'FontSize',8)
    title('2D Pointcloud')
    xlabel('x')
    ylabel('z')
    axis equal
    
    % plot lower middle Camera Positions
    subplot(5,8,[20:22,28:30,36:38])
    cla;
    if size(t_BA,2)<21 
        plot([0 t_BA(1,:) T(1,4)],[0 t_BA(3,:) T(3,4)],'rx','linewidth',1)
        legend({'Camera BA'},'FontSize',5)
    else
        plot([t_BA(1,end-19:end) T(1,4)],[t_BA(3,end-19:end) T(3,4)],'rx',...
            [0 t_BA(1,1:end-20)],[0 t_BA(3,1:end-20)],'kx',...
            T(1,4),T(3,4),'bo','linewidth',1)
        legend({'cam BA','cam fixed'},'FontSize',8)
    end
    title('Camera Position')
    xlabel('x')
    ylabel('z')
    axis equal
    
    subplot(5,8,[7 8 15 16 23 24])
    cla;
    if size(t_BA)<21 
        plot(t_BA(1,:),t_BA(3,:),'.b','linewidth',3)
    else
        plot(t_BA(1,end-20:end),t_BA(3,end-20:end),'.b','linewidth',3)
    end
    hold on;
    plot(X(:,1),X(:,3),'rx','linewidth',1)
    plot(cam(1,:),cam(3,:),'g','linewidth',2)
    legend({'cam pos','landmarks','cam viewing angle'},'FontSize',8)
    title('last 20 cam pos and current Landmarks')
    axis equal;
    
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