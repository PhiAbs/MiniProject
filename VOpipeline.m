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

% parameters
bidirect_thresh = 0.3; % TODO  0.3: good
last_bootstrap_frame_index = 2;  %TODO
baseline_thresh = 0.1;
maxDistance_essential = 0.001;  % 0.1 is too big for parking!! 0.01 might work as well
maxNumTrials_Essential = 20000;
max_allowed_point_dist = 150;  %TODO  100: good 150: good especially for parking
minQuality_Harris = 0.01;  %TODO  0.1: good
harris_rejection_radius = 5; %TODO 10: good for kitti
p3p_pixel_thresh = 1;  % TODO 1: good. 5: not so good
p3p_num_iter = 5000;
BA_iter = 2;
num_BA_frames = 20;
reprojection_thresh = 30;  %15: good. 10000: not so good for kitti, good for parking


%% Bootstrapping

img0 = uint8(single(imread([path '/00/image_0/' sprintf('%06d.png',0)])));
img1 = uint8(single(imread([path '/00/image_0/' sprintf('%06d.png',1)])));

points = detectHarrisFeatures(img0,'MinQuality',0.1);
kps = points.Location;  %keypoint_start

pointTracker = vision.PointTracker('MaxBidirectionalError',0.3);
initialize(pointTracker, kps, img0);
setPoints(pointTracker,kps);
[kpl,keep,confidence] = pointTracker(img1);   % keypoints_latest
kps = kps(keep,:);
kpl = kpl(keep,:);
confidence = confidence(keep);

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
% showMatchedFeatures(img0,img1,kps,kpl)

[R_W_C,t_W_C] = relativeCameraPose(E,cameraParams,kps,kpl);
% transform to our system
R_W_C = R_W_C';
t_W_C = t_W_C';
T_WC = [R_W_C,t_W_C];

T_CW = [R_W_C',-R_W_C'*t_W_C];

%% triangulate Points
M = cameraMatrix(cameraParams,R_W_C', t_W_C'); 
M = (K*T_WC)';

S.P = kpl;
[S.X, reprojectionErrors] = triangulate( kps, kpl, (K*[eye(3) zeros(3,1)])', M);
% subplot(2,1,1)
% plot3(S.X(:,1),S.X(:,2),S.X(:,3),'*'); view(0,90);
% xlabel('x')
% ylabel('y')
% zlabel('z')
% grid on;

%% continious

keep = all((abs(S.X(:,1:2))<[anglex angley].*S.X(:,3)) & reprojectionErrors < reprojection_thresh ,2);
S.X = S.X(keep,:);
S.P = S.P(keep,:);
% extract new features in 2nd image
points = detectHarrisFeatures(img1,'MinQuality', minQuality_Harris);
kpl = points.Location;
keep = ~ismember(kpl,S.P,'rows');
S.C = kpl(keep,:);
S.T = T_CW(:) * ones(1,nnz(keep));
S.F = S.C;

img = img1;
T = T_CW;
clear img0 img1 reprojection_error F T_CW;

for i=2:last_frame
    
    % get new image
    img_prev = img;
    img = uint8(single(imread([path '/00/image_0/' sprintf('%06d.png',i)])));
    % track features into new image
    % initialize KLT trackers for continuous mode
    trackP = vision.PointTracker('MaxBidirectionalError', bidirect_thresh);
    initialize(trackP, S.P, img_prev);
    trackC = vision.PointTracker('MaxBidirectionalError', bidirect_thresh);
    initialize(trackC, S.C, img_prev);
    
    [points, keepP] = trackP(img);
    S.P = points(keepP,:);
    S.X = S.X(keepP,:);
    [points, keepC] = trackC(img);
    S.C = points(keepC,:);
    S.T = S.T(:,keepC);
    S.F = S.F(keepC,:);
    
    [R_WC, t_WC, keepP] = estimateWorldCameraPose(S.P,S.X,cameraParams,...
        'MaxNumTrials', p3p_num_iter, 'MaxReprojectionError', p3p_pixel_thresh);
    R_WC = R_WC';
    t_WC = t_WC';
    T_CW = [R_WC',-R_WC'*t_WC];
    nnz(keepP)/length(keepP);
    S.X = S.X(keepP,:);
    S.P = S.P(keepP,:);
%     plot(t_W_C(1),t_W_C(3),'o','linewidth',2)
%     hold on;

    % triangulate S.C and S.F
    Xnew = []; reprojectionErrors = [];
    for i=1: size(S.C,1)
        [new, reprojectionError] = triangulate( S.F(i,:), S.C(i,:), (K*reshape(S.T(:,1),3,4))', (K*T_CW)');
        Xnew = [Xnew; new];
        reprojectionErrors = [reprojectionErrors; reprojectionError];
    end

    keep = all((abs(Xnew(:,1:2)-t_WC(1:2))<[anglex angley].*(Xnew(:,3)-t_WC(3))) & reprojectionErrors < 2 ,2);   
    % plot for debugging and tuning
    plotall(img, S.X, S.P, Xnew, S.C, reprojectionErrors<2, ...
        all((Xnew > [-200 -100 0] & Xnew < [200 100 100]),2), t_WC)
%     pause(2);
    
    Xnew = Xnew(keep,:);
    S.X = [S.X; Xnew];
    S.P = [S.P; S.C(keep,:)];
    S.C = S.C(~keep,:);
    S.T = S.T(:,~keep);
    S.F = S.F(~keep,:);
    % plot3(Xnew(:,1),Xnew(:,2),Xnew(:,3),'*'); view(0,90);
    
    % extract new Keypints
    points = detectHarrisFeatures(img,'MinQuality', minQuality_Harris);
    kpl = points.Location;  %keypoint_latest
    keep = ~ismember(kpl,[S.P; S.C],'rows');
    S.C = [S.C; kpl(keep,:)];
    S.T = [S.T, T_CW(:) * ones(1,nnz(keep))];
    S.F = [S.F; kpl(keep,:)];
    
end



%% Functions

function plotall(image,X,P,Xnew,C,keepReprojection,keepGreater0,t_WC,sizes)

%     figure('name','11','units','normalized','outerposition',[0.1 0.1 0.85 0.8]);
    figure(11)
    set(gcf,'units','normalized','outerposition',[0.1 0.1 0.85 0.8]);
    % plot upper image with 3D correspondences
    subplot(3,3,1:6)
    hold off;
    imshow(image);
    hold on;
    plot(C(~keepReprojection,1),C(~keepReprojection,2),'bs','linewidth',1.5)
    plot(C(~keepGreater0,1),C(~keepGreater0,2),'cd','linewidth',1.5)
    plot(C(keepGreater0&keepReprojection,1),C(keepGreater0&keepReprojection,2),...
        'yo','linewidth',1.5)
    plot(P(:,1),P(:,2),'r*','linewidth',1.5)
    legend('ReprojectionError is too big',...
        'triangulated behind camera',...
        'newly triangulated',...
        'old 3D correspondences')
    title('image with 3D correspondences and new kps')
    
    
    % bar diagram with sizes of different matrices
    
    
    % plot lower left as 2D point cloud
    subplot(3,3,8)
    hold on;
    plot(Xnew(keepGreater0&keepReprojection,1),Xnew(keepGreater0&keepReprojection,3),...
        'bx','linewidth',1.5);
    plot(X(:,1),X(:,3),'rx','linewidth',1.5);
    title('2D Pointcloud')
    xlabel('x')
    ylabel('z')
    
    
    % plot Camera Positions
    subplot(3,3,9)
    hold on;
    plot(t_WC(1),t_WC(3),'rx','linewidth',3)
    title('Camera Position')
    xlabel('x')
    ylabel('z')
    axis([-10 10 0 inf])

end

