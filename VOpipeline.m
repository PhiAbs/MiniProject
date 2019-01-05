%% initialize VO Pipeline with Kitty Dataset
clear all; close all; clc;

ds = 1; % 0: KITTI, 1: Malaga, 2: parking

%% define Path and load parameters
% path = 'datasets/kitti00/kitti';
% % need to set kitti_path to folder containing "00" and "poses"
% assert(exist('path', 'var') ~= 0);
% ground_truth = load([path '/poses/00.txt']);
% ground_truth = ground_truth(:, [end-8 end]);
% last_frame = 4540;
% % K = load([path '/00/K.txt']);
%     K = [7.188560000000e+02 0 6.071928000000e+02
%         0 7.188560000000e+02 1.852157000000e+02
%         0 0 1];
% cameraParams = cameraParameters('IntrinsicMatrix',K');

if ds == 0
    path = 'datasets/kitti00/kitti';
    % need to set kitti_path to folder containing "00" and "poses"
    assert(exist('path', 'var') ~= 0);
    ground_truth = load([path '/poses/00.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
elseif ds == 1 
    path = 'datasets/malaga-urban-dataset-extract-07';
    % Path containing the many files of Malaga 7.
    assert(exist('path', 'var') ~= 0);
    images = dir([path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif ds == 2
    path = 'datasets/parking';
    % Path containing images, depths and all...
    assert(exist('path', 'var') ~= 0);
    last_frame = 598;
    K = load([path '/K.txt']);
     
    ground_truth = load([path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
else
    assert(false);
end

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
minQuality_Harris = 0.001;  %TODO  0.1: good
p3p_pixel_thresh = 1;  % TODO 1: good. 5: not so good
p3p_num_iter = 10000;
reprojection_thresh = 1;  %15: good. 10000: not so good for kitti, good for parking
reprojection_thresh_p3p = 4;
triangAngleThres = 0.01;
nonmax_suppression_radius = 10;
harris_rejection_radius = 10; %TODO 10: good for kitti
BA_iter = 2; 
num_BA_frames = 10;
max_iter_BA = 100;
num_fixed_frames_BA = 1;
absoluteTolerance_BA = 0.001;
enable_BA = true;
enable_plot = false;

% % parameters
% bidirect_thresh = 3; % TODO  0.3: good
% maxDistance_essential = 0.01;  % 0.1 is too big for parking!! 0.01 might work as well
% maxNumTrials_Essential = 20000;
% minQuality_Harris = 0.001;  %TODO  0.1: good
% p3p_pixel_thresh = 1;  % TODO 1: good. 5: not so good
% p3p_num_iter = 10000;
% reprojection_thresh = 1;  %15: good. 10000: not so good for kitti, good for parking
% reprojection_thresh_p3p = 4;
% triangAngleThres = 0.01;
% nonmax_suppression_radius = 10;
% harris_rejection_radius = 10; %TODO 10: good for kitti
% BA_iter = 2; 
% num_BA_frames = 10;
% max_iter_BA = 100;
% num_fixed_frames_BA = 1;
% absoluteTolerance_BA = 0.001;
% enable_BA = true;
% enable_plot = false;


%% Bootstrapping

% store first image
if ds == 1
    img0 = uint8(loadImage(ds, 1, path, left_images));
    img1 = uint8(loadImage(ds, 2, path, left_images));
    img2 = uint8(loadImage(ds, 3, path, left_images));
else
    img0 = uint8(loadImage(ds, 0, path, path));
    img1 = uint8(loadImage(ds, 1, path, path));
    img2 = uint8(loadImage(ds, 2, path, path));
end

% img0 = uint8(single(imread([path '/00/image_0/' sprintf('%06d.png',0)])));
% img1 = uint8(single(imread([path '/00/image_0/' sprintf('%06d.png',1)])));
% img2 = uint8(single(imread([path '/00/image_0/' sprintf('%06d.png',2)])));

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

cam_x = [];
cam_z = [];

%% start continuous

for i=3:last_frame
    
    img_prev = img;
    
    % get new image
    if ds == 0
        img = uint8(single(imread([path '/00/image_0/' ...
            sprintf('%06d.png',i)])));
    elseif ds == 1
        img = uint8(single(rgb2gray(imread([path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i+1).name]))));
    elseif ds == 2
        img = uint8(single(im2uint8(rgb2gray(imread([path ...
            sprintf('/images/img_%05d.png',i)])))));
    else
        assert(false);
    end
    
    % get new image
%     img_prev = img;
%     img = uint8(single(imread([path '/00/image_0/' sprintf('%06d.png',i)])));
    
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
%     S.P_BA(~keep_P_BA, :, :) = [];
%     S.X_BA(~keep_P_BA, :, :) = [];
%     keep_P_BA(~keep_P_BA) = [];
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
%     plotall(img, S.X, S.P, Xnew, S.C(NonLms,:), reprojectionErrors < reprojection_thresh, ...
%         triangulationAngles > triangAngleThres,...
%         all((abs(Xnew_cam(1:2, :))<[anglex; angley].*Xnew_cam(3, :))',2),...
%         t_WC, cameraPoses_all)%, sizes)
    
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

function plotall(image,X,P,Xnew,C,keepReprojection,keepAngle,keepBehind,...
    t_WC, cameraPoses_all)

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
%     hold on;
%     plot(t_WC(1,:),t_WC(3,:),'rx','linewidth',1)
%     title('Camera Position')
%     xlabel('x')
%     ylabel('z')
%     axis equal
cla;
for i = 1:size(cameraPoses_all, 1)
    cam_x(i) = cameraPoses_all.Location{i}(1);
    cam_z(i)= cameraPoses_all.Location{i}(3);
end
%     plot(cameraPoses_all.Location{i}(1), cameraPoses_all.Location{i}(3), 'Marker', 'x');
plot(cam_x, cam_z, 'rx');
    xlabel('x')
    ylabel('z')
axis equal;
hold on;
title('Camera Position')

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