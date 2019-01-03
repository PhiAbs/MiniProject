%% Setup
clear; close all; clc;
ds = 0; % 0: KITTI, 1: Malaga, 2: parking

if ds == 0 || 1
    % good params for kitti and malaga
    num_first_image = 1; % nr 1 would refer to the first image in the folder
    bidirect_thresh = inf; % 0.3: good. larger number means more features (but worse quality)
    last_bootstrap_frame_index = 2; 
    baseline_thresh = 0.02; % larger number means less features (but better triangulation)
    maxDistance_essential = 0.1;  % 0.1 is too big for parking!! 0.01 might work as well
    maxNumTrials_Essential = 20000;
    max_allowed_point_dist = 80;  %100: good 150: good especially for parking
    harris_num_image_splits = 1;
    minQuality_Harris = 0.1;  %0.001: good. smaller number means more features!
    nonmax_suppression_radius = 15; % larger number means less features
    harris_rejection_radius = 15; %TODO: make it same as nonmax suppression radius? 10: good for kitti
    p3p_pixel_thresh = 2;  % 1: good. 5: not so good. larger number means more features, but worse quality
    p3p_num_iter = 100000;
    BA_iter = 2;
    num_BA_frames = 20;
    max_iter_BA = 100;
    num_fixed_frames_BA = 1;
    absoluteTolerance_BA = 0.001;
    reprojection_thresh = 10;  
    plot_stuff = false;
    enable_plotall = true;
    disp_stuff = false;
    enable_bootstrap = true;
end

if ds == 2
%     ok for parking
    bidirect_thresh = 0.2; % TODO  0.3: good. larger number means more features (but worse quality)
    last_bootstrap_frame_index = 2;  %TODO
    baseline_thresh = 0.1; % 0.05: good. larger number means less features (but better triangulation)
    maxDistance_essential = 0.01;  % 0.1 is too big for parking!! 0.01 might work as well
    maxNumTrials_Essential = 20000;
    max_allowed_point_dist = 120;  %TODO  100: good 150: good especially for parking
    harris_num_image_splits = 1;
    minQuality_Harris = 0.0001;  %TODO  0.001: good. smaller number means more features!
    nonmax_suppression_radius = 10;
    harris_rejection_radius = 10; %TODO 10: good for kitti
    p3p_pixel_thresh = 0.5;  % TODO 1: good. 5: not so good. larger number means more features, 
                             % but worse quality
    p3p_num_iter = 10000;
    BA_iter = 2;
    num_BA_frames = 20;
    max_iter_BA = 100;
    num_fixed_frames_BA = 1;
    absoluteTolerance_BA = 0.001;
    reprojection_thresh = 1;  
    plot_stuff = false;
    disp_stuff = true;
    enable_bootstrap = true;
end


if ds == 0
    path = '../datasets/kitti00/kitti';
    % need to set kitti_path to folder containing "00" and "poses"
    assert(exist('path', 'var') ~= 0);
    ground_truth = load([path '/poses/00.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
elseif ds == 1 
    path = '../datasets/malaga-urban-dataset-extract-07';
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
    path = '../datasets/parking';
    % Path containing images, depths and all...
    assert(exist('path', 'var') ~= 0);
    last_frame = 598;
    K = load([path '/K.txt']);
     
    ground_truth = load([path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
else
    assert(false);
end

% camera angles
anglex = K(1,3)/K(1,1);
angley = K(2,3)/K(2,2);

% add paths to subfolders
addpath('plot');
addpath('essentialMatrix');
addpath('p3p');
addpath('bundleAdjustment');

%% Bootstrap

% store first image
if ds == 1
    image = uint8(loadImage(ds, num_first_image, path, left_images));
else
    image = uint8(loadImage(ds, num_first_image-1, path));
end

% points = detectHarrisFeatures(image, 'MinQuality', minQuality_Harris);
points = harris_sector_wise(harris_num_image_splits, ...
    minQuality_Harris, nonmax_suppression_radius, image);

keypoints_start = points.Location;

if disp_stuff
    disp('Start Bootstrapping');
    disp(['extracted Harris Keypoints: ', num2str(points.Count)]);
end

% add pointtracker object for KLT
pointTracker = vision.PointTracker('MaxBidirectionalError', bidirect_thresh);
initialize(pointTracker, keypoints_start, image);

image_prev = image;

for i = 1:last_bootstrap_frame_index
    % iteratively add more images if neccessary
    if ds == 1
        image = uint8(loadImage(ds, i+num_first_image, path, left_images));
    else
        image = uint8(loadImage(ds, i+num_first_image-1, path));
    end
    
    % find matching keypoints in newest image using KLT
    [points, point_validity] = pointTracker(image);
    
    keypoints_latest = points(point_validity, :);
    keypoints_start = keypoints_start(point_validity, :);
    
    % Estimate the essential matrix E 
    cameraParams = cameraParameters('IntrinsicMatrix',K');
    [E, in_essential] = estimateEssentialMatrix(keypoints_start, ...
        keypoints_latest, cameraParams, 'MaxNumTrials', ...
        maxNumTrials_Essential, 'MaxDistance', maxDistance_essential);

    [Rots,u3] = decomposeEssentialMatrix(E);
    
    p1 = [keypoints_start'; ones(1, length(keypoints_start))];
    pend = [keypoints_latest'; ones(1, length(keypoints_latest))];
    
    [R_C2_W, t_C2_W] = disambiguateRelativePose(Rots,u3,p1,pend,K,K);

    M1 = K * eye(3,4);
    Mend = K * [R_C2_W, t_C2_W];
    [worldPoints, reprojectionErrors] = triangulate(p1(1:2, :)', ...
        pend(1:2, :)', M1', Mend');
    Points3D = [worldPoints'; ones(1, size(worldPoints, 1))];

%     remove points that have a large reprojection error
    keep_reprojected = (reprojectionErrors < reprojection_thresh);
    Points3D = Points3D(:, keep_reprojected);
    in_essential = in_essential(keep_reprojected);
    keypoints_start = keypoints_start(keep_reprojected, :);
    keypoints_latest = keypoints_latest(keep_reprojected, :);
    
%     T_W_C = [R_C2_W', -R_C2_W' * t_C2_W; [0, 0, 0, 1]];
%     T_World = [eye(3) ones(3,1); [0, 0, 0, 1]];
%     [keep_triang, worldPoints] = triangulatePoints(keypoints_latest', keypoints_start', ...
%         T_W_C,T_World(:)*ones(1,size(keypoints_latest,1)), ...
%         ones(1,size(keypoints_latest,1)), K, baseline_thresh, reprojection_thresh);
%     worldPoints = worldPoints(:, keep_triang);
%     Points3D = [worldPoints'; ones(1, size(worldPoints, 1))];
%     
%     in_essential = in_essential(keep_triang);
%     keypoints_start = keypoints_start(keep_triang, :);
%     keypoints_latest = keypoints_latest(keep_triang, :);

    
    if disp_stuff
        disp([' number of removed keypoints: ' num2str(sum(~keep_reprojected))]);
    end
    
    % only keep valid points in point tracker
    setPoints(pointTracker, keypoints_latest);
    
    % Plot stuff
    if plot_stuff
        plotBootstrap(image_prev, image, keypoints_start, keypoints_latest, ...
            Points3D, R_C2_W, t_C2_W);
    end
    
    if disp_stuff
        disp(['Iteration ', num2str(i)]);
        disp(['Keypoints in Image nr', num2str(i+1), ': ', ...
            num2str(length(keypoints_start))]);
    end
    
    % Plot reprojected Points
%     if plot_stuff
%         P_reprojected = reprojectPoints(Points3D(1:3,:)',K^(-1)*Mend, K);
%         error = max(abs(sum(P_reprojected(in_essential,:)-keypoints_latest(in_essential,:),2)));
%/nnz(in_essential);
%         fprintf(['error: ',num2str(error),'\t with ',num2str(nnz(in_essential)),' inlier. \n']);
%         figure(83)
%         clf;
%         imshow(image)
%         hold on;
%         plot(P_reprojected(in_essential,1),P_reprojected(in_essential,2),'bx','linewidth',1.5)
%         hold on;
%         plot(keypoints_latest(in_essential,1),keypoints_latest(in_essential,2),'ro','linewidth',1.5)
%         hold on;
%         pause(0.01);
%     end
    
    image_prev = image;
end

% When keyframe for bootstrapping has been found, we redo the calculation
% for the essential matrix only with inliers!
keypoints_start = keypoints_start(in_essential, :);
keypoints_latest = keypoints_latest(in_essential, :);

p1 = [keypoints_start'; ones(1, length(keypoints_start))];
pend = [keypoints_latest'; ones(1, length(keypoints_latest))];

cameraParams = cameraParameters('IntrinsicMatrix',K');
[E, in_essential] = estimateEssentialMatrix(keypoints_start, ...
    keypoints_latest, cameraParams, 'MaxNumTrials', ...
        maxNumTrials_Essential, 'MaxDistance', maxDistance_essential);

[Rots,u3] = decomposeEssentialMatrix(E);

[R_C2_W, t_C2_W] = disambiguateRelativePose(Rots,u3,p1,pend,K,K);

M1 = K * eye(3,4);
Mend = K * [R_C2_W, t_C2_W];
[worldPoints, reprojectionErrors] = triangulate(p1(1:2, :)', ...
    pend(1:2, :)', M1', Mend');
Points3D = [worldPoints'; ones(1, size(worldPoints, 1))];

T=[R_C2_W, t_C2_W;0 0 0 1]^(-1);
[error, inlier_reprojection] = estimate_projection_error( ...
    keypoints_latest, Points3D(1:3,:)', T, K, 1);
nnz(~inlier_reprojection)

% remove points that have a large reprojection error
keep_reprojected = (reprojectionErrors < reprojection_thresh);
Points3D = Points3D(:, keep_reprojected);
keypoints_start = keypoints_start(keep_reprojected, :);
keypoints_latest = keypoints_latest(keep_reprojected, :);
% 
%     T_W_C = [R_C2_W', -R_C2_W' * t_C2_W; [0, 0, 0, 1]];
%     T_World = [eye(3) ones(3,1); [0, 0, 0, 1]];
%     [keep_triang, worldPoints] = triangulatePoints(keypoints_latest', keypoints_start', ...
%         T_W_C,T_World(:)*ones(1,size(keypoints_latest,1)), ...
%         ones(1,size(keypoints_latest,1)), K, baseline_thresh, reprojection_thresh);
%     worldPoints = worldPoints(:, keep_triang);
%     Points3D = [worldPoints'; ones(1, size(worldPoints, 1))];
%     
%     in_essential = in_essential(keep_triang);
%     keypoints_start = keypoints_start(keep_triang, :);
%     keypoints_latest = keypoints_latest(keep_triang, :);

% Plot stuff
if plot_stuff
    plotBootstrap(image_prev, image, keypoints_start, keypoints_latest, ...
        Points3D, R_C2_W, t_C2_W);
end

if disp_stuff
    disp(['Iteration ', num2str(i)]);
    disp(['Keypoints in Image nr', num2str(i+1), ': ', ...
        num2str(length(keypoints_start))]);
end

% Extract Harris Features from last bootstrapping image
points = harris_sector_wise(harris_num_image_splits, ...
    minQuality_Harris, nonmax_suppression_radius, image);

kp_new_latest_frame = points.Location;

%

% remove points that lie behind the first camera or that are far away
T_C_W = inv(T);
Points3D_cam = T_C_W(1:3,:) * Points3D;
within_min = Points3D_cam(3, :) > 0;
within_max = Points3D_cam(3,:) < max_allowed_point_dist;
keep_camera_angle = all((abs(Points3D_cam(1:2, :))<[anglex; angley].*(Points3D_cam(3, :))),1);
Points3D = Points3D(:, (within_min & within_max & keep_camera_angle));
keypoints_latest = keypoints_latest((within_min & within_max & keep_camera_angle), :);

% new Harris keypoints must NOT already be tracked points!! 
%These points are then candidate Keypoints for the continuous mode.
kp_new_sorted_out = checkIfKeypointIsNew(kp_new_latest_frame', ...
    keypoints_latest', harris_rejection_radius);

% Create Structs for continuous operation
% Struct S contains
prev_img = image;
S = struct;
S.P = keypoints_latest';
S.X = Points3D(1:3,:);
S.C = kp_new_sorted_out;
S.F = S.C;
T = [R_C2_W', -R_C2_W'*t_C2_W; 0,0,0,1];
S.T = T(:)* ones(1, size(S.C, 2));
S.Frames = last_bootstrap_frame_index * ones(1, size(S.C, 2));

% store data for bundle adjustment
cameraPoses_all = table;
cameraPoses_all.ViewId(1) = uint32(last_bootstrap_frame_index);
cameraPoses_all.Orientation{1} = T(1:3, 1:3);
cameraPoses_all.Location{1} = T(1:3, end)';

S.P_BA(:, num_BA_frames, 1) = S.P(1,:)';
S.P_BA(:, num_BA_frames, 2) = S.P(2,:)';
S.X_BA = S.X'; 
keep_P_BA = ones(size(S.P, 2), 1);
S.C_trace_tracker(:, num_BA_frames, 1) = S.C(1,:)';
S.C_trace_tracker(:, num_BA_frames, 2) = S.C(2,:)';

% initialize KLT trackers for continuous mode
tracker_P = vision.PointTracker('MaxBidirectionalError',bidirect_thresh,'MaxIterations',50);
initialize(tracker_P, S.P', image);
tracker_C = vision.PointTracker('MaxBidirectionalError',bidirect_thresh);
initialize(tracker_C, S.C', image);

if plot_stuff
    figure(21);
    plot3(0,0,0,'x');
    plotContinuous(prev_img, S.X, S.X, S.P, S.C, T, K);
end
%% Continuous operation

sizes = [last_bootstrap_frame_index size(S.P,2) 0 size(S.C,2) nnz(~keep_camera_angle) nnz(~keep_reprojected)];

range = (last_bootstrap_frame_index+1):last_frame;
for i = range
    fprintf('\n\nProcessing frame %d\n=====================\n', i+num_first_image-1);
    if ds == 0
        image = imread([path '/00/image_0/' sprintf('%06d.png',i+num_first_image)]);
    elseif ds == 1
        image = rgb2gray(imread([path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));
    elseif ds == 2
        image = im2uint8(rgb2gray(imread([path ...
            sprintf('/images/img_%05d.png',i+num_first_image-1)])));
    else
        assert(false);
    end
    % Makes sure that plots refresh.    
    pause(0.01);
    
    if disp_stuff
        disp('run KLT on S.P');
    end
    
    % tic
%     do KLT tracking for keypoints
    [points_KLT_P, keep_P] = tracker_P(image);
    % toc
    Pold = points_KLT_P;
    keep_KLT_Pold = keep_P;
    
    S.P = points_KLT_P(keep_P,:)';
    S.X = S.X(:, keep_P);
    
    % update data for bootstrap: update only the points that can still be
    % tracked!
    keep_Index = find(keep_P_BA);
    keep_Index = keep_Index(keep_P);
    keep_P_BA = false(size(keep_P_BA));
    keep_P_BA(keep_Index) = 1;
    S.P_BA(:, 1, :) = [];
    S.P_BA(keep_P_BA, end+1, 1) = S.P(1,:);
    S.P_BA(keep_P_BA, end, 2) = S.P(2,:);
    
%     Estimate new camera pose using p3p
    if disp_stuff
        disp('run p3p with Ransac on S.P and S.X');
    end
    
    tic
    [T, S.P, S.X, best_inlier_mask] = ... % T=T_W_C;
        ransacLocalization(S.P, S.X, K, p3p_pixel_thresh, p3p_num_iter);
    toc
    
    keep_p3p_Pklt = best_inlier_mask;
    
    % update data for bootstrap: update only the points that can still be
    % tracked!
    keep_Index = find(keep_P_BA);
    keep_Index = keep_Index(best_inlier_mask);
    keep_P_BA = false(size(keep_P_BA));
    keep_P_BA(keep_Index) = 1;
    S.P_BA(~keep_P_BA, end, 1) = 0;
    S.P_BA(~keep_P_BA, end, 2) = 0;
    
%     Track 
    if disp_stuff
        disp('run KLT on S.C');
    end
    
    % tic
    [points_KLT_C, keep_C] = tracker_C(image);
    % toc;
    
    Frames_old = S.Frames;
    
    S.C = points_KLT_C(keep_C,:)';
    S.T = S.T(:, keep_C);
    S.F = S.F(:, keep_C);
    S.Frames = S.Frames(keep_C);  
    S.C_trace_tracker(:, 1, :) = [];
    S.C_trace_tracker(~keep_C, :, :) = [];
    S.C_trace_tracker(:, end+1, :) = points_KLT_C(keep_C,:);
        
    % Triangulate new points
    if disp_stuff
        disp('try to triangulate S.C and S.F');
    end
    
    % tic
%     [keep_triang, keep_reprojected, X_new] = triangulatePoints(S.C, S.F, T, S.T, ...
%         S.Frames, K, baseline_thresh, reprojection_thresh);
    [keep_triang, X_new] = triangulatePoints(S.C, S.F, T, S.T, ...
        S.Frames, K, baseline_thresh, reprojection_thresh);
    X_new = X_new(:, keep_triang);
    % toc
    C_plotall = S.C;
    
    if ~isempty(X_new)
        % delete points that are far away or that lie behind the camera
        [S, keep_P_BA, X_new,keep_camera_angle] = pointSanityCheck(S, keep_P_BA, T, ...
            X_new, keep_triang, ...
            max_allowed_point_dist, anglex, angley);
    else
        keep_camera_angle = [];
    end
    
        % add newest camera pose to all camera poses
    cameraPoses_new = table;
    cameraPoses_new.ViewId(1) = uint32(i);
    cameraPoses_new.Orientation{1} = T(1:3, 1:3);
    cameraPoses_new.Location{1} = T(1:3, end)';
    cameraPoses_all = [cameraPoses_all; cameraPoses_new];
    
    if enable_plotall
        sizes = [sizes; ...
            i size(S.P,2)-size(X_new,2) size(X_new,2) size(C_plotall,2) nnz(~keep_camera_angle) nnz(~keep_triang)];
        if length(sizes) >= 10
            sizes(1,:) = [];
        end
        t_WC_BA = [];
        if i > num_BA_frames+1
            t_WC_BA = cameraPoses_all.Location{end-num_BA_frames};
        end
        plotall(image,S.X',S.P',X_new',C_plotall',keep_triang,keep_camera_angle,T(1:3,4),sizes,t_WC_BA,...
            Pold,keep_KLT_Pold,keep_p3p_Pklt)
    end
    
    % bundle adjustment
    if BA_iter == num_BA_frames && enable_bootstrap
        % delete all rows in the BA matrices which only contain zeros or
        % only one valid point (they cannot be used for BA

        valid_points = S.P_BA(:,:,1) > 0;
        untracked_landmark_idx = find(sum(valid_points, 2) == 0);
        S.P_BA(untracked_landmark_idx, :, :) = [];
        S.X_BA(untracked_landmark_idx, :) = [];
        keep_P_BA(untracked_landmark_idx) = [];
        
        [S, keep_P_BA, T, cameraPoses_all] = ...
            bundle_adjustment(S, cameraPoses_all, i, num_BA_frames, ...
            keep_P_BA, K, max_iter_BA, num_fixed_frames_BA, absoluteTolerance_BA);
        
        if plot_stuff  
            disp('plot bundle adjustment')
            % tic
            plotBundleAdjustment(cameraPoses_all)
            % toc
        end
        
%         BA_iter = 11; % use 2 to make sure that the last camera 
%                       % from the last bundle adjustment is used again!
        
    else
        BA_iter = BA_iter + 1;
    end
    
    
    % extract new keypoints
    if disp_stuff
        disp('extract Keypoint');
    end
    
    % tic
    points = harris_sector_wise(harris_num_image_splits, ...
        minQuality_Harris, nonmax_suppression_radius, image);

    kp_new_latest_frame = points.Location;
    % toc
    if disp_stuff
        disp('check if extracted Keypoints are new')
    end
    % tic
    kp_new_sorted_out = checkIfKeypointIsNew(kp_new_latest_frame', ...
        [S.P, S.C], harris_rejection_radius);  
    % toc
    
    S.C = [S.C, kp_new_sorted_out];
    S.F = [S.F, kp_new_sorted_out];
    S.T = [S.T, T(:)*ones(1, size(kp_new_sorted_out, 2))]; 
    S.Frames = [S.Frames, i*ones(1, size(kp_new_sorted_out, 2))]; 
    S.C_trace_tracker(end+1:end+size(kp_new_sorted_out, 2), end, :) = kp_new_sorted_out';
    
    setPoints(tracker_P, S.P');
    setPoints(tracker_C, S.C'); 
    
    prev_img = image;
    
    if disp_stuff
        disp(['Number of 3D points: ' num2str(size(S.X,2))]);
        disp(['Number of new keypoints: ' num2str(size(kp_new_sorted_out,2))]);
        disp(['Number of candidate keypoints: ' num2str(size(S.C, 2))]);
    end
    
    if plot_stuff
        disp('plot continuous')
        % tic
        plotContinuous(image, X_new, S.X, S.P, S.C, T, K);
        % toc

        disp(['Frames still in S.C' num2str(unique(S.Frames))]);
        plotBarDiagram(S.Frames,Frames_old);
    end
end