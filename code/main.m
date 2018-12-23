%% Setup
clear; close all; clc;
ds = 0; % 0: KITTI, 1: Malaga, 2: parking

bidirect_thresh = 0.3;
last_bootstrap_frame_index = 10;
baseline_thresh = 0.1;
max_allowed_point_dist = 100;
harris_rejection_radius = 2;
p3p_pixel_thresh = 5;
p3p_num_iter = 5000;
BA_iter = 1;
num_BA_frames = 10;

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

% add paths to subfolders
addpath('helpers');
addpath('klt');
addpath('harris');
addpath('plot');
addpath('essentialMatrix');
addpath('fundamentalMatrix');
addpath('p3p');
addpath('bundleAdjustment');

% Bootstrap

% store first image
if ds == 1
    image = uint8(loadImage(ds, 1, path, left_images));
else
    image = uint8(loadImage(ds, 1, path));
end

% number of keypoints we want to extract from first imaeg
% num_keypoints = 200;
% keypoints get stored as cell array with arrays Nx2, [col, row]
% keypoints_start = extractHarrisKeypoints(imgb{1}, num_keypoints);
points = detectHarrisFeatures(image);
keypoints_start = points.Location;

disp('Start Bootstrapping');
disp(['extracted Harris Keypoints: ', num2str(points.Count)]);
% disp(['extracted Harris Keypoints: ', num2str(length(points))]);

% add pointtracker object for KLT
pointTracker = vision.PointTracker('MaxBidirectionalError', bidirect_thresh);
initialize(pointTracker, keypoints_start, image);

image_prev = image;

for i = 1:last_bootstrap_frame_index
    % iteratively add more images if neccessary
    if ds == 1
        image = uint8(loadImage(ds, i+1, path, left_images));
    else
        image = uint8(loadImage(ds, i+1, path));
    end
    
    % find matching keypoints in newest image using KLT
    [points, point_validity] = pointTracker(image);
    
    keypoints_latest = points(point_validity, :);
    keypoints_start = keypoints_start(point_validity, :);
    
    % Estimate the essential matrix E 
    p1 = [keypoints_start'; ones(1, length(keypoints_start))];
    pend = [keypoints_latest'; ones(1, length(keypoints_latest))];

    [E, in_essential] = estimateEssentialMatrix(p1, pend, K, K);

    [Rots,u3] = decomposeEssentialMatrix(E);

    [R_C2_W, t_C2_W] = disambiguateRelativePose(Rots,u3,p1,pend,K,K);

    M1 = K * eye(3,4);
    Mend = K * [R_C2_W, t_C2_W];
    Points3D = linearTriangulation(p1, pend, M1, Mend);

    % only keep valid points in point tracker
    setPoints(pointTracker, keypoints_latest);
    
    % Plot stuff
    plotBootstrap(image_prev, image, keypoints_start, keypoints_latest, Points3D, R_C2_W, t_C2_W);
    
    disp(['Iteration ', num2str(i)]);
    disp(['Keypoints in Image nr', num2str(i+1), ': ', num2str(length(keypoints_start))]);

    % check if camera moved far enough to finish bootstrapping
    avg_depth = mean(Points3D(3,:));
    baseline_dist = norm(t_C2_W);
    
    disp(['Ratio baseline_dist / avg_depth: ', ...
        num2str((baseline_dist / avg_depth)), ...
        ' must be > ', num2str(baseline_thresh)]);

    if (baseline_dist / avg_depth) > baseline_thresh
        % leave bootstrapping if keyframes are enough far apart
        last_bootstrap_frame_index = i;
        break;
    end
    
    image_prev = image;
end

% When keyframe for bootstrapping has been found, we redo the calculation
% for the essential matrix only with inliers!
keypoints_start = keypoints_start(in_essential, :);
keypoints_latest = keypoints_latest(in_essential, :);

p1 = [keypoints_start'; ones(1, length(keypoints_start))];
pend = [keypoints_latest'; ones(1, length(keypoints_latest))];

[E, in_essential] = estimateEssentialMatrix(p1, pend, K, K);

[Rots,u3] = decomposeEssentialMatrix(E);

[R_C2_W, t_C2_W] = disambiguateRelativePose(Rots,u3,p1,pend,K,K);

M1 = K * eye(3,4);
Mend = K * [R_C2_W, t_C2_W];
Points3D = linearTriangulation(p1, pend, M1, Mend);

% Plot stuff
plotBootstrap(image_prev, image, keypoints_start, keypoints_latest, Points3D, R_C2_W, t_C2_W);

disp(['Iteration ', num2str(i)]);
disp(['Keypoints in Image nr', num2str(i+1), ': ', num2str(length(keypoints_start))]);

% Extract Harris Features from last bootstrapping image
% kp_new_latest_frame = extractHarrisKeypoints(image, num_keypoints);
points = detectHarrisFeatures(image);
kp_new_latest_frame = points.Location;

%%

% remove points that lie behind the first camera or that are far away
within_min = Points3D(3, :) > 0;
within_max = Points3D(3,:) < max_allowed_point_dist;
Points3D = Points3D(:, (within_min & within_max));
keypoints_latest = keypoints_latest((within_min & within_max), :);

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


% initialize KLT trackers for continuous mode
tracker_P = vision.PointTracker('MaxBidirectionalError',bidirect_thresh);
initialize(tracker_P, S.P', image);
tracker_C = vision.PointTracker('MaxBidirectionalError',bidirect_thresh);
initialize(tracker_C, S.C', image);

figure(21);
plot3(0,0,0,'x');
plotContinuous(prev_img, S.X, S.P, S.C, T);

%% Continuous operation
% TODO: 
% why does the garage set give bad camera poses??? WTF
% reproject triangulated points and only keep those that have a small error
% bundle adjustment

range = (last_bootstrap_frame_index+1):last_frame;
for i = range
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    if ds == 0
        image = imread([path '/00/image_0/' sprintf('%06d.png',i)]);
    elseif ds == 1
        image = rgb2gray(imread([path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));
    elseif ds == 2
        image = im2uint8(rgb2gray(imread([path ...
            sprintf('/images/img_%05d.png',i)])));
    else
        assert(false);
    end
    % Makes sure that plots refresh.    
    pause(0.01);
    
    disp('run KLT on S.P');
    tic
%     [keep_P, P_delta] = runKLTContinuous(S.P, image, prev_img);
    % do KLT tracking for keypoints
    [points_KLT_P, keep_P] = tracker_P(image);
    toc
    
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
    disp('run p3p with Ransac on S.P and S.X');
    tic
    [T, S.P, S.X, best_inlier_mask] = ...
        ransacLocalization(S.P, S.X, K, p3p_pixel_thresh, p3p_num_iter);
    toc
    
    % update data for bootstrap: update only the points that can still be
    % tracked!
    keep_Index = find(keep_P_BA);
    keep_Index = keep_Index(best_inlier_mask);
    keep_P_BA = false(size(keep_P_BA));
    keep_P_BA(keep_Index) = 1;
    S.P_BA(~keep_P_BA, end, 1) = 0;
    S.P_BA(~keep_P_BA, end, 2) = 0;
    
%     Track 
    disp('run KLT on S.C');
    tic
%     [keep_C, C_delta] = runKLTContinuous(S.C, image, prev_img);
    [points_KLT_C, keep_C] = tracker_C(image);
    toc;
    
    Frames_old = S.Frames;
    
    S.C = points_KLT_C(keep_C,:)';
    S.T = S.T(:, keep_C);
    S.F = S.F(:, keep_C);
    S.Frames = S.Frames(keep_C);  
    
    % Triangulate new points
    disp('try to triangulate S.C and S.F');
    tic
    [keep_triang, X_new] = triangulatePoints(S.C, S.F, T, S.T, ...
        S.Frames, K, baseline_thresh);
    toc
    
    if ~isempty(X_new)
        % delete points that are far away or that lie behind the camera
        [S.X, S.P, S.C, S.F, S.T, S.Frames, S.P_BA, S.X_BA, keep_P_BA, X_new] = ...
            pointSanityCheck(S.X, S.P, S.C, S.F, S.T, S.Frames, S.P_BA, S.X_BA, ...
            keep_P_BA, T, K, X_new, keep_triang, max_allowed_point_dist);
    end
        
    % extract new keypoints
    disp('extract Keypoint')
    tic
%     num_keypoints = 100;
%     kp_new_latest_frame = extractHarrisKeypoints(image, num_keypoints);
    points = detectHarrisFeatures(image);
    kp_new_latest_frame = points.Location;
    toc
    disp('check if extracted Keypoints are new')
    tic
    kp_new_sorted_out = checkIfKeypointIsNew(kp_new_latest_frame', ...
        [S.P, S.C], harris_rejection_radius);  
    toc
    
    S.C = [S.C, kp_new_sorted_out];
    S.F = [S.F, kp_new_sorted_out];
    S.T = [S.T, T(:)*ones(1, size(kp_new_sorted_out, 2))]; 
    S.Frames = [S.Frames, i*ones(1, size(kp_new_sorted_out, 2))]; 
    setPoints(tracker_P, S.P');
    setPoints(tracker_C, S.C');
    
    prev_img = image;
    
    % add newest camera pose to all camera poses
    cameraPoses_all.ViewId(i-last_bootstrap_frame_index+1) = uint32(i);
    cameraPoses_all.Orientation{i-last_bootstrap_frame_index+1} = T(1:3, 1:3);
    cameraPoses_all.Location{i-last_bootstrap_frame_index+1} = T(1:3, end)';
    
    % bundle adjustment
    if BA_iter == num_BA_frames
        [S.T, S.X, T, cameraPoses_all] = ...
            bundle_adjustment(S, cameraPoses_all, i, num_BA_frames, keep_P_BA, K);
        plotBundleAdjustment(cameraPoses_all)
        BA_iter = 1;
    else
        BA_iter = BA_iter + 1;
    end

    disp(['Number of 3D points:' num2str(size(S.X,2))]);
    disp(['Number of new keypoints:' num2str(size(kp_new_sorted_out,2))]);
    disp(['Number of candidate keypoints:' num2str(size(S.C, 2))]);
    
    plotContinuous(image, X_new, S.P, S.C, T);
    
%     disp(['Frames still in S.C' num2str(unique(S.Frames))];

    plotBarDiagram(S.Frames,Frames_old);
end