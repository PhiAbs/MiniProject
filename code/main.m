%% Setup
clear; close all; clc;
ds = 2; % 0: KITTI, 1: Malaga, 2: parking

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


%% Bootstrap

% Use more than two images to keep more keypoints
last_bootstrap_frame_index = 20;
imgb = cell(1, 1);

% store first image
if ds == 1
    imgb{1} = loadImage(ds, 1, path, left_images);
else
    imgb{1} = loadImage(ds, 1, path);
end
% number of keypoints we want to extract from first imaeg
num_keypoints = 200;

% keypoints get stored as cell array with arrays Nx2, [col, row]
kp_m = cell(1, 1);
kp_m{1} = extractHarrisKeypoints(imgb{1}, num_keypoints);

disp('Start Bootstrapping');
disp(['extracted Harris Keypoints: ', num2str(num_keypoints)]);
    
% for the first k images, do only KLT. The fundamental matrix is only
% calculated afterwards (avoids wrong matrices)
% TODO: this first loop might also be skipped (adjust the start index of
% the next loop in this case)

k_max = 8;
for k = 1:k_max-1
    % iteratively add more images if neccessary
    if ds == 1
        imgb{k+1} = loadImage(ds, k+1, path, left_images);
    else
        imgb{k+1} = loadImage(ds, k+1, path);
    end
    
    kp_m = runKLT(kp_m, imgb, k);
    
    figure(21)
    imshow(uint8(imgb{end}));
    hold on;
    plot(kp_m{end}(:,1)', kp_m{end}(:,2)', 'rx', 'Linewidth', 2);
    plotMatches(1:length(kp_m{1}), kp_m{1}, kp_m{end});
    
    disp(['Iteration ', num2str(k)]);
    disp(['Keypoints in Image nr ', num2str(k+1), ': ', num2str(length(kp_m{1}))]);
end

for i = k_max:last_bootstrap_frame_index
    % iteratively add more images if neccessary
    if ds == 1
        imgb{i+1} = loadImage(ds, i+1, path, left_images);
    else
        imgb{i+1} = loadImage(ds, i+1, path);
    end
    
    % find matching keypoints in second image using lucas-kanade-tracker. Code
    % from solution for exercise 8
    kp_m = runKLT(kp_m, imgb, i);

    % Estimate the essential matrix E 
    p1 = [kp_m{1}'; ones(1, length(kp_m{1}))];
    pend = [kp_m{end}'; ones(1, length(kp_m{end}))];

    [E, in_essential] = estimateEssentialMatrix(p1, pend, K, K);

    % TODO: check if det is controlled correctly????
    [Rots,u3] = decomposeEssentialMatrix(E);

    [R_C2_W, t_C2_W] = disambiguateRelativePose(Rots,u3,p1,pend,K,K);

    M1 = K * eye(3,4);
    Mend = K * [R_C2_W, t_C2_W];
    Points3D = linearTriangulation(p1(:, in_essential),pend(:, in_essential),M1,Mend);
    
    % store points in our main cell array
    kp_m{1} = p1(1:2, in_essential)';
    kp_m{end} = pend(1:2, in_essential)';

    % remove points that lie behind the first camera or that are far away
    max_thresh = 120;
    within_min = Points3D(3, :) > 0;
    within_max = Points3D(3,:) < max_thresh;
    Points3D = Points3D(:, (within_min & within_max));
    
    % Plot stuff
    plotBootstrap(imgb, kp_m, Points3D, R_C2_W, t_C2_W);
    
    disp(['Iteration ', num2str(i)]);
    disp(['Keypoints in Image nr', num2str(i+1), ': ', num2str(length(kp_m{1}))]);

    % check if camera moved far enough to finish bootstrapping
    avg_depth = mean(Points3D(3,:));
    baseline_dist = norm(t_C2_W);
    keyframe_selection_thresh = 0.2;
    
    disp(['Ratio baseline_dist / avg_depth: ', ...
        num2str((baseline_dist / avg_depth)), ...
        ' must be > ', num2str(keyframe_selection_thresh)]);

    if (baseline_dist / avg_depth) > keyframe_selection_thresh
        % leave bootstrapping if keyframes are enough far apart
        last_bootstrap_frame_index = i;
        break;
    end
end

% Extract Harris Features from last bootstrapping image
kp_new_latest_frame = extractHarrisKeypoints(imgb{end}, num_keypoints);

%%
% new Harris keypoints must NOT already be tracked points in kp_m!! 
%These points are then candidate Keypoints for the continuous mode.
rejection_radius = 5;
kp_new_sorted_out = checkIfKeypointIsNew(kp_new_latest_frame', ...
    kp_m{end}', rejection_radius);

% Create Structs for continuous operation
% Struct S contains
prev_img = imgb{end};
S = struct;
S.P = kp_m{end}';
S.X = Points3D(1:3,:);
S.C = kp_new_sorted_out;
S.F = S.C;
S.T = [R_C2_W', -R_C2_W'*t_C2_W; 0,0,0,1];
S.T = S.T(:)* ones(1, size(S.C, 2));
S.Frames = last_bootstrap_frame_index * ones(1, size(S.C, 2));

% figure(1)
% plot(S.T(13), S.T(15),'x');
% hold on;

% check if p3p gives the same as bootstrap
%% Continuous operation
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
    [keep_P, P_delta] = runKLTContinuous(S.P, image, prev_img);
    toc
       
    S.P = S.P + P_delta;
    S.P = S.P(:, keep_P);
    S.X = S.X(:, keep_P);
    
%     Estimate new camera pose using p3p
    disp('run p3p with Ransac on S.P and S.X');
    tic
    [T, S.P, S.X] = ransacLocalization(S.P, S.X, K);
    toc
    
%     Track 
    disp('run KLT on S.C');
    tic
    [keep_C, C_delta] = runKLTContinuous(S.C, image, prev_img);
    toc;
    
    Frames_old = S.Frames;
    
    S.C = S.C + C_delta;
    S.C = S.C(:, keep_C);
    S.T = S.T(:, keep_C);
    S.F = S.F(:, keep_C);
    S.Frames = S.Frames(keep_C);  
    
    % Triangulate new points
    disp('try to triangulate S.C and S.F');
    tic
    [keep_triang, X_new] = triangulatePoints(S.C, S.F, T, S.T);
    toc
    
    S.P = [S.P, S.C(:, keep_triang)];
    S.C = S.C(:, ~keep_triang);
    S.T = S.T(:, ~keep_triang);
    S.F = S.F(:, ~keep_triang);
    S.X = [S.X, X_new];
    S.Frames = S.Frames(~keep_triang);
    
    % extract new keypoints
    disp('extract Keypoint')
    tic
    kp_new_latest_frame = extractHarrisKeypoints(image, num_keypoints);
    toc
    disp('check if extracted Keypoints are new')
    tic
    kp_new_sorted_out = checkIfKeypointIsNew(kp_new_latest_frame', ...
        [S.P, S.C], rejection_radius);  
    toc
    
    S.C = [S.C, kp_new_sorted_out];
    S.F = [S.F, kp_new_sorted_out];
    S.T = [S.T, T(:)*ones(1, size(kp_new_sorted_out, 2))]; 
    S.Frames = [S.Frames, i*ones(1, size(kp_new_sorted_out, 2))]; 
    
    prev_img = image;

    disp(['Number of 3D points:' num2str(size(S.X,2))]);
    disp(['Number of new keypoints:' num2str(size(kp_new_sorted_out,2))]);
    disp(['Number of candidate keypoints:' num2str(size(S.C, 2))]);
%     disp(['Frames still in S.C' num2str(unique(S.Frames))];

    plotBarDiagram(S.Frames,Frames_old);
    figure(1)
    plot(T(1,end), T(3,end),'x');
    hold on;
end