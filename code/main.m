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


%% Bootstrap

% Use more than two images to keep more keypoints
max_iterations = 10;
imgb = cell(1, 1);

% store first image
imgb{1} = loadImage(ds, 1, path);

% number of keypoints we want to extract from first imaeg
num_keypoints = 200;

% keypoints get stored as cell array with arrays Nx2, [col, row]
kp_m = cell(1, 1);
kp_m{1} = extractHarrisKeypoints(imgb{1}, num_keypoints);

% find matching keypoints in second image using lucas-kanade-tracker. Code
% from solution for exercise 8
r_T = 15;
num_iters = 50;
lambda = 0.1;

for i = 1:max_iterations
    % iteratively add more images if neccessary
    imgb{i+1} = loadImage(ds,i+2, path);
    
    % find matching keypoints in second image using lucas-kanade-tracker. Code
    % from solution for exercise 8
    dkp = [];
    keep = [];

    for j = 1:size(kp_m{i}, 1)
        kp_latest = kp_m{i};
        [dkp(:,j), keep(j)] = trackKLTRobustly(...
            imgb{i}, imgb{i+1}, kp_latest(j,:), r_T, num_iters, lambda);
        disp(j);
    end

    kp_latest = kp_latest + dkp';
    kp_latest = kp_latest(logical(keep), :);
    
    % add the keypoints of latest image
    kp_m{i+1} = kp_latest;
    
    % update the keypoints of first image
    kp1_m = kp_m{1};
    kp_m{1} = kp1_m(logical(keep), :);

    % Estimate the essential matrix E 
    p1 = [kp_m{1}'; ones(1, length(kp_m{1}))];
    pend = [kp_m{end}'; ones(1, length(kp_m{end}))];

    [E, in_essential] = estimateEssentialMatrix(p1, pend, K, K);

    [Rots,u3] = decomposeEssentialMatrix(E);

    [R_C2_W, T_C2_W] = disambiguateRelativePose(Rots,u3,p1,pend,K,K);

    M1 = K * eye(3,4);
    Mend = K * [R_C2_W, T_C2_W];
    P_C2_W = linearTriangulation(p1(:, in_essential),pend(:, in_essential),M1,Mend);
    
    % store points in our main cell array
    kp_m{1} = p1(1:2, in_essential)';
    kp_m{end} = pend(1:2, in_essential)';

    % remove points that lie behind the first camera or that are far away
    max_thresh = 100;
    within_min = P_C2_W(3, :) > 0;
    within_max = P_C2_W(3,:) < max_thresh;
    P_C2_W = P_C2_W(:, (within_min & within_max));
    
    % Plot stuff
    plotBootstrap(imgb, kp_m, P_C2_W, R_C2_W, T_C2_W);

    % check if camera moved far enough to finish bootstrapping
    avg_depth = mean(P_C2_W(3,:));
    baseline_dist = norm(T_C2_W);
    keyframe_selection_thresh = 0.2;

    if (baseline_dist / avg_depth) > keyframe_selection_thresh
        % leave bootstrapping if keyframes are enough far apart
        break;
    end
end
%% Continuous operation
range = (bootstrap_frames(end)+1):last_frame;
for i = range
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    if ds == 0
        image = imread([kitti_path '/00/image_0/' sprintf('%06d.png',i)]);
    elseif ds == 1
        image = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));
    elseif ds == 2
        image = im2uint8(rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',i)])));
    else
        assert(false);
    end
    % Makes sure that plots refresh.    
    pause(0.01);
    
    prev_img = image;
end