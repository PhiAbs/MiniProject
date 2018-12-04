%% Setup
clear; close all; clc;
ds = 2; % 0: KITTI, 1: Malaga, 2: parking

if ds == 0
    kitti_path = '../datasets/kitti00/kitti';
    % need to set kitti_path to folder containing "00" and "poses"
    assert(exist('kitti_path', 'var') ~= 0);
    ground_truth = load([kitti_path '/poses/00.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
elseif ds == 1
    % Path containing the many files of Malaga 7.
    assert(exist('malaga_path', 'var') ~= 0);
    images = dir([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif ds == 2
    parking_path = '../datasets/parking';
    % Path containing images, depths and all...
    assert(exist('parking_path', 'var') ~= 0);
    last_frame = 598;
    K = load([parking_path '/K.txt']);
     
    ground_truth = load([parking_path '/poses.txt']);
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
% TODO: change this parameter to change baseline
% Use more than two images to keep more keypoints
bootstrap_frames = [1, 2, 3];
imgb = cell(length(bootstrap_frames), 1);

for i = 1:length(bootstrap_frames)
    if ds == 0
        imgb{i} = single(imread([kitti_path '/00/image_0/' ...
            sprintf('%06d.png',bootstrap_frames(i))]));
    elseif ds == 1
        imgb{i} = single(rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(bootstrap_frames(i)).name])));
    elseif ds == 2
        imgb{i} = single(rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',bootstrap_frames(i))])));
    else
        assert(false);
    end
end

% corner detection and keypoint extraction using Harris in the first image 
% (used solution from exercise 03
num_keypoints = 200;

% keypoints get stored as cell array with arrays Nx2, [col, row]
kp1_m = extractHarrisKeypoints(imgb{1}, num_keypoints);

% find matching keypoints in second image using lucas-kanade-tracker. Code
% from solution for exercise 8
r_T = 15;
num_iters = 50;
lambda = 0.1;

% iterate over several successive images to find as many matches as possible
keypoints = kp1_m;
for i = 1:length(bootstrap_frames)-1
    dkp = [];
    keep = [];
    
    for j = 1:size(keypoints, 1)
        % TODO: also use intermediate frames
        [dkp(:,j), keep(j)] = trackKLTRobustly(...
            imgb{i}, imgb{i+1}, keypoints(j,:), r_T, num_iters, lambda);
        disp(j);
    end
    
    kpold = keypoints(logical(keep), :);
    keypoints = keypoints + dkp';
    keypoints = keypoints(logical(keep), :);
    
    kp1_m = kp1_m(logical(keep), :);    
end

% kp_m: matched keypoints. stores keypoints of first and last image from
% bootstrapping
kp_m = cell(2, 1);
kp_m{1} = kp1_m;
kp_m{2} = keypoints;
%%

% Estimate the essential matrix E 
p1 = [kp_m{1}'; ones(1, length(kp_m{1}))];
pend = [kp_m{end}'; ones(1, length(kp_m{end}))];

[E, inliers_boot] = estimateEssentialMatrix(p1, pend, K, K);

[Rots,u3] = decomposeEssentialMatrix(E);

[R_C2_W, T_C2_W] = disambiguateRelativePose(Rots,u3,p1,pend,K,K);

M1 = K * eye(3,4);
Mend = K * [R_C2_W, T_C2_W];
P_C2_W = linearTriangulation(p1(:, inliers_boot),pend(:, inliers_boot),M1,Mend);

% remove points that lie behind the first camera or outside of a certain
% range
max_thresh = 60;
within_min = P_C2_W(3, :) > 0;
within_max = P_C2_W(3,:) < max_thresh;
P_C2_W = P_C2_W(:, (within_min & within_max));


% Show keypoint matches
figure(1);
subplot(2,2,1);
imshow(uint8(imgb{1}));
hold on;
plot(kp_m{1}(inliers_boot, 1)', kp_m{1}(inliers_boot, 2)', 'rx', 'Linewidth', 2);
subplot(2,2,2);
imshow(uint8(imgb{end}));
hold on;
plot(kp_m{end}(inliers_boot, 1)', kp_m{end}(inliers_boot, 2)', 'rx', 'Linewidth', 2);
subplot(2,2,[3, 4]);
imshow(uint8(imgb{end}));
hold on;
plot(kp_m{end}(inliers_boot, 1)', kp_m{end}(inliers_boot, 2)', 'rx', 'Linewidth', 2);
plotMatches(1:sum(inliers_boot), kp_m{1}(inliers_boot, :), kp_m{end}(inliers_boot, :));

% show triangulated points
figure(2);
plot3(P_C2_W(1,:), P_C2_W(2,:), P_C2_W(3,:), 'o');
grid on;
axis equal;
xlabel('x');
ylabel('y');
zlabel('z');

% show coordinate systems of the two cameras
plotCoordinateFrame(eye(3),zeros(3,1), 0.8);
text(-0.1,-0.1,-0.1,'Cam 1','fontsize',10,'color','k','FontWeight','bold');

center_cam2_W = -R_C2_W'*T_C2_W;
plotCoordinateFrame(R_C2_W',center_cam2_W, 0.8);
text(center_cam2_W(1)-0.1, center_cam2_W(2)-0.1, center_cam2_W(3)-0.1,'Cam 2','fontsize',10,'color','k','FontWeight','bold');


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