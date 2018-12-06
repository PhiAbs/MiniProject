function [kp_m, keep] = runKLT(kp_m, imgb, i)
%RUNKLT calls KLT functions
%input: 
% kp_m: keypoint cell array with old keypoints
% imgb: image array
% i: index of current image
% output: 
% kp_m: keypoint cell array containing all keypoints for all images
% keep: tells you which keypoinst still can be used and which have to be
% discarded

% Parameters for KLT tracker
r_T = 15;
num_iters = 50;
lambda = 0.1;

% find matching keypoints in second image using lucas-kanade-tracker. Code
% from solution for exercise 8

for j = 1:size(kp_m{i}, 1)
    kp_latest = kp_m{i};
    [dkp(:,j), keep(j)] = trackKLTRobustly(...
        imgb{i}, imgb{i+1}, kp_latest(j,:), r_T, num_iters, lambda);
end

keep = logical(keep);

kp_latest = kp_latest + dkp';
kp_latest = kp_latest(keep, :);

% add the keypoints of latest image
kp_m{i+1} = kp_latest;

    % update the keypoints of first image
    kp1_m = kp_m{1};
    kp_m{1} = kp1_m(keep, :);
end

