function [kp_m] = runKLT(kp_m, imgb, i)
%RUNKLT Summary of this function goes here
%   Detailed explanation goes here

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

kp_latest = kp_latest + dkp';
kp_latest = kp_latest(logical(keep), :);

% add the keypoints of latest image
kp_m{i+1} = kp_latest;

    % update the keypoints of first image
    kp1_m = kp_m{1};
    kp_m{1} = kp1_m(logical(keep), :);
end

