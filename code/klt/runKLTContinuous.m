function [keep, delta] = runKLTContinuous(points, image, prev_img)
%RUNKLT runs the KLT tracker on several points
%input: 
% points: stores the keypoints that have to be tracked, 2xN
% image: image into which points have to be tracked
% prev_img: image from which points have to be tracked from
% output: 
% keep: logical vector, telling you which points were tracked correctly and which not 
% delta: shift in pixels of the keypoint from prev_img to image, 2xN

% Parameters for KLT tracker
r_T = 15;  % 15
num_iters = 30;  % 50
lambda = 0.1;

% find matching keypoints in second image using lucas-kanade-tracker. Code
% from solution for exercise 8

for i = 1:size(points, 2)
    [delta(:,i), keep(i)] = trackKLTRobustly(...
        prev_img, image, points(:,i)', r_T, num_iters, lambda);
end

