function keypoints = extractHarrisKeypoints(img, num_keypoints)
% Uses the Harris Corner detection to extract keypoints from an image
% input: 
% img: image from which we want to extract keypoints, nxm
% num_keypoints: the number of keypoints we want to extract
% output: 
% keypoints: keypoints in pixel coordinates, 2xnum_keypoints, first row:
% pixel row coordinates, second row: pixel column coordinates

harris_patch_size = 9;
harris_kappa = 0.08;
nonmaximum_supression_radius = 8;

harris_scores = harris(img, harris_patch_size, harris_kappa);
assert(min(size(harris_scores) == size(img)));

keypoints = selectKeypoints(...
    harris_scores, num_keypoints, nonmaximum_supression_radius);

keypoints = fliplr(keypoints');
% figure(2);
% imshow(img);
% hold on;
% plot(keypoints(2, :), keypoints(1, :), 'rx', 'Linewidth', 2);
end

