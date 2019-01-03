function [T_W_C, P_inlier, X_inlier, inlier_idx] = ransacLocalization(P, X, K, pixel_thresh, num_iter)
% query_keypoints should be 2x1000
% all_matches should be 1x1000 and correspond to the output from the
%   matchDescriptors() function from exercise 3.
% best_inlier_mask should be 1xnum_matched (!!!) and contain, only for the
%   matched keypoints (!!!), 0 if the match is an outlier, 1 otherwise.


cameraParams = cameraParameters('IntrinsicMatrix',K');

[R_W_C, t_W_C, inlier_idx] = estimateWorldCameraPose(double(P'),double(X'),...
    cameraParams, 'MaxNumTrials', num_iter, 'MaxReprojectionError', pixel_thresh);

P_inlier = P(:,inlier_idx);
X_inlier = X(:,inlier_idx);

% redo pose estimation only with inliers
% [R_W_C, t_W_C, ~] = estimateWorldCameraPose(double(P_inlier'),double(X_inlier'),...
%     cameraParams, 'MaxNumTrials', num_iter, 'MaxReprojectionError', pixel_thresh);

T_W_C = [R_W_C', t_W_C'; 0 0 0 1];

end
