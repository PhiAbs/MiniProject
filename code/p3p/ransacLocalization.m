function [T_W_C, P_inlier, X_inlier] = ransacLocalization(P, X, K)
% all_matches should be 1x1000 and correspond to the output from the
%   matchDescriptors() function from exercise 3.
% best_inlier_mask should be 1xnum_matched (!!!) and contain, only for the
%   matched keypoints (!!!), 0 if the match is an outlier, 1 otherwise.
% database_keypoints: 2x516
% p_W_landmarks: 3 x 516

% parameters
max_reprojection_error = 10;
keypoint_selection = 3;
max_num_iteration = 200;

% store only the matched 3D and 2D points
% P = flipud(query_keypoints(:, all_matches > 0));
% X = p_W_landmarks(:, all_matches(all_matches > 0));

for i = 1:max_num_iteration
    % choose 3 random points
    [P_sample, idx] = datasample(P, keypoint_selection, 2, 'Replace', false);
    X_sample = X(:,idx);

    % normalize and transform query coordinates
    P_sample_norm = K \ [P_sample; ones(1, length(P_sample))];
    P_norm = vecnorm(P_sample_norm);
    P_sample_norm = P_sample_norm ./ P_norm;

    poses = p3p( X_sample , P_sample_norm );

    % store the 2 valid transformations
    j = 1;
    for ii = 1:4:5
        R_C_W_p(:,:,j) = real(poses(:,ii+1:ii+3)');
        T_C_W_p(:,:,j) = - R_C_W_p(:,:,j) * real(poses(:,ii));
        j = j + 1;
    end

    % reproject all points using the two transformation matrices
    p_reprojected_1 = reprojectPoints(X', [R_C_W_p(:,:,1), T_C_W_p(:,:,1)], K)';
    p_reprojected_2 = reprojectPoints(X', [R_C_W_p(:,:,2), T_C_W_p(:,:,2)], K)';

    % calculate reprojection error and use the reprojection matrix with the
    % smaller error
    p_error_1 = sum((p_reprojected_1 - P).^2); 
    p_error_2 = sum((p_reprojected_2 - P).^2);

    if sum(p_error_1) < sum(p_error_2)
        M = [R_C_W_p(:,:,1), T_C_W_p(:,:,1)];
        p_error = p_error_1;

    else
        M = [R_C_W_p(:,:,2), T_C_W_p(:,:,2)];
        p_error = p_error_2;
    end

    % sort inliers and outliers
    inliers = p_error < max_reprojection_error^2;

    if i == 1 | sum(inliers) > max_num_inliers_history
        best_inlier_mask = inliers;
        max_num_inliers_history(i) = sum(inliers);
        R_C_W = M(1:3,1:3);
        t_C_W = M(:,4);
    else
        max_num_inliers_history(i) = max_num_inliers_history(i-1);
    end
end

P_inlier = P(:, best_inlier_mask);
X_inlier = X(:, best_inlier_mask);

T_W_C = inv([M; [0, 0, 0, 1]]);


end