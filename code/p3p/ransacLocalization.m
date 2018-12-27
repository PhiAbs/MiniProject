function [T_W_C, P_inlier, X_inlier, best_inlier_mask] = ransacLocalization(P, X, K, pixel_thresh, num_iter)
% query_keypoints should be 2x1000
% all_matches should be 1x1000 and correspond to the output from the
%   matchDescriptors() function from exercise 3.
% best_inlier_mask should be 1xnum_matched (!!!) and contain, only for the
%   matched keypoints (!!!), 0 if the match is an outlier, 1 otherwise.


cameraParams = cameraParameters('IntrinsicMatrix',K');
% not sure about the exact output R_W_C or R_C_W

[R_W_C, t_W_C, inlier_idx] = estimateWorldCameraPose(double(P'),X',cameraParams);

T_W_C = [R_W_C', t_W_C'; 0 0 0 1];
P_inlier = P(:,inlier_idx);
X_inlier = X(:,inlier_idx);

% k = 3;
% 
% best_inlier_mask = zeros(1, size(P, 2));
% max_num_inliers_history = zeros(1, num_iter);
% max_num_inliers = 0;
% 
% % RANSAC
% for i = 1:num_iter
%     % Model from k samples (DLT or P3P)
%     [X_sample, idx] = datasample(X, k, 2, 'Replace', false);
%     P_sample = P(:, idx);
%     
%     % Backproject keypoints to unit bearing vectors.
%     normalized_bearings = K\[P_sample; ones(1, 3)];
%     for ii = 1:3
%         normalized_bearings(:, ii) = normalized_bearings(:, ii) / ...
%             norm(normalized_bearings(:, ii), 2);
%     end
% 
%     poses = p3p(X_sample, normalized_bearings);
% 
%     % Decode p3p output
%     R_C_W_guess = zeros(3, 3, 2);
%     t_C_W_guess = zeros(3, 1, 2);
%     for ii = 0:1
%         R_W_C_ii = real(poses(:, (2+ii*4):(4+ii*4)));
%         t_W_C_ii = real(poses(:, (1+ii*4)));
%         R_C_W_guess(:,:,ii+1) = R_W_C_ii';
%         t_C_W_guess(:,:,ii+1) = -R_W_C_ii'*t_W_C_ii;
%     end
%     
%     % CAREFUL - 2 other solutions added! 
%     
%     % Count inliers:
%     projected_points = projectPoints(... 
%         (R_C_W_guess(:,:,1) * X) + ...
%         repmat(t_C_W_guess(:,:,1), ...
%         [1 size(X, 2)]), K);
%     difference = P - projected_points;
%     errors = sum(difference.^2, 1);
%     is_inlier = errors < pixel_thresh^2;
%     R_C_W_inlier = R_C_W_guess(:,:,1);
%     t_C_W_inlier = t_C_W_guess(:,:,1);
%     
%     % also consider inliers for the alternative solution.
%     projected_points = projectPoints(...
%         (R_C_W_guess(:,:,2) * X) + ...
%         repmat(t_C_W_guess(:,:,2), ...
%         [1 size(X, 2)]), K);
%     difference = P - projected_points;
%     errors = sum(difference.^2, 1);
%     alternative_is_inlier = errors < pixel_thresh^2;
%     if nnz(alternative_is_inlier) > nnz(is_inlier)
%         is_inlier = alternative_is_inlier;
%         R_C_W_inlier = R_C_W_guess(:,:,2);
%         t_C_W_inlier = t_C_W_guess(:,:,2);
%     end
%     
%     % TODO remove this threshold??
%     min_inlier_count = 6;
%     
%     if nnz(is_inlier) > max_num_inliers && ...
%             nnz(is_inlier) >= min_inlier_count
%         max_num_inliers = nnz(is_inlier);        
%         best_inlier_mask = is_inlier;
%         R_C_W_best = R_C_W_inlier;
%         t_C_W_best = t_C_W_inlier;
%     end
%     
%     max_num_inliers_history(i) = max_num_inliers;
% end
% 
% P_inlier = P(:, best_inlier_mask);
% X_inlier = X(:, best_inlier_mask);
% 
% T_W_C = [R_C_W_best', -R_C_W_best'*t_C_W_best; 0,0,0,1];

end

