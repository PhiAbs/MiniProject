function  [X, P, C, F, T_array, Frames, P_BA, X_BA, keep_P_BA, X_new] = ...
    pointSanityCheck(X, P, C, F, T_array, Frames, P_BA, X_BA, keep_P_BA,...
    T, K, X_new, keep_triang, max_allowed_point_dist)
%POINTSANITYCHECK makes sure that points very far away or points behind the
%camera are thrown away.

T_C_W = inv(T);
M_C_W = K * T_C_W(1:3,:);
X_new_cam = M_C_W * [X_new; ones(1, size(X_new, 2))];
points_behind_cam = X_new_cam(3,:) < 0;
points_far_away = X_new_cam(3,:) > max_allowed_point_dist;

X_new = X_new(:, (~points_far_away & ~points_behind_cam));
I = find(keep_triang);
I = I(points_behind_cam | points_far_away);
keep_triang(I) = 0;

P = [P, C(:, keep_triang)];
X = [X, X_new];

% update data for bootstrap
keep_P_BA = [keep_P_BA; ones(sum(keep_triang), 1)];
P_BA(end+1:end+sum(keep_triang), end, :) = C(:, keep_triang)';
X_BA = [X_BA; X_new'];

C = C(:, ~keep_triang);
T_array = T_array(:, ~keep_triang);
F = F(:, ~keep_triang);
Frames = Frames(~keep_triang);
end

