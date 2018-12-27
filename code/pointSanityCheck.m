function  [S, keep_P_BA, X_new] = pointSanityCheck(S, keep_P_BA, T, K, ...
    X_new, keep_triang, max_allowed_point_dist)
%POINTSANITYCHECK makes sure that points very far away or points behind the
%camera are thrown away.

T_C_W = inv(T);
M_C_W = K * T_C_W(1:3,:);
X_new_cam = M_C_W * [X_new; ones(1, size(X_new, 2))];
points_behind_cam = X_new_cam(3,:) < 0;
points_far_away = X_new_cam(3,:) > max_allowed_point_dist;

% sort out points that are too far away or behind the camera
X_new = X_new(:, (~points_far_away & ~points_behind_cam));
I = find(keep_triang);
I = I(points_behind_cam | points_far_away);
keep_triang(I) = 0;

S.P = [S.P, S.C(:, keep_triang)];
S.X = [S.X, X_new];

% update data for bootstrap
keep_P_BA = [keep_P_BA; ones(sum(keep_triang), 1)];
S.P_BA(end+1:end+sum(keep_triang), end, :) = S.C(:, keep_triang)';
S.X_BA = [S.X_BA; X_new'];

S.C = S.C(:, ~keep_triang);
S.T = S.T(:, ~keep_triang);
S.F = S.F(:, ~keep_triang);
S.Frames = S.Frames(~keep_triang);
end

