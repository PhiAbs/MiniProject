function  [S, keep_P_BA, X_new] = pointSanityCheck(S, keep_P_BA, T, K, ...
    X_new, keep_triang, keep_reprojected, max_allowed_point_dist)
%POINTSANITYCHECK makes sure that points very far away or points behind the
%camera are thrown away.

% sort out all points with a too large reprojection error
C_keep_triang = S.C(:, keep_triang);
C_keep_reprojected = C_keep_triang(:, keep_reprojected);
X_new = X_new(:, keep_reprojected);

T_C_W = inv(T);
M_C_W = K * T_C_W(1:3,:);
X_new_cam = M_C_W * [X_new; ones(1, size(X_new, 2))];
points_behind_cam = X_new_cam(3,:) < 0;
points_far_away = X_new_cam(3,:) > max_allowed_point_dist;

% sort out points that are too far away or behind the camera
X_new = X_new(:, (~points_far_away & ~points_behind_cam));
% I = find(keep_triang);
% I = I(points_behind_cam | points_far_away);
% keep_triang(I) = 0;

C_valid = C_keep_reprojected(:, (~points_far_away & ~points_behind_cam));

S.P = [S.P, C_valid];
S.X = [S.X, X_new];

% update data for bootstrap
keep_P_BA = [keep_P_BA; ones(size(X_new, 2), 1)];
S.P_BA(end+1:end+size(X_new, 2), end, :) = C_valid';
S.X_BA = [S.X_BA; X_new'];

S.C = S.C(:, ~keep_triang);
S.T = S.T(:, ~keep_triang);
S.F = S.F(:, ~keep_triang);
S.Frames = S.Frames(~keep_triang);
end

