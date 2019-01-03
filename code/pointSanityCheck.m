function  [S, keep_P_BA, X_new] = pointSanityCheck(S, keep_P_BA, T, ...
    X_new, keep_triang, anglex, angley)
%POINTSANITYCHECK makes sure that points very far away or points behind the
%camera are thrown away.

% sort out all points with a too large reprojection error
C_keep = S.C(:, keep_triang);
C_trace_keep = S.C_trace_tracker(keep_triang, :, :);

T_C_W = inv(T);
X_new_cam = T_C_W(1:3,:) * [X_new; ones(1, size(X_new, 2))];
keep_camera_angle = all((abs(X_new_cam(1:2, :))<[anglex; angley].*(X_new_cam(3, :))),1);

% sort out points that are too far away, behind the camera or outside the
% camera viewing angle
X_new = X_new(:, (keep_camera_angle));

C_valid = C_keep(:, (keep_camera_angle));
C_trace_valid = C_trace_keep((keep_camera_angle), :, :);

S.P = [S.P, C_valid];
S.X = [S.X, X_new];

% update data for bootstrap
keep_P_BA = [keep_P_BA; ones(size(X_new, 2), 1)];
S.P_BA(end+1:end+size(X_new, 2), :, :) = C_trace_valid;
S.X_BA = [S.X_BA; X_new'];

S.C = S.C(:, ~keep_triang);
S.T = S.T(:, ~keep_triang);
S.F = S.F(:, ~keep_triang);
S.Frames = S.Frames(~keep_triang);
S.C_trace_tracker = S.C_trace_tracker(~keep_triang, :, :);
end

