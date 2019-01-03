function [S, keep_P_BA, T_refined, cameraPoses_all] = ...
    bundle_adjustment(S, cameraPoses_all, iter, num_BA_frames, keep_P_BA, ...
    K, max_iterations, num_fixed_frames, absoluteTolerance_BA)
% Do bundle adjustment over a sliding window

% view_ids = iter+1-num_BA_frames:iter;
view_ids = cameraPoses_all.ViewId(end+1-num_BA_frames:end);

for BA_i = 1:size(S.P_BA, 1)
    % extract the x and y coordinate in every column and store them as
    % pointTrack objects
    valid_points = (S.P_BA(BA_i, :, 1) > 0);
    
    points_BA = ...
        [S.P_BA(BA_i, valid_points, 1); S.P_BA(BA_i, valid_points, 2)]';
    pointTracks(BA_i) = pointTrack(view_ids(valid_points), points_BA);
end

% store the num_BA_frames last camera poses and IDs in a new table (used
% for bundle adjustment). Invert the rotation matrix, since our convention
% is different than the one from matlab!
cameraPoses = table;
cameraPoses.ViewId(1:num_BA_frames) = ...
    cameraPoses_all.ViewId(end+1-num_BA_frames:end);
for j = 1:num_BA_frames
    cameraPoses.Orientation{j} = ...
        cameraPoses_all.Orientation{end+j-num_BA_frames}';
    cameraPoses.Location{j} = ...
        cameraPoses_all.Location{end+j-num_BA_frames};
end

% Invert K since matlab has another convention than we do
cameraParams = cameraParameters('IntrinsicMatrix', K'); 

fixed_views = cameraPoses.ViewId(1):cameraPoses.ViewId(num_fixed_frames);


% while max(reprojectionErrors) > reprojection_thresh
    [refinedPoints3D, refinedPoses, reprojectionErrors] = ...
        bundleAdjustment(S.X_BA, pointTracks, ...
        cameraPoses, cameraParams, 'FixedViewId', fixed_views, ...
        'PointsUndistorted', true, 'MaxIterations', max_iterations, ...
        'AbsoluteTolerance', absoluteTolerance_BA,  ...
        'RelativeTolerance', 1e-5);

orientation_change_sum = zeros(3,3);
location_change_sum = zeros(1,3);
for i = 1:num_BA_frames
    orientation_change_sum = orientation_change_sum + ...
        refinedPoses.Orientation{i} - cameraPoses.Orientation{i};
    location_change_sum = location_change_sum + refinedPoses.Location{i} ...
        - cameraPoses.Location{i};
end

disp(['max reprojection error after BA: ' num2str(max(reprojectionErrors))]);
disp('BA change in orientation of all matrices combined')
disp(num2str(orientation_change_sum));
disp('BA change in location of all matrices combined')
disp(num2str(location_change_sum));

% % update all 3D points that were refined
S.X_BA = refinedPoints3D;

% Sort out all the 3D points that are still visible in the newest image and
% store them only in the 3D points that were tracked longer than 1 frame
S.X = S.X_BA(logical(keep_P_BA), :)';

% only keep points with a small reprojection error
% keep_BA_reprojection = reprojectionErrors < 1;
% keep_reprojection = keep_BA_reprojection(logical(keep_P_BA));
% keep_P_BA = keep_P_BA(keep_BA_reprojection);
% S.X_BA = S.X_BA(keep_BA_reprojection, :);
% S.P_BA = S.P_BA(keep_BA_reprojection, :, :);
% S.X = S.X(:, keep_reprojection);
% S.P = S.P(:, keep_reprojection);

% store the latest refined camera pose
T_refined = [refinedPoses.Orientation{end}', ...
    refinedPoses.Location{end}'; [0,0,0,1]];

% store the refined camera poses in the table containing all camera poses
for j = 1:num_BA_frames
    cameraPoses_all.Orientation{end+j-num_BA_frames} = ...
        refinedPoses.Orientation{j}';
    cameraPoses_all.Location{end+j-num_BA_frames} = ...
        refinedPoses.Location{j};
end

% update the camera poses in S.T
for k = 1:num_BA_frames
    idx = find(S.Frames == refinedPoses.ViewId(k));
    T_new = [refinedPoses.Orientation{k}', ...
        refinedPoses.Location{k}'; [0,0,0,1]];
    if isempty(idx) == false
        for iter = 1:size(idx, 1)
            S.T(:, idx(iter)) = T_new(:);
        end
    end
end

end

