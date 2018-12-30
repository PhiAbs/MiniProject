function [X_BA_refined, S_T, X_refined, T_refined, cameraPoses_all] = ...
    bundle_adjustment(S, cameraPoses_all, iter, num_BA_frames, keep_P_BA, K, max_iterations)
% Do bundle adjustment over a sliding window

iterator = 1;
for BA_i = 1:size(S.P_BA, 1)
    view_ids = iter+1-num_BA_frames:iter;
    % extract the x and y coordinate in every column and store them as
    % pointTrack objects
    valid_points = (S.P_BA(BA_i, :, 1) > 0);
    
%     only keep points that were tracked for more than one frame!
    if sum(valid_points) > 1
        tracked_long_enough(BA_i) = true;
        points_BA = ...
            [S.P_BA(BA_i, valid_points, 1); S.P_BA(BA_i, valid_points, 2)]';
        pointTracks(iterator) = pointTrack(view_ids(valid_points), points_BA);
        iterator = iterator + 1;
    else
        tracked_long_enough(BA_i) = false;
    end
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

fixed_views = cameraPoses.ViewId(1):cameraPoses.ViewId(4);

% reprojection_thresh = 10; 
% reprojectionErrors = reprojection_thresh + 1;
X_tracked_long_enough = S.X_BA(tracked_long_enough, :);
% points_well_reprojected = true(size(X_tracked_long_enough, 1), 1);

% while max(reprojectionErrors) > reprojection_thresh
    [refinedPoints3D, refinedPoses, reprojectionErrors] = ...
        bundleAdjustment(X_tracked_long_enough, pointTracks, ...
        cameraPoses, cameraParams, 'FixedViewId', fixed_views, ...
        'PointsUndistorted', true, 'MaxIterations', max_iterations);
    
    % find all points with large reprojection errors and rerun BA without
    % these points
%     max(reprojectionErrors)
%     indexes = find(points_well_reprojected);
%     points_well_reprojected_iter = (reprojectionErrors < reprojection_thresh);
%     points_well_reprojected(indexes(~points_well_reprojected_iter)) = false;
%     X_tracked_long_enough = X_tracked_long_enough(points_well_reprojected_iter, :);
%     pointTracks = pointTracks(points_well_reprojected_iter);
% end

% update all points with a small reprojection error
% index = find(tracked_long_enough);
% index = index(points_well_reprojected);
% X_BA_refined = S.X_BA;
% X_BA_refined(index, :) = refinedPoints3D;

% % update all 3D points that were refined
S.X_BA(tracked_long_enough, :) = refinedPoints3D;
X_BA_refined = S.X_BA;

% Sort out all the 3D points that are still visible in the newest image and
% store them only in the 3D points that were tracked longer than 1 frame
X_refined = X_BA_refined(logical(keep_P_BA), :)';

% store the latest refined camera pose
T_refined = [refinedPoses.Orientation{num_BA_frames}', ...
    refinedPoses.Location{num_BA_frames}'; [0,0,0,1]];

% store the refined camera poses in the table containing all camera poses
for j = 1:num_BA_frames
    cameraPoses_all.Orientation{end+j-num_BA_frames} = ...
        refinedPoses.Orientation{j}';
    cameraPoses_all.Location{end+j-num_BA_frames} = ...
        refinedPoses.Location{j};
end

% update the camera poses in S.T
S_T = S.T;
for k = 1:num_BA_frames
    idx = find(S.Frames == refinedPoses.ViewId(k));
    T_new = [refinedPoses.Orientation{k}', ...
        refinedPoses.Location{k}'; [0,0,0,1]];
    if isempty(idx) == false
        for iter = 1:size(idx, 1)
            S_T(:, idx(iter)) = T_new(:);
        end
    end
end

end

