function [S_T, X_refined, T_refined, cameraPoses_all] = ...
    bundle_adjustment(S, cameraPoses_all, iter, num_BA_frames, keep_P_BA, K)
%BA Summary of this function goes here

for BA_i = 1:size(S.P_BA, 1)
    view_ids = iter+1-num_BA_frames:iter;
    % extract the x and y coordinate in every column and store them as
    % pointTrack objects
    valid_points = (S.P_BA(BA_i, :, 1) > 0);
    points_BA = ...
        [S.P_BA(BA_i, valid_points, 1); S.P_BA(BA_i, valid_points, 2)]';
    pointTracks(BA_i) = pointTrack(view_ids(valid_points), points_BA);
end

% store the num_BA_frames last camera poses and IDs in a new table (used
% for bundle adjustment)
cameraPoses = table;
cameraPoses.ViewId(1:num_BA_frames) = ...
    cameraPoses_all.ViewId(end+1-num_BA_frames:end);
for j = 1:num_BA_frames
    cameraPoses.Orientation{j} = ...
        cameraPoses_all.Orientation{end+j-num_BA_frames};
    cameraPoses.Location{j} = ...
        cameraPoses_all.Location{end+j-num_BA_frames};
end

% do bundle adjustment
radialDistortion = [0 0]; 
cameraParams = cameraParameters('IntrinsicMatrix', K', ...
    'RadialDistortion',radialDistortion); 
[refinedPoints3D, refinedPoses] = ...
    bundleAdjustment(S.X_BA, pointTracks, cameraPoses, cameraParams);

% Sort out all the 3D points that are still visible in the newest image
X_refined = refinedPoints3D(logical(keep_P_BA), :)';

% store the latest refined camera pose
T_refined = [refinedPoses.Orientation{num_BA_frames}, ...
    refinedPoses.Location{num_BA_frames}'; [0,0,0,1]];

% store the refined camera poses in the table containing all camera poses
for j = 1:num_BA_frames
    cameraPoses_all.Orientation{end+j-num_BA_frames} = ...
        cameraPoses.Orientation{j};
    cameraPoses_all.Location{end+j-num_BA_frames} = ...
        cameraPoses.Location{j};
end

% update the camera poses in S.T
S_T = S.T;
for k = 1:num_BA_frames
    idx = find(S.Frames == cameraPoses.ViewId(k));
    T_new = [refinedPoses.Orientation{k}, ...
        refinedPoses.Location{k}'; [0,0,0,1]];
    if isempty(idx) == false
        for iter = 1:size(idx, 1)
            S_T(:, idx(iter)) = T_new(:);
        end
    end
end

end

% 
% for BA_i = 1:size(S.P_BA, 1)
%             view_ids = i+1-num_BA_frames:i;
%             % extract the x and y coordinate in every column
%             valid_points = (S.P_BA(BA_i, :, 1) > 0);
%             points_BA = ...
%                 [S.P_BA(BA_i, valid_points, 1); S.P_BA(BA_i, valid_points, 2)]';
%             pointTracks(BA_i) = pointTrack(view_ids(valid_points), points_BA);
%         end
%         
%         cameraPoses = table;
%         cameraPoses.ViewId(1:num_BA_frames) = ...
%             cameraPoses_all.ViewId(end+1-num_BA_frames:end);
%         for j = 1:num_BA_frames
%             cameraPoses.Orientation{j} = ...
%                 cameraPoses_all.Orientation{end+j-num_BA_frames};
%             cameraPoses.Location{j} = ...
%                 cameraPoses_all.Location{end+j-num_BA_frames};
%         end
%         
%         radialDistortion = [0 0]; 
%         cameraParams = cameraParameters('IntrinsicMatrix', K', ...
%             'RadialDistortion',radialDistortion); 
%         [xyzRefinedPoints, refinedPoses] = ...
%             bundleAdjustment(S.X_BA, pointTracks, cameraPoses, cameraParams);

