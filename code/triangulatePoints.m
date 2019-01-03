function [keep_triang, X_new] = ...
    triangulatePoints(C, F, T, T_mat, Frames,K, baseline_thresh, reprojection_thresh)
%triangulatePoints triangulates every point. 
% input:
% C: all keypoints in current frame in pixel coordinates, 2xM
% F: all keypoints in previous frames in pixel coordinates, 2xM
% T: Transformation matrix from current camera to World, 4x4
% T_mat: Matrix containing all transformation matrices for every keypoint
% in F as a row vector, 16xM
% K: camera calibration matrix
% output: 
% keep_triang: logical vector, containing ones for points in C for 
% which baseline is large enough 1xM
% X_new: stores newly triangulated points, 3xM

keep_triang = [];
X_new = [];
triangulationAngles = [];
reprojectionErrors = [];

% invert matrix to get T_C_W
T_C_W = inv(T);
Mc = K * T_C_W(1:3,:);

for frame=unique(Frames)
    
    sameFrame = Frames == frame;
    c = [C(:,sameFrame); ones(1,nnz(sameFrame))];
    f = [F(:,sameFrame); ones(1,nnz(sameFrame))];
    
    % invert matrix to get Tf_C_W
    Tf = (reshape(T_mat(:,find(sameFrame == 1,1)),[4 4]));
    Tf_C_W = inv(Tf);
    Mf = K * Tf_C_W(1:3,:);
    
    [worldPoints, reprojectionError] = triangulate(f(1:2, :)', ...
        c(1:2, :)', Mf', Mc');
    
    reprojectionErrors = [reprojectionErrors; reprojectionError];
    X_new = [X_new worldPoints'];
    
    % check triangulation error as proposed in mini project script
    c_inf_homo = Tf_C_W(1:3,1:3)*T(1:3,1:3)*inv(K)*c;
    f_inf_homo = inv(K)*f;
    
    triangulationAngle = acos(sum(c_inf_homo.*f_inf_homo,1)./...
        (vecnorm(c_inf_homo,2).*vecnorm(f_inf_homo,2)));
    
    triangulationAngles = [triangulationAngles; triangulationAngle'];
end

% [real(triangulationAngles) triangulationAngles > baseline_thresh...
%     real(reprojectionErrors) reprojectionErrors < reprojection_thresh]

% disp(['max angle: ', num2str(max(triangulationAngles))])
% disp(['reprojection error percentage: ',...
%     num2str(nnz(reprojectionErrors < reprojection_thresh)/length(reprojectionErrors))])

keep_triang = (triangulationAngles > baseline_thresh & reprojectionErrors < reprojection_thresh)';

end





% function [keep_triang, keep_reprojected, X_new] = ...
%     triangulatePoints(C, F, T, T_mat, Frames,K, baseline_thresh, reprojection_thresh)
% %triangulatePoints triangulates every point. 
% % input:
% % C: all keypoints in current frame in pixel coordinates, 2xM
% % F: all keypoints in previous frames in pixel coordinates, 2xM
% % T: Transformation matrix from current camera to World, 4x4
% % T_mat: Matrix containing all transformation matrices for every keypoint
% % in F as a row vector, 16xM
% % K: camera calibration matrix
% % output: 
% % keep_triang: logical vector, containing ones for points in C for 
% % which baseline is large enough 1xN
% % X_new: stores newly triangulated points, 3xnnz(keep_triang)
% % keep_reprojected: boolean vector containing information whether a point
% % can be kept or if its reprojection error is too large
% 
% keep_triang = false(1,size(C,2));
% keep_reprojected = [];
% X_new = [];
% 
% % invert matrix to get T_C_W
% T_C_W = inv(T);
% Mc = K * T_C_W(1:3,:);
% 
% for frame=unique(Frames)
%     
%     sameFrame = Frames == frame;
%     c = [C(:,sameFrame); ones(1,nnz(sameFrame))];
%     f = [F(:,sameFrame); ones(1,nnz(sameFrame))];
%     
%     % invert matrix to get Tf_C_W
%     Tf = (reshape(T_mat(:,find(sameFrame == 1,1)),[4 4]));
%     Tf_C_W = inv(Tf);
%     Mf = K * Tf_C_W(1:3,:);
%     
% %     % TODO: only keep points that are triangulated with a min. accuracy!
%     [worldPoints, reprojectionErrors] = triangulate(f(1:2, :)', ...
%         c(1:2, :)', Mf', Mc');
%     X = worldPoints';
%     
%     % check if baseline is reached AND if points lie within allowed range
%     if(checkBaseline(X, T, Tf(:,end), baseline_thresh))
%         % remove points that have a large reprojection error. the
%         % keep_reprojected vector has the same length as the newly added 3D
%         % points!
%         keep_reprojected = ...
%             [keep_reprojected; (reprojectionErrors < reprojection_thresh)];
%         
%         keep_triang = keep_triang | sameFrame;
%         X_new = [X_new X];
%     end
% end
% 
% 
% 
% end
% 
% 
% 
