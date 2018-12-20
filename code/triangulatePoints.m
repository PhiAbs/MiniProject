function [keep_triang, X_new] = triangulatePoints(C, F, T, T_mat, Frames,K, baseline_thresh)
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
% which baseline is large enough 1xN
% X_new: stores newly triangulated points, 3xnnz(keep_triang)

keep_triang = false(1,size(C,2));
X_new = [];

% invert matrix to get T_C_W
T_C_W = inv(T);
Mc = K * T_C_W(1:3,:);
% Mc = K * T(1:3,:);


% for i=1:size(C,2)
for frame=unique(Frames)
    
    sameFrame = Frames == frame;
    c = [C(:,sameFrame); ones(1,nnz(sameFrame))];
    f = [F(:,sameFrame); ones(1,nnz(sameFrame))];
    
    % invert matrix to get Tf_C_W
    Tf = (reshape(T_mat(:,find(sameFrame == 1,1)),[4 4]));
    Tf_C_W = inv(Tf);
    Mf = K * Tf_C_W(1:3,:);
%     Mf = K * Tf(1:3,:);
    
    % TODO: only keep points that are triangulated with a min. accuracy!
    X = linearTriangulation(f,c,Mf,Mc);
    X = X(1:3,:);
    
    if(checkBaseline(X, T, Tf(:,end), baseline_thresh))
        keep_triang = keep_triang | sameFrame;
        X_new = [X_new X];
    end
end

% keep_triang = logical(keep_triang);

end



