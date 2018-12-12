function [E, inliers_b] = estimateEssentialMatrix(p1, p2, K1, K2)
% estimateEssentialMatrix_normalized: estimates the essential matrix
% given matching point coordinates, and the camera calibration K
%
% Input: point correspondences
%  - p1(3,N): homogeneous coordinates of 2-D points in image 1
%  - p2(3,N): homogeneous coordinates of 2-D points in image 2
%  - K1(3,3): calibration matrix of camera 1
%  - K2(3,3): calibration matrix of camera 2
%
% Output:
%  - E(3,3) : fundamental matrix
%

% normalize points to avoid numerical errors
[p1_n,T1] = normalise2dpts(p1);
[p2_n,T2] = normalise2dpts(p2);

[F, inliers_b] = estimateFundamentalMatrix(p1_n(1:2,:)', p2_n(1:2,:)', ...
    'Method','RANSAC', 'NumTrials',50000,'DistanceThreshold',1e-2);

% Undo the normalization
F = (T2.') * F * T1;

% % TODO instead of this function, we could also use the above code!
% [F, inliers_b] = fundamentalMatrixRANSAC(p1, p2);

% Compute the essential matrix from the fundamental matrix given K
E = K2'*F'*K1;

end
