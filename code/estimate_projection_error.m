function [error, inlier] = estimate_projection_error( P, X, T_W_C, K, threshold)
% P: [Mx2] Array with corresponding image points in image coordinates
% X: [Mx3] Array with corresponding Landmarks in World Frame
% T: [4x4] Transfromation Matrix from Camera to World
% K: [3x3] intrinsic Camera Matrix
% threshold: value for which the L2 norm of P-P_reprojected get disgarded

M=T_W_C^(-1);

P_reprojected = reprojectPoints(X,M(1:3,:), K);
error = sum((P_reprojected-P).^2,2);

inlier = error <= threshold;

end
