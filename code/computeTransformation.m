function [R, T] = computeTransformation(F, K, kp_0, kp_1)
% Computes the essential Matrix E and decomposes it into R and t
% input:
% F: Fundamental matrix
% K: internal camera calibration matrix
% kp_0: matched keypoints from one image, inliers (from RANSAC), size nx2
% kp_1: matched keypoints from another image, inliers (from RANSAC), size nx2
%output: 
% R: Rotation matrix
% T: translation vector

E = K'*F*K;

% decompose E. This returns 2 solutions for R and 2 solutions for t. 
% Function taken from Exercise 6
[R,u3] = decomposeEssentialMatrix(E);

% show coordinate systems of all possible cameras
figure(20);
grid on;
axis equal;
plotCoordinateFrame(eye(3),zeros(3,1), 0.2);
text(-0.1,-0.1,-0.1,'Cam 1','fontsize',10,'color','k','FontWeight','bold');

center_cam = -R(:,:,1)'*u3;
plotCoordinateFrame(R(:,:,1)',center_cam, 0.2);
text(center_cam(1)-0.1, center_cam(2)-0.1, center_cam(3)-0.1,'Cam 2','fontsize',10,'color','k','FontWeight','bold');

center_cam = -R(:,:,1)'*-u3;
plotCoordinateFrame(R(:,:,1)',center_cam, 0.2);
text(center_cam(1)-0.1, center_cam(2)-0.1, center_cam(3)-0.1,'Cam 3','fontsize',10,'color','k','FontWeight','bold');

center_cam = -R(:,:,2)'*u3;
plotCoordinateFrame(R(:,:,2)',center_cam, 0.2);
text(center_cam(1)-0.1, center_cam(2)-0.1, center_cam(3)-0.1,'Cam 4','fontsize',10,'color','k','FontWeight','bold');

center_cam = -R(:,:,2)'*-u3;
plotCoordinateFrame(R(:,:,2)',center_cam, 0.2);
text(center_cam(1)-0.1, center_cam(2)-0.1, center_cam(3)-0.1,'Cam 5','fontsize',10,'color','k','FontWeight','bold');


% find the correct R and t such that 3D points lie in front of camera
% Function taken from Exercise 6
[R,T] = disambiguateRelativePose(R, u3, [(kp_0), ones(length(kp_0), 1)]', ...
    [(kp_1), ones(length(kp_1), 1)]', K, K);


center_cam = -R'*T;
plotCoordinateFrame(R',center_cam, 0.2);
text(center_cam(1)-0.1, center_cam(2)-0.1, center_cam(3)-0.1,'Cam chosen','fontsize',10,'color','k','FontWeight','bold');
end

