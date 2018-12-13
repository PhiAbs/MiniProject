function plotBootstrap(image_prev, image, kp_start, kp_latest, Points, R_C2_W, T_C2_W)
% Plots 2D matches and 3D points for bootstrapping process

    % Show keypoint matches
    figure(1);
    subplot(2,2,1);
    imshow(image_prev);
    hold on;
    plot(kp_start(:,1)', kp_start(:,2)', 'rx', 'Linewidth', 2);
    subplot(2,2,2);
    imshow(image);
    hold on;
    plot(kp_latest(:,1)', kp_latest(:,2)', 'rx', 'Linewidth', 2);
    subplot(2,2,[3, 4]);
    imshow(image);
    hold on;
    plot(kp_latest(:,1)', kp_latest(:,2)', 'rx', 'Linewidth', 2);
    plotMatches(1:length(kp_start), kp_start, kp_latest);

    % show triangulated points
    figure(2);
    plot3(Points(1,:), Points(2,:), Points(3,:), 'o');
    grid on;
    axis equal;
    xlabel('x');
    ylabel('y');
    zlabel('z');

    % show coordinate systems of the two cameras
    plotCoordinateFrame(eye(3),zeros(3,1), 0.8);
    text(-0.1,-0.1,-0.1,'Cam 1','fontsize',10,'color','k','FontWeight','bold');

    center_cam2_W = double(-R_C2_W'*T_C2_W);
    plotCoordinateFrame(R_C2_W',center_cam2_W, 0.8);
    text(center_cam2_W(1)-0.1, center_cam2_W(2)-0.1, center_cam2_W(3)-0.1,'Cam 2','fontsize',10,'color','k','FontWeight','bold');

end

