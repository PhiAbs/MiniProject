function plotBootstrap(imgb, kp_m, P_C2_W, R_C2_W, T_C2_W)
% Plots 2D matches and 3D points for bootstrapping process

    % Show keypoint matches
    figure(1);
    subplot(2,2,1);
    imshow(uint8(imgb{1}));
    hold on;
    plot(kp_m{1}(:,1)', kp_m{1}(:,2)', 'rx', 'Linewidth', 2);
    subplot(2,2,2);
    imshow(uint8(imgb{end}));
    hold on;
    plot(kp_m{end}(:,1)', kp_m{end}(:,2)', 'rx', 'Linewidth', 2);
    subplot(2,2,[3, 4]);
    imshow(uint8(imgb{end}));
    hold on;
    plot(kp_m{end}(:,1)', kp_m{end}(:,2)', 'rx', 'Linewidth', 2);
    plotMatches(1:length(kp_m{1}), kp_m{1}, kp_m{end});

    % show triangulated points
    figure(2);
    plot3(P_C2_W(1,:), P_C2_W(2,:), P_C2_W(3,:), 'o');
    grid on;
    axis equal;
    xlabel('x');
    ylabel('y');
    zlabel('z');

    % show coordinate systems of the two cameras
    plotCoordinateFrame(eye(3),zeros(3,1), 0.8);
    text(-0.1,-0.1,-0.1,'Cam 1','fontsize',10,'color','k','FontWeight','bold');

    center_cam2_W = -R_C2_W'*T_C2_W;
    plotCoordinateFrame(R_C2_W',center_cam2_W, 0.8);
    text(center_cam2_W(1)-0.1, center_cam2_W(2)-0.1, center_cam2_W(3)-0.1,'Cam 2','fontsize',10,'color','k','FontWeight','bold');

end

