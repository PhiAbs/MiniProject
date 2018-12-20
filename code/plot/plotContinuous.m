function plotContinuous(img_latest, X, P, C, T_newest)
    
    % show keypoints that have 3D correspondence
    figure(20);
    subplot(2,2,1);
    cla;
    imshow(uint8(img_latest));
    hold on;
    title('keypoints with 3D correspondence')
    plot(P(1,:), P(2,:)', 'rx', 'Linewidth', 2);
    
    % show candidate keypoints
    figure(20);
    subplot(2,2,2);
    cla;
    imshow(uint8(img_latest));
    hold on;
    title('candidate keypoints')
    plot(C(1,:), C(2,:), 'bx', 'Linewidth', 2);
    
    % plot camera centers (seen top-down)
    figure(20);
    subplot(2,2,4);
    hold on;
    axis equal;
    plot(T_newest(1,end), T_newest(3,end), 'Marker', 'o');
    title('camera centers (seen top-down)')

    % show triangulated points in top-down-2D-plot
    figure(21);
    hold on;
    if isempty(X) == 0
        plot3(X(1,:), X(2,:), X(3,:), 'rx');
    end
    hold on;
    grid on;
    axis equal;
    xlabel('x');
    ylabel('y');
    zlabel('z');

    % show coordinate system of the newest camera
    center_cam = T_newest(1:3,4);
    plotCoordinateFrame(eye(3),zeros(3,1), 0.8);
    plotCoordinateFrame(T_newest(1:3,1:3),center_cam, 0.8);
end

