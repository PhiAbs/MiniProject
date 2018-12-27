function plotContinuous(img_latest, X_new, X, P, C, T_newest, K)
    
% show triangulated points in top-down-2D-plot
    figure(21);
    hold on;
    if isempty(X_new) == 0
        plot3(X_new(1,:), X_new(2,:), X_new(3,:), 'rx');
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


    % show keypoints that have 3D correspondence
    % check reprojection error
    M = T_newest^(-1);
    P_reprojected = reprojectPoints(X',M(1:3,:), K);
    error = median(abs(sum(P_reprojected-P',2)));
    fprintf(['error: ',num2str(error)]);
    
%     FIG1=figure('name','summarized figure','units','normalized','outerposition',[0.1 0.1 0.85 0.8]);
    figure(1);
    subplot(2,2,1:2);
    cla;
    imshow(uint8(img_latest));
    hold on;
    title('keypoints with 3D correspondence and reprojection')
    plot(P_reprojected(:,1),P_reprojected(:,2),'bx','linewidth',1.5)
    hold on;
    plot(P(1,:),P(2,:),'ro','linewidth',1.5)
    hold on;
    [maximums, idx] = maxk(abs(sum(P_reprojected-P',2)),3);
    plot(P_reprojected(idx,1),P_reprojected(idx,2),'yo','linewidth',2.5)
    
    % show candidate keypoints
    subplot(2,2,3);
    cla;
    imshow(uint8(img_latest));
    hold on;
    title('candidate keypoints')
    plot(C(1,:), C(2,:), 'bx', 'Linewidth', 2);
    
    % plot camera centers (seen top-down)
    subplot(2,2,4);
    hold on;
    axis equal;
    plot(T_newest(1,end), T_newest(3,end), 'Marker', 'o');
    title('camera centers (seen top-down)')

    % show triangulated points
    figure(21);
    cla;
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

