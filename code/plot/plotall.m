function plotall(image,X,P,Xnew,C,keepAngleThresh,keepWithinViewingAngle,t_WC,sizes)

%     figure('name','11','units','normalized','outerposition',[0.1 0.1 0.85 0.8]);
    figure(11)
    set(gcf,'units','normalized','outerposition',[0.1 0.1 0.85 0.8]);
    % plot upper image with 3D correspondences
    subplot(2,4,1:4)
    hold off;
    imshow(image);
    [ys,xs]=size(image);
    hold on;
    Ckeep=C(keepAngleThresh,:);
    plot([xs;C(~keepAngleThresh,1)],[ys;C(~keepAngleThresh,2)],'bs','linewidth',1.5)
    plot([xs;Ckeep(~keepWithinViewingAngle,1)],[ys;Ckeep(~keepWithinViewingAngle,2)],'cd','linewidth',1.5)
    plot(P(:,1),P(:,2),'ro','linewidth',1.5)
    plot([xs;Ckeep(keepWithinViewingAngle,1)],[ys;Ckeep(keepWithinViewingAngle,2)],'yo','linewidth',1.5)
    legend('Triangulation Angle too small',...
        'outside Cam Viewing angle',...
        'old 3D correspondences',...
        'newly triangulated')
    title('image with 3D correspondences and new kps')
    
    
    % bar diagram with sizes of different matrices
    subplot(2,4,5:6)
    bar(sizes(:,1),sizes(:,2:end))
    legend('#-landmarks','#-tracked keypoints','#-newly triangulated',...
        '#-outside viewing angle','#-angle too small')
    title('Array sizes over frames')
    
    
    % plot lower left as 2D point cloud
    subplot(2,4,7)
    hold on;
    plot(X(:,1),X(:,3),'rx',Xnew(:,1),Xnew(:,3),'bx','linewidth',1);
    legend('old landmarks','newly triangulated')
    title('2D Pointcloud')
    xlabel('x')
    ylabel('z')
    
    
    % plot Camera Positions
    subplot(2,4,8)
    hold on;
    plot(0,0,'bx',t_WC(1),t_WC(3),'rx','linewidth',1)
    title('Camera Position')
    xlabel('x')
    ylabel('z')
    axis equal
%     axis([-10 10 0 inf])

end