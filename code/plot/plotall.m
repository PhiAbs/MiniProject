function plotall(image,X,P,Xnew,C,keepReprojection,keepGreater0,t_WC)

%     figure('name','11','units','normalized','outerposition',[0.1 0.1 0.85 0.8]);
    figure(11)
    set(gcf,'units','normalized','outerposition',[0.1 0.1 0.85 0.8]);
    % plot upper image with 3D correspondences
    subplot(2,3,1:3)
    hold off;
    imshow(image);
    hold on;
    plot(C(~keepReprojection,1),C(~keepReprojection,2),'bs','linewidth',1.5)
    plot(C(~keepGreater0,1),C(~keepGreater0,2),'cd','linewidth',1.5)
    plot(C(keepGreater0&keepReprojection,1),C(keepGreater0&keepReprojection,2),...
        'yo','linewidth',1.5)
    plot(P(:,1),P(:,2),'r*','linewidth',1.5)
    legend('ReprojectionError is too big',...
        'triangulated behind camera',...
        'newly triangulated',...
        'old 3D correspondences')
    title('image with 3D correspondences and new kps')
    
    
%     % bar diagram with sizes of different matrices
%     subplot(2,3,4)
%     bar(sizes(:,1),sizes(:,2:end))
%     legend('#-landmarks','#-tracked keypoints','#-newly triangulated',...
%         '#-outside viewing angle','#-error too big')
%     title('Array sizes over frames')
    
    
    % plot lower left as 2D point cloud
    subplot(2,3,5)
    hold on;
    plot(Xnew(keepGreater0&keepReprojection,1),Xnew(keepGreater0&keepReprojection,3),...
        'bx','linewidth',1.5);
    plot(X(:,1),X(:,3),'rx','linewidth',1.5);
    legend('newly triangulated','old landmarks')
    title('2D Pointcloud')
    xlabel('x')
    ylabel('z')
    
    
    % plot Camera Positions
    subplot(2,3,6)
    hold on;
    plot(t_WC(1),t_WC(3),'rx','linewidth',3)
    title('Camera Position')
    xlabel('x')
    ylabel('z')
    axis([-10 10 0 inf])

end