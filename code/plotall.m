%% Functions

function plotall(image,Xhist,P,X,C,keepReprojection,keepAngle,keepBehind,...
    T, camera_positions, sizes, anglex, num_BA_frames)

    for i=1:size(camera_positions, 1)
        t_BA(:,i) = camera_positions{i}';
    end

    cam = T*[-anglex*5 0 anglex*5; 0 0 0; 5 0 5; 1 1 1];

    % figure('name','11','units','normalized','outerposition',[0.1 0.1 0.85 0.8]);
    figure(11)
    set(gcf,'units','normalized','outerposition',[0.1 0.1 0.85 0.8]);
    % plot upper image with 3D correspondences
   
    subplot(5,8,[1:6,9:14])
    hold off;
    imshow(image);
    hold on;
    plot([0;C(~keepReprojection,1)],[0;C(~keepReprojection,2)],'bs','linewidth',1.5)
    plot([0;C(~keepAngle,1)],[0;C(~keepAngle,2)],'cd','linewidth',1.5)
    plot([0;C(~keepBehind,1)],[0;C(~keepBehind,2)],'b+','linewidth',1.5)
    plot([0;C(keepAngle&keepReprojection&keepBehind,1)],...
        [0;C(keepAngle&keepReprojection&keepBehind,2)],...
        'yo','linewidth',1.5)
    plot(P(:,1),P(:,2),'r*','linewidth',1.5)
    legend({'ReprojectionError is too big',...
        'Angle too small',...
        'outside viewing angle',...
        'newly triangulated',...
        'old 3D landmarks'},'FontSize',8)
    title('image with 3D correspondences and new kps')
    
    % bar diagram with sizes of different matrices
    subplot(5,8,[31 32 39 40])
    bar(sizes(:,1),sizes(:,2:end))
    legend({'#-landmarks','#-tracked keypoints','#-newly triangulated'},'FontSize',6)
    title('Array sizes over frames')
    
    % plot lower left as 2D point cloud
%     subplot(5,8,[17:19,25:27,33:35])
%     plot(Xhist(:,1),Xhist(:,3),'bx',X(:,1),X(:,3),'rx',T(1,4),T(3,4),'g+',...
%         cam(1,:),cam(3,:),'g','linewidth',1);
%     legend({'old landmarks','currently visible'},'FontSize',8)
%     title('2D Pointcloud')
%     xlabel('x')
%     ylabel('z')
%     axis equal
    
    % plot lower middle Camera Positions
    subplot(5,8,[17:22,25:30,33:38])
    cla;
    if size(t_BA,2)<num_BA_frames+1 
        plot([0 t_BA(1,:) T(1,4)],[0 t_BA(3,:) T(3,4)],'rx','linewidth',1)
        legend({'Camera BA'},'FontSize',5)
    else
        plot([t_BA(1,end+1-num_BA_frames:end) T(1,4)],[t_BA(3,end+1-num_BA_frames:end) T(3,4)],'rx',...
            [0 t_BA(1,1:end-num_BA_frames)],[0 t_BA(3,1:end-num_BA_frames)],'kx',...
            T(1,4),T(3,4),'bo','linewidth',1)
        legend({'cam BA','cam fixed'},'FontSize',8)
    end
    title('Camera Position')
    xlabel('x')
    ylabel('z')
    axis equal
    
    subplot(5,8,[7 8 15 16 23 24])
    cla;
    if size(t_BA)<21 
        plot(t_BA(1,:),t_BA(3,:),'.b','linewidth',3)
    else
        plot(t_BA(1,end-20:end),t_BA(3,end-20:end),'.b','linewidth',3)
    end
    hold on;
    plot(X(:,1),X(:,3),'rx','linewidth',1)
    plot(cam(1,:),cam(3,:),'g','linewidth',2)
    legend({'cam pos','landmarks','cam viewing angle'},'FontSize',8)
    title('last 20 cam pos and current Landmarks')
    axis equal;
    
end