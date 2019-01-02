function plotBundleAdjustment(cameraPoses_all)
% plot all camera centers after bundle adjustment
figure(30);
clf;

for i = 1:size(cameraPoses_all, 1)
    cam_x(i) = cameraPoses_all.Location{i}(1);
    cam_z(i)= cameraPoses_all.Location{i}(3);
end
%     plot(cameraPoses_all.Location{i}(1), cameraPoses_all.Location{i}(3), 'Marker', 'x');
plot(cam_x, cam_z, 'rx');
axis equal;
hold on;
title('camera centers after bundle adjustment')
end

