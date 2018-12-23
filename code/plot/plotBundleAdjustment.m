function plotBundleAdjustment(cameraPoses_all)
% plot all camera centers after bundle adjustment
figure(30);
clf;
for i = 1:size(cameraPoses_all, 1)
    plot(cameraPoses_all.Location{i}(1), cameraPoses_all.Location{i}(3), 'Marker', 'x');
    axis equal;
    hold on;
end
title('camera centers after bundle adjustment')
end

