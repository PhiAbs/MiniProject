function keep = checkBaseline(P, T_W_Current, t_old, threshold)
%checkBaseline returns if 3D point is kept or not
% input: 
% points3D: 3D points, 3xN
% T_current: translation from base frame to current camera frame, 3x1
% T_old: translation from base frame to older camera frame, 3x1
% threshold: If value below threshold, 3D point is not kept
% output: 
% keep: boolean, tells if 3D point is kept or not

baseline = norm(t_current - t_old);
for i=1:size(P,2)
    P_current = T_W_Current\P(:,i);
    depth(i) = P_current(3);    
end

avg_depth = mean(depth);

if(baseline/avg_depth > threshold)
    keep = true;
else
    keep = false;
end

end