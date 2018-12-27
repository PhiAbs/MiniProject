function keep = checkBaseline(X, T, t_old, threshold)
%checkBaseline returns if 3D point is kept or not
% input: 
% X: 3D points, 3xN
% T: translation from camera to world frame, 4x4
% T_old: translation from base frame to older camera frame, 4x1
% threshold: If value below threshold, 3D point is not kept
% output: 
% keep: boolean, tells if 3D point is kept or not

baseline = norm(T(:,4) - t_old);
for i=1:size(X,2)
    P_current = T\[X(:,i); 1];
    depth(i) = P_current(3);    
end

avg_depth = median(depth); % USED TO BE MEAN

if(baseline/avg_depth > threshold)
    keep = true;
else
    keep = false;
end

end