function kp_new = checkIfKeypointIsNew(kp_new, kp_tracked, threshold)
% checkIfKeypointIsNew checks if a newly found keypoint (Harris) already
% existed as a keypoint which was tracked into the image from an earlier stage.
% Discard a point if it is within a certain radius around an existing
% keypoint
% input: 
% kn_new: newly found keypoints in Image, size 2xN
% kp_tracked: keypoints tracked from earlier image, size 2xM
% threshold: If a new point is closer to an existing keypoint than this
% threshold, the point gets discarded
% output: 
% kp_new: sorted out keypoints, size 2xK

for i = 1:length(kp_tracked)
    distance = sqrt(sum((kp_new - kp_tracked(:,i)).^2));
    kp_new = kp_new(:,distance > threshold);
end

