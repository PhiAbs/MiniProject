%% initialize VO Pipeline
clear all; close all; clc;

% choose your dataset
ds = 0; % 0: KITTI, 1: Malaga, 2: parking

%% define Path and load parameters
if ds == 0
    path = 'datasets/kitti00/kitti';
    % need to set kitti_path to folder containing "00" and "poses"
    assert(exist('path', 'var') ~= 0);
    ground_truth = load([path '/poses/00.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
elseif ds == 1 
    path = 'datasets/malaga-urban-dataset-extract-07';
    % Path containing the many files of Malaga 7.
    assert(exist('path', 'var') ~= 0);
    images = dir([path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif ds == 2
    path = 'datasets/parking';
    % Path containing images, depths and all...
    assert(exist('path', 'var') ~= 0);
    last_frame = 598;
    K = load([path '/K.txt']);
     
    ground_truth = load([path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
else
    assert(false);
end

cameraParams = cameraParameters('IntrinsicMatrix',K');

% camera angles
anglex = K(1,3)/K(1,1);
angley = K(2,3)/K(2,2);

% paths to functions
addpath('code')

% parameters
last_bootstrap_frame = 2;
bidirect_thresh = 3; % TODO  0.3: good
maxDistance_essential = 0.1;  % 0.1 is too big for parking!! 0.01 might work as well
maxNumTrials_Essential = 20000;
minQuality_Harris = 0.01;  %TODO  0.1: good
p3p_pixel_thresh = 1;  % TODO 1: good. 5: not so good
p3p_num_iter = 10000;
reprojection_thresh = 1;  %15: good. 10000: not so good for kitti, good for parking
reprojection_thresh_p3p = 2;
triangAngleThres = 0.015;
nonmax_suppression_radius = 10;
harris_rejection_radius = 10; %TODO 10: good for kitti
BA_iter = 2; 
num_BA_frames = 20;
max_iter_BA = 100;
num_fixed_frames_BA = 1;
absoluteTolerance_BA = 0.001;
enable_BA = true;
enable_plot = false;


%% Bootstrapping

% store first image
if ds == 1
    img0 = uint8(loadImage(ds, 1, path, left_images));
    img1 = uint8(loadImage(ds, 2, path, left_images));
    img2 = uint8(loadImage(ds, 3, path, left_images));
else
    img0 = uint8(loadImage(ds, 0, path, path));
    img1 = uint8(loadImage(ds, 1, path, path));
    img2 = uint8(loadImage(ds, 2, path, path));
end


points = detectHarrisFeatures(img0,'MinQuality',minQuality_Harris);
points = nonmaxSuppression(points, nonmax_suppression_radius);
kps = points.Location;  %keypoint_start

pointTracker = vision.PointTracker('MaxBidirectionalError',bidirect_thresh);
initialize(pointTracker, kps, img0);
setPoints(pointTracker,kps);
[kpl,keep] = pointTracker(img1);   % keypoints_latest
kps = kps(keep,:);
kpl = kpl(keep,:);

% track a second time
setPoints(pointTracker,kpl);
[kpl,keep] = pointTracker(img2);   % keypoints_latest
kps = kps(keep,:);
kpl = kpl(keep,:);


%% estimate Fundamental Matrix and initial pose
[E, keep] = estimateEssentialMatrix(kps, ...
    kpl, cameraParams, 'MaxNumTrials', ...
    maxNumTrials_Essential, 'MaxDistance', maxDistance_essential);
kps = kps(keep,:);
kpl = kpl(keep,:);

[R_WC,t_WC] = relativeCameraPose(E,cameraParams,kps,kpl);

% transform to our convention
R_WC = R_WC';
t_WC = t_WC';
T_WC = [R_WC,t_WC];

T_CW = [R_WC',-R_WC'*t_WC];

%% triangulate Points
M = (K*T_CW)';
[Xnew, reprojectionErrors] = triangulate( kps, kpl, (K*[eye(3) zeros(3,1)])', M);

c_inf_homo = T_WC(1:3,1:3)*inv(K)*[kpl ones(length(kpl),1)]';
f_inf_homo = inv(K)*[kps ones(length(kps),1)]';
triangulationAngles = acos(sum(c_inf_homo.*f_inf_homo,1)./...
(vecnorm(c_inf_homo,2).*vecnorm(f_inf_homo,2)));

Xnew_cam = T_CW*[Xnew ones(size(Xnew,1),1)]';
keep = all((abs(Xnew_cam(1:2, :))<[anglex; angley].*Xnew_cam(3, :))'...
    & reprojectionErrors < reprojection_thresh...
    & triangulationAngles' > triangAngleThres,2);

%% setup for continuous

S.X = Xnew(keep,:);
S.P = kpl(keep,:);
S.C = kpl;
S.F = kps;
S.findP = find(keep);
S.keepX = find(keep(S.findP));
T0 = [eye(3) zeros(3,1)];
S.T = T0(:)*ones(1,size(S.C,1));
S.Frames = 0*ones(1,size(S.C,1));

% extract new features in 2nd image
points = detectHarrisFeatures(img2,'MinQuality', minQuality_Harris);
points = nonmaxSuppression(points, nonmax_suppression_radius);
kpl=checkIfKeypointIsNew(points.Location', S.P', harris_rejection_radius);
kpl=kpl';
S.C = [S.C; kpl];
S.T = [S.T T_WC(:)*ones(1,size(kpl,1))];
S.F = [S.F; kpl];
S.Frames = [S.Frames, last_bootstrap_frame * ones(1,size(S.C,1))]; 

% store data for bundle adjustment
cameraPoses_all = table;
cameraPoses_all.ViewId(1) = uint32(last_bootstrap_frame);
cameraPoses_all.Orientation{1} = T_WC(1:3, 1:3);
cameraPoses_all.Location{1} = T_WC(1:3, end)';
S.P_BA(:, num_BA_frames, 1) = S.P(:,1);
S.P_BA(:, num_BA_frames, 2) = S.P(:,2)';
S.X_BA = S.X; 
keep_P_BA = ones(size(S.P, 1), 1);
S.C_trace_tracker(:, num_BA_frames, 1) = S.C(:,1);
S.C_trace_tracker(:, num_BA_frames, 2) = S.C(:,2);

img = img2;
clear img0 img1 img2 reprojection_error F T_CW;
img_prev = img;


sizes = [0 0 0 0; 2 size(S.P,2) nnz(~keep) nnz(keep)];  
t_WC_BA = [0 0 0; T_WC(1:3,4)'];
X_hist = S.X;

plotall(img_prev, S.X, S.P, S.X, S.C([keep;true(length(kpl),1)],:), ...
    reprojectionErrors < reprojection_thresh, ...
    triangulationAngles' > triangAngleThres,...
    all((abs(Xnew_cam(1:2, :))<[anglex; angley].*Xnew_cam(3, :))',2),...
    T_WC, t_WC_BA', sizes, anglex)

pause(5)

% initialize KLT trackers for continuous mode
trackP = vision.PointTracker('MaxBidirectionalError', bidirect_thresh);
initialize(trackP, S.P, img_prev);
trackC = vision.PointTracker('MaxBidirectionalError', bidirect_thresh);
initialize(trackC, S.C, img_prev);

cam_x = [];
cam_z = [];

%% start continuous

for i=3:last_frame
    
%     fprintf('\n\nProcessing frame %d\n=====================\n', i);
    
    img_prev = img;
    
    % get new image
    if ds == 0
        img = uint8(single(imread([path '/00/image_0/' ...
            sprintf('%06d.png',i)])));
    elseif ds == 1
        img = uint8(single(rgb2gray(imread([path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i+1).name]))));
    elseif ds == 2
        img = uint8(single(im2uint8(rgb2gray(imread([path ...
            sprintf('/images/img_%05d.png',i)])))));
    else
        assert(false);
    end
    
    % track features into new image
    [points, keepC] = trackC(img);
    S.keepX = find(keepC(S.findP));
    keepP = keepC(S.findP);
    S.findP = S.findP(keepC(S.findP));
    S.P = points(S.findP,:); % order is crucial, first S.P then S.C
    S.X = S.X(S.keepX,:);   
    S.C = points(keepC,:);
    S.T = S.T(:,keepC);
    S.F = S.F(keepC,:);
    S.C_trace_tracker(:, 1, :) = [];
    S.C_trace_tracker(~keepC, :, :) = [];
    S.C_trace_tracker(:, end+1, :) = points(keepC,:);
    S.Frames = S.Frames(keepC);
    
    % update data for bootstrap: update only the points that can still be
    % tracked!
    keep_Index = find(keep_P_BA);
    keep_Index = keep_Index(keepP);
    keep_P_BA = false(size(keep_P_BA));
    keep_P_BA(keep_Index) = 1;
    S.P_BA(:, 1, :) = [];
    S.P_BA(keep_P_BA, end+1, 1) = S.P(:,1);
    S.P_BA(keep_P_BA, end, 2) = S.P(:,2);
    keepCidx = find(keepC);
    S.findP = find(ismember(S.findP,keepCidx));
    
    [R_WC, t_WC, ~] = estimateWorldCameraPose(S.P,S.X,cameraParams,...
        'MaxNumTrials', p3p_num_iter, 'MaxReprojectionError', p3p_pixel_thresh);
    R_WC = R_WC';
    t_WC = t_WC';
    
    % we reproject the points on our own and define the threshold for the
    % points that should get discarded
    R_C_W = R_WC';
    t_C_W = -R_C_W*t_WC;
    % careful: transpose rotation matrix due to other convention
    projected_points = worldToImage(cameraParams,R_C_W',t_C_W',S.X);
    keepP = abs(sqrt(sum((projected_points - S.P).^2, 2))) ...
        < reprojection_thresh_p3p;
    
    T_WC = [R_WC, t_WC];
    T_CW = [R_WC',-R_WC'*t_WC];
    S.X = S.X(keepP,:);
    S.P = S.P(keepP,:);
    
    % update data for bootstrap: update only the points that can still be
    % tracked!
    keep_Index = find(keep_P_BA);
    keep_Index = keep_Index(keepP);
    keep_P_BA = false(size(keep_P_BA));
    keep_P_BA(keep_Index) = 1;
    S.P_BA(~keep_P_BA, end, :) = 0;
    S.findP = S.findP(keepP);
    S.keepX = S.keepX(keepP);

    % triangulate S.C and S.F
    Xnew = []; 
    reprojectionErrors = []; 
    triangulationAngles = [];
    M = (K*T_CW)';
    NonLms = find(~ismember([1:size(S.C,1)],S.findP));
    for j=NonLms 
        T_WCold = reshape(S.T(:,j),3,4);
        T_ColdW = [T_WCold(:,1:3)',-T_WCold(:,1:3)'*T_WCold(:,4)];
        Mold = (K*T_ColdW)';
        [new, reprojectionError] = triangulate( S.F(j,:), S.C(j,:), Mold, M);
        Xnew = [Xnew; new];
        reprojectionErrors = [reprojectionErrors; reprojectionError];
        
        c_inf_homo = T_ColdW(1:3,1:3)*T_WC(1:3,1:3)*inv(K)*[S.C(j,:) 1]';
        f_inf_homo = inv(K)*[S.F(j,:) 1]';
    
        triangulationAngle = acos(sum(c_inf_homo.*f_inf_homo,1)./...
        (vecnorm(c_inf_homo,2).*vecnorm(f_inf_homo,2)));
        triangulationAngles = [triangulationAngles; triangulationAngle];
            
    end
    
    Xnew_cam = T_CW*[Xnew ones(size(Xnew,1),1)]';
    keep = all((abs(Xnew_cam(1:2, :))<[anglex; angley].*Xnew_cam(3, :))' ...
                    & (Xnew_cam(3,:) > 0)' ...
                    & reprojectionErrors < reprojection_thresh ...
                    & triangulationAngles > triangAngleThres,2);

    sizes=[sizes; i size(S.P,1) size(S.C(NonLms,:),1) nnz(keep)];    
    t_WC_BA = [t_WC_BA; cameraPoses_all.Location{end}];
    X_hist = [X_hist; S.X];
    
    if size(sizes,1)>5
        sizes(1,:)=[];
    end
    
    plotall(img, X_hist, S.P, S.X, S.C(NonLms,:), reprojectionErrors < reprojection_thresh, ...
        triangulationAngles > triangAngleThres,...
        all((abs(Xnew_cam(1:2, :))<[anglex; angley].*Xnew_cam(3, :))',2),...
        T_WC, t_WC_BA', sizes, anglex)
    
    S.findP = [S.findP; NonLms(keep)'];
    S.keepX = [S.keepX; ones(nnz(keep),1)];
    
    Xnew = Xnew(keep,:);
    S.X = [S.X; Xnew];
    S.P = [S.P; S.C(NonLms(keep),:)];

    % update data for bootstrap
    keep_P_BA = [keep_P_BA; ones(size(Xnew, 1), 1)];
    S.P_BA(end+1:end+size(Xnew, 1), :, :) = S.C_trace_tracker(NonLms(keep), :, :);
    S.X_BA = [S.X_BA; Xnew];
    
    % extract new Keypints
    points = detectHarrisFeatures(img,'MinQuality', minQuality_Harris);
    points = nonmaxSuppression(points, nonmax_suppression_radius);
    kpl = points.Location;  %keypoint_latest
    
    % make sure that the same keypoint does not get tracked more than once
    kp_new_sorted_out = checkIfKeypointIsNew(kpl', ...
          (S.C)', harris_rejection_radius);
    S.C = [S.C; kp_new_sorted_out'];
    S.C_trace_tracker(end+1:end+size(kp_new_sorted_out, 2), end, :) = kp_new_sorted_out';
    S.F = [S.F; kp_new_sorted_out'];
    S.T = [S.T, T_WC(:) * ones(1,size(kp_new_sorted_out, 2))];
    S.Frames = [S.Frames, i*ones(1, size(kp_new_sorted_out, 2))]; 
    setPoints(trackP, S.P);
    setPoints(trackC, S.C);
    
    % add newest camera pose to all camera poses
    cameraPoses_new = table;
    cameraPoses_new.ViewId(1) = uint32(i);
    cameraPoses_new.Orientation{1} = T_WC(1:3, 1:3);
    cameraPoses_new.Location{1} = T_WC(1:3, end)';
    cameraPoses_all = [cameraPoses_all; cameraPoses_new];
    
    % bundle adjustment
    if BA_iter == num_BA_frames && enable_BA
        % delete all rows in the BA matrices which only contain zeros or
        % only one valid point (they cannot be used for BA

        valid_points = S.P_BA(:,:,1) > 0;
        untracked_landmark_idx = find(sum(valid_points, 2) == 0);
        S.P_BA(untracked_landmark_idx, :, :) = [];
        S.X_BA(untracked_landmark_idx, :) = [];
        keep_P_BA(untracked_landmark_idx) = [];
        
        [S, keep_P_BA, T_WC, cameraPoses_all] = ...
            bundle_adjustment(S, cameraPoses_all, num_BA_frames, ...
            keep_P_BA, K, max_iter_BA, num_fixed_frames_BA, absoluteTolerance_BA);
        
    else
        BA_iter = BA_iter + 1;
    end
    
    if enable_plot
        plotBundleAdjustment(cameraPoses_all);
    end    
end
