%% initialize VO Pipeline with Kitty Dataset
clear all; close all; clc;

%% define Path and load parameters
path = 'datasets/kitti00/kitti';
% need to set kitti_path to folder containing "00" and "poses"
assert(exist('path', 'var') ~= 0);
ground_truth = load([path '/poses/00.txt']);
ground_truth = ground_truth(:, [end-8 end]);
last_frame = 4540;
K = load([path '/00/K.txt']);
cameraParams = cameraParameters('IntrinsicMatrix',K');


%% Bootstrapping

img0 = uint8(single(imread([path '/00/image_0/' sprintf('%06d.png',0)])));
img1 = uint8(single(imread([path '/00/image_0/' sprintf('%06d.png',1)])));

points = detectHarrisFeatures(img0,'MinQuality',0.1);
kps = points.Location;  %keypoint_start

pointTracker = vision.PointTracker('MaxBidirectionalError',0.3);
initialize(pointTracker, kps, img0);
setPoints(pointTracker,kps);
[kpl,keep,confidence] = pointTracker(img1);   % keypoints_latest
kps = kps(keep,:);
kpl = kpl(keep,:);
confidence = confidence(keep);

% plot to validate
% figure(1)
% subplot(2,1,1)
% showMatchedFeatures(img0,img1,kps,kpl)
% subplot(2,1,2)
% stem3(kps(:,1),kps(:,2),confidence,'fill'); view(-25,30);

%% estimate Fundamental Matrix and initial pose


[F, keep] = estimateFundamentalMatrix(kps,kpl);
% E2 = K'*F*K;
% [E, keep] = estimateEssentialMatrix(kps,kpl,cameraParams);

kps = kps(keep,:);
kpl = kpl(keep,:);
% subplot(2,1,2)
% showMatchedFeatures(img0,img1,kps,kpl)

% [R_C_W,t_C_W] = relativeCameraPose(E,cameraParams,kps,kpl)
[R_W_C,t_W_C] = relativeCameraPose(F,cameraParams,kps,kpl);
T_CW = [R_W_C',-R_W_C'*t_W_C'];

%% triangulate Points

M = cameraMatrix(cameraParams,R_W_C',-R_W_C*t_W_C'); % = T_CW' but not exactly equal

S.P = kpl;
[S.X, reprojectionErrors] = triangulate( kps, kpl, (K*[eye(3) zeros(3,1)])', M);
% subplot(2,1,1)
% plot3(S.X(:,1),S.X(:,2),S.X(:,3),'*'); view(0,90);
% xlabel('x')
% ylabel('y')
% zlabel('z')
% grid on;

%% continious

keep = all((S.X > [-25 -10 0] & S.X < [25 10 100]) & reprojectionErrors<1 ,2);
S.X = S.X(keep,:);
S.P = S.P(keep,:);
% extract new features in 2nd image
points = detectHarrisFeatures(img1,'MinQuality',0.1);
kpl = points.Location;
keep = ~ismember(kpl,S.P,'rows');
S.C = kpl(keep,:);
S.T = T_CW(:) * ones(1,nnz(keep));
S.F = S.C;

img = img1;
T = T_CW;
clear img0 img1 reprojection_error F T_CW;

for i=2:last_frame
    
    % get new image
    img_prev = img;
    img = uint8(single(imread([path '/00/image_0/' sprintf('%06d.png',i)])));
    % track features into new image
    % initialize KLT trackers for continuous mode
    trackP = vision.PointTracker('MaxBidirectionalError',0.3);
    initialize(trackP, S.P, img_prev);
    trackC = vision.PointTracker('MaxBidirectionalError',0.3);
    initialize(trackC, S.C, img_prev);
    
    [points, keepP] = trackP(img);
    S.P = points(keepP,:);
    S.X = S.X(keepP,:);
    [points, keepC] = trackC(img);
    S.C = points(keepC,:);
    S.T = S.T(:,keepC);
    S.F = S.F(keepC,:);
    
    [R_W_C, t_W_C, keepP] = estimateWorldCameraPose(S.P,S.X,cameraParams);
    T_CW = [R_W_C',-R_W_C'*t_W_C'];
    nnz(keepP)/length(keepP)
    S.X = S.X(keepP,:);
    S.P = S.P(keepP,:);
%     plot(t_W_C(1),t_W_C(3),'o','linewidth',2)
%     hold on;

    % triangulate S.C and S.F
    Xnew = []; reprojectionErrors = [];
    for i=1: size(S.C,1)
        [new, reprojectionError] = triangulate( S.F(i,:), S.C(i,:), reshape( S.T(:,1),3,4)', T_CW');
        Xnew = [Xnew; new];
        reprojectionErrors = [reprojectionErrors; reprojectionError];
    end
    % This doesn't work, values wax to big
    Xnew
    keep = all((Xnew > [-200 -100 0] & Xnew < [200 100 100]) & reprojectionErrors<10 ,2);
    Xnew = Xnew(keep,:);
    S.C = S.C(~keep,:);
    S.T = S.T(:,~keep);
    S.F = S.F(~keep,:);
    plot3(Xnew(:,1),Xnew(:,2),Xnew(:,3),'*'); view(0,90);
    
    % extract new Keypints
    points = detectHarrisFeatures(img,'MinQuality',0.1);
    kpl = points.Location;  %keypoint_latest
    keep = ~ismember(kpl,[S.P; S.C],'rows');
    S.C = [S.C; kpl(keep,:)];
    S.T = [S.T, T_CW(:) * ones(1,nnz(keep))];
    S.F = [S.F; kpl(keep,:)];
    
end



