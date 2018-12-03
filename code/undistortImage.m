function I_u_bilinear = undistortImage(I_d_gray, K, D)
% function taken from exercise 1, file practice_01_distortion.m
% idea: assume we have a pixel coordinate in the undistorted image. 
% remap this coordinate to normalized camera coordinates using K^-1
% distort the coordinate using the lens distortion function
% remap the coordinate with K to the pixel space (don't forget lambda!)
% now we have a correspondence between the pixel coordinates in the
% undistorted and in the distorted image. 
% interpolate the pixel value in the distorted image and put this value
% at the pixel coordinate in the distorted image

% input: 
% I_d_gray: greyscale input image, size nxm
% K: Camera matrix, size 3x3
% D: Distortion matrix, size 2x1

% output: 
% I_u_bilinear: undistorted image, size nxm

% create a matrix for the undistorted image
[n, m] = size(I_d_gray);
I_u_bilinear = I_d_gray;

for y = 1:n
    for x = 1:m
        %remap pixel coordinate to camera coordinates
        p_camera = K\[x;y;1];
        
        % apply distortion function
        r_2 = p_camera(1)^2 + p_camera(2)^2;
        distortion = 1 + D(1)*r_2 + D(2)*r_2^2;
        p_camera_distorted = distortion * p_camera;
        p_camera_distorted(3) = 1;
        
        % remap the pixels from the camera to the pixel space (now
        % distorted coordinates!
        p_image_distorted = K * p_camera_distorted;
        
        % normalize with z-coordinate
        u = p_image_distorted(1) / p_image_distorted(3);
        v = p_image_distorted(2) / p_image_distorted(3);
        
        % do bilinear interpolation
        u_c = ceil(u);
        u_f = floor(u);
        v_c = ceil(v);
        v_f = floor(v);
        
        % interpolation along v
        u_int_c = (u_c - u) * I_d_gray(v_c, u_c) + (u - u_f) * I_d_gray(v_c, u_c);
        u_int_f = (u_c - u) * I_d_gray(v_f, u_c) + (u - u_f) * I_d_gray(v_f, u_c);
        
        % interpolation along u
        pixel_value = (v_c - v) * u_int_f + (v - v_f) * u_int_c;
        I_u_bilinear(y,x) = pixel_value;

    end
end

figure();
imshow(I_u_bilinear);
title('with bilinear interpolation');

end

