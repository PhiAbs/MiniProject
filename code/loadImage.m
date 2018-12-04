function img = loadImage(ds,index, path)
% Loads images from image folder 
% input: 
% ds: choose dataset
% index: image index
% path: path to image set
% output: 
% img: loaded image

if ds == 0
    img = single(imread([path '/00/image_0/' ...
        sprintf('%06d.png',index)]));
elseif ds == 1
    img = single(rgb2gray(imread([path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(index).name])));
elseif ds == 2
    img = single(rgb2gray(imread([path ...
        sprintf('/images/img_%05d.png',index)])));
else
    assert(false);
end


end

