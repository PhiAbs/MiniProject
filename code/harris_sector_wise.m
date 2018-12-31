function points = harris_sector_wise(harris_num_image_splits, ...
    minQuality_Harris, nonmax_suppression_radius, image)
%harris_sector_wise: extract harris corners in several sectors. in each 
%sector we suppress the maxima in a certain area

region_size = floor(size(image,2) / harris_num_image_splits);
edge = 1;
for i = 1:harris_num_image_splits
    region = [edge, 1, region_size, size(image,1)];
    
    points_new = detectHarrisFeatures(image, 'ROI', region, 'MinQuality', minQuality_Harris);
    
    if i == 1
        points = nonmaxSuppression(points_new, nonmax_suppression_radius);
    else
        points = [points; nonmaxSuppression(points_new, nonmax_suppression_radius)];
    end
    
    if i == harris_num_image_splits - 1
        edge = edge + region_size;
        region_size = size(image,2) - edge;
    else
        edge = edge + region_size;
    end
end
end

