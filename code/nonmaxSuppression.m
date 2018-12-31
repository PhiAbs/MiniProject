function points = nonmaxSuppression(points, radius)
%nonmaxSuppression suppresses all pixels within a certain radius around a
%maximum

% sort points according to their strength
[points.Metric, I] = sort(points.Metric, 'descend');
points.Location = points.Location(I, :);

% go through metrics and suppress other points within a certain radius
i = 1;
while i <= points.Count
    
    suppress_idx = sqrt(sum((points.Location(i+1:end, :) - ...
        points.Location(i,:)).^2, 2)) < radius;
    
    suppress_idx = [false(i, 1); suppress_idx];
    points(suppress_idx) = [];
    
    i = i + 1;
end

