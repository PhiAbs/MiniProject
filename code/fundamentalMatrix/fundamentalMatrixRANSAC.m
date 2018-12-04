function [F, bestInliers] = fundamentalMatrixRANSAC(x1s, x2s)
% This method calculates the fundamental matrix by applying the
% 8-point-algorithm with 8 random points and by checking the inliers
% input: 

    % threshold
    d = 0.5;
    
    bestInliers = [];
    bestNbInliers = 0;
    
    [nx1s, T1] = normalizePoints2d(x1s);
    [nx2s, T2] = normalizePoints2d(x2s);
    
    for k = 1:1000
       rnd = randi([1, size(nx1s,2)], [1, 8]);
       F = fundamentalMatrix(nx1s(:,rnd), nx2s(:,rnd));
       dist = sampsonError(T2'*F*T1, x1s, x2s); 
    
       idx = find(dist < d);
       nbInliers = size(idx,2);
       
       if (bestNbInliers < nbInliers)
          bestInliers = idx;
          bestNbInliers = nbInliers;
       end
       
    end
    
    % final fit
    F = fundamentalMatrix(nx1s(:, bestInliers), nx2s(:, bestInliers));
    dist = sampsonError(T2'*F*T1, x1s, x2s);
    bestInliers = (dist < d);
    
    F = T2'*F*T1;
    
end