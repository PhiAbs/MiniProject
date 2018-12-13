function plotBarDiagram(Frames_new,Frames_old);



frame_min = min(Frames_old);
frame_max = max(Frames_new);
x = frame_min:frame_max;
counts_new = sum(Frames_new == x',2);
counts_old = sum(Frames_old == x',2);
if(length(x)~=1)
    figure(111)
    bar(x',[counts_old, counts_new,]);
    legend('previous images', 'current image');
end

end

