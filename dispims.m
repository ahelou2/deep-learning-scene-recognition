function [imdisp] = dispims(imstack,drows,dcols)
% [imdisp] = dispims(imstack,drows,dcols,flip,border,frame_rows)
%
% display a stack of images
% Originally written by Sam Roweis 

N = size(imstack,2);
n2 = ceil(sqrt(N));
n1 = ceil(N/n2);
%figure;
for i=1:N
    daimg = reshape(imstack(:,i),drows,dcols);
    subplot(n1, n2, i);
    imagesc(daimg) ;colormap gray; axis image;
end
drawnow;
