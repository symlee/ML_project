% before incorporating atan2

clc
close all

% Code to preprocess the images
tempdir = pwd;
rootFolder = fullfile(tempdir, 'data/1/train');

imPath = fullfile(rootFolder, 'left/4.png');
im = imread(imPath);

im_bin = (medfilt2(im, [5 5]) < 255);
im_bin = (medfilt2(im_bin, [10 10]));  % helps get rid of specles inside
m_bin = bwareaopen(im_bin, 10); 

[BW,thresh,gv,gh] = edge(im_bin,'sobel');
%edgeDir = atan2(gv, gh);

CC = bwconncomp(BW);
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);

[row, col] = ind2sub(size(im), CC.PixelIdxList{idx});
minColInd = find(col == min(col));
maxColInd = find(col == max(col));

ptA = [row(minColInd(1)) col(minColInd(1))]
ptB = [row(maxColInd(1)) col(maxColInd(1))]
BW(CC.PixelIdxList{idx}) = 0;

if ptB(1) < ptA(1)
    slope_up = 1
else
    slope_up = 0
end
if slope_up == 1
%     slope = abs(ptB(1) - ptA(1)) / (ptB(2) - ptA(2))  
%     rad = pi/2 - atan(slope)
    rad = atan2((ptB(1) - ptA(1)), (ptB(2) - ptA(2))) + pi/2
else
%     slope = -(ptB(1) - ptA(1)) / (ptB(2) - ptA(2))
%     rad = pi/2 - atan(slope)
    rad = atan2((ptB(1) - ptA(1)), (ptB(2) - ptA(2))) + pi/2
end


figure(1)
subplot(2,2,1)
imshow(im)
subplot(2,2,2)
imshow(im_bin)
subplot(2,2,3)
imshow(m_bin)
subplot(2,2,4)
imshow(BW)

figure
imshow(imrotate(im, rad2deg(rad)))

%{
ind = find(im_bin == 1);

[rowInd, colInd] = ind2sub(size(im), ind);
rowMean = mean(rowInd);
colMean = mean(colInd);

X = [rowInd - rowMean, colInd - colMean];
cov = X' * X;
[U, S, V] = eigs(cov);
angle_rad = atan(U(2,1)/U(1,1));
 angle_rad = pi/2 + angle_rad
angle_deg = rad2deg(angle_rad);

B = imrotate(im, -angle_deg);


figure(1)
subplot(2,2,1)
imshow(im)
subplot(2,2,2)
imshow(im_bin)
subplot(2,2,3)
imshow(B)
%}


