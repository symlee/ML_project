clc
close all

%% Code to preprocess the images

%% Correct for arbitrary orientation

tempdir = pwd;
rootFolder = fullfile(tempdir, 'data/1/train');
imPath = fullfile(rootFolder, 'left/16.png');

im = imread(imPath);
% obtain mask over track (use median filter as not to blur edges too much)
im_bin = (medfilt2(im, [5 5]) < 255);
% further preprocessing to get clean mask if necessary
% im_bin = (medfilt2(im_bin, [10 10]));  % get rid of speckles inside the tracks 
% m_bin = bwareaopen(im_bin, 10); % get rid of speckles around the border of the image

% find edges of mask
BW = edge(im_bin,'sobel');

% final longest connected component (longest edge)
CC = bwconncomp(BW);
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);

% find leftmost and rightmost points of longest edge
[row, col] = ind2sub(size(im), CC.PixelIdxList{idx});
minColInd = find(col == min(col));
maxColInd = find(col == max(col));
 
ptLeft = [row(minColInd(1)) col(minColInd(1))];
ptRight = [row(maxColInd(1)) col(maxColInd(1))];
BW(CC.PixelIdxList{idx}) = 0;  % remove it for visual verification

rad = atan2((ptRight(1) - ptLeft(1)), (ptRight(2) - ptLeft(2))) + pi/2;

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
% old code trying to use PCA to correct for arbitrary orientation
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