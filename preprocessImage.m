%% Code to preprocess the images

%% Correct for arbitrary orientation
function im_rotated = preprocessImage(im)
% tempdir = pwd;
% rootFolder = fullfile(tempdir, 'data/1/train');
% imPath = fullfile(rootFolder, 'left/16.png');
% 
% im = imread(imPath);

% obtain mask over track (use median filter as not to blur edges too much)
im_bin = (medfilt2(im, [5 5]) < 255);
% further preprocessing to get clean mask if necessary
im_bin = (medfilt2(im_bin, [10 10]));  % get rid of corner outliers
% m_bin = bwareaopen(im_bin, 10); % get rid of speckles around the border of the image

% remove pixels outside of footprint track (white border is unnecessary)
im(im_bin == 0) = 0;

% find edges of mask
BW = edge(im_bin,'sobel');

% final longest connected component (longest edge)
CC = bwconncomp(BW);
numPixels = cellfun(@numel,CC.PixelIdxList);
[largestComponent,idx] = max(numPixels);

% if largestComponent < 30  % I think I needed this last time but it fails
% on right/2861.png
if isempty(idx)
    % im is already correctly oriented
    im_rotated = im;
    
    % return cropped image
    mask = im_bin;
    [rowMask, colMask] = ind2sub(size(mask), find(mask == 1));
    
    minRow = min(rowMask);
    maxRow = max(rowMask);
    minCol = min(colMask);
    maxCol = max(colMask);
    
    im_rotated = im_rotated(minRow:maxRow, minCol:maxCol);
elseif largestComponent < 30
    % im is already correctly oriented
    im_rotated = im;
    
    % return cropped image
    mask = im_bin;
    [rowMask, colMask] = ind2sub(size(mask), find(mask == 1));
    
    minRow = min(rowMask);
    maxRow = max(rowMask);
    minCol = min(colMask);
    maxCol = max(colMask);
    
    im_rotated = im_rotated(minRow:maxRow, minCol:maxCol);
else
    % find leftmost and rightmost points of longest edge
    [row, col] = ind2sub(size(im), CC.PixelIdxList{idx});
    minColInd = find(col == min(col));
    maxColInd = find(col == max(col));

    ptLeft = [row(minColInd(1)) col(minColInd(1))];
    ptRight = [row(maxColInd(1)) col(maxColInd(1))];
%     BW(CC.PixelIdxList{idx}) = 0;  % remove longest edge for visual verification

    % rotate image according to rotation of longest edge
    rad = atan2((ptRight(1) - ptLeft(1)), (ptRight(2) - ptLeft(2))) + pi/2;
    im_rotated = imrotate(im, rad2deg(rad));
    
    % return cropped, rotated image
    mask = imrotate(im_bin, rad2deg(rad));
    [rowMask, colMask] = ind2sub(size(mask), find(mask == 1));

    minRow = min(rowMask);
    maxRow = max(rowMask);
    minCol = min(colMask);
    maxCol = max(colMask);
    
    im_rotated = im_rotated(minRow:maxRow, minCol:maxCol);
end

% % Debugging code
% 
% figure(1)
% subplot(2,2,1)
% imshow(im)
% subplot(2,2,2)
% imshow(im_bin)
% subplot(2,2,3)
% imshow(m_bin)
% subplot(2,2,4)
% imshow(BW)
% 
% a = 2

%{
figure
imshow(imrotate(im, rad2deg(rad)))
%}