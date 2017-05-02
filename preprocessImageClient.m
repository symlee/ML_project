close all

% im = imread('C:\Users\msl\Dropbox\ML_project\data\1\train\right\2861.png');
im = imread('C:\Users\msl\Dropbox\ML_project\data\1\train\right\55.png');

im_rotated = preprocessImage(im);
imshow(im_rotated)