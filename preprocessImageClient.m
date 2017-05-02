close all

% % Simple test of preprocessImage()
% 
% im = imread('C:\Users\msl\Dropbox\ML_project\data\1\train\right\2861.png');
% % im = imread('C:\Users\msl\Dropbox\ML_project\data\1\train\right\55.png');
% 
% im_rotated = preprocessImage(im);
% imshow(im_rotated)

%%

% Offline image preprocessor

% define data path
tempdir = pwd;
rootFolder = fullfile(tempdir, 'data/1/train');
categories = {'left', 'right'};

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

for j = 1:10
    imgPath = imds.Files{j};
    im = imread(imgPath);
    im_rotated = preprocessImage(im);
    [height, width] = size(im_rotated);
    
%     im_rotated = imresize(im_rotated, [210 210]);
    disp(size(im_rotated))
    imwrite(im_rotated, strrep(imgPath, '\train\', '\train\processed\'));    
end