close all

% Simple test of preprocessImage()

% im = imread('C:\Users\msl\Dropbox\ML_project\data\1\train\right\2861.png');
% % im = imread('C:\Users\msl\Dropbox\ML_project\data\1\train\right\15.png');
% tic
% im_rotated = preprocessImage(im);
% toc
% imshow(im_rotated)

%%

% Offline image preprocessor

% profile on

% define data path
tempdir = pwd;
rootFolder = fullfile(tempdir, 'data/1/train');
categories = {'left', 'right'};

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
[numImages, ~] = size(imds.Files);
% numImages = 200;
tic
for j = 1:numImages
    if mod(j,50) == 0
        j
    end
    imgPath = imds.Files{j};
    im = imread(imgPath);
    im_rotated = preprocessImage(im);
    im_rotated = imresize(im_rotated, [470 230]);
    imwrite(im_rotated, strrep(imgPath, '\train\', '\train\processed\'));
end
toc

% imgPath = 'C:\Users\msl\Dropbox\ML_project\data\1\train\right\2861.png';
% im = imread(imgPath);
% im_rotated = preprocessImage(im);
% [height, width] = size(im_rotated);
% imwrite(im_rotated, strrep(imgPath, '\train\', '\train\processed\'));
