% close all
% 
% % Simple test of preprocessImage()
% 
% % im = imread('/home/msl/Dropbox/ML_project/data/1/train/right/2861.png');
% im = imread('/home/msl/Dropbox/ML_project/data/1/train/right/15.png');
% tic
% im_rotated = preprocessImage(im);
% toc
% figure
% imshow(imbinarize(im_rotated))
% figure
% imshow(im_rotated)

%%

% Offline image preprocessor

% profile on

% define data path
cd ..
tempdir = pwd;
cd ML_project
rootFolder = fullfile(tempdir, 'data/1/test');

% % for the left and right images
% categories = {'left', 'right'};
% imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

% for validation data
imds = imageDatastore(rootFolder, 'LabelSource', 'foldernames');

[numImages, ~] = size(imds.Files);
% numImages = 200;
tic
for j = 1:numImages
    if mod(j,50) == 0
        j
    end
    img_path = imds.Files{j};
    im = imread(img_path);
    im_rotated = preprocessImage(im);
    im_rotated = imresize(im_rotated, [235 115]);  % resize image
    % save_path = strrep(img_path, '/train/', '/train/processed_small/');
    save_path = strrep(img_path, '/test/', '/test_processed/');
    
    imwrite(im_rotated, save_path);
end
toc



% imgPath = 'C:\Users\msl\Dropbox\ML_project\data\1\train\right\2861.png';
% im = imread(imgPath);
% im_rotated = preprocessImage(im);
% [height, width] = size(im_rotated);
% imwrite(im_rotated, strrep(imgPath, '\train\', '\train\processed\'));
