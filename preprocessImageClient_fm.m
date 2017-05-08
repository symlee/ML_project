% close all
% 
% % Simple test of preprocessImage()
% im = imread('/home/msl/Dropbox/data/2/valid/13.png');
% % figure
% % imshow(im)
% 
% filter_row = 13;
% filter_col = (filter_row + 1)/2 - 1;
% threshold = 160;
% 
% denoisedImage = wiener2(im, [filter_row, filter_col]);
% denoisedImage(denoisedImage > threshold) = 255;
% denoisedImage(denoisedImage <= threshold) = 0;
% figure
% imshow((denoisedImage))

%%

% Offline image preprocessor

% profile on

% define data path
cd ..
tempdir = pwd;
cd ML_project
rootFolder = fullfile(tempdir, 'data/2/test');

% for validation data
imds = imageDatastore(rootFolder, 'LabelSource', 'foldernames');

[numImages, ~] = size(imds.Files);
% numImages = 200;
tic

filter_row = 13;
filter_col = (filter_row + 1)/2 - 1;
threshold = 160;

for j = 1:numImages
    if mod(j,50) == 0
        j
    end
    img_path = imds.Files{j};
    im = imread(img_path);
    denoisedImage = wiener2(im, [filter_row, filter_col]);
    denoisedImage(denoisedImage > threshold) = 255;
    denoisedImage(denoisedImage <= threshold) = 0;
%    denoisedImage = im;
    denoisedImage = imresize(denoisedImage, [224 224]);
    %save_path = strrep(img_path, '/train/', '/train_bin/');
    save_path = strrep(img_path, '/test/', '/test_processed_small3/');
    
    imwrite(denoisedImage, save_path);
end
toc



% imgPath = 'C:\Users\msl\Dropbox\ML_project\data\1\train\right\2861.png';
% im = imread(imgPath);
% im_rotated = preprocessImage(im);
% [height, width] = size(im_rotated);
% imwrite(im_rotated, strrep(imgPath, '\train\', '\train\processed\'));
