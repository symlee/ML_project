%% For finding the average size of the images (avgImageSize = [585 210])

% define data path
tempdir = pwd;
rootFolder = fullfile(tempdir, 'data/1/train');
%% Load Images
% Instead of operating on all of Caltech 101, which is time consuming, use
% three of the categories: airplanes, ferry, and laptop. The image category
% classifier will be trained to distinguish amongst these six categories.

% rootFolder = fullfile(dataDir, '101_ObjectCategories');
categories = {'left', 'right'};

%%
% Create an |ImageDatastore| to help you manage the data. Because
% |ImageDatastore| operates on image file locations, images are not loaded
% into memory until read, making it efficient for use with large image
% collections.
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

cumulativeSize = [0 0];
for j = 1:size(imds.Files, 1)
    im = imread(imds.Files{j});
    indSize = size(im);
    cumulativeSize = cumulativeSize + indSize;
end

avgSize = cumulativeSize / size(imds.Files, 1)