% sort into full or partial group and normalize images
cd ..
tempdir = pwd;
cd ML_project
rootFolder = fullfile(tempdir, 'data/3');

% for validation data
imds = imageDatastore(rootFolder, 'LabelSource', 'foldernames');

[numImages, ~] = size(imds.Files);
% numImages = 200;
tic

full_size = [550 200];
partial_size = [300 200];

for j = 1:numImages
    if mod(j,50) == 0
        j
    end
    img_path = imds.Files{j};
    im = imread(img_path);
    [height, width] = size(im);
    if (height/width) > 2
        save_path = strrep(img_path, '/3/', '/3/full/');
        im = imresize(im, full_size);
    else
        save_path = strrep(img_path, '/3/', '/3/partial/');
        im = imresize(im, partial_size);
    end
    
    imwrite(im, save_path);
end
toc
