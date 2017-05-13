tic
% generate normalized image dataset
base_dir = '/home/msl/Dropbox';
rootFolder_full = fullfile(base_dir, 'data/3/full');
rootFolder_partial = fullfile(base_dir, 'data/3/partial');

% create image manager and retrieve relevant information to the full and
% partial image datasets
imds_full = imageDatastore(rootFolder_full, 'LabelSource', 'foldernames');
imds_partial = imageDatastore(rootFolder_partial, 'LabelSource', 'foldernames');

full_size = [550 200];
partial_size = [300 200];

[numImages_full, ~] = size(imds_full.Files);
[numImages_partial, ~] = size(imds_partial.Files);
numPixels_full = full_size(1) * full_size(2);
numPixels_partial = partial_size(1) * partial_size(2);

x_full = zeros(numImages_full, numPixels_full);
x_partial = zeros(numImages_partial, numPixels_partial);

for j = 1:numImages_full
    img_path = imds_full.Files{j};
    im = imread(img_path);
    x_full(j, :) = reshape(im, 1, numPixels_full);
end
for j = 1:numImages_partial
    img_path = imds_partial.Files{j};
    im = imread(img_path);
    x_partial(j, :) = reshape(im, 1, numPixels_partial);
end

% normalize the data (zero mean and unit std dev)
for j = 1:numPixels_full
    feat_vec = x_full(:,j);
    x_full(:,j) = (feat_vec - mean(feat_vec)) / std(feat_vec);
end

% normalize the data (zero mean and unit std dev)
for j = 1:numPixels_partial
    feat_vec = x_partial(:,j);
    x_partial(:,j) = (feat_vec - mean(feat_vec)) / std(feat_vec);
end

toc


%% 

cov_full = x_full * x_full';
cov_partial = x_partial * x_partial';

% use work around and find eigenvector X^TX
[V_full_xtx, D_full] = eig(cov_full);
[D_full, I_full] = sort(diag(D_full), 'descend');
V_full_xtx = V_full_xtx(:, I_full);

[V_partial_xtx, D_partial] = eig(cov_partial);
[D_partial, I_partial] = sort(diag(D_partial), 'descend');
V_partial_xtx = V_partial_xtx(:, I_partial);

% these are the eigenfeet
V_full = (x_full' * V_full_xtx)';
V_partial = (x_partial' * V_partial_xtx)';

% some of the values are below 0 since they were normalized. need to make
% them positive and scale them
V_vis_full = V_full;
V_vis_partial = V_partial;

% add min of zero centered arrays to make them all positive
V_full_min = min(V_full, [], 2);
V_vis_full = V_vis_full - V_full_min;
V_partial_min = min(V_partial, [], 2);
V_vis_partial = V_vis_partial - V_partial_min;

% normalize to [0, 1] for visualization 
V_full_max = max(V_vis_full, [], 2);
V_vis_full = rdivide(V_vis_full, V_full_max);
V_partial_max = max(V_vis_partial, [], 2);
V_vis_partial = rdivide(V_vis_partial, V_partial_max);

% visualize some of the eigenfeet
figure
imshow(reshape(V_vis_full(1, :), 550, 200))
figure
imshow(reshape(V_vis_full(2, :), 550, 200))
figure
imshow(reshape(V_vis_full(3, :), 550, 200))

figure
imshow(reshape(V_vis_partial(1, :), 300, 200))
figure
imshow(reshape(V_vis_partial(2, :), 300, 200))
figure
imshow(reshape(V_vis_partial(3, :), 300, 200))

%% 

% find new features (each row is new image, each column is comparison with
% different eigenfoot)
eigenfeet_ft_full = zeros(numImages_full);
eigenfeet_ft_partial = zeros(numImages_partial);

for j = 1:numImages_full
    image = x_full(j, :);
    for k = 1:size(V_full, 1)
        ft = sum(abs(image - V_full(k, :)));
        eigenfeet_ft_full(j, k) = ft;
    end
end

for j = 1:numImages_partial
    image = x_partial(j, :);
    for k = 1:size(V_partial, 1)
        ft = sum(abs(image - V_partial(k, :)));
        eigenfeet_ft_partial(j, k) = ft;
    end
end

%%

close all

proj_dir = '/home/msl/Dropbox/ML_project';
cluster_dir_full = '/home/msl/Dropbox/data/3/clustered/full/';
cluster_dir_partial = '/home/msl/Dropbox/data/3/clustered/partial/';

k = 7;   % number of clusters (use 10 for full

% perform k means clustering directly on pixels was features
idx_full = kmeans(eigenfeet_ft_full, k);
idx_partial = kmeans(eigenfeet_ft_partial, k);


clustered_filenames_full = cell(k, 1);
clustered_filenames_partial = cell(k, 1);

% visualize cluster as montage
for j = 1:length(idx_full)
   cluster =  idx_full(j);
   clustered_filenames_full{cluster}{end+1} = imds_full.Files{j};
end

for j = 1:length(idx_partial)
   cluster =  idx_partial(j);
   clustered_filenames_partial{cluster}{end+1} = imds_partial.Files{j};
end

for j = 1:k
    figure
    montage(clustered_filenames_partial{j})
end



