function featureImg = loadUserImage(filePath)
%LOADUSERIMAGE Loads and pre-processes a user image suitably for use with 
%the neural network.
%   Loads the image saved in "filePath", and performs required thresholding
%   and centering as part of pre-processing. The returned variable is a
%   28 x 28 gray-scale image that can be directly feeded to the network
%   after vectorizing.
%
% P.S. Thank you Rakshith for flipping idea :D

% Load image and correct orientation for JPG files if required
img = imread(filePath);
imgInfo = imfinfo(filePath);
if imgInfo.Orientation == 6
        img = imrotate(img, -90);
end
% Resize to a tangible size for clustering, threshold dynamically
imgBW = imresize(rgb2gray(img), [300 300]);
C = ClusterValues(imgBW(:), 1);
binImg = imgBW < mean([C, 0]);
% Smoothen image to get a gradient at digit edges
binImg = double(imgaussfilt(uint8(binImg), 2));
% Find digit centre based on average pixel L2 distance
[x, y] = find(binImg);
cX = floor(mean(x));
cY = floor(mean(y));
stdX = floor(std(x));
stdY = floor(std(y));
xRange = (cX - 3*stdX) : (cX + 3*stdX);
yRange = (cY - 3*stdY) : (cY + 3*stdY);
newRange = max(range(xRange), range(yRange));
% Crop image about centre within 3 standard deviations and resize for
% feature vector
featureImg = imresize(permute(...
    binImg(max(1, cX - newRange/2) : min(size(binImg,1), cX + newRange/2), ...
    max(1, cY - newRange/2) : min(size(binImg,2), cY + newRange/2)), ...
    [2 1]), [28 28]);
featureImg = floor(featureImg * 255);

end


function C = ClusterValues(vals, N)
%CLUSTERVALUES Find the centres of N clusters on vals using k-means.
C = zeros(N, 1);
idx = kmeans(vals(:), N);
for i = 1:N
    C(i) = mean(vals(idx == i));
end
end