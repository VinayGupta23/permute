function featureImg = loadUserImage(filePath)
%LOADUSERIMAGE Loads and pre-processes a user image suitably for use with 
%the neural network.
%   Loads the image saved in "filePath", and performs required thresholding
%   and centering as part of pre-processing. The returned variable is a
%   28 x 28 gray-scale image that can be directly feeded to the network
%   after "vectorizing".

% TODO: Waiting to resolve issues #2 and #3.
% TODO: Changes required to automate centering , gray-scaling and
%       thresholidng. Currently all hardcoded.

% Select red component (cause it somehow works :P)
img = imread(filePath);
redMask = img(:,:,1);
binImg = redMask < 50;
featureImg = imresize(permute(binImg(200:end-200, 200:end-200), [2 1]), [28 28]);

end

