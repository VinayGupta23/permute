%% Introduction

%  This script contains the code to design the neural network, load input
%  parameters and also train and test the network.
%
%  Training is done using conjugate gradient algorithm (which is an
%  advanced version of gradient descent), and the network is a
%  back-propagation network.


%% Setup the parameters of network

input_layer_size  = 28*28;
hidden_layer_size = 20;
num_labels = 10; % There are 10 digits (0-9)
% The digit "0" will be mapped to label 10, since MATLAB indexing starts
% at 1.


%% Loading and Visualizing Data
% The public MNIST dataset is used, obtained from the Kaggle website.

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')
trainData = csvread('data/train.csv', 1, 0);

% Keep first 80% of data for training, and last 20% for testing.
% NOTE: The digits are already randomly ordered, so it is not done
% explicitly.
trainInd = floor(size(trainData,1)*0.8);

X = trainData(1:trainInd, 2:end);
y = trainData(1:trainInd, 1);
y(y == 0) = 10;

Xtest = trainData(trainInd+1:end, 2:end);
ytest = trainData(trainInd+1:end, 1);
ytest(ytest == 0) = 10;

% Visualize some digits
displayData(X(1:25,:));


%% Initialize network weights

% Compute random initial weights for the network
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters to a vector for use with optimization function
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% Training the network

% The algorithm for weight update is the delta-rule and the weights changes 
% are computed by the optimization algorithm "Conjugate Gradient".
% Further, we also use regularization to prevent overfitting.

fprintf('\nTraining Neural Network... \n')

% Number of iterations
options = optimset('MaxIter', 100);
% Regularization parameter
lambda = 1;

% Create function handle for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
% The training phase
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params that were vectorized
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% Visualize Weights

%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% Evaluate Prediction Accuracy

%  After training the neural network, test performance on training and
%  testing data.

predTrain = predict(Theta1, Theta2, X);
predTest = predict(Theta1, Theta2, Xtest);

fprintf('\nTraining Set Accuracy: %.2f%%\n', mean(double(predTrain == y)) * 100);
fprintf('Testing Set Accuracy: %.2f%%\n', mean(double(predTest == ytest)) * 100);


%% Test on some user-sketched images
% Save images in the Data folder with name "imgX.jpg", where X is image ID.

imageIDs = 1:5;
figure;
for i = 1:length(imageIDs)
    imgPath = ['data/img' num2str(imageIDs(i)) '.jpg'];
    img = imread(imgPath);
    imgInfo = imfinfo(imgPath);
    if imgInfo.Orientation == 6
        img = imrotate(img, -90);
    end
    subplot(2, length(imageIDs), i);
    image(img); axis equal;
    subplot(2, length(imageIDs), i+length(imageIDs));
    imProc = loadUserImage(imgPath);
    imagesc(imProc); axis equal;
    text(6, 32, sprintf('Predicted: %d', predict(Theta1, Theta2, imProc(:)')));
end

%% References

%   1. Andrew N G's online Machine Learning course at Coursera
%   2. MATLAB Documentation
