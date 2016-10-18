function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input image vector.
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given
%   trained weights of a neural network (Theta1, Theta2)
%   NOTE: "0" is represented by "10" as explained in "main.m".

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% Feed-forward to get final activations
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
% Max-net the output nodes
[~, p] = max(h2, [], 2);

end
