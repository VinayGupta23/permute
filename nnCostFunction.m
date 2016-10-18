function [J, grad] = nnCostFunction(nn_params, ...
                                    input_layer_size, ...
                                    hidden_layer_size, ...
                                    num_labels, ...
                                    X, y, lambda)
%NNCOSTFUNCTION Implements log-likelihood cost function for a two layer
%neural network (classification).
%   "J" is the cost and "grad" is the gradient of the weights unrolled into 
%   a single vector for use with "fmincg".
%   "nn_params" is the unrolled version of weights, which needs to be
%   reshaped before use.

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight
% matrices for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% Return variables
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Part 1: Feedforward the neural network and return the cost in the
%         variable J.
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad.
%         NOTE:
%         The vector y is a vector of labels containing values from 1 to 
%         10. It must be mapped to a binary vector of 1's and 0's
%
% Part 3: Implement regularization with the cost function and gradients.

% Feed-forward
z2 = [ones(m,1) X] * Theta1';
a2 = sigmoid(z2);
z3 = [ones(m,1) a2] * Theta2';
a3 = sigmoid(z3);

% Compute error (cost) function
for i = 1:num_labels
    h = a3(:,i);
    yi = y == i;
    J = J + -1/m * sum(yi.*log(h) + (1-yi).*log(1-h));
end
J = J + 0.5 * lambda/m * sum([sum(Theta1(:,2:end).^2) sum(Theta2(:,2:end).^2)]);

% Find error by back-propagation
for t = 1:m
    y_nn = (1:num_labels)' == y(t);
    del3 = a3(t,:)' - y_nn;
    del2 = (Theta2' * del3) .* sigmoidGradient([1 z2(t,:)]');
    Theta1_grad = Theta1_grad + del2(2:end)*[1 X(t,:)];
    Theta2_grad = Theta2_grad + del3*[1 a2(t,:)];
end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda / m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda / m * Theta2(:,2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
