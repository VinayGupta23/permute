function g = sigmoidGradient(z)
%SIGMOIDGRADIENT Gradient of the sigmoid function evaluated at point z.

g = sigmoid(z) .* (1 - sigmoid(z));

end
