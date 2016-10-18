function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%
%  NOTE: W is set to a matrix of size(L_out, 1 + L_in) as the column row 
%  of W handles the "bias" terms.

% Randomly initialize the weights to small values to break symmetry
epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

end
