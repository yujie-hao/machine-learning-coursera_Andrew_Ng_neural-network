function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); % θ1(25, 401)

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % Θ2(10, 26)

% Setup some useful variables
% m, sample size 5000
m = size(X, 1);
         
% You need to return the following variables correctly 
% J = 0;
% Theta1_grad = zeros(size(Theta1));
% Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% add x0 all ones to first column
X = [ones(m,1) X]; % (5000, 401)
z2 = Theta1 * X'; % (25, 401) x (401, 5000) = (25, 5000)
a2 = sigmoid(z2); % (25, 5000)
a2 = [ones(1, m); a2]; % (26, 5000), add 1st row all 1s

% step.1: 5 steps of backpropagation algorithm
H = sigmoid(Theta2 * a2); % a3 = H(x) --> (10, 26) x (26, 5000) = (10, 5000)

% convert (5000, 1) digit labels into (5000, 10) label by 0s/1. 
% Example: Number 7 --> [0 0 0 0 0 0 1 0 0 0]
Y = zeros(m, num_labels);
% for r = 1:m
%     Y(r, y(r,1)) = 1; % (5000, 10)
% end
eye_matrix = eye(num_labels);
% if y(i) is 6, select 6th row(: means whole row) of the eye matrix --> [0 0 0 0 0 1 0 0 0 0]
% Υ(5000, 10)
Y = eye_matrix(y,:); 

% J = sum(sum(-1 * log(H) .* Y' - (1 - Y') .* log(1 - H))) / m; % non-regularized

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% step.2
% δ(10, 5000)
% Υ(5000, 10)
% a3 = H(10, 5000)
% delta3 = zeros(num_labels, m);% no need to init
delta3 = H - Y'; %(10, 5000)

% step.3
% (26, 5000)   = (26, 10)* (10,5000) .*  (26, 5000)
z2 = [ones(1, m); z2];
delta2 = (Theta2' * delta3) .* sigmoidGradient(z2); 

% step.4
%     (10,5000) * (5000, 26) = (10, 26)
Delta2 = delta3 * a2';

delta2 = delta2(2:end, :);
Delta1 = delta2 * X; % (25, 5000) * (5000, 401) = (25, 401)

% step.5
% regularizing
regularized_term = ones(size(Theta1, 1), size(Theta1, 2) - 1);
regularized_term = [zeros(size(Theta1, 1), 1) regularized_term];
Theta1_grad = Delta1 / m + lambda / m * Theta1 .* regularized_term; %(25, 401)

regularized_term = ones(size(Theta2, 1), size(Theta2, 2) - 1);
regularized_term = [zeros(size(Theta2, 1), 1) regularized_term];
Theta2_grad = Delta2 / m + lambda / m * Theta2 .* regularized_term; %(10, 26)

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

% Note that you should not be regularizing the terms that correspond to the bias. For the matrices Theta1 and Theta2, this corresponds to the first column of each Θ matrix. 
J = sum(sum(-1 * log(H) .* Y' - (1 - Y') .* log(1 - H))) / m + (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * lambda / (2 * m); % Regularized cost function

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
