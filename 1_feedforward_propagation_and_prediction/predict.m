function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
% num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
% p = zeros(size(X, 1), 1);
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m, 1) X];
a2 = sigmoid(Theta1 * X');
a2 = [ones(1, m); a2];
a3 = sigmoid(Theta2 * a2);
[~, p] = max(a3, [], 1);
p = p';

% note: https://drive.google.com/file/d/1run6oN1P04mgNwnBod8aeVriPxeLkyPO/view?usp=sharing
% Training the NN to get the Theta values is the subject of Week 5.
% 25 units is a tradeoff between giving good enough results, and not taking too long to train. It's determined by experiment.
% =========================================================================
end
