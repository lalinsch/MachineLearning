function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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

% We do the second layer predictions (a(2)) based on our real data (Theta1)
% and apply hypothesis (returns the same as one vs all)
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
h_z2 = sigmoid(z2);

%We then use the predictions and apply the hidden layers values (Theta2) 
%which gives us the overall hypothesis
a2 = [ones(m, 1) h_z2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

%returns a column vector with the highest value in the matrix and it's
%corresponding index(prediction "p")
[pval, p] = max(a3, [], 2);

% =========================================================================


end
