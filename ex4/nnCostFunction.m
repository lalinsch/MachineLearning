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
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% 

    %initiate first layer activation
    a1 = X;
    a1 = [ones(size(a1, 1), 1) a1];
    %Calculate z2 based on a1 and the initial Theta1 values
    z2 = a1 * Theta1';
    %Apply the hypothesis to find g(z)
    a2 = sigmoid(z2);
    
    %add another column of ones (bias unit)
    a2 = [ones(size(a2, 1), 1) a2];
    
    %calculate the next z value for the third layer (output layer)
    z3 = a2 * Theta2';
    %apply sigmoid function
    a3 = sigmoid(z3);
    
    %define output value (h)
    h = a3;
    %create matrix with booleans depending on the number of labels (10 in
    %this case)
    temp = eye(num_labels);
    %turn the matrix into a vector
    yVec = temp(y, :);
    
    %cost function (vectorised)
    J = (1/m) * sum(sum((-yVec .* log(h)) - ((1-yVec) .* log(1 - h))));

   %Step 2: Regularization 
   %Inititate thetas without bias term
   Theta1_reg = Theta1(:, 2:end);
   Theta2_reg = Theta2(:, 2:end);
   
   %Compute regularization
   reg = (lambda / (2*m)) * (sum(sum(Theta1_reg .^2 )) + sum(sum(Theta2_reg .^ 2)));
    
   J = J + reg;
%END of feedforward propagation
%Start of back propagation 

    d3 = h - yVec;
    
    %compute delta 2
    d2 = (d3 * Theta2) .* [ones(size(z2, 1), 1) sigmoidGradient(z2)]; 
    %remove bias unit
    d2 = d2(:, 2:end);
    
    %Unregularised gradients
    Theta2_grad = (1 / m) * d3' * a2;
    Theta1_grad = (1 / m) * d2' * a1;
    
    %Add regularization to gradients 
    %Adding a row of zeros to multiply the first element of Theta(0) by
    %zero
    Theta2_grad = Theta2_grad + ((lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)]);
    Theta1_grad = Theta1_grad + ((lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)]);
    
     
    















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
