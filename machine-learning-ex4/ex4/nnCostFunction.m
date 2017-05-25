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

[r,~]=size(X);
X_temp=[ones(r,1) X];
temp=sigmoid(X_temp*Theta1');
[r1,~]=size(temp);
temp=[ones(r1,1) temp];
h=sigmoid(temp*Theta2');
temp_y=zeros(m,num_labels);
for i=1:m
temp_y(i,y(i,1))=1;
endfor
J=(-1./m)*sum((temp_y.*log(h)+(1-temp_y).*log(1-h))(:));
a=Theta1(1:end,2:end);
b=Theta2(1:end,2:end);
reg=(lambda./(2*m))*(sum((a.*a)(:))+sum((b.*b)(:)));
J=J+reg;


#Calculating the second part
data1=zeros(size(Theta1));
data2=zeros(size(Theta2));
for t=1:m
a1=X(t,:);
a1=[1 a1];
z2=a1*Theta1';
a2=sigmoid(z2);
a2=[1 a2];
z3=a2*Theta2';
a3=sigmoid(z3);
delta_3=a3;
delta_3(1,y(t,1))=delta_3(1,y(t,1))-1;
delta_2=delta_3*Theta2;
#delta_2=delta_2(2:end);
delta_2=delta_2.*sigmoidGradient([1 z2]);
delta_2=delta_2(2:end);
#size(delta_2)
#size(a1)
#size(Theta1)
data1=(data1+delta_2'*a1);
data2=(data2+delta_3'*a2);

%size(delta_2)
#size(Theta2)
#size(delta_3)
#size(z2)
#size(y)
#size(delta_3)
#size(a2)
#size(z2)
#size(a1)
#size(Theta1)


end;

Theta1_grad=data1/m;
Theta2_grad=data2/m;


#for i=1:m
#a1=X(i,:);
#size(a1)
#z2=a1*Theta1';
#a2=sigmoid(z2);
#a2=[1 a2];
#z3=a2*Theta2';
#a3=sigmoid(z3);
#delta_3=a3;
#delta_3(1,y(i,1))=delta_3(1,y(i,1))-1;
#temp=Theta2(:,2:end);
#delta_2=(delta_3*temp).*sigmoidGradient(z2);
#size(delta_2)
#size(delta_3)
#endfor


Theta1_grad(:, 2:input_layer_size+1) = Theta1_grad(:, 2:input_layer_size+1) + lambda / m * Theta1(:, 2:input_layer_size+1);
Theta2_grad(:, 2:hidden_layer_size+1) = Theta2_grad(:, 2:hidden_layer_size+1) + lambda / m * Theta2(:, 2:hidden_layer_size+1);












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
