function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

size(X)
size(y)
size(theta)
alpha
num_iters

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

%size(X(1,:)
%size((X*theta)'*X(1,:))
%temp1=(alpha/m)*((X*theta-y)'*X(:,1));
%temp2=(alpha/m)*([X*theta-y]'*X(:,2));
%temp=[temp1;temp2];
%theta=theta-temp;
temp=(alpha/m)*(X'*(X*theta-y));
theta=theta-temp;






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end