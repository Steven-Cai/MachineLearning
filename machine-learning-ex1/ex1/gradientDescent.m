function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % theta1
    sum = 0;
    hypothesis = X * theta;
    for n = 1:m
        diff = hypothesis(n, 1) - y(n, 1);
        sum += (diff * X(n, 1));
    end
    theta1 = theta(1, 1) - alpha * (1 / m) * sum;

    % theta2
    sum = 0;
    hypothesis = X * theta;
    for n = 1:m
      diff = hypothesis(n ,1) - y(n, 1);
      sum += (diff * X(n, 2));
    end
    theta2 = theta(2, 1) - alpha * (1 / m) * sum;

    % updata theta
    theta(1, 1) = theta1;
    theta(2, 1) = theta2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
