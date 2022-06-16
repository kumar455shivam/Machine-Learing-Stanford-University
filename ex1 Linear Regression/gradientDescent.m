function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
% s = size(X);
% temp = zeros(m, s(2)+1);
% temp(:,1) = 1;
% temp(:,2:end) = X;
% Xnew=temp;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
%     d1 =0; d2=0;
%         for i =1:length(m)
%         d1 = d1 + ((theta(1) + theta(2)*X(i) - y(i)));
%         d2 = d2 + ((theta(1) + theta(2)*X(i) - y(i))*X(i));
%         end
old  = theta;
    theta(1) = theta(1) - alpha*(1/m)*sum(X*old - y);
    theta(2) = theta(2) - alpha*(1/m)*sum((X*old - y).*X(:,2));
%     theta(1) = theta(1) - alpha*(1/m)*d1;
%     theta(2) = theta(2) - alpha*(1/m)*d2;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
   % J_history(iter)

end

end
