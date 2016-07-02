function J = computeCost(X, y, theta)
% J = COMPUTECOST(X, y, theta) computes the cost for linear regression 
% using theta as the parameter for linear regression to fit the data 
% points in X and y

m = length(y);

i = 1:m;
J = (1/(2*m)) * sum( ((theta(1) + theta(2) .* X(i,2)) - y(i)) .^ 2); % Un-Vectorized

end
