function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  
  t = sigmoid(theta' * X);
  g = (( t - y) * X')';
  f = - sum( (y .* log(t))  +  ((1-y) .* log(1 - t)) );
  
