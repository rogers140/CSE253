function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes) matrix.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2);

  temp = exp(theta'* X);
  denominator = repmat(sum(temp, 1), [num_classes,1]);
  P = temp ./ denominator; %probability
  
  I = sub2ind(size(P), y, 1:m);
  
  value = P(I); %vector that only contain the non-zero probability
  f = -sum(log(value));
  
  P(I) = P(I) - 1; %subtract 1 for those c_n=j
  g = X * P';
 
  g=g(:); % make gradient a long vector for minFunc
end

