function [theta_gd, iter, errors] = ...
    gradient_descent( fun, theta0, X, y, max_iter, grad_threshold)
%GRADIENT_DESCENT Implements gradient descent
% Arguments:
%   fun - function returning cost and gradient.
%   theta0 - Starting weights.
%   X - The examples stored in a matrix.  
%       X(i,j) is the i'th coordinate of the j'th example.
%   y - The label for each example. y(j) is the j'th example's label.
%   max_iter - max number of iterations to run.
%   grad_threshold - threshold value for gradients to stop.
theta_gd = theta0;
iter = 0;
eta = 1 / size(X, 2);

[f, g] = fun(theta_gd, X, y);

fprintf('Start error: %f\n',f);
errors = zeros(max_iter,1);

tic;
while(iter < max_iter)
  theta_gd(:) = theta_gd(:) - eta * g;
  
  [f, g_new] = fun(theta_gd, X, y); 
  iter = iter + 1;
  errors(iter) = f;
  grad_sum = sum(abs(g_new));
  if grad_sum < grad_threshold
    fprintf('Gradient small enough.\n');
    break;
  end
  g = g_new;
  if mod(iter, 1000) == 0
      fprintf('iteration: %f, error:%f, sum of gradient: %f\n', ...
          iter, f, grad_sum);
  end
end

fprintf('Gradient Descent took %f seconds.\n', toc);
end

