function [theta_sgd, iteration, errors] = ...
    stochastic_gd( fun, theta0, X, y, max_iter, batch_size, threshold)
%STOCHASTIC_GD Implements stochastic gradient descent
%   Detailed explanation goes here
theta_sgd = theta0;
old_theta_sgd = theta_sgd;    % To test for convergence
converged = 0;
errors = zeros(max_iter, 1);

eta = 0.001;
iteration = 0;

[f, ~] = fun(theta_sgd(:), X, y);

fprintf('Batch Start Error: %f\n', f);
   
tic;
while converged == 0 && iteration < max_iter 
    for batch_number = 0:(floor(size(X, 2) / batch_size) - 1)
        batch_start = batch_number * batch_size + 1;
        batch_end = batch_start + batch_size - 1;
        batch.X = X(:,batch_start:batch_end);
        batch.y = y(batch_start:batch_end);
        if(iteration > 0)
            delta = 1 / size(X, 2);
        else
            delta = batch_size / (batch_size * (batch_number+1) );
        end
        
        [~, g] = fun( theta_sgd(:), batch.X, batch.y );
        theta_sgd(:) = theta_sgd(:) - delta * eta * g;
    end
    iteration = iteration + 1;
    [f, ~] = fun(theta_sgd(:), X, y);
    theta_change = sum(sum(abs(old_theta_sgd - theta_sgd)));
    old_theta_sgd = theta_sgd;
    errors(iteration) = f;
    
    if(theta_change < threshold)
        converged = 1;
    end
    
    fprintf('iteration: %d, error:%f, sum of theta diff: %f\n', ...
        iteration, f, theta_change);
end

fprintf('Batch of %d Gradient Descent took %f seconds after %d full iterations.\n', batch_size, toc, iteration);

end

