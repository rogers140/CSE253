function [theta_sgd, iteration, errors] = ...
    stochastic_gd( fun, theta0, ei, X, y, max_iter, batch_size, threshold)
%STOCHASTIC_GD Implements stochastic gradient descent
%   Detailed explanation goes here
theta_sgd = theta0;
old_theta_sgd = theta_sgd;    % To test for convergence
g_prev = [];     % previous gradient for computing momentum term.
converged = 0;
errors = zeros(max_iter, 1);

eta = ei.eta;
iteration = 0;

[f, ~] = fun(theta_sgd(:), ei, X, y);

fprintf('Start Error: %f\n', f);
   
tic;
X = X';
y = y';

while converged == 0 && iteration < max_iter
    permuted_indices = randperm(size(X, 2), size(X, 2));
    for batch_number = 0:(floor(size(X, 2) / batch_size) - 1)
        batch_start = batch_number * batch_size + 1;
        batch_end = batch_start + batch_size - 1;
        batch.X = X(:,permuted_indices(1, batch_start:batch_end));
        batch.y = y(:,permuted_indices(1, batch_start:batch_end));
        if(iteration > 0)
            delta = 1 / size(X, 2);
        else
            delta = batch_size / (batch_size * (batch_number+1) );
        end
        
        [~, g] = fun( theta_sgd(:), ei,batch.X', batch.y' );
        g_current = - delta * eta * g;
        if ~isempty(g_prev)
            g_current = g_current + ei.beta * g_prev;
        end
        theta_sgd(:) = theta_sgd(:) + g_current;
        g_prev = g_current;
    end
    iteration = iteration + 1;
    [f, ~] = fun(theta_sgd(:), ei, X', y');
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

