function [theta_sgd, iteration, errors] = ...
    stochastic_gd( fun,theta, X, y, layers, options)
% Runs stochastic gradient descent with momentum to optimize the
% parameters for the given objective.
%
% Parameters:
%  funObj     -  function handle which accepts as input theta,
%                data, labels and returns cost and gradient w.r.t
%                to theta.
%  theta      -  unrolled parameter vector
%  data       -  stores data in m x n x numExamples tensor
%  labels     -  corresponding labels in numExamples x 1 vector
%  options    -  struct to store specific options for optimization
%
% Returns:
%  opttheta   -  optimized parameter vector
%
% Options (* required)
%  epochs*     - number of epochs through data
%  alpha*      - initial learning rate
%  minibatch*  - size of minibatch
%  momentum    - momentum constant, defualts to 0.9
%STOCHASTIC_GD Implements stochastic gradient descent
%   Detailed explanation goes here

%%======================================================================
%% Setup
assert(all(isfield(options,{'epochs','alpha','minibatch'})),...
        'Some options not defined');
if ~isfield(options,'momentum')
    options.momentum = 0.9;
end;
epochs = options.epochs;
alpha = options.alpha;
batch_size = options.minibatch;

% Setup for momentum
mom = 0.5;
momIncrease = 20;

theta_sgd = theta;
old_theta_sgd = theta_sgd;    % To test for convergence
g_prev = [];     % previous gradient for computing momentum term.
errors = zeros(epochs, 1);

iteration = 0;

[f, ~] = fun(theta_sgd, X, y, layers, options);

fprintf('Start Error: %f\n', f);
   
tic;

it = 0;
ep = 0;
while iteration < epochs
    ep = ep + 1;
    permuted_indices = randperm(size(X, 3), size(X, 3));
    for batch_number = 0:(floor(size(X, 3) / batch_size) - 1)
        it = it + 1;
        % increase momentum after momIncrease iterations
        if it == momIncrease
            mom = options.momentum;
        end
    
        batch_start = batch_number * batch_size + 1;
        batch_end = batch_start + batch_size - 1;
        batch.X = X(:,:,permuted_indices(1, batch_start:batch_end));
        batch.y = y(:,permuted_indices(1, batch_start:batch_end));
        delta = batch_size / (batch_size * (batch_number+1));
        
        [~, g] = fun( theta_sgd, batch.X, batch.y, layers, options);
        g_current = - delta * alpha * g;
%       g_current = - alpha * g;
        if ~isempty(g_prev)
            g_current = g_current + mom * g_prev;
        end
        old_batch_theta = theta_sgd;
        theta_sgd = theta_sgd + g_current;
        g_prev = g_current;
        
        batch_theta_change = sum(sum(abs(old_batch_theta - theta_sgd)));
        fprintf('Batch: %d, theta change:%f\n', ...
            batch_number, batch_theta_change);
    end
    iteration = iteration + 1;
    [f, ~] = fun(theta_sgd, X, y, layers, options);
    theta_change = sum(sum(abs(old_theta_sgd - theta_sgd)));
    old_theta_sgd = theta_sgd;
    errors(iteration) = f;
    
    % aneal learning rate by factor of two after each epoch
    alpha = alpha/2.0;
    
    fprintf('epochs: %d, error:%f, sum of theta diff: %f\n', ...
        ep, f, theta_change);
end

fprintf('Batch of %d Gradient Descent took %f seconds after %d full iterations.\n', batch_size, toc, iteration);

end

