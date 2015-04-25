close all;clear all;clc;
addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled

% Load the MNIST data for this exercise.
% train.X and test.X will contain the training and testing images.
%   Each matrix has size [n,m] where:
%      m is the number of examples.
%      n is the number of pixels in each image.
% train.y and test.y will contain the corresponding labels (0 or 1).
binary_digits = true;
[train,test] = ex1_load_mnist(binary_digits);

% Add row of 1s to the dataset to act as an intercept term.
train.X = [ones(1,size(train.X,2)); train.X]; 
test.X = [ones(1,size(test.X,2)); test.X];

% Training set dimensions
m=size(train.X,2);
n=size(train.X,1);

% Train logistic regression classifier using minFunc
options = struct('MaxIter', 100);

% First, we initialize theta to some small random values.
theta = rand(n,1)*0.001;

% Call minFunc with the logistic_regression.m file as the objective function.
%
% TODO:  Implement batch logistic regression in the logistic_regression.m
% file! Remember to use MATLAB's vectorization features to speed up your code.
%


tic;
theta=minFunc(@logistic_regression, theta, options, train.X, train.y);
fprintf('Optimization took %f seconds.\n', toc);

t = sigmoid(theta' * train.X);
f = - sum( (train.y .* log(t))  +  ((1-train.y) .* log(1 - t)) );
fprintf('error: %f\n',f);
   


% TODO:  1) Write your own gradient check code and check the gradient
%           calculated above.
%        2) Use stochastic gradient descent for this problem.
%        *3) Use batch gradient descent.
%        4) Plot speed of convergence for 2 (and 3) (loss function - # of iteration)
%        5) Compute training time and accuracy of train & test data.

%gradient check
g = (( t - train.y) * train.X')';
epsilon = 0.00001;
success = 1;
for i=1:size(g,1)
    theta_plus = theta;
    theta_minus = theta;
    
    theta_plus(i) = theta_plus(i) + epsilon;
    theta_minus(i) = theta_minus(i) - epsilon;
    
    t_plus = sigmoid(theta_plus' * train.X);
    f_plus = - sum( (train.y .* log(t_plus))  +  ((1-train.y) .* log(1 - t_plus)) );
    
    t_minus = sigmoid(theta_minus' * train.X);
    f_minus = - sum( (train.y .* log(t_minus))  +  ((1-train.y) .* log(1 - t_minus)) );
    difference = (f_plus - f_minus)/(2*epsilon) - g(i);
    if(abs(difference) > 0.0001)
        success = 0;
        break;
    end
end
if(success == 1)
    fprintf('Gradient check passed.\n');
else
    fprintf('Gradient check failed.\n');
end

% Gradient Descent
theta_gd = rand(n,1)*0.001;
count = 0;
eta = 1 / size(train.X, 2);

t = sigmoid(theta_gd' * train.X);
f = - sum( (train.y .* log(t))  +  ((1-train.y) .* log(1 - t)) );
fprintf('Start error: %f\n',f);
gd_y = [];

tic;
while(count < 2000000)
  g = (( sigmoid(theta_gd' * train.X) - train.y) * train.X')';
  theta_gd = theta_gd - eta * g;
  
  t = sigmoid(theta_gd' * train.X);
  f = - sum( (train.y .* log(t))  +  ((1-train.y) .* log(1 - t)) );
  % fprintf('error: %f, theta change: %f\n',f, sum(abs(g)));
  gd_y = [gd_y f];
  count = count + 1;
  if(sum(abs(g)) < 100)
    fprintf('Small enough.\n');
    break;
  end
end
fprintf('Gradient Descent took %f seconds.\n', toc);

% Stoichastic/Batch Gradient Descent
batch_size = 100;
theta_sgd = rand(n,1)*0.001;
old_theta_sgd = theta_sgd;    % To test for convergence
converged = 0;
sgd_y = [];

eta = 0.001;
iteration = 0;

t = sigmoid(theta_sgd' * train.X);
f = - sum( (train.y .* log(t))  +  ((1-train.y) .* log(1 - t)) );
theta_change = sum(abs(old_theta_sgd - theta_sgd));
% fprintf('Batch Start Error: %f\n', f);
   
tic;
while(converged == 0)
    for batch_number = 0:(floor(size(train.X, 2) / batch_size) - 1)
        batch_start = batch_number * batch_size + 1;
        batch_end = batch_start + batch_size - 1;
        batch.X = train.X(:,batch_start:batch_end);
        batch.y = train.y(batch_start:batch_end);
        if(iteration > 0)
            delta = 1 / size(train.X, 2);
        else
            delta = batch_size / (batch_size * (batch_number+1) );
        end
        
        g = (( sigmoid(theta_sgd' * batch.X) - batch.y) * batch.X')';
        theta_sgd = theta_sgd - delta * eta * g;
    end
    iteration = iteration + 1;
    t = sigmoid(theta_sgd' * train.X);
    f = - sum( (train.y .* log(t))  +  ((1-train.y) .* log(1 - t)) );
    theta_change = sum(abs(old_theta_sgd - theta_sgd));
    % fprintf('Iteration: %d, error: %f, theta change: %f, wrongness: %f\n',iteration, f, theta_change, sum(abs(theta_sgd - theta)));
    old_theta_sgd = theta_sgd;
    sgd_y = [sgd_y f];
    
    if(theta_change < 0.0005)
        converged = 1;
    end
end
fprintf('Batch of %d Gradient Descent took %f seconds after %d full iterations.\n', batch_size, toc, iteration);


% Example of printing out training/test accuracy.
accuracy = binary_classifier_accuracy(theta,train.X,train.y);
fprintf('MinFunc Training accuracy: %2.1f%%\n', 100*accuracy);
accuracy = binary_classifier_accuracy(theta,test.X,test.y);
fprintf('MinFunc Test accuracy: %2.1f%%\n', 100*accuracy);

% Example of printing out GD training/test accuracy.
accuracy = binary_classifier_accuracy(theta_gd,train.X,train.y);
fprintf('GD Training accuracy: %2.1f%%\n', 100*accuracy);
accuracy = binary_classifier_accuracy(theta_gd,test.X,test.y);
fprintf('GD Test accuracy: %2.1f%%\n', 100*accuracy);

% Example of printing out Stoichastic GD training/test accuracy.
accuracy = binary_classifier_accuracy(theta_sgd,train.X,train.y);
fprintf('SGD Training accuracy: %2.1f%%\n', 100*accuracy);
accuracy = binary_classifier_accuracy(theta_sgd,test.X,test.y);
fprintf('SGD Test accuracy: %2.1f%%\n', 100*accuracy);

title('2-D Line Plot');
plot(1:iteration, sgd_y, 1:count, gd_y);
ylabel('Objective Function');
xlabel('Iteration');
legend('Batch Gradient Descent', 'Gradient Descent');