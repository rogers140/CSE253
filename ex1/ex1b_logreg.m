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

%% ------------------------------------------------------------------------
% Call minFunc with the logistic_regression.m file as the objective function.
%
theta = rand(n,1)*0.001;
tic;
theta=minFunc(@logistic_regression, theta, options, train.X, train.y);
fprintf('Optimization took %f seconds.\n', toc);

t = sigmoid(theta' * train.X);
f = - sum( (train.y .* log(t))  +  ((1-train.y) .* log(1 - t)) );
fprintf('error: %f\n',f);

%% gradient check ---------------------------------------------------------
epsilon = 0.00001;

gradient = (( sigmoid(theta' * train.X) - train.y) * train.X')';
errors = gradient_checker( ...
    @logistic_regression, theta, gradient, train, size(theta,1), epsilon );

fprintf('gradient checker maxium error: %1.5e\n', max(errors));

%% Gradient Descent -------------------------------------------------------
theta_gd = rand(n,1)*0.001;
[theta_gd, gd_iters, gd_errs] = gradient_descent( ... 
    @logistic_regression, theta_gd, train.X, train.y, 200000, 100);

%% Stoichastic ---------------------- -------------------------------------
theta_sgd = rand(n,1)*0.001;
[theta_sgd, sgd_iters, sgd_errs] = stochastic_gd( ... 
    @logistic_regression, theta_sgd, train.X, train.y, 100000, 1, 0.001);

%% Batch Gradient Descent -------------------------------------------------
theta_bsgd = rand(n,1)*0.001;
[theta_bsgd, bsgd_iters, bsgd_errs] = stochastic_gd( ... 
    @logistic_regression, theta_bsgd, train.X, train.y, 100000, 100, 0.001);

%% ------------------------------------------------------------------------
% minFun training/test accuracy.
accuracy = binary_classifier_accuracy(theta,train.X,train.y);
fprintf('MinFunc Training accuracy: %2.1f%%\n', 100*accuracy);
accuracy = binary_classifier_accuracy(theta,test.X,test.y);
fprintf('MinFunc Test accuracy: %2.1f%%\n', 100*accuracy);

% GD training/test accuracy.
accuracy = binary_classifier_accuracy(theta_gd,train.X,train.y);
fprintf('GD Training accuracy: %2.1f%%\n', 100*accuracy);
accuracy = binary_classifier_accuracy(theta_gd,test.X,test.y);
fprintf('GD Test accuracy: %2.1f%%\n', 100*accuracy);

% Stoichastic GD training/test accuracy.
accuracy = binary_classifier_accuracy(theta_sgd,train.X,train.y);
fprintf('SGD Training accuracy: %2.1f%%\n', 100*accuracy);
accuracy = binary_classifier_accuracy(theta_sgd,test.X,test.y);
fprintf('SGD Test accuracy: %2.1f%%\n', 100*accuracy);

% Batch Stoichastic GD training/test accuracy.
accuracy = binary_classifier_accuracy(theta_bsgd,train.X,train.y);
fprintf('Batch SGD Training accuracy: %2.1f%%\n', 100*accuracy);
accuracy = binary_classifier_accuracy(theta_bsgd,test.X,test.y);
fprintf('Batch SGD Test accuracy: %2.1f%%\n', 100*accuracy);

plot(1:gd_iters, gd_errs(gd_errs ~= 0), 1:sgd_iters, ...
    sgd_errs(sgd_errs ~=0), 1:bsgd_iters, bsgd_errs(bsgd_errs ~= 0));
title('2-D Line Plot Logistic Regression');
ylabel('Objective Function');
xlabel('Iteration');
legend('Gradient Descent', 'Stochastic Gradient Descent', 'Batch Gradient Descent');