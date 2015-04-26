close all; clear all;clc;
addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled

% Load the MNIST data for this exercise.
% train.X and test.X will contain the training and testing images.
%   Each matrix has size [n,m] where:
%      m is the number of examples.
%      n is the number of pixels in each image.
% train.y and test.y will contain the corresponding labels (0 to 9).
binary_digits = false;
num_classes = 10;
[train,test] = ex1_load_mnist(binary_digits);

% Add row of 1s to the dataset to act as an intercept term.
train.X = [ones(1,size(train.X,2)); train.X]; 
test.X = [ones(1,size(test.X,2)); test.X];
train.y = train.y+1; % make labels 1-based.
test.y = test.y+1; % make labels 1-based.

% Training set info
m=size(train.X,2);
n=size(train.X,1);

% Train softmax classifier using minFunc
options = struct('MaxIter', 200);

% Initialize theta.  We use a matrix where each column corresponds to a class,
% and each row is a classifier coefficient for that class.
% Inside minFunc, theta will be stretched out into a long vector (theta(:)).
theta = rand(n,num_classes)*0.001;

%% ------------------------------------------------------------------------
theta = rand(n,num_classes)*0.001;

% Call minFunc with the softmax_regression.m file as objective.
%
% TODO:  Implement batch softmax regression in the softmax_regression.m
% file using a vectorized implementation.
%
tic;
theta(:)=minFunc(@softmax_regression, theta(:), options, train.X, train.y);
fprintf('Optimization took %f seconds.\n', toc);

% minFunc training/test accuracy.
accuracy = multi_classifier_accuracy(theta,train.X,train.y);
fprintf('Training accuracy: %2.1f%%\n', 100*accuracy);
accuracy = multi_classifier_accuracy(theta,test.X,test.y);
fprintf('Test accuracy: %2.1f%%\n', 100*accuracy);

%% gradient check ---------------------------------------------------------
epsilon = 0.00001;
num_trials = 785;

[f, gradient] = softmax_regression(theta(:), train.X, train.y);
errors = gradient_checker( ...
    @softmax_regression, theta(:), g, train, num_trials, epsilon);

fprintf('gradient checker maxium error: %1.5e\n', max(errors));

%% gradient descent -------------------------------------------------------
theta_gd = rand(n,num_classes)*0.001;
[theta_gd, gd_iters, gd_errs] = gradient_descent( ... 
    @softmax_regression, theta_gd, train.X, train.y, 200000, 1000);

%% stochastic gradient descent --------------------------------------------
theta_sgd = rand(n,num_classes)*0.001;
[theta_sgd, sgd_iters, sgd_errs] = stochastic_gd( ... 
    @softmax_regression, theta_sgd, train.X, train.y, 100000, 1, 0.001);

%% Batch stochastic gradient descent --------------------------------------------
theta_bsgd = rand(n,num_classes)*0.001;
[theta_bsgd, bsgd_iters, bsgd_errs] = stochastic_gd( ... 
    @softmax_regression, theta_bsgd, train.X, train.y, 100000, 100, 0.001);

%% Test errors ------------------------------------------------------------
% gradient descent.
accuracy = multi_classifier_accuracy(theta_gd,train.X,train.y);
fprintf('Gradient Descent training accuracy: %2.1f%%\n', 100*accuracy);
accuracy = multi_classifier_accuracy(theta_gd,test.X,test.y);
fprintf('Gradient Descent  test accuracy: %2.1f%%\n', 100*accuracy);

% stochastic gradient descent.
accuracy = multi_classifier_accuracy(theta_sgd,train.X,train.y);
fprintf('Stochastic Gradient Descent training accuracy: %2.1f%%\n', ...
    100*accuracy);
accuracy = multi_classifier_accuracy(theta_sgd,test.X,test.y);
fprintf('Stochastic Gradient Descent  test accuracy: %2.1f%%\n', ... 
    100*accuracy);

% Batch stochastic gradient descent.
accuracy = multi_classifier_accuracy(theta_bsgd,train.X,train.y);
fprintf('Batch Stochastic Gradient Descent training accuracy: %2.1f%%\n', ...
    100*accuracy);
accuracy = multi_classifier_accuracy(theta_bsgd,test.X,test.y);
fprintf('Batch Stochastic Gradient Descent  test accuracy: %2.1f%%\n', ... 
    100*accuracy);

plot(1:gd_iters, gd_errs(gd_errs ~= 0), 1:sgd_iters, ...
    sgd_errs(sgd_errs ~=0), 1:bsgd_iters, bsgd_errs(bsgd_errs ~= 0));
title('2-D Line Plot Softmax');
ylabel('Objective Function');
xlabel('Iteration');
legend('Gradient Descent', 'Stochastic Gradient Descent', 'Batch Gradient Descent');