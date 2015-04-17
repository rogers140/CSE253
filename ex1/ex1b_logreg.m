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


% TODO:  1) Write your own gradient check code and check the gradient
%           calculated above.
%        2) Use gradient descent for this problem. 
%        3) Plot speed of convergence (loss function - # of iteration)
%        4) Compute accuracy of train & test data.




% Example of printing out training/test accuracy.
accuracy = binary_classifier_accuracy(theta,train.X,train.y);
fprintf('Training accuracy: %2.1f%%\n', 100*accuracy);
accuracy = binary_classifier_accuracy(theta,test.X,test.y);
fprintf('Test accuracy: %2.1f%%\n', 100*accuracy);

