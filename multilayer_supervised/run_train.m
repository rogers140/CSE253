close all; clear all;clc;
%% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));
addpath(genpath('../common/gabor'));

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)


%TODO: decide proper hyperparameters.
% dimension of input features FOR YOU TO DECIDE
ei.input_dim = ;
% number of output classes FOR YOU TO DECIDE
ei.output_dim = ;
% sizes of all hidden layers and the output layer FOR YOU TO DECIDE
ei.layer_sizes = [, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = ;
% which type of activation function to use in hidden layers
% feel free to implement support for different activation function
ei.activation_fun = 'logistic';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';

%% load data

%% run training
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);

% TODO:  1) check the gradient calculated by supervised_dnn_cost.m
%        2) Decide proper hyperparamters and train the network.
%        3) Implement SGD version of solution.
%        4) Plot speed of convergence for 1 and 3.
%        5) Compute training time and accuracy of train & test data.

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f\n', acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f\n', acc_train);
