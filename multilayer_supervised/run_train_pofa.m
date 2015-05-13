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

%% load face data
% data: n by input_dim number of features.
% label:  n by output_dim number of 1s and 0s
[processed_training_data, processed_test_data] = ...
    load_preprocess('POFA', [64 64], [96 96], [8 8]);

label_map = containers.Map({'AN', 'DI' ,'FE', 'HA', 'SA', 'SP'}, ...
            1:6 );
        
[data_train, labels_train, data_test, labels_test] = ...
    proprocessed_data_to_nn_data( processed_training_data, ...
    processed_test_data, 2, label_map );

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = size(data_train, 2);
% number of output classes
ei.output_dim = size(label_map, 1);
% sizes of all hidden layers and the output layer
ei.layer_sizes = [25, ei.output_dim];   % TODO: adjust?
% scaling parameter for l2 weight regularization penalty
ei.lambda = 0.06;   % TODO: adjust?
% which type of activation function to use in hidden layers
% feel free to implement support for different activation function
ei.activation_fun = 'logistic';
ei.eta = .01; % TODO: Change

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';

%% run training
fprintf('Start Training\n');
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);

% TODO:  1) check the gradient calculated by supervised_dnn_cost.m
%        2) Decide proper hyperparamters and train the network.
%        3) Implement SGD version of solution.
%        4) Plot speed of convergence for 1 and 3.
%        5) Compute training time and accuracy of train & test data.

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~, pred_list] = max(pred');
[~, label_list] = max(labels_test');
acc_test = mean(pred_list==label_list);
fprintf('test accuracy: %f\n', acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~, pred_list] = max(pred');
[~, label_list] = max(labels_train');
acc_train = mean(pred_list==label_list);
fprintf('train accuracy: %f\n', acc_train);
