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

%% TODO: load face data
[processed_training_data, processed_test_data] = ...
    load_preprocess('NimStim', [64 64], [96 96], [8 8]);

label_map = containers.Map({'23M','24M','25M','26M','27M','28M','29M',...
    '29m','30M','31M','32M','33M','34M','35M','36M','37M','38M','39M',...
    '40M','41M','42M','43M','45M'}, 1:23 );
        
[data_train, labels_train, data_test, labels_test] = ...
    proprocessed_data_to_nn_data( processed_training_data, ...
    processed_test_data, 1, label_map );


%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)


%TODO: decide proper hyperparameters.
% dimension of input features FOR YOU TO DECIDE
ei.input_dim = 40;
% number of output classes FOR YOU TO DECIDE
ei.output_dim = 23;
% sizes of all hidden layers and the output layer FOR YOU TO DECIDE
ei.layer_sizes = [25, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 1;
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
[~, pred_list] = max(pred');
[~, label_list] = max(labels_test');
acc_test = mean(pred_list==label_list);
fprintf('test accuracy: %f\n', acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~, pred_list] = max(pred');
[~, label_list] = max(labels_train');
acc_train = mean(pred_list==label_list);
fprintf('train accuracy: %f\n', acc_train);
