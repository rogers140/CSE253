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
label_map = containers.Map({'AN', 'DI' ,'FE', 'HA', 'SA', 'SP'}, ...
            1:6 );
        
pofa_ids = {'aa','cc','em','gs','jb','jj','jm','mf','mo','nr','pe','pf','sw','wf'};

% preliminary run
[processed_training_data, processed_test_data] = ...
    load_preprocess_POFA(pofa_ids{1}, [64 64], [96 96], [8 8], 8);

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
ei.layer_sizes = [30, ei.output_dim];   % TODO: adjust?
% scaling parameter for l2 weight regularization penalty
ei.lambda = .6;   % TODO: adjust?
% which type of activation function to use in hidden layers
% feel free to implement support for different activation function
ei.activation_fun = 'logistic';
ei.eta = .0007; % SGD step size
ei.beta = 0.7; % SGD momentum step

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';

%% run training
minfunc_outputs = cell(6, size(pofa_ids,2));
for id_ind = 1:size(pofa_ids,2)
    [processed_training_data, processed_test_data] = ...
    load_preprocess_POFA(pofa_ids{id_ind}, [64 64], [96 96], [8 8], 8);

    [data_train, labels_train, data_test, labels_test] = ...
        proprocessed_data_to_nn_data( processed_training_data, ...
        processed_test_data, 2, label_map );
    
    [opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
        params,options, ei, data_train, labels_train);

    %epsilon = 0.00001;
    %num_trials = 785;
    %[cost, grad, pred_prob] = supervised_dnn_cost(params, ei, data_train, labels_train);
    %errors = gradient_checker( ...
    %    @supervised_dnn_cost, params, grad, num_trials, epsilon, ei, data_train, labels_train);

    minfunc_outputs{1,id_ind} = opt_params;
    minfunc_outputs{2,id_ind} = output;
    minfunc_outputs{3,id_ind} = data_train;
    minfunc_outputs{4,id_ind} = labels_train;
    minfunc_outputs{5,id_ind} = data_test;
    minfunc_outputs{6,id_ind} = labels_test;
end
%% compute accuracy on the test and train set
for id_ind = 1:size(pofa_ids,2)
    fprintf('Results for actor %s - ', pofa_ids{id_ind});
    
    opt_params = minfunc_outputs{1,id_ind};
    output = minfunc_outputs{2,id_ind};
    data_train = minfunc_outputs{3,id_ind};
    labels_train = minfunc_outputs{4,id_ind};
    data_test = minfunc_outputs{5,id_ind};
    labels_test = minfunc_outputs{6,id_ind};
    
    [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
    [~, pred_list] = max(pred');
    [~, label_list] = max(labels_test');
    acc_test = mean(pred_list==label_list);
    fprintf('test accuracy: %f, ', acc_test);

    [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
    [~, pred_list] = max(pred');
    [~, label_list] = max(labels_train');
    acc_train = mean(pred_list==label_list);
    fprintf('train accuracy: %f\n', acc_train);
end

%%
[opt_params, iteration, errors] = stochastic_gd(@supervised_dnn_cost...
    , params, ei, data_train, labels_train, 5000, 1, 0);

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

%%
plot(minfunc_outputs{2,1}.trace.funcCount, minfunc_outputs{2,1}.trace.fval, ...
    minfunc_outputs{2,2}.trace.funcCount, minfunc_outputs{2,2}.trace.fval, ...
    minfunc_outputs{2,3}.trace.funcCount, minfunc_outputs{2,3}.trace.fval, ...
    minfunc_outputs{2,4}.trace.funcCount, minfunc_outputs{2,4}.trace.fval, ...
    minfunc_outputs{2,5}.trace.funcCount, minfunc_outputs{2,5}.trace.fval, ...
    minfunc_outputs{2,6}.trace.funcCount, minfunc_outputs{2,6}.trace.fval, ...
    minfunc_outputs{2,7}.trace.funcCount, minfunc_outputs{2,7}.trace.fval, ...
    minfunc_outputs{2,8}.trace.funcCount, minfunc_outputs{2,8}.trace.fval, ...
    minfunc_outputs{2,9}.trace.funcCount, minfunc_outputs{2,9}.trace.fval, ...
    minfunc_outputs{2,10}.trace.funcCount, minfunc_outputs{2,10}.trace.fval, ...
    minfunc_outputs{2,11}.trace.funcCount, minfunc_outputs{2,11}.trace.fval, ...
    minfunc_outputs{2,12}.trace.funcCount, minfunc_outputs{2,12}.trace.fval, ...
    minfunc_outputs{2,13}.trace.funcCount, minfunc_outputs{2,13}.trace.fval, ...
    minfunc_outputs{2,14}.trace.funcCount, minfunc_outputs{2,14}.trace.fval );
title('2-D Line Plot Minfunc 14 Actors');
ylabel('Objective Function');
xlabel('Iteration');