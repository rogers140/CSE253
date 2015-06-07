close all;clear all;clc;
%% Convolution Neural Network Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started in building a single.
%  layer convolutional nerual network. In this exercise, you will only
%  need to modify cnnCost.m and cnnminFuncSGD.m. You will not need to 
%  modify this file.

%%======================================================================
%% STEP 0: Initialize Parameters and Load Data
%  Here we initialize some parameters used for the exercise.

% Configuration
layers = parseNetwork('network_required.txt');
imageDimX = layers{1}.X;
imageDimY = layers{1}.Y;

% Load MNIST Train
addpath ../common/;
images = loadMNISTImages('../common/train-images-idx3-ubyte');
images = reshape(images,imageDimX,imageDimY,[]);
labels = loadMNISTLabels('../common/train-labels-idx1-ubyte');

% TODO: Remove ... this is added to speed up the testing
%images = images(:, :, 1:500);
%labels = labels(1:500, :);

labels(labels==0) = 10; % Remap 0 to 10
label_mat = labels2mat(labels);

options = [];
options.lambda = 0.06;

% Initialize Parameters
layers = aTeamCnnInitParams(layers);
theta = layers2params(layers);
%%======================================================================
%% STEP 1: Implement convNet Objective
%  Implement the function cnnCost.m.

%%======================================================================
%% STEP 2: Gradient Check
%  Use the file computeNumericalGradient.m to check the gradient
%  calculation for your cnnCost.m function.  You may need to add the
%  appropriate path or copy the file to this directory.

DEBUG=false;  % set this to true to check gradient
if DEBUG
    % To speed up gradient checking, we will use a reduced network and
    % a debugging data set
    sample_images = images(:,:,1:5);
    sample_labels = label_mat(:,1:5);
    [~, grad, ~] = aTeamCnnCost(theta, sample_images, sample_labels, ...
        layers, options);

    % Check gradients
    numGrad = computeNumericalGradient( @(x) aTeamCnnCost(x, ...
        sample_images, sample_labels, layers, options), theta );
 
    % Use this to visually compare the gradients side by side
    disp([numGrad grad]);
    
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    % Should be small. In our implementation, these values are usually 
    % less than 1e-9.
    disp(diff); 
 
    assert(diff < 1e-9,...
        'Difference too large. Check your gradient computation again');
    
end;

%%======================================================================
%% STEP 3: Learn Parameters
%  Implement minFuncSGD.m, then train the model.

options.epochs = 3;
options.minibatch = 200; %200 for relu, 256 for tanh, 500 for sigmoid, 
options.alpha = 0.0003;  %0.0003 for relu, 0.001 for tanh and sigmoid
options.momentum = .95;

% opttheta = minFuncSGD(@(x,y,z,l) aTeamCnnCost( ...
%                 x,y,z,l,numClasses,filterDim, ...
%                 numFilters,poolDim, options), ...
%                 theta,images,label_mat,layers,options);
train_images = images(:,:,1:60000);
train_labels = label_mat(:,1:60000);
[opttheta, mom_iteration, mom_errors] = stochastic_gd(@(x,y,z,l,o) aTeamCnnCost(...
                  x,y,z,l,o), theta, train_images, train_labels,layers, options);

%%======================================================================
%% STEP 4: Test
%  Test the performance of the trained model using the MNIST test set. Your
%  accuracy should be above 97% after 3 epochs of training

testImages = loadMNISTImages('../common/t10k-images-idx3-ubyte');
testImages = reshape(testImages,imageDimX,imageDimY,[]);
testLabels = loadMNISTLabels('../common/t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10
test_label_mat = labels2mat(testLabels);

testImages = testImages(:,:,:);
testLabels = test_label_mat(:,:);

[cost, grads, preds]=aTeamCnnCost(opttheta, testImages,...
    testLabels, layers, options, true);
[~,real_result] = max(test_label_mat, [], 1);
acc = sum(preds==real_result')/size(real_result, 2);

% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f\n',acc);

% Plot
plot(1:mom_iteration, mom_errors);
title('CNN SGD');
ylabel('Objective Function');
xlabel('Epoc');
legend('Momentum SGD');