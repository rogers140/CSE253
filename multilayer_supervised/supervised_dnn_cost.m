function [ cost, grad, pred_prob] = ...
    supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);

%% forward prop
reg_term = 0;
for i = 1:(numHidden+1)
    if i == 1
        hAct{i} = stack{i}.W * data' + repmat(stack{i}.b, 1, size(data,1));
    else
        hAct{i} = stack{i}.W * ...
            sigmoid(hAct{i-1}) + repmat(stack{i}.b, 1, size(data,1));
    end
    regMat = stack{i}.W .^ 2;
    reg_term = sum(regMat(:)) + sum((stack{i}.b .^ 2));
end
reg_term = ei.lambda * sqrt(reg_term);

%% predict, compute cost.
raw_output = sigmoid(hAct{numHidden+1});

[pred_prob, cost, cost_matrix, der_matrix]...
    = crossEntropy(raw_output', labels);


if ~isempty(labels)
    cost = cost + reg_term; % L2 regularization
end

%% return here if only predictions desired.

if po
    grad = [];
    return;
end;


%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

[gradStack , updated_weighted_stack] = backprop(ei, stack, hAct, der_matrix, data);

% Note: bias update does not care about inputs and the activation function
% bias[j] -= gamma_bias * 1 * delta[j]

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector


[grad] = stack2params(gradStack);
end



