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
reg_term = ei.lambda / 2 * reg_term;

%% predict, compute cost.
% raw_output = sigmoid(hAct{numHidden+1});
raw_output = hAct{numHidden+1};
[pred_prob, cost, der_matrix]...
    = crossEntropy(raw_output', labels);


%if ~isempty(labels)
%    cost = cost + reg_term; % L2 regularization TODO needs to consider in error.
%end

%% return here if only predictions desired.
if po
    grad = [];
    return;
end;

%% compute gradients using backpropagation
gradStack = backprop(ei, stack, hAct, der_matrix, data);
cost = cost + reg_term;
[grad] = stack2params(gradStack);
end



