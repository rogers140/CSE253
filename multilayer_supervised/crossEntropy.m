function [pred, cost, der_matrix] = ...
    crossEntropy(output, label)
%output is data_size*class_num matrix before softmax (directly from network)
%label is data_size*class_size matrix for real label
%it will give output of crossEntropy
cost = -1;
num_classes = size(output,2);
pred = exp(output);
denominator = repmat(sum(pred, 2), [1, num_classes]);
pred = pred ./ denominator; %softmax
der_matrix = zeros(size(output));
if isempty(label)
    return;
end
cost_matrix = (-1)*log(pred).* label;
der_matrix = (pred - label);
cost = sum(cost_matrix(:));
end