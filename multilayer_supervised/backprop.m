function [ grad_stack, updated_weight_stack ]...
    = backprop(ei, weight_stack, activation_stack, output_error, data)
% ei: network information
% weight_stack: the original stack with all the weights
% activation_stack: has the activation of all the units
% output_error: cross entroy n*c matrix

% Create 3 stacks with the same dimensions as the original stack
deltas_stack = weight_stack;
updated_weight_stack = weight_stack;
gradient_stack = weight_stack;
grad_stack = weight_stack;

% Calculate sigma for the output layer
layer_count = numel(ei.layer_sizes);

% Calculate gradient + delta for the output unit
temp = sigmoid(activation_stack{layer_count})'; % n*c matrix
gradient_stack{layer_count}.W = sum(temp .* (1-temp)); % [1*c]
deltas_stack{layer_count}.W = sum(output_error .* temp .* (1-temp)); % [1*c]

temp = deltas_stack{layer_count}.W' * sum(activation_stack{layer_count - 1},2)';% [hu * hl]
updated_weight_stack{layer_count}.W = weight_stack{layer_count}.W + (ei.eta .* temp);
grad_stack{layer_count}.W = temp;

% Calculate delta for the other layers
for l = (layer_count-1) : -1 : 1
    % Sum deltas of with the previous/upper layer
    % [1 * hl ] = [1 * hu] * [hu * hl]
    sum_of_deltas = deltas_stack{l+1}.W * weight_stack{l+1}.W;
    
    % calculate gradient for each unit at each training sample
    temp = sigmoid(activation_stack{l}');    % [n * hl]
    gradient_stack{l}.W = sum(temp .* (1-temp)); % [1 * hl]
    
    % Calculate deltas for the current layer
    deltas_stack{l}.W = sum_of_deltas .* gradient_stack{l}.W; % [1 * hl]
    
    % Update the weights
    % [hu * hl ] = [hu * 1] * [1 * hl]
    if(l==1)
        % Get the input instead of the activation function
        temp = deltas_stack{l}.W' * sum(data);
    else
        temp = deltas_stack{l}.W' * sum(activation_stack{l - 1},2)';
    end
    updated_weight_stack{l}.W = weight_stack{l}.W + (ei.eta .* temp);
    grad_stack{l}.W = temp;
end

end