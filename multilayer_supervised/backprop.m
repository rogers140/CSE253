function [ grad_stack ]...
    = backprop(ei, weight_stack, activation_stack, output_error, data)
% ei: network information
% weight_stack: the original stack with all the weights
% activation_stack: has the activation of all the units
% output_error: cross entroy n*c matrix
% Create 3 stacks with the same dimensions as the original stack

    deltas_stack = cell(2,1);
    grad_stack = weight_stack;

    layer_count = numel(ei.layer_sizes);

    % Calculate delta for the other layers
    for l = layer_count : -1 : 1
        % lower layer's input
        if l==1
            % Get the input instead of the activation function
            input = data';
        else
            input = sigmoid(activation_stack{l-1});
        end

        % compute delta for the current layer.
        if l < layer_count
            g = sigmoid(activation_stack{l});
            deltas_stack{l} = ...
            (g .* ( 1 - g ))' .* ...
            (deltas_stack{l+1} * weight_stack{l+1}.W);
        else 
            deltas_stack{layer_count} = output_error;
        end

        % gradients.
        grad_stack{l}.W = (input * deltas_stack{l})';
        grad_stack{l}.b = sum(deltas_stack{l})';
    end
end