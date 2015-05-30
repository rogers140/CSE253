function [ grad_layers ]...
    = backprop(ei, layers, output_error)
% ei: network information
% weight_stack: the original stack with all the weights
% activation_stack: has the activation of all the units
% output_error: cross entroy n*c matrix
% Create 3 stacks with the same dimensions as the original stack

    deltas_stack = cell(size(layers));
    grad_layers = layers;

    layer_count = numel(layers);

    % Calculate delta for the other layers
    for l = layer_count : -1 : 1
        % compute delta for the current layer.
        switch layers{l}.name
            case 'input'
                error('backprop for input not implemented');
            case 'pooling'
                error('backprop for not implemented');
            case 'convolution'
                error('backprop for not implemented');
            case 'fully'
                deltas_stack{l} = ...
                actVal2Deriv(layers{l}.activation)' .* ...
                (deltas_stack{l+1} * layers{l+1}.weights);
            case 'output'
                deltas_stack{l} = output_error;
            otherwise
                warning('Unexpected layer identifier.');
        end

        % gradients.
        grad_layers{l}.W = (input * deltas_stack{l})' ...
            + ei.lambda * weight_stack{l}.W;
        grad_layers{l}.b = sum(deltas_stack{l}, 1)';
    end
end