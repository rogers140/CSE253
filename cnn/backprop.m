function grad_layers = backprop(options, layers, output_error)
% options: training options
% layers: the network
% output_error: cross entroy n*c matrix
% Create new layers with the input layers to store gradients.

    % TODO: this will take up unnecessary memory: only the current layer
    % and the previous layer is ever used.
    deltas_stack = cell(size(layers));
    layer_input= [];
    grad_layers = layers;

    layer_count = numel(layers);

    % Calculate delta for the other layers
    for l = layer_count : -1 : 1
        % compute delta for the current layer.
        switch layers{l}.name
            case 'input'
                error('backprop for input not implemented');
            case {'pooling', 'convolution'}
                
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
        if strcmp(layers{l}.name, 'convolution') ...
           || strcmp(layers{l}.name, 'fully') ...
           || strcmp(layers{l}.name, 'output')
            grad_layers{l}.W = ...
                (layers{l}.input * deltas_stack{l})' ...
                + options.lambda * layers{l}.W;
            grad_layers{l}.b = sum(deltas_stack{l}, 1)';
        end
    end
end