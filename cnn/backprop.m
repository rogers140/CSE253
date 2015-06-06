function grad_layers = backprop(options, layers, output_error)
% options: training options
% layers: the network
% output_error: cross entroy n*c matrix
% Create new layers with the input layers to store gradients.

    % TODO: this will take up unnecessary memory: only the current layer
    % and the previous layer is ever used.
    deltas_stack = cell(size(layers));
    grad_layers = cell(size(layers));

    layer_count = numel(layers);

    % Calculate delta for the other layers
    for l = layer_count : -1 : 1
        grad_layers{l}.name = layers{l}.name;
        % compute delta for the current layer.
        switch layers{l}.name
            case 'input'
            case {'pooling', 'convolution'}
                x = size(layers{l}.activation, 1);
                y = size(layers{l}.activation, 2);
                z = size(layers{l}.activation, 3);
                n = size(layers{l}.activation, 4);
                
                if strcmp(layers{l+1}.name, 'convolution') 
                    padX = layers{l+1}.X - 1;
                    padY = layers{l+1}.Y - 1;
                    deltas_stack{l} = zeros(x, y, z, n);
                    deltas = padarray(deltas_stack{l+1}, [padX padY]);
                    %filters = rot90(layers{l+1}.weights, 2);
                    for imageNum = 1:n
                        for p = 1:z
                            for q = 1:size(layers{l+1}.weights, 3)
                                filter = layers{l+1}.weights(:, :, q, p);
                                
                                deltas_stack{l}(:, :, p, imageNum) = ...
                                    deltas_stack{l}(:, :, p, imageNum) + ...
                                    conv2(deltas(:, :, q, imageNum), ...
                                        filter, 'valid');
                            end
                        end
                    end
                    deltas_stack{l} = deltas_stack{l} .* ...
                        actVal2Deriv(layers{l}.activation,...
                        layers{l}.actFunc);
                    
                elseif strcmp(layers{l+1}.name, 'pooling')
                    dup = ones(layers{l+1}.X, layers{l+1}.Y);
                    map = arrayfun(@(x) kron(x, dup), ...
                        deltas_stack{l+1},'UniformOutput', false);
                    if strcmp(layers{l+1}.type, 'max')
                        deltas_stack{l} = cell2mat(map) .* ...
                            layers{l+1}.maxMap;
                    elseif strcmp(layers{l+1}.type, 'mean')
                        deltas_stack{l} = cell2mat(map) ./ ...
                            (layers{l}.X * layers{l}.Y);
                    end
                    
                else
                    % Everything else: Output, Fully
                    numImages = size(layers{l+1}.activation, 2);
                    deltas = actVal2Deriv(layers{l+1}.input, ...
                        layers{l}.actFunc)' .* ...
                        (deltas_stack{l+1} * layers{l+1}.weights);
                    deltas_stack{l} = reshape(deltas', x, y, z, numImages);
                    
                end
            case 'fully'
                deltas_stack{l} = ...
                actVal2Deriv(layers{l}.activation, ...
                        layers{l}.actFunc)' .* ...
                (deltas_stack{l+1} * layers{l+1}.weights);
            case 'output'
                deltas_stack{l} = output_error;
            otherwise
                warning('Unexpected layer identifier.');
        end 
        
        % gradients.
        if strcmp(layers{l}.name, 'convolution')
            z = size(layers{l}.activation, 3);
            n = size(layers{l}.activation, 4);
            
            % create a gradient for each filter by convolving each feature
            % of every sample of the input into the layer by the delta of
            % each filter, and summing them up.
            grad_layers{l}.weights = zeros(size(layers{l}.weights));
            for filterNum = 1:z
                for featureNum = 1:size(layers{l}.input,3)
                    for imageNum = 1:n
                        rotdelta = rot90( ...
                            deltas_stack{l}(:, :, filterNum, imageNum), 2);
                        grad_layers{l}.weights(:, :, featureNum, filterNum) = ...
                            grad_layers{l}.weights(:, :, featureNum, filterNum) + ...
                            conv2( ...
                                layers{l}.input(:,:,featureNum,imageNum), ...
                                rotdelta, 'valid');
                    end
                end
            end
            grad_layers{l}.weights = grad_layers{l}.weights + ...
                options.lambda * layers{l}.weights;
            
            % the gradient of the bias is the sum of all filter dimensions
            % and samples.
            grad_layers{l}.bias = sum(sum(sum(deltas_stack{l}, 4), 2), 1);
        elseif strcmp(layers{l}.name, 'fully') ...
            || strcmp(layers{l}.name, 'output')
            grad_layers{l}.weights = ...
                (layers{l}.input * deltas_stack{l})' ...
                + options.lambda * layers{l}.weights;
            grad_layers{l}.bias = sum(deltas_stack{l}, 1)';
        end
    end
end