function params = layers2params( layers )
%LAYERS2PARAMS Extract a vector of params from layers
%   Optimized param extraction from layers.

depth = numel(layers);

% weighted_layers = {'convolution' 'fully' 'output'};

% Compute the space needed.
num_params = 0;
for l = 1:depth
    if strcmp(layers{l}.name, 'convolution') ...
       || strcmp(layers{l}.name, 'fully') ...
       || strcmp(layers{l}.name, 'output')
            num_params = num_params + numel(layers{l}.weights) ...
                + numel(layers{l}.bias);
    end
end

params = zeros(num_params, 1);
param_count = 1;

% store relevant layers' weights and bias in to params.
for l = 1:depth
    if strcmp(layers{l}.name, 'convolution') ...
       || strcmp(layers{l}.name, 'fully') ...
       || strcmp(layers{l}.name, 'output')
        
        layer_param_count = numel(layers{l}.weights) ...
                + numel(layers{l}.bias);
        params(param_count : param_count + layer_param_count - 1, 1) = ...
            [layers{l}.weights(:); layers{l}.bias(:)];
        
        param_count = param_count + numel(layers{l}.weights) ...
                + numel(layers{l}.bias);

        % TODO some asserts maybe useful here during debugging:
        % - matching dimensions for densely connected layers
        % - bias/weight dimensions agree with filter dimensions in
        %   convolution layers ...
        % etc.
    end
end

end

