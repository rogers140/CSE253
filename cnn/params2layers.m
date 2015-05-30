function layers = params2layers( params, layers )
%PARAMS2LAYERS Return a layers based on input params and layers
%    Convert params back into weights and bias and put them in layers.

depth = numel(layers);
param_count = 1;

for l = 1:depth
    if strcmp(layers{l}.name, 'convolution') ...
       || strcmp(layers{l}.name, 'fully') ...
       || strcmp(layers{l}.name, 'output')
        num_weights = numel(layers{l}.weights);
        num_bias = numel(layers{l}.bias);
        
        % convert params to weights
        layers{l}.weights(:) = ...
            params(param_count : param_count + num_weights - 1);
        
        param_count = param_count + num_weights;
        
        % convert params to bias
        layers{l}.bias(:) = ...
            params(param_count : param_count + num_bias - 1);
        
        param_count = param_count + num_bias;
    end
end

