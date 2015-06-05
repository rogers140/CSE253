function layers = aTeamCnnInitParams(layers)
% Initialize parameters for a single layer convolutional neural
% network followed by a softmax layer.
%                            
% Parameters:
%  layers - cell matrix that contains all the layer information
%
%
% Returns:
%  theta      -  unrolled parameter vector with initialized weights
%  layers     -  initialize the weights for all the layers


%% Loop through layers and initialize them
hiddenSize = 0;
dim = [0 0 0];
for l = 1:size(layers, 1)
    switch layers{l}.name
        case 'input'
            dim(1) = layers{l}.X;
            dim(2) = layers{l}.Y;
            dim(3) = layers{l}.Z;
            hiddenSize = dim(1) * dim(2) * dim(3);
        case 'pooling'
            assert(mod(dim(1), layers{l}.X)==0,...
                'X poolDim must divide imageDim - filterDim + 1');
            assert(mod(dim(2), layers{l}.Y)==0,...
                'Y poolDim must divide imageDim - filterDim + 1');
            assert(mod(dim(3), layers{l}.Z)==0,...
                'Z poolDim must divide imageDim - filterDim + 1');
   
            dim(1) = dim(1) / layers{l}.X;
            dim(2) = dim(2) / layers{l}.Y;
            dim(3) = dim(3) / layers{l}.Z;
            hiddenSize = dim(1) * dim(2) * dim(3);
        case 'convolution'
            assert(layers{l}.X < dim(1), ...
                'filterDim X must be less than imageDim X');
            assert(layers{l}.Y < dim(2), ...
                'filterDim Y must be less than imageDim Y');
            assert(layers{l}.Z == dim(3), ...
                'filterDim Z must equal that of imageDim Z');
            
            layers{l}.weights = 1e-1*randn(...
                layers{l}.X, layers{l}.Y, layers{l}.Z, ...
                layers{l}.numFilters);
            layers{l}.bias = zeros(layers{l}.numFilters, 1);
            
            dim(1) = dim(1) - layers{l}.X + 1;
            dim(2) = dim(2) - layers{l}.Y + 1;
            dim(3) = layers{l}.numFilters;
            hiddenSize = dim(1) * dim(2) * dim(3);
        case 'fully'
            r  = sqrt(6) / sqrt(layers{l}.units + hiddenSize+1);
            layers{l}.weights = rand(layers{l}.units, hiddenSize) * 2 * r - r;
            layers{l}.bias = zeros(layers{l}.units, 1);
            
            hiddenSize = layers{l}.units;
            dim = [0 0 0];
        case 'output'
            r  = sqrt(6) / sqrt(layers{l}.units + hiddenSize+1);
            layers{l}.weights = rand(layers{l}.units, hiddenSize) * 2 * r - r;
            layers{l}.bias = zeros(layers{l}.units, 1);
            
            hiddenSize = layers{l}.units;
            dim = [0 0 0];
        otherwise
            warning('Unexpected layer identifier.');
    end
end

end

