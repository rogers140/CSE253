function [pooledFeatures, maxMap] = aTeamCnnPool(poolDim, convolvedFeatures, type)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region, aka, patch dimension.
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%  type - 'mean' or 'max'
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%  maxMap - matrix storing the max position for max-pooling, in the form
%                   maxMap(maxRow, maxCol, featureNum, imageNum)

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);
maxMap = zeros(convolvedDim, convolvedDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%
if strcmp(type, 'mean')
    % mean-pooling
    W = ones(poolDim, poolDim);
    totalNumInPatch = poolDim * poolDim;
    for imageNum = 1:numImages
        for filterNum = 1:numFilters
            % convoluting with ones matrix
            convolvedImage = conv2(convolvedFeatures(:, :, filterNum, imageNum), W, 'valid');
            % subsampling
            indices = 1:poolDim:convolvedDim;
            sample = convolvedImage(indices, indices);
            % averaging the sample
            pooledFeatures(:, :, filterNum, imageNum) = sample / totalNumInPatch;
        end
    end
else
    % max-pooling, and store the indices of max position into maxMap
    for imageNum = 1:numImages
        for filterNum = 1:numFilters
            %calculate the split boundary for the matrix
            slice = repmat(poolDim,1, convolvedDim / poolDim);
            cells = mat2cell(convolvedFeatures(:, :, filterNum, imageNum)...
                , slice, slice);
            for block = 1:size(cells(:), 1)
                % cell is indexed vertically, like
                % 1 3 
                % 2 4
                [val, loc] = max(reshape(cells{block}, 1, poolDim * poolDim));
                [loc_row, loc_col] = ind2sub([poolDim poolDim], loc);
                
                pooledFeatures_col = ceil(block / (convolvedDim / poolDim));
                pooledFeatures_row = mod(block, (convolvedDim / poolDim));
                if pooledFeatures_row == 0
                    pooledFeatures_row = convolvedDim / poolDim;
                end

                pooledFeatures(pooledFeatures_row, pooledFeatures_col, filterNum, imageNum) = val;

                loc_row = (pooledFeatures_row - 1) * poolDim + loc_row;
                loc_col = (pooledFeatures_col - 1) * poolDim + loc_col;
                maxMap(loc_row, loc_col, filterNum, imageNum) = 1;
            end
        end
    end
end


end

