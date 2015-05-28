function [pooledFeatures, maxMap] = cnnPool(poolDim, convolvedFeatures, type)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
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
            d = size(X, 1) / poolDim;
            C = mat2cell(convolvedFeatures(:, :, filterNum, imageNum)...
                , [d d], [d d]);

            for block = 1:size(C(:), 1)
                [val loc] = max(reshape(C{block}, 1,d*d));
                [loc_row loc_col] = ind2sub([d d], loc);
                pool_col = ceil(block/poolDim);
                pool_row = mod(block,poolDim);
                if pool_row == 0
                    pool_row = poolDim;
                end

                pooledFeatures(pool_row, pool_col, filterNum, imageNum) = val;

                loc_row = (pool_row - 1) * d + loc_row;
                loc_col = (pool_col - 1) * d + loc_col;
                maxMap(loc_row, loc_col, filterNum, imageNum) = 1;
            end
        end
    end
end


end

