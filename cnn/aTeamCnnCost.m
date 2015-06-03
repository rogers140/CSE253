function [cost, grad, preds] = aTeamCnnCost( ...
    theta, images, labels, pred, layers, options)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  layers     -  information about the layers in the network
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;

layers = params2layers(theta, layers);

imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images
%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.


% signal - is the current activation functions that is going through
% network.
signal = images;
for l = 1:size(layers, 1)
    switch layers{l}.name
        case 'input'
        case 'pooling'
            [signal, maxMap] = cnnPool(layers{l}.X, signal, layers{l}.type);
            layers{l}.maxMap = maxMap;
        case 'convolution'
            signal = cnnConvolve(layers{l}.X, layers{l}.numFilters, ...
                signal, layers{l}.weights, layers{l}.bias, ...
                layers{l}.actFunc);
        case 'fully'
            signal = reshape(signal,[],numImages);
            layers{l}.input = signal;
            signal = layers{l}.weights * signal + ...
                repmat(layers{l}.bias, 1, numImages);
            signal = actFunction(signal, layers{l}.actFunc);
        case 'output'
            signal = reshape(signal,[],numImages);
            layers{l}.input = signal;
            signal = layers{l}.weights * signal + ...
                repmat(layers{l}.bias, 1, numImages); 
            numClasses = layers{l}.units;
    end
    layers{l}.activation = signal;
end

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

[pred_prob, cost, der_matrix] = crossEntropy(signal', labels');

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(pred_prob, [], 1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

for l = size(layers, 1):-1:1
    switch layers{l}.name
        case 'input'
        case {'pooling', 'convolution'}
            x = size(layers{l}.activation, 1);
            y = size(layers{l}.activation, 2);
            z = size(layers{l}.activation, 3);
            n = size(layers{l}.activation, 4);
            if strcmp(layers{l}.name, 'convolution')
                derivative = actDerivative(layers{l}.activation,...
                    layers{l}.actFunc);
            else
                derivative = ones(size(layers{l}.activation));
            end
            
            if(strcmp(layers{l+1}.name, 'convolution'))
                padX = layers{l+1}.X - 1;
                padY = layers{l+1}.Y - 1;
                layers{l}.deltas = zeros(x, y, z, n);
                weights = flip(flip(layers{l+1}.weights, 1), 2);
                deltas = padarray(layers{l+1}.deltas, [padX padY]);
                for imageNum = 1:n
                    for p = 1:z
                        for q = 1:size(layers{l+1}.weights, 3)
                            layers{l}.deltas(:, :, p, imageNum) = layers{l}.deltas(:, :, p, imageNum) +...
                                conv2(deltas(:, :, q, imageNum), weights(:, :, p), 'valid');
                        end
                    end
                end
                layers{l}.deltas = layers{l}.deltas .* derivative;
            elseif(strcmp(layers{l+1}.name, 'pooling'))
                dup = ones(layers{l+1}.X, layers{l+1}.Y);
                map = arrayfun(@(x) kron(x, dup), layers{l+1}.deltas,'UniformOutput', false);
                if strcmp(layers{l+1}.type, 'max')
                    layers{l}.deltas = derivative .* cell2mat(map) .* layers{l+1}.maxMap;
                elseif strcmp(layers{l+1}.type, 'mean')
                    scale = 1 / (layers{l}.X * layers{l}.Y);
                    layers{l}.deltas = derivative .* cell2mat(map) .* scale;
                end
            else
                % Everything else: Output, Fully
                deltas =  layers{l+1}.weights' * reshape(layers{l+1}.deltas, [],numImages);
                layers{l}.deltas = derivative .* reshape(deltas,x, y, z,numImages);   
            end
        case 'fully'
            derivative = actDerivative(layers{l}.activation,...
                layers{l}.actFunc);
            layers{l}.deltas = derivative .* (layers{l+1}.weights'...
                * layers{l+1}.deltas);
        case 'output'
            layers{l}.deltas = der_matrix';
    end
end

%% STEP 1d: Gradient Calculation
>>>>>>> c66787fa1c58d3f6eee6822135bf3d5e3ccc7960
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

grad = backprop(options, layers, der_matrix)

% TODO: implement actual gradient
grad_layers = layers;

%% Unroll gradient into grad vector for minFunc
grad = layers2param(grad_layers);

end