function [cost, grad, preds] = aTeamCnnCost( ...
    theta, images, labels, layers, options, pred)
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
reg_term = 0;
for l = 1:size(layers, 1)
    switch layers{l}.name
        case 'input'
        case 'pooling'
            [signal, maxMap] = cnnPool(layers{l}.X, signal, layers{l}.type);
            layers{l}.maxMap = maxMap;
        case 'convolution'
            if ndims(signal) == 3
                signal = reshape(signal, size(signal, 1), ...
                    size(signal, 2), 1, size(signal, 3));
            end
            layers{l}.input = signal;
            signal = aTeamCnnConvolve(layers{l}.X, layers{l}.numFilters, ...
                signal, layers{l}.weights, layers{l}.bias, ...
                layers{l}.actFunc);
            reg_term = reg_term + sum(layers{l}.weights(:) .^2 ); 
        case 'fully'
            signal = reshape(signal,[],numImages);
            layers{l}.input = signal;
            signal = layers{l}.weights * signal + ...
                repmat(layers{l}.bias, 1, numImages);
            signal = actFunction(signal, layers{l}.actFunc);
            reg_term = reg_term + sum(layers{l}.weights(:) .^2 ); 
        case 'output'
            signal = reshape(signal,[],numImages);
            layers{l}.input = signal;
            signal = layers{l}.weights * signal + ...
                repmat(layers{l}.bias, 1, numImages); 
            reg_term = reg_term + sum(layers{l}.weights(:) .^2 ); 
            % numClasses = layers{l}.units;
    end
    layers{l}.activation = signal;
end

reg_term = options.lambda / 2 * reg_term;

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
else
    preds = [];
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
% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.
grad_layers = backprop(options, layers, der_matrix);

%% Unroll gradient into grad vector for minFunc
grad = layers2params(grad_layers);

%% update cost
cost = cost + reg_term; 
end