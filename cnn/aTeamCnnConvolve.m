function convolvedFeatures = aTeamCnnConvolve(filterDim, numFilters, ...
    images, W, b, actFunc)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  filterDim - filter (feature) dimension
%  numFilters - number of feature maps
%  images - large images to convolve with, matrix in the form
%           images(r, c, image number)
%  W, b - W, b for features from the sparse autoencoder
%         W is of shape (filterDim,filterDim,numFilters)
%         b is of shape (numFilters,1)
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
if ~exist('actFunc','var')
    actFunc = 'sigmoid';
end;

numImages = size(images, 4);
numFeatures = size(images, 3);
imageDim = size(images, 1);
convDim = imageDim - filterDim + 1;

convolvedFeatures = zeros(convDim, convDim, numFilters, numImages);

% Instructions:
%   Convolve every filter with every image here to produce the 
%   (imageDim - filterDim + 1) x (imageDim - filterDim + 1) x numFeatures x numImages
%   matrix convolvedFeatures, such that 
%   convolvedFeatures(imageRow, imageCol, featureNum, imageNum) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to (imageRow + filterDim - 1, imageCol + filterDim - 1)
%
% Expected running times: 
%   Convolving with 100 images should take less than 30 seconds 
%   Convolving with 5000 images should take around 2 minutes
%   (So to save time when testing, you should convolve with less images, as
%   described earlier)

% Flip the feature matrix because of the definition of convolution, as explained later
filters = rot90(W, 2);

for imageNum = 1:numImages
  for filterNum = 1:numFilters
    convolvedFeature = zeros(convDim, convDim);
    for featureNum = 1:numFeatures
        % Obtain the feature (filterDim x filterDim) needed during the convolution
        filter = filters(:,:,featureNum,filterNum);

        % Obtain the image
        im = squeeze(images(:, :, featureNum, imageNum));

        % Convolve "filter" with "im", adding the result to convolvedImage
        % be sure to do a 'valid' convolution
        % Add the bias unit
        convolvedFeature = convolvedFeature + conv2(im, filter, 'valid');
    end
    % Sum up features, add bias unit, apply activation function to get
    % hidden activation
    convolvedFeatures(:, :, filterNum, imageNum) = ... 
        actFunction(convolvedFeature + b(filterNum), actFunc);
  end
end

end

