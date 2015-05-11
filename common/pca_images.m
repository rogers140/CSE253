function [pca_training, pca_test] = pca_images...
    (training, test, n_components)
% Input
% original = Input 3D matrix
% n_components = degrees of freedom
%
% The function will store the variables in folder pca inside the dataset
% folder
%
% Output
% Transformed matrix

    % Set Default Parameters
    if nargin < 3
        n_components = 8;
    end

    joined = 8;
    [n, f, p] = size(training);
    [nt, ft, pt] = size(test);
    pca_training = [];
    pca_test = [];
    for s = 1:joined:f
        end_index = s + joined - 1;
        partial_training = training(:,s:end_index,:);
        [coeff, score, ~, ~, ~, mu] = pca(...
            reshape(partial_training, [n, joined*p]));
        pca_training = horzcat(pca_training, score(:,1:n_components));
        
        partial_test = test(:,s:end_index,:);
        pca_partial_test = (reshape(partial_test, [nt, joined*p])...
            - repmat(mu,nt,1)) * coeff;
        pca_test = horzcat(pca_test, pca_partial_test(:,1:n_components));
    end
end
