function [transformed_matrix] = pca_images...
    (original, n_components)
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
    if nargin < 2
        n_components = 8;
    end

    joined = 8;
    [n, f, p] = size(original);
    transformed_matrix = [];
    for s = [1:joined:f]
        end_index = s + joined - 1;
        partial_matrix = original(:,s:end_index,:);
        [coeff, score] = pca(reshape(partial_matrix, [n, joined*p]));
        transformed_matrix = horzcat(transformed_matrix, score(:,1:n_components));
    end
end
