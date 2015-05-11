function [transformed_matrix] = pca_images...
    (cellmatrix, n_components)
% Input
% cellmatrix = Input cell matrix
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

    fprintf('Calculate PCA.\n');
    images = [];
    [w h] = size(cellmatrix{1,1,1});
    vector_length = w * h;
    for n = 1:size(cellmatrix, 1)
        for s = 1:size(cellmatrix, 2)
            image = [];
            for j = 1:size(cellmatrix, 3)
                i = double(reshape(cellmatrix{n, s, j},[1 vector_length]));
                image = [image; i];
            end
            images = [images; image];
        end
    end

    [coeff, score] = pca(images);
    temp_matrix = score(:,1:n_components);

    transformed_matrix = cell(size(cellmatrix, 1), size(cellmatrix, 2));
    for n = 1:size(cellmatrix, 1)
        for s = 1:size(cellmatrix, 2)
            index = (n-1) * s + s;
            transformed_matrix{n, s} = temp_matrix(index,:);
        end
    end

end
