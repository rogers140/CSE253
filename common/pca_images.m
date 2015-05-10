function [transformed_matrix] = pca_images...
    (cellmatrix, n_components, force)
% Input
% cellmatrix = Input cell matrix
% n_components = degrees of freedom
% force = (default = false | true) recalculate if necessary
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
    if nargin < 3
        force = false;
    end

    target = strcat('../', dataset, '/pca/');
    filename = strcat('pca_',num2str(n_components),'.mat');
    
    if ~exist(target, 'dir')
        mkdir(target);
    end

    if force && exist(filename, 'file')
        delete(filename);
    end
    
    
    if exist(filename, 'file') == 2
        fprintf('PCA data loaded from file.\n');
        loaded = load(filename, 'transformed_matrix');
        transformed_matrix = loaded.transformed_matrix;
    else
        fprintf('Calculate PCA.\n');
        images = [];
        [w h] = cellmatrix(1,1,1);
        vector_length = w * h;
        for n = 1:size(cellmatrix, 1)
            for s = 1:size(cellmatrix, 2)
                image = [];
                for j = 1:size(cellmatrix, 3)
                    i = double(reshape(cellmatrix(n, s, j),[1 vector_length]));
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
                transformed_matrix(n, s) = temp_matrix(index,:);
            end
        end
        save(filename, 'transformed_matrix');
    end
end
