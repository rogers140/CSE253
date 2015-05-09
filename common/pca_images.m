function [transformed_matrix] = pca_images...
    (dataset, folder, n_components, force)
% Input
% dataset = (POFA|NimStim)
% folder = (original|rescaled|zscored...)
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
        folder = 'zscored';
    end
    if nargin < 3
        n_components = 8;
    end
    if nargin < 4
        force = false;
    end

    path = strcat('../', dataset, '/', folder, '/');
    target = strcat('../', dataset, '/pca/');
    filename = strcat(target, folder, '_',...
        num2str(n_components),'.mat');
    
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
        fprintf('Calculating PCA.\n');
        files = dir(path);
        
        images = [];
        vector_length = 0;
        for file = files'
            if file.isdir || strcmp(file.name, '.gitignore') ...
                    || strcmp( file.name, '.DS_Store')
                continue;
            end
            
            image = imread(strcat(path, file.name));
            if vector_length == 0
                [w, h] = size(image);
                vector_length = w * h;
            end
            
            images = [images; double(reshape(image,[1 vector_length]))];
        end
        [coeff, score] = pca(images);
        transformed_matrix = score(:,1:n_components);
        
        save(filename, 'transformed_matrix');
    end
end
