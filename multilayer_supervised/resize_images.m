function resize_images(dataset, size, force)
% Rescale images for the specified dataset, it rescales images in
% 'original' folder and output the rescaled images into 'rescaled' folder.
% The function checks if rescaling is already executed by checking the
% existance of 'rescaled' folder. "force" will overwrite previous images
% if it is set to true.
%
% This function also converts RGB images to grayscale.
%
% Default values:
% 'size' default value [64 64]
% 'force' default value false
%
% Examples:
% resize_images('POFA');
% resize_images('POFA', [100, 50]);
% resize_images('NimStim', [100, 50], true);

    % Set Default Parameters
    if nargin == 1
        size = [64 64];
    end
    if nargin < 3
        force = false;
    end

    path = strcat('../', dataset, '/original/');
    target = strcat('../', dataset, '/rescaled/');

    if force 
        rmdir(target, 's');
    end

    if ~exist(target, 'dir')
        mkdir(target);
        files = dir(path);
        
        for file = files'
            if file.isdir || strcmp(file.name, '.gitignore') ...
                    || strcmp( file.name, '.DS_Store')
                continue;
            end
            image = imread(strcat(path, file.name));
            transformed = imresize(image, size);
            
            if ~ismatrix(transformed)
                transformed = rgb2gray(transformed);
            end
            
            imwrite(transformed, strcat(target, file.name));
        end
    end
end

