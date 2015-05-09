function processed_data = ...
    load_preprocess(dataset, image_dim, gabor_dim, final_dim, force)
    % dataset = 'POFA'/'NimStim'
    %  TODO return an array of train/test partitioned data.
   
    if nargin == 1
        image_dim = [64 64]; % TODO remove
    end
    
    if nargin < 3
        gabor_dim = [96 96]; % TODO remove
    end
    
    if nargin < 4
        final_dim = [8 8]; % TODO remove
    end
    
    if nargin < 5
        force = false;
    end
    
    dataFname = sprintf('data_%s.mat', dataset(1,:));
    
    if force && exist(dataFname, 'file')
        delete(dataFname);
    end
    
    if exist(dataFname, 'file')
        load(dataFname);
        
        % TODO partition data to training and testing;
        
        return;
    end
    
    % resize images.
    resize_images(dataset, image_dim, force);

    % Apply Gabor Filter --------------------------------------------------
    scale_step = 2^(3/4); % best scaling factor for k is 2^(3/4)

    % Create filters if none exists.
    gaborFname = ...
        sprintf('gaborFilters_%dx%d.mat', gabor_dim(1), gabor_dim(2));
    if ~exist(gaborFname,'file')    
        G = createGabor(gabor_dim, scale_step);
    else
        load(gaborFname);
    end
    
    
    % load images.
    files = dir(['../',dataset,'/rescaled/']);
    num_files = 0;
    for i=1:size(files, 1)
        if files(i).isdir || strcmp(files(i).name, '.gitignore') ...
                || strcmp(files(i).name, '.DS_Store')
            continue;
        end
        num_files = num_files + 1;
    end
    processed_data = cell(2, num_files);
    
    % filter every image.
    fprintf ('Preprocessing image ...');
    % file_count is the actual image count. i is everything in the
    % directory.
    file_count = 1;
    for i=1:size(files, 1)
        if files(i).isdir || strcmp(files(i).name, '.gitignore') ...
                || strcmp(files(i).name, '.DS_Store')
            continue;
        end
        
        % parse the label
        [~, file_label, ~] = fileparts(files(i).name);
        processed_data{1,file_count} = strsplit(file_label,'[\-_]', ...
            'DelimiterType','RegularExpression');
        
        % create 40 new images
        filtered_counter = 1;
        filtered_images = cell(1, 40);
        for s = 1:5
            for j = 1:8
                % filtered images resized to final_size.
                raw_image = imread(['../',dataset,'/rescaled/', ...
                    files(i).name]);
                filtered_images{filtered_counter} = imresize(imfilter( ...
                    raw_image, G{s,j}, 'same'), final_dim);
                filtered_counter = filtered_counter + 1;
            end
        end
        
        % NOTE: the first row of processed_data is the label, the second is
        % the processed image.
        % TODO z score
        % TODO PCA
        
        processed_data{2,file_count} = filtered_images;
        
        fprintf(' %d', file_count);
        file_count = file_count + 1;
    end
    fprintf(' done.\n');
    
    % TODO partition data to training and testing
    save (dataFname,'processed_data');
end