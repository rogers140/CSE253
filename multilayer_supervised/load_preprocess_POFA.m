function [processed_training_data, processed_test_data] = ...
    load_preprocess_POFA(test_id, image_dim, gabor_dim, final_dim, pca_components, force)
    % test command: load_preprocess('aa',[64 64], [96 96], [8 8], 8, true);
    
    dataset = 'POFA';
    
    if nargin < 6
        force = false;
    end
    
    dataFname_filtered_images = sprintf('POFA_gabor_filtered.mat');
    dataFname_training = sprintf('data_training_POFA_%s.mat', test_id);
    dataFname_test = sprintf('data_test_POFA_%s.mat', test_id);
    
    if force && exist(dataFname_filtered_images, 'file')
        delete(dataFname_filtered_images);
    end
    
    if force && exist(dataFname_training, 'file')
        delete(dataFname_training);
    end
    
    if force && exist(dataFname_test, 'file')
        delete(dataFname_test);
    end
    
    if exist(dataFname_training, 'file') && exist(dataFname_test, 'file')
        load(dataFname_training);
        load(dataFname_test); 
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
    
    if exist(dataFname_filtered_images, 'file')
        load(dataFname_filtered_images);
    else
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

        fprintf ('Preprocessing image ...\n');
        % file_count is the actual image count. i is everything in the
        % directory.
        file_count = 0;
        for i=1:size(files, 1)
            if files(i).isdir || strcmp(files(i).name, '.gitignore') ...
                    || strcmp(files(i).name, '.DS_Store')
                continue;
            end

            % is a image file.
            file_count = file_count + 1;

            % parse the label
            [~, file_label, ~] = fileparts(files(i).name);
            processed_data{1,file_count} = strsplit(file_label,'[\-_]', ...
                'DelimiterType','RegularExpression');

            % create 5x8 new images
            filtered_images = cell(5, 8);
            for s = 1:5
                for j = 1:8
                    % filtered images resized to final_size.
                    raw_image = imread(['../',dataset,'/rescaled/', ...
                        files(i).name]);
                    filtered_images{s,j} = imresize(imfilter( ...
                        raw_image, G{s,j}, 'same'), final_dim);
                end
            end

            % NOTE: the first row of processed_data is the label, the second is
            % the processed image.

            processed_data{2,file_count} = filtered_images;

            fprintf(' %d', file_count);
        end
        
        save(dataFname_filtered_images, 'processed_data');
    end
    
    % Partition data into training and test
    file_count = size(processed_data, 2);
    train_test_files = false(1, file_count);
    for f=1:file_count
        train_test_files(f) = ...
            ~isempty(strfind(processed_data{1,f}{1},test_id));
    end
    
    training_indices = find(train_test_files == false);
    test_indices = find(train_test_files == true);
    training_size = size(training_indices, 2);
    test_size = file_count - training_size;
    
    processed_training_data = processed_data(:,training_indices);
    processed_test_data = processed_data(:, test_indices);
    
    % zscore---only calculate mean and std on training data, and then apply
    % them on test data.
    fprintf ('\nComputing zscore...\n'); 
    zscore_training = zeros(training_size, 5*final_dim(1), 8*final_dim(2));
    for i = (1:training_size)
        % 40*64
        zscore_training(i,:,:) = abs(double(cell2mat(processed_training_data{2, i})));
    end 
    [zscore_training, zscore_mean, zscore_std] = ...
        zscore(zscore_training, 0, 1); % traing data matrix after zscore
    
    % test data matrix after zscore
    zscore_test = zeros(test_size, 5*final_dim(1), 8*final_dim(2));
    
    for i = (1: test_size)
        zscore_test(i,:,:) = (abs(double(cell2mat(processed_test_data{2, i}))) - squeeze(zscore_mean))./squeeze(zscore_std);      
    end
    
    %PCA
    fprintf ('Applying PCA...\n');
    [pca_training, pca_test] = pca_images(zscore_training, zscore_test, pca_components);
    
    labels_train = processed_training_data(1, :);
    labels_test = processed_test_data(1, :);
    
    processed_training_data = [];
    processed_test_data = [];
    processed_training_data.labels = labels_train;
    processed_test_data.labels = labels_test;
    processed_training_data.data = pca_training;
    processed_test_data.data = pca_test;
    
    fprintf(' done.\n');
    % Save data into files
    save(dataFname_training,'processed_training_data');
    save(dataFname_test,'processed_test_data');
end