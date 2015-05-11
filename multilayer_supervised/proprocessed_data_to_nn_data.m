function [data_train, labels_train, data_test, labels_test] = ...
    proprocessed_data_to_nn_data( processed_training_data, ...
    processed_test_data, label_ind, label_map )
%PROPROCESSED_DATA_TO_NN_DATA Convert preprocessed data to nn input output
%   processed_train_data = preprocessed training data, output of
%       load_preprocess.m. 2xn cell matrix where the first row contains
%       cell arrays of strings containing a label. The second row are 1xf
%       arrays of f features.
%   processed_test_data = preprocessed test data, output same as above.
%   label_ind = the index of the cell arrays of strings that is the label.
%   label_map = maps explicit lables to nn output index.
%   
    data_train = cell2mat(processed_training_data(2,:)');
    data_test = cell2mat(processed_test_data(2,:)');
    labels_train = zeros(size(data_train, 1), size(label_map, 1));
    labels_test = zeros(size(data_test, 1), size(label_map, 1));
    
    % there are very limited number of pictures, loop is fine.
    for i=1:size(data_train, 1)
        labels_train( ...
            i, label_map(processed_training_data{1, 1}(label_ind))) = 1;
    end
    for i=1:size(data_test, 1)
        labels_test( ...
            i, label_map(processed_test_data{1, 1}(label_ind))) = 1;
    end
end

