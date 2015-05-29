function [ layers ] = parseNetwork(filename)
%filename must be a text file in the working directory

network = fileread(filename);
raw_layers = strsplit(network, '->');

layers = cell(size(raw_layers, 2), 1);

for l = 1:size(raw_layers, 2)
    params = strsplit(strtrim(raw_layers{l}));
    layers{l}.name = params{1};
    switch params{1}
        case 'input'
            dims = cellfun(@str2num, strsplit(params{2}, '*'));
            layers{l}.X = dims(1);
            layers{l}.Y = dims(2);
            if size(dims) > 2
                layers{l}.Z = dims(3);
            else
                layers{l}.Z = 1;
            end
        case 'convolution'
            dims = cellfun(@str2num, strsplit(params{5}, '*'));
            layers{l}.X = dims(1);
            layers{l}.Y = dims(2);
            layers{l}.numFilters = dims(3);
            layers{l}.actFunc = params{8};
        case 'max_pooling'
            layers{l}.name = 'pooling';
            layers{l}.type = 'max';
            dims = cellfun(@str2num, strsplit(params{3}, '*'));
            layers{l}.X = dims(1);
            layers{l}.Y = dims(2);
            if size(dims) > 2
                layers{l}.Z = dims(3);
            else
                layers{l}.Z = 1;
            end
        case 'mean_pooling'
            layers{l}.name = 'pooling';
            layers{l}.type = 'mean';
            dims = cellfun(@str2num, strsplit(params{3}, '*'));
            layers{l}.X = dims(1);
            layers{l}.Y = dims(2);
            if size(dims) > 2
                layers{l}.Z = dims(3);
            else
                layers{l}.Z = 1;
            end
        case 'fully'
            layers{l}.units = str2double(params{4});
            layers{l}.actFunc = params{9};
        case 'output'
            layers{l}.units = str2double(params{3});
            layers{l}.lossFunc = params{12};
        otherwise
            warning('Unexpected layer identifier.');
    end
end


end

