function derivative_vector = crossEntropyDerivative(output, label)
% output is a data_size * class_size matrix before softmax
% label is a data_size * class_size matrix
% derivative_vector is a data_size vector
num_classes = size(output,2);
temp = exp(output);
denominator = repmat(sum(temp, 2), [1, num_classes]);
temp = temp ./ denominator; %softmax
derivative_vector = sum((-1)*label ./ temp, 2);
end