function cost = crossEntropy(output, label)
%output is data_size*class_num matrix before softmax (directly from network)
%label is data_size*class_size matrix for real label
%it will give output of crossEntropy
num_classes = size(output,2);
temp = exp(output);
denominator = repmat(sum(temp, 2), [1, num_classes]);
temp = temp ./ denominator; %softmax
cost_matrix = (-1)*log(temp).* label;
cost = sum(sum(cost_matrix));
end