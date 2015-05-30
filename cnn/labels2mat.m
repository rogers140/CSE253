function labelmat = labels2mat( labels )
%LABELS2MAT Summary of this function goes here
%   Detailed explanation goes here

labelmat = zeros(10, size(labels,1));
labelmat(sub2ind(size(labelmat), labels', 1:size(labels,1))) = 1;

end

