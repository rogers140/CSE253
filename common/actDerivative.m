function h = actDerivative( a, method )
%ACTDERIVATIVE Derivative of the activation function
%   MAKE SURE THIS MATCHES actFunction.m!

switch method
    case 'tanh'
        h = 1.1439*(sech(2/3*a)^2);
    case 'relu'
        h = max(a, 0);
        max_val = max(a(:));
        if max_val > 0
            h = h ./ max(a(:));
            h = ceil(h);
        end
    case 'sigmoid'
        g = sigmoid(a);
        h = (g .* ( 1 - g ));
end

end

