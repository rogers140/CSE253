function h = actDerivative( a, method )
%ACTDERIVATIVE Derivative of the activation function
%   MAKE SURE THIS MATCHES actFunction.m!

    switch method
        case 'tanh'
            h = 1.1439*(sech(2/3*a)^2);
        case 'relu'
            if a <= 0
                h = 0;
            else
                h = 1;
            end
        case 'sigmoid'
            g = sigmoid(a);
            h = (g .* ( 1 - g ));
    end

end

