function deriv = actVal2Deriv( z )
%ACTVAL2DERIV Compute derivative directly from activation value
%   For all of our activations functions, derivative can be computed
%   directly from activation output
    switch method
    case 'tanh'
        deriv = 1.1439*(1-z^2);
    case 'relu'
        if z <= 0
            deriv = 0;
        else
            deriv = 1;
        end
    case 'sigmoid'
        deriv = (z .* ( 1 - z ));
    end
    
end

