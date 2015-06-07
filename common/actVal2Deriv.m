function deriv = actVal2Deriv( z, method )
%ACTVAL2DERIV Compute derivative directly from activation value
%   For all of our activations functions, derivative can be computed
%   directly from activation output
    switch method
    case 'tanh'
        deriv = 1.14393*(1 - ((z / 1.7159) .^2));
    case 'relu'
        deriv = z~=0;
    case 'sigmoid'
        deriv = (z .* ( 1 - z ));
    end
    
end

