function h = actFunction( a, method )

    switch method
        case 'tanh'
            h=1.7159*tanh(2/3*a);
        case 'relu'
            h=max(0,a);
        case 'sigmoid'
            h=sigmoid(a);
    end
        
end

