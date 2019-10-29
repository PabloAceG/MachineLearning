function [o, a] = layerActivation(weights, inputs, fun)
    % Neuron activation
    a = weights * inputs;
    
    % Function selection
    switch fun
        case 'perceptron'
            o = perceptron(a);
        case 'sigmoid'
            o = sigmoid(a);
        case 'tanh'
            o = hyperbolicTangent(a);
        otherwise
            o = perceptron(a);
    end
    
    o = [o; ones(1, size(o, 2))];
end