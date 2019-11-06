% Params:
%   W - Layer Weight
%   v - Layer Input Vector (previous layer outputs 
%                           + extra row with value 1)
% Returns:
%   o - Layer Output       (column)
%   a - Neurons activation (row)
function [o, a] = layerActivation(W, v, fun)
    % Neuron activation
    a = W * v;
    
    % Function selection
    switch fun
        case 'sigmoid'
            o = sigmoid(a);
        case 'tanh'
            o = hyperbolicTangent(a);
        otherwise
            o = sigmoid(a);
    end
    
    o = o';
end