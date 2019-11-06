% Params:
%   W   - Layer Weight
%   v   - Layer Input Vector (previous layer outputs 
%                             + extra row with value 1)
%   fun - Function to be executed
% Returns:
%   o - Layer Output (column)
function o = layerActivation(W, v, fun)
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