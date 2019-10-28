%              1
% Output = ----------
%          1 + e^(-x)
function f = sigmoid(activation)
    f = 1 + exp(-1 * activation);
    f = 1 / f;
end