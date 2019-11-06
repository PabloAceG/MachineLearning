%              1
% Output = ----------
%          1 + e^(-x)
function f = sigmoid(activation)
    f = 1 + exp(-activation);
    f = 1 / f;
end