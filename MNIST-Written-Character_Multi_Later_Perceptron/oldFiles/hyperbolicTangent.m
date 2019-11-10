% Not necessarily implemented. MATLAB has already this function implemented
% as tanh(X).
%                        2
% Output = tanh(x) = ---------- - 1
%                    1 + e^(-2x)
function f = hyperbolicTangent(activation)
    f = 1 + exp(-1 * activation);
    f = 2 / f;
    f = f - 1;
end