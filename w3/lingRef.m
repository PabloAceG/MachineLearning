% Simple Linear Regresion
%
% Definitions:
%
% C = (cov(A, A) cov(A, B))
%     (cov(B, A) cov(B, B))
function [a, b] = lingRef(Xs, Ys)
    invN   = 1 / length(Xs);
    % x   = (1 / N) ? xi
    x   = invN .* sum(Xs);
    % y   = (1 / N) ? yi
    y   = invN .* sum(Ys);
    % Cxx = (1 / N) ? xi * xi
    Cxx = invN .* sum(Xs .* Xs);
    % Cxy = (1 / N) ? xi * yi
    Cxy = invN .* sum(Xs .* Ys);
    
    %     (Cxy - x * y)   Cov(x, y)
    % b = ------------- = ---------
    %     (Cxx - x * x)     Var(x)
    b = (Cxy - x .* y) / (Cxx - x .* x);
    % Could also be used
    % b = (cov(Xs, Ys) / var(Xs));
    % b = b(2, 1); % Selects only the one needed.
    
    % a = y - b * x)
    a = y - b .* x;
end