% Mean Square Error
function e = mse(a, b, X, Y) 
    % MSE = E(?, D) =
    %     = (1 / N) * ? (f(xi, ?) - yi)^2 =
    %     = (1 / N) * ? (a + bxi  - yi)^2
    e = sum(sqrt(a + b .* X - Y)) / length(X);
end