function [train, test] = crossValidationSet(set, labels, k_value)
    
    train = cell(k_value, 2);
    test  = cell(k_value, 2);
    
    N    = length(labels);
    jump = floor(N / k_value);
    
% Divide data in k pieces
    for i = 1 : 1 : k_value
        if     i == 1

            testX  = set(:, 1        : jump);
            testY  = labels(1        : jump);
            trainX = set(:, jump + 1 : end);
            trainY = labels(jump + 1 : end);

        elseif i * jump <= N && i < k_value

            testX  = set    (:, (i - 1) * jump + 1 : i * jump);
            testY  = labels ((i - 1) * jump + 1 : i * jump);
            trainX = [set(:, 1 : (i - 1) * jump) , ...
                      set(:, i * jump + 1 : end)];
            trainY = [labels(1 : (i - 1) * jump) ; ...
                      labels(i * jump + 1 : end)];

        else

            testX  = set   (:, (i - 1) * jump + 1 : end);
            testY  = labels((i - 1) * jump + 1 : end);
            trainX = set   (:, 1                  : (i - 1) * jump);
            trainY = labels(1                  : (i - 1) * jump);

        end
        
        train(i, :) = {trainX ; trainY};
        test (i, :) = {testX  ; testY};
    end
    
    % Return
    [train, test];
    
end