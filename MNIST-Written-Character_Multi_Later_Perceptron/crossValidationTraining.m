function [] = crossValidationTraining(w,               ...
                                      samples, labels, ...
                                      trainFunction,   ...
                                      k_value)

    %                          DATA DIVISION
    %                         ===============
    [train, test] = crossValidationSet(samples, labels, k_value);
    
    numLayers = length(w);
    sizeOutputLayer = size(cell2mat(w(end)), 1);

    % Cross validations
    for i = 1 : k_value
        % Training and Testing sets for iteration
        setTrainImage = [cell2mat(train(i, 1)); ...
                         ones(1, size(cell2mat(train(i, 1)), 2))];
        setTrainLabel = cell2mat(train(i, 2));
        setTestImage  = cell2mat(test (i, 1));
        setTestLabel  = cell2mat(test (i, 2));

        for j = 1 : 10%length(labels)
            input = setTrainImage(:, j);

            % Cells activation per layer.
            cellsAct = cell(numLayers, 1);

            % Number of layers
            for k = 1 : numLayers

                [input, cellsAct] = layerActivation(cell2mat(w(k)), ...
                                                    input,          ...
                                                    trainFunction);
            end

            % Prediction made by neural network.
            estimation = input(1 : end - 1);
            % Desired output for given input (correct value of prediction).
            desired = zeros(sizeOutputLayer, 1);
            desired(setTrainLabel(j) + 1) = 1;

            error = estimation - desired;

        end
    end

end