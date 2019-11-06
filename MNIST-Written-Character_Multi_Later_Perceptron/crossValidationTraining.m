function [] = crossValidationTraining(W,               ...
                                      samples, labels, ...
                                      trainFunction,   ...
                                      k_value)

    %                          DATA DIVISION
    %                         ===============
    [train, test] = crossValidationSet(samples, labels, k_value);
    
    % Output layer, despite not having weights attached after itself, is a
    % layer nontheless
    numLayers = length(W) + 1;
    sizeOutputLayer = size(cell2mat(W(end)), 1);
    
    % Cross validations
    for i = 1 : k_value
        % Training and Testing sets for iteration
        setTrainImage = cell2mat(train(i, 1));
        setTrainLabel = cell2mat(train(i, 2));
        setTestImage  = cell2mat(test (i, 1));
        setTestLabel  = cell2mat(test (i, 2));
        
        % Data obtained after processing each layer
        % Output of each layer
        outputs = cell(numLayers, 1);

        for j = 1 : 10%length(labels)
            % Input layer outputs
            o          = setTrainImage(:, j);
            outputs(1) = mat2cell(o, size(o, 1), size(o, 2));
            
            % Iterates through layers, except for the output layer
            
            for k = 1 : numLayers - 1
                
                o = layerActivation(cell2mat(W(k)), ...
                                    [o; 1],         ...
                                    trainFunction);
                outputs(k + 1) = mat2cell(o, size(o, 1), size(o, 2));
            end

            W = gradientDescentUpdate(outputs,          ...
                                      W,                ...
                                      setTrainLabel(j), ...
                                      0);

        end
    end

end

% TODO: 
% - Delete 10 and substitute by number of labels in j loop
