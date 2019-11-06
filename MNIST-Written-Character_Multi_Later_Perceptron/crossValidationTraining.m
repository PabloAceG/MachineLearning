function [] = crossValidationTraining(W,               ...
                                      samples, labels, ...
                                      trainFunction,   ...
                                      k_value)

    %                          DATA DIVISION
    %                         ===============
    [train, test] = crossValidationSet(samples, labels, k_value);
    
    numLayers = length(W);
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
        outputs     = cell(numLayers, 1);
        % Cells activation per layer.
        activations = cell(numLayers, 1);

        for j = 1 : 10%length(labels)
            o = setTrainImage(:, j);

            % Number of layers
            for k = 1 : numLayers
                
                [o, a] = layerActivation(cell2mat(W(k)), ...
                                         [o; 1],         ...
                                         trainFunction);
                outputs(k)     = mat2cell(o, size(o, 1), size(o, 2));
                activations(k) = mat2cell(a, size(a, 1), size(a, 2));
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
% - activations might no be needed. Delete from:
% - Set learning rate gradientDescentUpdate(_,_,_, X)
%   -- this file
%   -- layerActivation.m
