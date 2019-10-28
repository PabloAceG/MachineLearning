% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% COURSEWORK 2: MNIST WRITTEN CHARACTER CLASSIFICATION WITH A MULTI-LAYER
% PERCEPTRON
% AUTHOR: PABLO ACEREDA
% FILE:   

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% LINES THAT HAVE TO BE IN EVERY SCRIPT: 
% clear variables, 
% close opened windows 
% and clean the comand windows
%-------------------------------------------------------------------------%
clear variables;
close all;
clc;
%-------------------------------------------------------------------------%

%                               DATASET
%                              =========
% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as
% train-images.idx3-ubyte / train-labels.idx1-ubyte
% The datasets obtained from http://yann.lecun.com/exdb/mnist/ are
% concatenated to perform k-folding afterwards.
images = [loadMNISTImages('./dataset/train-images.idx3-ubyte'), ...
          loadMNISTImages('./dataset/t10k-images.idx3-ubyte')];
labels = [loadMNISTLabels('./dataset/train-labels.idx1-ubyte');
          loadMNISTLabels('./dataset/t10k-labels.idx1-ubyte')];

%                             DISPLAY DATA
%                            ==============
% We are using display_network from the autoencoder code
% TODO: Remove comment 
% display_network(images(:, 1 : 100)); % Show the first 100 images
% disp(labels(1 : 10));

%                       NEURAL NETWORK TRAINING
%                      =========================
% Trainig using k-fold
k_value = 10;

% Number of layers
numLayers = 3;

% Data division
[train, test] = crossValidationSet(images, labels, k_value);

numberLayers = 3;
numberNeuronsLayer = [size(images, 1), 10, 10, 10];
Weights = cell(numberLayers, 1);

for i = 2 : length(numberNeuronsLayer)
       w = rand(numberNeuronsLayer(i), ...
                numberNeuronsLayer(i - 1) + 1);
       Weights(i - 1) = mat2cell(w, size(w, 1), size(w, 2));

end

% Cross validations
for i = 1 : k_value
    % Training and Testing sets for iteration
    setTrainImage = cell2mat(train(i, 1));
    setTrainLabel = cell2mat(train(i, 2));
    setTestImage  = cell2mat(test (i, 1));
    setTestLabel  = cell2mat(test (i, 2));
    
    for j = 1 : 10%length(labels)
        input = [setTrainImage(:, j); 1];
        
        % Number of layers
        for k = 1 : numLayers
            input = [layerActivation(cell2mat(Weights(k)), ...
                                    input,                 ...
                                    'perceptron');         ...
                     1];
        end

        % Error
        label = setTrainLabel(j);
    end
end

"I'm done"

