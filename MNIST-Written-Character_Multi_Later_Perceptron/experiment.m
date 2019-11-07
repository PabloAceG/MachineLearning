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
images = loadMNISTImages('./dataset/train-images.idx3-ubyte');%, ...
          %loadMNISTImages('./dataset/t10k-images.idx3-ubyte')];
labels = loadMNISTLabels('./dataset/train-labels.idx1-ubyte');%;
          %loadMNISTLabels('./dataset/t10k-labels.idx1-ubyte')];

%                             DISPLAY DATA
%                            ==============
% We are using display_network from the autoencoder code
% TODO: Remove comment 
% display_network(images(:, 1 : 100)); % Show the first 100 images
% disp(labels(1 : 10));

%                       NEURAL NETWORK TRAINING
%                      =========================

% Size of the output layer (from 0 to 9)
sizeOutputLayer = 10;
% Number of Neurons Per Layer
numberNeuronsLayer = [size(images, 1), 10, sizeOutputLayer];
% Number of layers. Output and Input layers as counted as layers
numLayers = length(numberNeuronsLayer);

% Weights initialization
% Output layer does not have weights
Weights = cell(numLayers - 1, 1);

for i = 2 : length(numberNeuronsLayer)
       w = rand(numberNeuronsLayer(i), ...
                numberNeuronsLayer(i - 1) + 1);
       Weights(i - 1) = mat2cell(w, size(w, 1), size(w, 2));

end

% Trainig using k-fold
k_value = 10;

%                           CROSS VALIDATION 
%                          ==================
crossValidationTraining(Weights, images, labels, 'sigmoid', k_value);

"I'm done, no errors"

