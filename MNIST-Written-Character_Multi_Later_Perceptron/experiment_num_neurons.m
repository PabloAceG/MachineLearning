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

% TRAINING
train_images = loadMNISTImages('./dataset/train-images.idx3-ubyte');
train_labels = loadMNISTLabels('./dataset/train-labels.idx1-ubyte');
% TESTING
test_images  = loadMNISTImages('./dataset/t10k-images.idx3-ubyte');
test_labels  = loadMNISTLabels('./dataset/t10k-labels.idx1-ubyte');

%                             DISPLAY DATA
%                            ==============
% We are using display_network from the autoencoder code
display_network(train_images(:, 1 : 100)); % Show the first 100 images
disp(train_labels(1 : 10));

% Preparation for different consecutive experiments
numExperiments = 3;
hiddenDims     = [10, 20, 35, 50];
errors = zeros(length(hiddenDims), numExperiments);
times  = zeros(length(hiddenDims), numExperiments);

%                       NEURAL NETWORK TRAINING
%                      =========================

for h = 1 : 1 : length(hiddenDims)
    for e = 1 : 1 : numExperiments
        tic 
        % Network Dimesions
        inputDim = size(train_images, 1);
        hiddenDim = hiddenDims(h);
        outputDim = 10;
        % Network creation
        net = MLP(inputDim, hiddenDim, outputDim, 1);
        net = net.initWeight(1.0);

        for i = 1 : 1 : length(train_labels)
            net.adapt_to_target(train_images(:, i), train_labels(i), 0.1);
        end

        predictions = zeros(length(test_labels), 1);

        for t = 1 : 1 : length(test_labels) 
            predictions(t) =  net.compute_output(test_images(:, t));
        end

        errors(h, e) = computeError(predictions, test_labels);
        times(h, e) = toc;
    end
end

errors
times

x = [1 : length(hiddenDims)];

figure;
subplot(2, 1, 1)
gscatter(x, errors, x)
xlabel('experiments')
ylabel('errors')
subplot(2, 1, 2)
gscatter(x, times,  x)
xlabel('experiments')
ylabel('times')
legend('off')
