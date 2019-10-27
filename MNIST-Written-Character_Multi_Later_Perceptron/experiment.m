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

% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as
% train-images.idx3-ubyte / train-labels.idx1-ubyte
% The datasets obtained from http://yann.lecun.com/exdb/mnist/ are
% concatenated to perform k-folding afterwards.
images = [loadMNISTImages('./dataset/train-images.idx3-ubyte'), ...
          loadMNISTImages('./dataset/t10k-images.idx3-ubyte')];
labels = [loadMNISTLabels('./dataset/train-labels.idx1-ubyte');
          loadMNISTLabels('./dataset/t10k-labels.idx1-ubyte')];

% We are using display_network from the autoencoder code
%display_network(images(:,1:100)); % Show the first 100 images
%disp(labels(1:10));

crossValidationSet(images, labels, 10)

