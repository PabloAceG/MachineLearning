% clear variables 
% close opened windows
% clean the comand windows
%-------------------------------------------------------------------------%
clear variables;
close all;
clc;
%-------------------------------------------------------------------------%

% Create 20 data of dimension stored in rows of data matrix x
% ---------------
% FIRST RUN:   20
% SECOND RUN: 200
% ---------------
numData = 200;
X = rand(numData, 2);

% First element
X(1, 1);
% First row, columns 1 and 2
X(1, 1 : 2);
% 2 to 4 row, first column
X(2 : 4, 1);
% All rows, first column
X(:, 1);

% Create an scattered plot
% scatter(X(:, 1), X(:, 2))

% XOR values as target outputs for data in X

% Checks wheter the values in the matrix X are bigger or smaller that 0.5.
% Therefore returns a boolean depending on that answer.
% TRUE: 1
% FALSE: 0
logicX = X < 0.5;

Y = double(xor(logicX(:, 1), logicX(:, 2)));

X1 = X(Y == 1, :);
X2 = X(Y ~= 1, :);

blueRGB = [0, 0, 255] / 256;
scatter(X1(:, 1), X1(:, 2), [], blueRGB)
hold on

redRGB = [255, 0, 0] / 256;
scatter(X2(:, 1), X2(:, 2), [], redRGB)

% Usage of fit K-nearest
% Neightbors = 1
Mdl = fitcknn(X(:, 1), X(:, 2), 'NumNeighbors',1);
k1 = predict(Mdl, Y)
% Neightbors = 3
Mdl = fitcknn(X(:, 1), X(:, 2), 'NumNeighbors',3);
k3 = predict(Mdl, Y)
% Neightbors = 10
Mdl = fitcknn(X(:, 1), X(:, 2), 'NumNeighbors',10);
k10 = predict(Mdl, Y)
% Neightbors = 100
Mdl = fitcknn(X(:, 1), X(:, 2), 'NumNeighbors',100);
k100 = predict(Mdl, Y)

hold on
scatter(X(:, 1), X(:, 2), 100, k1)

