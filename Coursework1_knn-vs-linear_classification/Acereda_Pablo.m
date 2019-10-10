% clear variables
% close opened windows
% clean the comand windows
%-------------------------------------------------------------------------%
clear variables;
close all;
clc;
%-------------------------------------------------------------------------%

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% COURSEWORK 1: EXPERIMENTAL COMPARISON OF K-NN AND LINEAR CLASSIFICATION
% ON THE IRIS DATA-SET
% AUTHOR: PABLO ACEREDA

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% VARIABLES
% --- Colors --- 
redRGB   = [255, 0, 0] / 256;
greenRGB = [0, 255, 0] / 256;
blueRGB  = [0, 0, 255] / 256;

% LOAD DATA
load fisheriris
% Want to order data randomly to apply cross-validation
P = randperm(length(species));
% Data from flower (all in cm):
% { Sepal length | Sepal width | Petal length | Petal width }
X = meas(P, :);
% Class label:
% - Setosa
% - Versicolor
% - Virginica
Y = species(P);

% Logical label
logicSetosa     = strcmp(Y, 'setosa');
logicVersicolor = strcmp(Y, 'versicolor');
logicVirginica  = strcmp(Y, 'virginica');
% Data separated in categories
setosa     = X(logicSetosa     == 1, :);
versicolor = X(logicVersicolor == 1, :);
virginica  = X(logicVirginica  == 1, :);

% Iris Data Graphs
% Rows
for i = 1 : 1 : 4
    % Columns
    for j = 1 : 1 : 4
        % Location in graph
        sp = subplot(4, 4, ((i - 1) * 4 + j));
        % Diagonal 
        if i ~= j
            % Graph
            scatter(setosa(:, j),     setosa(:, i),     [], redRGB,   '.')
            hold on
            scatter(versicolor(:, j), versicolor(:, i), [], greenRGB, '.')
            hold on
            scatter(virginica(:, j),  virginica(:, i),  [], blueRGB,  '.')
        else
            t = "";
            switch(i)
                case 1
                    t = "Sepal.Length";
                case 2
                    t = "Sepal.Width";
                case 3
                    t = "Petal.Length";
                case 4
                    t = "Petal.Width";
                otherwise
            end
            text(0.1, 0.5, t, "Parent", sp); axis off
        end
    end
end

Mdl_knn    = fitcknn(X, Y);
% TODO: Linear means there must be a RIGHT and a WRONG set of values. 
% Which is the right value here?
Mdl_linear = fitclinear(X, logicSetosa);

% TODO: Once predicions are found, what now?
pred_knn    = predict(Mdl_knn, X);
pred_linear = predict(Mdl_linear, X);

% TODO: Can 'loss' function be used to calculate error?

% TODO: Repeat subplot, with estimated values



