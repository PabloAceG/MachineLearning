% clear variables
% close opened windows
% clean the comand windows
%-------------------------------------------------------------------------%
clear variables;
close all;
clc;
%-------------------------------------------------------------------------%

%                              LOAD DATA
%                             ===========
load fisheriris meas species

%                                DATA 
%                               ======

% Want to order samples randomly to apply cross-validation
P = randperm(length(species));

% Data from flower (all in cm):
% { Sepal length | Sepal width | Petal length | Petal width }
X = meas(P, :);

% Class label:
% - Setosa
% - Versicolor
% - Virginica
Y = species(P);

%                                 LINEAR
%                                ========

iterations = 10; 
lin_errs   = zeros(iterations, 2);

for i = 1 : 1 : iterations
    lin_errs(i, :) = [i, ...
                      linear_classification(10 * i, ...
                                            strcat('linearExperimentk', ...
                                                   num2str(10 * i),     ...
                                                   '.log'))];
                      
end

%                                  KNN
%                                 =====

iterations = 150;
knn_errs   = zeros(iterations, 2); 

for i = 1 : 1 : iterations
    knn_errs(i, :) = [i,
                      knn_classification(10, i, ...
                                        strcat('knnExperimentk', ...
                                               num2str(i),       ...
                                               '.log'))];
end

%                              ==========
%                               PLOTTING
%                              ==========

% Iris Data Graphs
% Rows
for i = 1 : 1 : 4
    % Columns
    for j = 1 : 1 : 5
        if j <= 4
            % Location in graph
            sp = subplot(4, 8, ((i - 1) * 8 + j));
            % Diagonal             
            if mod(i, 4) ~= mod(j, 4)
                gscatter(X(:, j), X(:, i), Y, 'rgb', '.', 6);
                % Hide legend
                legend('off')
            else
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
            
        else
            % Linear Error
            subplot(4, 8, [5: 8, 13 : 16]);
            bar(lin_errs(:, 1), lin_errs(:, 2))
            % subplot(4, 8, [21: 24, 29 : 32]);
            % Statistics obtained kNN
            
        end 
    end
end

