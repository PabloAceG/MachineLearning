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

%                             LOGS FOLDERS
%                            ==============

if ~exist("logs/linearlog", 'dir')
   mkdir("logs/linearlog")
   
   disp("Directory logs/linearlog/ has been created.")
end

if ~exist("logs/knnlog", 'dir') 
   mkdir("logs/knnlog")
   
   disp("Directory logs/knnlog/ has been created.")
end

%                                 LINEAR
%                                ========

iterations = 10; 
lin_errs   = zeros(iterations, 2);

disp("Linear classification is executing")

for i = 1 : 1 : iterations
    lin_errs(i, :) = [i * 10,                                                ...
                      linear_classification(10 * i,                     ...
                                            strcat('linearExperimentk', ...
                                                   num2str(10 * i),     ...
                                                   '.log'))];
                      
end

% Minimal error Linear Classification
min_lin_err = min(lin_errs(:, 2));
min_lin_err = lin_errs(:, 2) == min_lin_err;
min_lin_err = lin_errs(min_lin_err == 1, :);

disp(strjoin(["The minimum error for linear classification is ", ...
              num2str(min_lin_err(2)),                           ...
              " for k = ",                                       ...
              num2str(min_lin_err(1))]))
disp("Linear classification has been executed successfully")
disp("")

%                                  KNN
%                                 =====

iterations = 150;
knn_errs   = zeros(iterations, 2); 

disp("kNN classification is about to be executed")

for i = 1 : 1 : iterations
    knn_errs(i, :) = [i,                                         ...
                      knn_classification(10, i,                  ...
                                        strcat('knnExperimentk', ...
                                               num2str(i),       ...
                                               '.log'))];
end

% Minimal error Linear Classification
min_knn_err = min(knn_errs(:, 2));
min_knn_err = knn_errs(:, 2) == min_knn_err;
min_knn_err = knn_errs(min_knn_err == 1, :);

disp(strjoin(["The minimum error for kNN classification is ",   ...
              num2str(min_knn_err(2)),                          ...
              " for k = 10 and the number of neighbors being ", ...
              num2str(min_knn_err(1))]))

disp("kNN classification has been executed successfully")

%                              ==========
%                               PLOTTING
%                              ==========

disp("Plotting data")

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
            bar(lin_errs(:, 1),    lin_errs(:, 2), 'b')
            hold on
            bar(min_lin_err(:, 1), min_lin_err(:, 2), 'g')
            % Title
            title("Linear Classification Error")
            % Labels
            xlabel("k-value")
            ylabel("Error")
            
            % kNN Error
            subplot(4, 8, [21: 24, 29 : 32]);
            bar(knn_errs(:, 1),    knn_errs(:, 2), 'b')
            hold on
            bar(min_knn_err(:, 1), min_knn_err(:, 2), 'g')
            % Title
            title("kNN Classification Error")
            % Labels
            xlabel("Num neightbors")
            ylabel("Error")
            
        end 
    end
end
