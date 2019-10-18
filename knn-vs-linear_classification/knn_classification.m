% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% COURSEWORK 1: EXPERIMENTAL COMPARISON OF K-NN AND LINEAR CLASSIFICATION
% ON THE IRIS DATA-SET
% AUTHOR: PABLO ACEREDA
% FILE:   KNN_CLASSIFICATION.M

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function knn_avg_err = knn_classification(kDiv, k_value, filename)

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

    %                     DATA PARTITIONING INFORMATION
    %                    ===============================

    N    = length(species);
    jump = floor(N / kDiv);

    %                          LOG FILE CREATION
    %                         ===================

    % Moves path to file's path
    cd(fileparts(mfilename('fullpath')));

    % Creates file in case it does not existe
    filename = strcat("logs/knnlog/", filename);
    % Open file 
    logFile = fopen(filename, 'w');

    % Writes in log file
    fprintf(logFile,                                           ...
            ['INFO: ',                                         ...
             '==== NEW EXPERIMENT ====\n',                     ...
             'INFO: ',                                         ...
             'Data is going to be divided into %i pieces.\n'], ...
            kDiv);

    % Valid number of divisions
    if kDiv <= N

        % Write into log
        fprintf(logFile,                             ...
                ['INFO: ',                           ...
                 'The value for k is valid ',        ...
                 '(k[%i] < number of samples[%i]) ', ...
                 '-> VALID.\n'],                     ...
                kDiv,                                ...
                N);

        %                            VARIABLES
        %                           ===========

        % Column1: Error
        % Column2: Meas
        % Column3: Species Predicted
        % Column4: Species Labeled
        knn = cell(length(kDiv), 4);
        knn(:, 2) = {[100]};

        %                            ========
        %                             K-FOLD
        %                            ========

        for i = 1 : 1 : kDiv

            % Writes in log
            fprintf(logFile,                     ...
                    '\n-- K-FOLD: %i/%i --\n\n', ...
                    i,                           ...
                    kDiv);

            if     i == 1

                testX  = X(       1 : jump, :);
                testY  = Y(       1 : jump, :);
                trainX = X(jump + 1 : end,  :);
                trainY = Y(jump + 1 : end,  :);

            elseif i * jump <= N && i < kDiv 

                testX  = X((i - 1) * jump + 1 : i * jump, :);
                testY  = Y((i - 1) * jump + 1 : i * jump, :);
                trainX = [X(1 : (i - 1) * jump, :); ...
                          X(i * jump + 1: end, :)];
                trainY = [Y(1 : (i - 1) * jump, :); ...
                          Y(i * jump + 1: end, :)];

            else

                testX  = X((i - 1) * jump + 1 : end, :);
                testY  = Y((i - 1) * jump + 1 : end, :);
                trainX = X(1                  : (i - 1) * jump     , :);
                trainY = Y(1                  : (i - 1) * jump     , :);

            end

            %                          =====
            %                           KNN
            %                          =====

            %                          MODEL
            Mdl_knn = fitcknn (trainX, trainY, 'NumNeighbors', k_value);
            %                         TESTING
            pred_knn = predict(Mdl_knn, testX);
            %                          ERROR
            err_knn = costfunction(testY, pred_knn) / length(testY);

            % Writes in log
            fprintf(logFile,                          ...
                    ['** KNN (k=%i) **\n',            ...
                     'Prediction for training: %s\n', ...
                     'Actual values:           %s\n', ...
                     'Error: %f\n'],                  ...
                      k_value,                        ...
                      strjoin(pred_knn),              ...
                      strjoin(testY),                 ...
                      err_knn);

            %                      ERROR STORAGE
            knn(i, :) = {[err_knn] [trainX] [pred_knn] [testY]};

            % Log: extra separation
            fprintf(logFile, '\n');
        end
    % ------------------------------------------------------------------- %

        knn_avg_err = sum(cell2mat(knn(:, 1))) / length(knn);

        fprintf(logFile,                                                ...
                ['\n===============================================\n', ...
                'kNN Mean Error: %f'],                                      ...
                knn_avg_err);

    else
        fprintf(logFile,                             ...
                ['ERROR: ',                          ...
                 'The value for k is too large ',    ...
                 '(k[%i] < number of samples[%i]) ', ...
                 '-> VALID.\n'],                     ...
                kDiv,                                ...
                N);
    end

    % Writes everything in log
    fclose(logFile);
    
    knn_avg_err;

end
