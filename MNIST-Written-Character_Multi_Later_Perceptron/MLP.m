classdef MLP < handle
    properties (SetAccess = private)
        % Dimensions
        inputDim
        hiddenDim
        outputDim
        
        % Number of Hidden Layers
        numHidden
        
        % Weights
        hiddenWeights
        outputWeights
    end
    
    methods
        function obj = MLP(inputD, hiddenD, outputD, numHid)
            % Dimensions
            obj.inputDim      = inputD;
            obj.hiddenDim     = hiddenD;
            obj.outputDim     = outputD;
            if nargin > 3
                % Number of Hidden Layers
                obj.numHidden     = numHid;
                % Hidden Weights
                obj.hiddenWeights = cell(numHid, 1);
                % Size of Weight Matrix
                for i = 1 : 1 : numHid
                    if i == 1 
                        size_cell = [hiddenD, inputD + 1];
                    else
                        size_cell = [hiddenD, hiddenD + 1];
                    end
                    % Matrix Initialization
                    obj.hiddenWeights(i) = mat2cell(zeros(size_cell), ...
                                                    size_cell(1),     ...
                                                    size_cell(2));
                end
            else
                % Number of Hidden Layers
                obj.numHidden = 1;
                % Weights Hidden
                obj.hiddenWeights = zeros(hiddenD, inputD  + 1);
            end
            % Weights
            obj.outputWeights = zeros(outputD, hiddenD + 1);
        end
        
        function obj = initWeight(obj, variance)
            % Hidden Layers
            if obj.numHidden == 1
               obj.hiddenWeights = - variance            ...
                                   + 2 * variance        ...
                                   * rand(obj.hiddenDim, ...
                                          obj.inputDim + 1);
            else
                for i = 1 : 1 : obj.numHidden
                    % Layer's Weight Dimensions
                    size_cell = size(cell2mat(obj.hiddenWeights(i)));
                    % Weight Initialization
                    obj.hiddenWeights(i) = mat2cell(- variance         ...
                                                    + 2 * variance     ...
                                                    * rand(size_cell), ...
                                                    size_cell(1),      ...
                                                    size_cell(2));
                end 
            end
            % Output Layer                   
            obj.outputWeights = - variance            ...
                                + 2 * variance        ...
                                * rand(obj.outputDim, ...
                                       obj.hiddenDim + 1);
        end
        
        function [hiddenNet, ...
                  hidden,    ...
                  outputNet, ...
                  output] = compute_net_activation(obj, ...
                                                   input)
            if obj.numHidden == 1
                % Hidden Layer
                hiddenNet = obj.hiddenWeights * [input; 1];
                hidden    = sigmoid(hiddenNet);
                % Output Layer
                outputNet = obj.outputWeights * [hidden; 1]; 
                output    = sigmoid(outputNet);
            else
                hiddenNet = cell(obj.numHidden, 1);
                hidden    = cell(obj.numHidden, 1);
                % Hidden Layers
                for i = 1 : 1 : obj.numHidden 
                    if i == 1 
                        in = input;
                    else
                        in = cell2mat(hidden(i - 1));
                    end
                    h = cell2mat(obj.hiddenWeights(i)) * [in; 1];
                    hiddenNet(i) = mat2cell(h,          ...
                                            size(h, 1), ...
                                            size(h, 2));
                    hidden(i)    = mat2cell(sigmoid(h), ...
                                            size(h, 1), ...
                                            size(h, 2));
                end
                % Output Layers
                outputNet = obj.outputWeights * ...
                           [cell2mat(hidden(obj.numHidden)); 1];
                output    = sigmoid(outputNet);
            end
        end
        
        function output = compute_output(obj, input)
            [hN, h, oN, o] = obj.compute_net_activation(input);
            % Output
            if obj.outputDim == 1
                output = o;
            else
                % Retrieves the index of the most likely solution
                [m, idx] = max(o);

                % Solutions from 0 to 9 (not 1 to 10)
                output = idx - 1;
            end
        end
        
        function obj = adapt_to_target(obj, input, target, rate)
            [hN, h, oN, o] = obj.compute_net_activation(input);
            
            % Target Output
            if obj.outputDim == 1
                t = target;
            else
                t = zeros(obj.outputDim, 1);
                t(target + 1) = 1;
            end
            
            % Output
            e_out  = o - t;
            d_out  = e_out .* (o .* (1 - o));
            
            if obj.numHidden == 1 
                % Weight Variation for Output 
                Aw_out = d_out * [h; 1].';
                
                % Hidden
                e_hidden = obj.outputWeights.' * d_out;
                e_hidden = e_hidden(1 : end - 1);
                d_hidden = e_hidden .* (h .* (1 - h));
                % Weight Variation for Hidden
                Aw_hidden = d_hidden * [input; 1].';
                % Weights Update
                obj.outputWeights = obj.outputWeights - rate * Aw_out;
                obj.hiddenWeights = obj.hiddenWeights - rate * Aw_hidden;
            else
                % Weight Variation for Output 
                Aw_out = d_out * [cell2mat(h(obj.numHidden)); 1].';
                
                % Hidden
                for i = obj.numHidden : -1 : 1
                    if i == obj.numHidden
                        d = d_out; 
                        w = obj.outputWeights;
                        v = [cell2mat(h(i - 1)); 1];
                    elseif i == 1
                        w = cell2mat(obj.hiddenWeights(i + 1));
                        v = [input; 1];
                    else
                        w = cell2mat(obj.hiddenWeights(i + 1));
                        v = [cell2mat(h(i - 1)); 1];
                    end
                     
                    % Output
                    o = cell2mat(h(i));
                    
                    e = w.' * d;
                    e = e(1 : end - 1);
                    d = e .* (o .* (1 - o)); 
                    Aw_hi = d * v.';
                    Aw_hidden(i) = mat2cell(Aw_hi,          ...
                                            size(Aw_hi, 1), ...
                                            size(Aw_hi, 2));
                end
                % Weights Update
                obj.outputWeights = obj.outputWeights - rate * Aw_out;
                for i = 1 : 1 : obj.numHidden
                    hW = cell2mat(obj.hiddenWeights(i)) ...
                         - (rate * cell2mat(Aw_hidden(i)));
                    obj.hiddenWeights(i) = mat2cell(hW,          ...
                                                    size(hW, 1), ...
                                                    size(hW, 2));
                end
            end     
        end
    end
end
