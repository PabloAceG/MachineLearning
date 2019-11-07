classdef MLP < handle
    properties (SetAccess = private)
        inputDim
        hiddenDim
        outputDim
        
        hiddenWeights
        outputWeights
    end
    
    methods
        function obj = MLP(inputD, hiddenD, outputD)
            obj.inputDim      = inputD;
            obj.hiddenDim     = hiddenD;
            obj.outputDim     = outputD;
            obj.hiddenWeights = zeros(hiddenD, inputD  + 1);
            obj.outputWeights = zeros(outputD, hiddenD + 1);
        end
        
        function obj = initWeight(obj, variance)
            % Hidden Layers
            obj.hiddenWeights = - variance            ...
                                + 2 * variance        ...
                                * rand(obj.hiddenDim, ...
                                       obj.inputDim  + 1);
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
            % Hidden Layers
            hiddenNet = obj.hiddenWeights * [input; 1];
            hidden    = sigmoid(hiddenNet)';
            % Output Layer
            outputNet = obj.outputWeights * [hidden; 1]; 
            output    = sigmoid(outputNet)';
        end
        
        function output = compute_output(obj, input)
            [hN, h, oN, output] = obj.compute_net_activation(input);
        end
        
        function obj = adapt_to_target(obj, input, target, rate)
            [hN, h, oN, o] = obj.compute_net_activation(input);
            
            % Output Update
            t = zeros(obj.outputDim, 1);
            t(target + 1) = 1;
            
            % Output
            % TODO: Change for t
            e_out  = o - target;
            d_out  = e_out .* (o .* (1 - o));
            Aw_out = d_out * [h; 1].';
                
            % Hidden
            e_hidden  = obj.outputWeights.' * d_out;
            e_hidden  = e_hidden(1 : end - 1);
            d_hidden  = e_hidden .* (h .* (1 - h));
            Aw_hidden = d_hidden * [input; 1].';
            
            % Weights Update
            obj.outputWeights = obj.outputWeights - rate * Aw_out;
            obj.hiddenWeights = obj.hiddenWeights - rate * Aw_hidden;
        end
    end
end
