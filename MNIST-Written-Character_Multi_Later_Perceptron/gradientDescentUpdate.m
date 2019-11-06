%              dE          dE       do^(1)
% d(n - 1) = -------- = -------- * --------
%             da^(1)     do^(1)     da^(1)
%
%                 dE
% w(n - 1) = ------------
%             dW^(n - 1) 
%
% d(n) = (w(n + 1)^T .* d(n + 1)) * (o(n) .* (1 - o(n))
%
% w(n) = d(n) .* v(n)^T
%
% Params:
%   a - Activations
%   v  - Outputs
%   W  - Weights
%   t  - Target
%   lr - Learning Rate
% Returns:
%   w - Weights updated
function w = gradientDescentUpdate(O, W, T, lr)

    % Weights Variation
    Aw = cell(size(W));
    
    for i = length(W) : -1 : 1
        % Layer Output
        o = cell2mat(O(i));
        % Input Vector
        v = [o; 1];
        
        if i == length(W) % Output Layer
            % Correct output
            t = zeros(size(cell2mat(W(i)), 1), 1);
            t(T + 1) = 1;
            % Error Output Layer
            e = o - t;
        else              % Hidden layers
            % Error Hidden Layers
            e = cell2mat(W(i + 1)).' * d;
            e = e(1 : end - 1);
        end
        
        % Desired Output
        d = e .* (o .* (1 - o));
        
        % Weight Variation for Layer i
        aw_i = d * v.';
        Aw(i) = mat2cell(aw_i, size(aw_i, 1), size(aw_i, 2));
    end
    
    w = Aw;
    
    % TODO: Implement newly calculated Weights
    
end




