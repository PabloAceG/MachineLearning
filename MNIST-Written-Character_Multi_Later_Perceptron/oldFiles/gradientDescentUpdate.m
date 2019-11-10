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
function W = gradientDescentUpdate(O, W, T, lr)

    % Weights Variation
    Aw = cell(size(W));
    
    for i = length(O) : -1 : 2
        % Layer Output
        o = cell2mat(O(i));
        % Input Vector
        v = [cell2mat(O(i - 1)); 1];
        
        if i == length(O) % Output Layer
            % Correct output
            t = zeros(size(cell2mat(O(i)), 1), 1);
            t(T + 1) = 1;
            % Error Output Layer
            e = o - t;
        else              % Hidden layers
            % Error Hidden Layers
            e = cell2mat(W(i)).' * d;
            e = e(1 : end - 1);
        end
        
        % Desired Output
        d = e .* (o .* (1 - o));
        
        % Weight Variation for Layer i
        aw_i  = d * v.';
        Aw(i - 1) = mat2cell(aw_i, size(aw_i, 1), size(aw_i, 2));
    end
    
    for i = 1 : 1 : length(W)
        W(i) = mat2cell(cell2mat(W(i)) - (lr * cell2mat(Aw(i))), size(cell2mat(W(i)), 1), size(cell2mat(W(i)), 2)); 
    end
   
    W;
end




