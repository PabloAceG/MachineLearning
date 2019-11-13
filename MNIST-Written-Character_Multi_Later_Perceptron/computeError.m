function e = computeError(predictions, desired)
    
    mistakes = (predictions ~= desired);
    
    e = double(sum(mistakes) / length(mistakes));
end