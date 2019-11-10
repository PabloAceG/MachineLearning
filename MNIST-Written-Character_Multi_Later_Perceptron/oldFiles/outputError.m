% E = ??y??t?^2 = (?y??t)^T ? (?y??t)
function e = outputError(output, desired)
    e = transpose(output - desired) * (output - desired);
end