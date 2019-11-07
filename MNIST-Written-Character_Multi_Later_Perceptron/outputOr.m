function a = outputOr(Xs)
    a = 0;
    for i=1:length(Xs)
        if(Xs(i) > 0.5)
            a = 1
        end
    end
end