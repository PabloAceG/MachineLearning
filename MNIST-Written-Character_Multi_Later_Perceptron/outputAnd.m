function a = outputAnd(Xs)
    a = 1;
    for i=1:length(Xs)
        if(Xs(i) < 0.5)
            a = 0;
        end
    end
end