function a = outputXOr(Xs)
    c = 0;
    for i=1:length(Xs)
        if(Xs(i) > 0.5)
            c = c+1;
        end
    end
    if(c==1)
        a = 1;
    else
        a = 0;
    end
end