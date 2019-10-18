function cost = costfunction(known, estimated)
    cost = sum(double(not(strcmp(known, estimated))));
end
