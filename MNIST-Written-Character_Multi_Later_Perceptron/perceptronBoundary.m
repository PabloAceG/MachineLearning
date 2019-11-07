function points = perceptronBoundary(w, beta)
    points = [];
    l = w(1)*w(1) + w(2)*w(2);
    if(l>0.00001)
        a = (beta-w(3)) / l;
        root = a * [w(1) ; w(2)];
        orth = [w(2) ; -w(1)];
        points = [ root + 100*orth , root - 100*orth ];
      else
        points = [ 0 0; 0 0 ];
    end
end