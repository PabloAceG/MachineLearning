clear all
close all

tic 

% random seed
rng(3);

% specify target output function (represented by function pointers here)
target = @outputXOr
targetName='XOR'

%choose number of hidden neurons
hiddenSize = 2;

% number of training steps between two plot renderings 
speedUp = 100;

record=0; % set to 1 to record video 
% WARNING: these videos are /uncompressed/ at first and VERY LARGE
if record
    mov(1:1)=struct('cdata',[],'colormap',[]);
    frame=1;
    title=['mlp-' targetName];
    writerObj = VideoWriter([title '.avi'], 'Uncompressed AVI');
    open(writerObj);
end

% create training data
Neach = 20;
Ntotal = 4*Neach
% 0/1 values
X = [repmat([0;0],1,Neach) repmat([0;1],1,Neach) repmat([1;0],1,Neach) repmat([1;1],1,Neach)];
% plus noise
X = X + randn(2, Ntotal)*0.1;
%... plus constant feature:
XF = [X; repmat([1],1,Ntotal)];
% outputs
Y = repmat(0, 1, Ntotal);
for i=1:Ntotal
    Y(i) = target(X(:,i));
end

%show data (color chosen by labels in Y)
scatter(X(1,:), X(2,:), 25, Y, 'filled')

colormap('jet');
colorbar();

pbaspect([1 1 1]) % quadratic aspect ratio;

hold on;

% initialize MLP
m = MLP(2, hiddenSize, 1);
m = m.initWeight(1.0);
    
% initialize hidden neuron visualization
for i=1:hiddenSize
    points = perceptronBoundary(m.hiddenWeights(i,1:3), 0);
    strength = m.outputWeights(1,i);
    boundary(i) = plot(points(1,:), points(2,:), 'LineWidth',2, 'Color', [0.5 0.5 0.5]);
end

% initialize output visualization (countours generareted through a grid of input values)
meshSize = 20;
as = linspace(-0.3,1.3,meshSize);
bs = linspace(-0.3,1.3,meshSize);
[A B] = meshgrid(as,bs);
E = zeros(meshSize,meshSize);
for i=1:meshSize
    for j=1:meshSize
        E(i,j) = m.compute_output([A(i,j); B(i,j)]);
    end
end
[colorMat outputContour] = contour(A,B,E);
caxis([-0.1 1.1]);

% enforce axes limits
xlim([-0.3 1.3]);
ylim([-0.3 1.3]);

% initialize marker indicating current data
marker = scatter([], [], 100, 's', 'filled');

hold off;
    
for t = 1:10000
    t
    
    for i=1:speedUp
        % choose random sample from data
        index = randi([1 Ntotal], 1, 1);
    
        % evaluate MLP's output (fwd prop)
        yest = m.compute_output(X(:,index));
    
        % perform learning step (back prop)
        m.adapt_to_target(X(:,index), Y(index), 0.05);
    end
    
    % update visualizations
    for i=1:hiddenSize
        points = perceptronBoundary(m.hiddenWeights(i,1:3), 0);
        strength = m.outputWeights(1,i);
        strength = 1+2*sqrt(abs(strength));
        if strength>15
            strength = 15;
        end
        set(boundary(i), 'XData', points(1,:), 'YData', points(2,:), 'LineWidth',strength);
    end
    for i=1:meshSize
        for j=1:meshSize
            E(i,j) = m.compute_output([A(i,j); B(i,j)]);
        end
    end
    set(outputContour, 'ZData', E);
        
    drawnow;
    
    % keep frame for video
    if record
        mov(frame)=getframe(gcf);
        writeVideo(writerObj,mov(frame));
        frame=frame+1;
    end
end

% store video
if record
    close(writerObj);
end   

toc