clear variables;
close all;
clc;
% Load and draw points
load weightData;

scatter(X, Y, 'filled')

xlabel('height')
ylabel('weight')
% %%%%%%%%%%%%%%%%%%%%

% Perceptron 
xMin = 1.6;
xMax = 1.9;

xlim([xMin xMax]);

a = -16;
b =  14;

% Calculate y-values
fxMin = a + b .* xMin;
fxMax = a + b .* xMax;

% Plot function
hold on
plot(xlim, [fxMin, fxMax])

% %%%%%%%%%%%%%%%%%%%%

% Mean Square Error 
mseXY = mse(a, b, X, Y)

% Linear Regression
% Using the 40 samples to train
[a, b] = lingRef(X, Y);

mseOpt = mse(a, b, X, Y)

fxMin = a + b .* xMin;
fxMax = a + b .* xMax;

% Visualizacion
hold on
plot(xlim, [fxMin, fxMax])

% Using the first 20 samples
[a, b] = lingRef(X(1:20,:), Y(1:20,:));
mseFirst = mse(a, b, X, Y)

fxMin = a + b .* xMin;
fxMax = a + b .* xMax;

% Visualizacion
hold on
plot(xlim, [fxMin, fxMax])

% Using the first 20 samples
[a, b] = lingRef(X(21:40,:), Y(21:40,:));
mseLast = mse(a, b, X, Y)

fxMin = a + b .* xMin;
fxMax = a + b .* xMax;

% Visualizacion
hold on
plot(xlim, [fxMin, fxMax])

legend({'Samples', 'Original', 'Optimal', '20 first', '20 last'}, ...
       'Location', 'northwest')

