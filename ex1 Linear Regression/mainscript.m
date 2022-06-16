clc;
clear all;
close all;

%% Load data
data = load('ex1data1.txt'); % read comma separated data

X = data(:, 1); y = data(:, 2);
s = size(X);
temp = zeros(s(1), s(2)+1);
temp(:,1) = 1;
temp(:,2:end) = X;
X=temp;

 X1 = [ones(20,1) (exp(1) + exp(2) * (0.1:0.1:2))'];
  Y1 = X1(:,2) + sin(X1(:,1)) + cos(X1(:,2));
  X = X1;
  y = Y1;
%% Plotting intial data
%plotData(X,y);

%% Computing Hypothesis

theta_initial  = [0.5 -0.5]';
alpha = 0.01;
num_iters = 10;

[theta, J_history] = gradientDescent(X, y, theta_initial, alpha, num_iters);

h = theta(1) + theta(2)*X(:,2);

%figure
plotData(X(:,2),y);
hold on;
plot(X(:,2), h,'k');

figure
plot(J_history);