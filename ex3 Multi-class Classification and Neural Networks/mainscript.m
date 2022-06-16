% % Load saved matrices from file
% load('ex3data1.mat');
% % The matrices X and y will now be in your MATLAB environment
% % a = zeros(20,20);
% % for i = 1:20
% %     a(:,i) = X(1, (20*(i-1)+1):20*(i));
% % end
% 
% m = size(X, 1);
% % Randomly select 100 data points to display
% rand_indices = randperm(m);
% sel = X(rand_indices(1:100), :);
% displayData(sel);
% 
% %% Model implementation
% 
% num_labels = 10; % 10 labels, from 1 to 10 
% lambda = 0.1;
% [all_theta] = oneVsAll(X, y, num_labels, lambda);
% pred = predictOneVsAll(all_theta, X);
% fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%% Nerual network
close all;
clc;
load('ex3data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));

% Load saved matrices from file
load('ex3weights.mat'); 
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26