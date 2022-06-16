% clear;
% % Load saved matrices X and y from file 
% load('ex3data1.mat');
% opts = templateLinear('Regularization','ridge','Learner','logistic','Lambda',0.001,'PassLimit',5);
% multiLogMdl = fitcecoc(X,y,'Learners',opts);
% fprintf('Training accuracy: %g%%',100*sum(y == predict(multiLogMdl,X))/length(y));
% i = randi(length(y));
% [class,~,~,prob] = predict(multiLogMdl,X(i,:));
% imshow(reshape(X(i,:),20,20))
% fprintf('True class: %d  |  Predicted class: %d | Probability of match: %.1f%%',y(i),class,100*prob(class));

%% Neural Network
clear;
close all;
load('ex3data1.mat');
load('ex3_companion.mat');
net ;% List the network variable properties
view(net); % Visualize the network
net.IW{1} ;% View the hidden layer weights

i = randi(length(y));
ysim = net(X(i,:)');
[~,class] = max(ysim);
imshow(reshape(X(i,:),20,20))
fprintf('True class: %d  |  Predicted class: %d | Probability of match: %.1f%%',y(i),class,100*ysim(class));

[~,ysim] = max(net(X'));
fprintf('Training accuracy: %g%%',100*sum(y == ysim')/length(y));