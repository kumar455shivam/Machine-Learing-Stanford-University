clc;
clear all;
method = 'regularization';

%% Model

if strcmpi(method, 'normal')
        data = readtable('ex2data1.txt');
        data.Properties.VariableNames = {'Exam1','Exam2','Admitted'};
        data.Admitted = logical(data.Admitted);
        summary(data)
        logMdl = fitglm(data,'Distribution','binomial');
        theta = logMdl.Coefficients.Estimate
        % Predict the probability for a student with scores of 45 and 85
        prob = predict(logMdl,[45 85]);
        fprintf('For a student with scores 45 and 85, we predict an admission probability of %f\n\n', prob);
        % Compute the training accuracy
        Admitted = predict(logMdl,data) > 0.5;
        fprintf('Train Accuracy: %f\n', mean(double(Admitted == data.Admitted)) * 100);
        figure; hold on;
        % Plot the positive and negative examples
        plotMdlData(data);

        % Plot the decision boundary
        xvals = [min(data.Exam1), max(data.Exam1)];
        yvals = -(theta(1)+theta(2)*xvals)/theta(3);
        plot(xvals,yvals); hold off;
        ylim([min(data.Exam2),max(data.Exam2)]);

        % Labels and Legend
        xlabel('Exam 1 score')
        ylabel('Exam 2 score')
        legend('Admitted','Not admitted','Decision Boundary')
        hold off;
elseif strcmpi(method, 'classification')
        data = readtable('ex2data1.txt');
        data.Properties.VariableNames = {'Exam1','Exam2','Admitted'};
        logMdl = trainedModel.GeneralizedLinearModel;
elseif strcmpi(method, 'polynomial')
        clear;
        data = readtable('ex2data2.txt');
        data.Properties.VariableNames = {'Test1','Test2','Pass'};
        data.Pass = logical(data.Pass);
        summary(data)
        logMdl = fitglm(data,'poly66','Distribution','binomial');
        % Compute accuracy on our training set
        Pass = predict(logMdl,data) > 0.5;
        fprintf('Train Accuracy: %f\n', mean(Pass == data.Pass) * 100);
        figure; hold on;
        % Plot the positive and negative examples
        plotMdlData(data);

        % Plot the decision boundary
        xvals = linspace(min(data.Test1), max(data.Test1));
        yvals = linspace(min(data.Test1), max(data.Test2));
        [X, Y] = meshgrid(xvals,yvals);
        p = predict(logMdl,[X(:),Y(:)]);
        contour(X,Y,reshape(p,size(X)),[0.5,0.5]); hold off;
        % Labels and legend
        xlabel('Test 1 score')
        ylabel('Test 2 score')
        legend('Pass', 'Fail','Decision Boundary')
elseif strcmpi(method, 'regularization')
        clear;
        X = load('ex2data2.txt');
        y = X(:,3);
        X(:,3) = [];
        % Create the polynomial feature matrix up to power 6
        powers = [nchoosek(0:6,2); fliplr(nchoosek(0:6,2));1 1;2 2;3 3]';
        powers(:,sum(powers)>6) = [];
        Xpoly = (X(:,1).^powers(1,:)).*(X(:,2).^powers(2,:));
        lambda = 0.001;
        logMdl = fitclinear(Xpoly,y,'Lambda',lambda,'Learner','logistic','Regularization','ridge');
        logMdl.Bias
        logMdl.Beta
        % Obtain the class labels and compute the training accuracy
        Pass = predict(logMdl,Xpoly);
        fprintf('Train Accuracy: %f\n', mean(Pass == y) * 100);
        % Plot the positve and negative examples
        figure; hold on;
        plotMdlData(array2table([X y],'VariableNames',{'Test1','Test2','Pass'})); 

        % Plot the decision boundary
        xvals = linspace(min(X(:,1)), max(X(:,1)));
        yvals = linspace(min(X(:,2)), max(X(:,2)));
        [Xgrid, Ygrid] = meshgrid(xvals,yvals);
        Xpolygrid = (Xgrid(:).^powers(1,:)).*(Ygrid(:).^powers(2,:));
        [~,Score] = predict(logMdl,Xpolygrid); % Obtain the probability scores
        contour(Xgrid,Ygrid,reshape(Score(:,2),size(Xgrid)),[0.5,0.5]); hold off;
        % Labels and legend
        xlabel('Test 1 score')
        ylabel('Test 2 score')
        legend('Pass', 'Fail','Decision Boundary')
            
end

    