%% Load datasets

trainData = csvread('Data/train.csv', 1, 0);
% testData = csvread('Data/test.csv', 1, 0);

trainInd = floor(size(trainData,1)*0.8);

Xtr = trainData(1:trainInd, 2:end);
Ytr = trainData(1:trainInd, 1);
Ylabels = zeros(size(Ytr,1), 10);
for i = 1:size(Ylabels, 1)
    Ylabels(i, Ytr(i)+1) = 1;
end

Xtest = trainData(trainInd+1:end, 2:end);
Ytest = trainData(trainInd+1:end, 1);
Ytestlabels = zeros(size(Ytest,1), 10);
for i = 1:size(Ytestlabels, 1)
    Ytestlabels(i, Ytest(i)+1) = 1;
end

% Xtest = testData(:,2:end);
% Ytest = testData(:,1);
% Ytestlabels = zeros(size(Ytest,1), 10);
% for i = 1:size(Ytestlabels,1)
%     Ytestlabels(i,Ytest(i)+1) = 1;
% end

%% Define neural network and train

net = patternnet([200 10]);
view(net);

net = train(net, Xtr.', Ylabels');

yOut = net(Xtr.');
[~, yOut] = max(yOut);
yOut = yOut - 1;
fprintf('\nTrain accuracy: %2.2f%%\n\n', 100*(1 - symerr(yOut,Ytr.')/size(Xtr,1)));

%% Test performance on testing set

ytestOut = net(Xtest.');
[~, ytestOut] = max(ytestOut);
ytestOut = ytestOut - 1;
fprintf('\nTest accuracy: %2.2f%%\n\n', 100*(1 - symerr(ytestOut,Ytest.')/size(Xtest,1)));
