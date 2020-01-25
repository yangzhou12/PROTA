% Load dataset
load FERETC80A45; % Each sample is a 32x32 matrix
[dc, dr, numSpl] = size(fea2D); % 32x32x320

% Partition the training and test sets
load DBpart; % 2 images per class for training, 2 image per class for test
fea2D_Train = fea2D(:, :, trainIdx);

% Train PORTA with moment-based concurrent regularization
P = 500; % Reduce the dimensionality to 500
model = PROTA_MCR(fea2D_Train, P, 'regParam', 5e2, ...
        'maxIters', 500, 'tol', 1e-5);
% PROTA_MCR Projection
newfea = projPROTA_MCR(fea2D, model);
    
% % Train PORTA with Bayesian concurrent regularization
% model = PROTA_BCR(fea2D_Train, 'regParam', 5e2, ...
%         'maxIters', 500, 'tol', 1e-5);
% % PROTA_BCR Projection
% newfea = projPROTA_BCR(fea2D, model);

% Sort the projected features by Fisher scores
[odrIdx, stFR] = sortProj(newfea(:,trainIdx), gnd(trainIdx));
newfea = newfea(odrIdx,:); 

% Classification via 1NN
dimTest = 100; % the number of features to be fed into a classifier
testfea = newfea(1:dimTest,:);
% In practice, it would be better to test different values of dimTest 
% for the best classification performance.

nnMd = fitcknn(testfea(:,trainIdx)', gnd(trainIdx));
label = predict(nnMd, testfea(:,testIdx)');
Acc = sum(gnd(testIdx) == label)/length(testIdx)
