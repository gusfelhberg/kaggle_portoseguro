function [trainedClassifier, validationAccuracy] = trainLDAClassifier(preprocessedtrain)
% [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
% returns a trained classifier and its accuracy. This code recreates the
% classification model trained in Classification Learner app. Use the
% generated code to automate training the same model with new data, or to
% learn how to programmatically train models.
%
%  Input:
%      trainingData: a table containing the same predictor and response
%       columns as imported into the app.
%
%  Output:
%      trainedClassifier: a struct containing the trained classifier. The
%       struct contains various fields with information about the trained
%       classifier.
%
%      trainedClassifier.predictFcn: a function to make predictions on new
%       data.
%
%      validationAccuracy: a double containing the accuracy in percent. In
%       the app, the History list displays this overall accuracy score for
%       each model.
%
% Use the code to train the model with new data. To retrain your
% classifier, call the function from the command line with your original
% data or new data as the input argument trainingData.
%
% For example, to retrain a classifier trained with the original data set
% T, enter:
%   [trainedClassifier, validationAccuracy] = trainClassifier(T)
%
% To make predictions with the returned 'trainedClassifier' on new data T2,
% use
%   yfit = trainedClassifier.predictFcn(T2)
%
% T2 must be a table containing at least the same predictor columns as used
% during training. For details, enter:
%   trainedClassifier.HowToPredict

% Auto-generated by MATLAB on 03-Dec-2017 17:34:40


% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = preprocessedtrain;
predictorNames = {'ps_ind_01', 'ps_ind_03', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin', 'ps_ind_02_cat_1', 'ps_ind_02_cat_2', 'ps_ind_02_cat_3', 'ps_ind_02_cat_4', 'ps_ind_04_cat_0', 'ps_ind_04_cat_1', 'ps_ind_05_cat_0', 'ps_ind_05_cat_1', 'ps_ind_05_cat_2', 'ps_ind_05_cat_3', 'ps_ind_05_cat_4', 'ps_ind_05_cat_5', 'ps_ind_05_cat_6', 'ps_car_02_cat_0', 'ps_car_02_cat_1', 'ps_car_04_cat_1', 'ps_car_04_cat_2', 'ps_car_04_cat_3', 'ps_car_04_cat_4', 'ps_car_04_cat_5', 'ps_car_04_cat_6', 'ps_car_04_cat_7', 'ps_car_04_cat_8', 'ps_car_04_cat_9', 'ps_car_07_cat_0', 'ps_car_07_cat_1', 'ps_car_08_cat_1', 'ps_car_09_cat_0', 'ps_car_09_cat_1', 'ps_car_09_cat_2', 'ps_car_09_cat_3', 'ps_car_09_cat_4', 'ps_car_10_cat_1', 'ps_car_10_cat_2', 'ps_reg_012', 'ps_reg_01ps_reg_02', 'ps_reg_01ps_reg_03', 'ps_reg_01ps_car_12', 'ps_reg_01ps_car_13', 'ps_reg_01ps_car_14', 'ps_reg_01ps_car_15', 'ps_reg_01ps_calc_01', 'ps_reg_01ps_calc_02', 'ps_reg_01ps_calc_03', 'ps_reg_022', 'ps_reg_02ps_reg_03', 'ps_reg_02ps_car_12', 'ps_reg_02ps_car_13', 'ps_reg_02ps_car_14', 'ps_reg_02ps_car_15', 'ps_reg_02ps_calc_01', 'ps_reg_02ps_calc_02', 'ps_reg_02ps_calc_03', 'ps_reg_032', 'ps_reg_03ps_car_12', 'ps_reg_03ps_car_13', 'ps_reg_03ps_car_14', 'ps_reg_03ps_car_15', 'ps_reg_03ps_calc_01', 'ps_reg_03ps_calc_02', 'ps_reg_03ps_calc_03', 'ps_car_122', 'ps_car_12ps_car_13', 'ps_car_12ps_car_14', 'ps_car_12ps_car_15', 'ps_car_12ps_calc_01', 'ps_car_12ps_calc_02', 'ps_car_12ps_calc_03', 'ps_car_132', 'ps_car_13ps_car_14', 'ps_car_13ps_car_15', 'ps_car_13ps_calc_01', 'ps_car_13ps_calc_02', 'ps_car_13ps_calc_03', 'ps_car_142', 'ps_car_14ps_car_15', 'ps_car_14ps_calc_01', 'ps_car_14ps_calc_02', 'ps_car_14ps_calc_03', 'ps_car_152', 'ps_car_15ps_calc_01', 'ps_car_15ps_calc_02', 'ps_car_15ps_calc_03', 'ps_calc_012', 'ps_calc_01ps_calc_02', 'ps_calc_01ps_calc_03', 'ps_calc_022', 'ps_calc_02ps_calc_03', 'ps_calc_032'};
predictors = inputTable(:, predictorNames);
response = inputTable.target;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationDiscriminant = fitcdiscr(...
    predictors, ...
    response, ...
    'DiscrimType', 'linear', ...
    'Gamma', 0, ...
    'FillCoeffs', 'off', ...
    'ClassNames', [0; 1]);

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
discriminantPredictFcn = @(x) predict(classificationDiscriminant, x);
trainedClassifier.predictFcn = @(x) discriminantPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'ps_ind_01', 'ps_ind_03', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin', 'ps_ind_02_cat_1', 'ps_ind_02_cat_2', 'ps_ind_02_cat_3', 'ps_ind_02_cat_4', 'ps_ind_04_cat_0', 'ps_ind_04_cat_1', 'ps_ind_05_cat_0', 'ps_ind_05_cat_1', 'ps_ind_05_cat_2', 'ps_ind_05_cat_3', 'ps_ind_05_cat_4', 'ps_ind_05_cat_5', 'ps_ind_05_cat_6', 'ps_car_02_cat_0', 'ps_car_02_cat_1', 'ps_car_04_cat_1', 'ps_car_04_cat_2', 'ps_car_04_cat_3', 'ps_car_04_cat_4', 'ps_car_04_cat_5', 'ps_car_04_cat_6', 'ps_car_04_cat_7', 'ps_car_04_cat_8', 'ps_car_04_cat_9', 'ps_car_07_cat_0', 'ps_car_07_cat_1', 'ps_car_08_cat_1', 'ps_car_09_cat_0', 'ps_car_09_cat_1', 'ps_car_09_cat_2', 'ps_car_09_cat_3', 'ps_car_09_cat_4', 'ps_car_10_cat_1', 'ps_car_10_cat_2', 'ps_reg_012', 'ps_reg_01ps_reg_02', 'ps_reg_01ps_reg_03', 'ps_reg_01ps_car_12', 'ps_reg_01ps_car_13', 'ps_reg_01ps_car_14', 'ps_reg_01ps_car_15', 'ps_reg_01ps_calc_01', 'ps_reg_01ps_calc_02', 'ps_reg_01ps_calc_03', 'ps_reg_022', 'ps_reg_02ps_reg_03', 'ps_reg_02ps_car_12', 'ps_reg_02ps_car_13', 'ps_reg_02ps_car_14', 'ps_reg_02ps_car_15', 'ps_reg_02ps_calc_01', 'ps_reg_02ps_calc_02', 'ps_reg_02ps_calc_03', 'ps_reg_032', 'ps_reg_03ps_car_12', 'ps_reg_03ps_car_13', 'ps_reg_03ps_car_14', 'ps_reg_03ps_car_15', 'ps_reg_03ps_calc_01', 'ps_reg_03ps_calc_02', 'ps_reg_03ps_calc_03', 'ps_car_122', 'ps_car_12ps_car_13', 'ps_car_12ps_car_14', 'ps_car_12ps_car_15', 'ps_car_12ps_calc_01', 'ps_car_12ps_calc_02', 'ps_car_12ps_calc_03', 'ps_car_132', 'ps_car_13ps_car_14', 'ps_car_13ps_car_15', 'ps_car_13ps_calc_01', 'ps_car_13ps_calc_02', 'ps_car_13ps_calc_03', 'ps_car_142', 'ps_car_14ps_car_15', 'ps_car_14ps_calc_01', 'ps_car_14ps_calc_02', 'ps_car_14ps_calc_03', 'ps_car_152', 'ps_car_15ps_calc_01', 'ps_car_15ps_calc_02', 'ps_car_15ps_calc_03', 'ps_calc_012', 'ps_calc_01ps_calc_02', 'ps_calc_01ps_calc_03', 'ps_calc_022', 'ps_calc_02ps_calc_03', 'ps_calc_032'};
trainedClassifier.ClassificationDiscriminant = classificationDiscriminant;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2017b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = preprocessedtrain;
predictorNames = {'ps_ind_01', 'ps_ind_03', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin', 'ps_ind_02_cat_1', 'ps_ind_02_cat_2', 'ps_ind_02_cat_3', 'ps_ind_02_cat_4', 'ps_ind_04_cat_0', 'ps_ind_04_cat_1', 'ps_ind_05_cat_0', 'ps_ind_05_cat_1', 'ps_ind_05_cat_2', 'ps_ind_05_cat_3', 'ps_ind_05_cat_4', 'ps_ind_05_cat_5', 'ps_ind_05_cat_6', 'ps_car_02_cat_0', 'ps_car_02_cat_1', 'ps_car_04_cat_1', 'ps_car_04_cat_2', 'ps_car_04_cat_3', 'ps_car_04_cat_4', 'ps_car_04_cat_5', 'ps_car_04_cat_6', 'ps_car_04_cat_7', 'ps_car_04_cat_8', 'ps_car_04_cat_9', 'ps_car_07_cat_0', 'ps_car_07_cat_1', 'ps_car_08_cat_1', 'ps_car_09_cat_0', 'ps_car_09_cat_1', 'ps_car_09_cat_2', 'ps_car_09_cat_3', 'ps_car_09_cat_4', 'ps_car_10_cat_1', 'ps_car_10_cat_2', 'ps_reg_012', 'ps_reg_01ps_reg_02', 'ps_reg_01ps_reg_03', 'ps_reg_01ps_car_12', 'ps_reg_01ps_car_13', 'ps_reg_01ps_car_14', 'ps_reg_01ps_car_15', 'ps_reg_01ps_calc_01', 'ps_reg_01ps_calc_02', 'ps_reg_01ps_calc_03', 'ps_reg_022', 'ps_reg_02ps_reg_03', 'ps_reg_02ps_car_12', 'ps_reg_02ps_car_13', 'ps_reg_02ps_car_14', 'ps_reg_02ps_car_15', 'ps_reg_02ps_calc_01', 'ps_reg_02ps_calc_02', 'ps_reg_02ps_calc_03', 'ps_reg_032', 'ps_reg_03ps_car_12', 'ps_reg_03ps_car_13', 'ps_reg_03ps_car_14', 'ps_reg_03ps_car_15', 'ps_reg_03ps_calc_01', 'ps_reg_03ps_calc_02', 'ps_reg_03ps_calc_03', 'ps_car_122', 'ps_car_12ps_car_13', 'ps_car_12ps_car_14', 'ps_car_12ps_car_15', 'ps_car_12ps_calc_01', 'ps_car_12ps_calc_02', 'ps_car_12ps_calc_03', 'ps_car_132', 'ps_car_13ps_car_14', 'ps_car_13ps_car_15', 'ps_car_13ps_calc_01', 'ps_car_13ps_calc_02', 'ps_car_13ps_calc_03', 'ps_car_142', 'ps_car_14ps_car_15', 'ps_car_14ps_calc_01', 'ps_car_14ps_calc_02', 'ps_car_14ps_calc_03', 'ps_car_152', 'ps_car_15ps_calc_01', 'ps_car_15ps_calc_02', 'ps_car_15ps_calc_03', 'ps_calc_012', 'ps_calc_01ps_calc_02', 'ps_calc_01ps_calc_03', 'ps_calc_022', 'ps_calc_02ps_calc_03', 'ps_calc_032'};
predictors = inputTable(:, predictorNames);
response = inputTable.target;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationDiscriminant, 'KFold', 5);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
