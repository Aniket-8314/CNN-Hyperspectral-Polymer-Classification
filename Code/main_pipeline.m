%% Master Pipeline: CNN-Based Classification of Hyperspectral Polymer Images
% Course: ME3101 - Autumn 2026
% Polymer Categories: HDPE, PET, PP, PS
%
% This pipeline uses the provided helper codes (readfile.m and reduceHypercubePCA.m)
% as the foundational data loading and PCA band transformation engine:
%   Task 1: Read and inspect hyperspectral data (using imhypercube from readfile.m)
%   Task 2: Partition data & select 10 common bands using reduceHypercubePCA on training data
%   Task 3: Standardize spatial size using balanced zero-padding only (80 x 80 x 10)
%   Task 4: Construct MATLAB datastores
%   Task 5: CNN architecture design and training
%   Task 6: Controlled design comparisons (depth, learning rate, regularization)
%   Task 7: Evaluation on held-out test images (confusion matrix, precision, recall, F1)
%   Task 8: Inference demonstration on an unseen image

clc; clear; close all;
rng(42); % Fixed seed for reproducibility

% Determine project root directory automatically
scriptDir = fileparts(mfilename('fullpath'));
if isfolder(fullfile(scriptDir, '..', 'Dataset'))
    rootDir = fullfile(scriptDir, '..');
elseif isfolder(fullfile(scriptDir, 'Dataset'))
    rootDir = scriptDir;
else
    rootDir = pwd;
end

addpath(fullfile(rootDir, 'Code'));
cd(rootDir);

fprintf('====================================================================\n');
fprintf('   ME3101 MINI-PROJECT 2: HYPERSPECTRAL POLYMER CLASSIFICATION CNN   \n');
fprintf('   Project Root: %s\n', rootDir);
fprintf('====================================================================\n\n');

%% TASK 1: Read and Inspect Hyperspectral Data (readfile.m approach)
fprintf('>>> TASK 1: Reading and Inspecting Hyperspectral Data (readfile.m approach)...\n');
datasetDir = 'Dataset';
classes = {'HDPE', 'PET', 'PP', 'PS'};
numClasses = numel(classes);

allFiles = {};
allLabels = {};

for c = 1:numClasses
    cls = classes{c};
    bils = dir(fullfile(datasetDir, cls, '*.bil'));
    for i = 1:numel(bils)
        allFiles{end+1} = fullfile(bils(i).folder, bils(i).name);
        allLabels{end+1} = cls;
    end
end
totalImages = numel(allFiles);
fprintf('Found %d total hyperspectral cubes across %d classes (50 per class).\n', ...
    totalImages, numClasses);

% Inspect first image dimensions using the readfile imhypercube approach
sampleCube = readHypercubeFile(allFiles{1});
[sRows, sCols, sBands] = size(sampleCube);
fprintf('Sample Image (%s):\n  Rows: %d | Cols: %d | Bands: %d\n', ...
    allFiles{1}, sRows, sCols, sBands);

%% TASK 2: Partition Data and Select 10 Common Bands using reduceHypercubePCA
fprintf('\n>>> TASK 2: Stratified Partitioning and PCA Spectral Band Selection...\n');

% Stratified 70% Train (140), 15% Validation (30), 15% Test (30)
trainFiles = {}; trainLabels = {};
valFiles   = {}; valLabels   = {};
testFiles  = {}; testLabels  = {};

for c = 1:numClasses
    cls = classes{c};
    clsIdx = find(strcmp(allLabels, cls));
    perm = randperm(numel(clsIdx));
    
    nTrain = 35; % 70% of 50
    nVal   = 8 - mod(c, 2); % 8 or 7
    nTest  = 50 - nTrain - nVal;
    
    trainIdx = clsIdx(perm(1:nTrain));
    valIdx   = clsIdx(perm(nTrain+1 : nTrain+nVal));
    testIdx  = clsIdx(perm(nTrain+nVal+1 : end));
    
    trainFiles  = [trainFiles, allFiles(trainIdx)];
    trainLabels = [trainLabels, allLabels(trainIdx)];
    valFiles    = [valFiles, allFiles(valIdx)];
    valLabels   = [valLabels, allLabels(valIdx)];
    testFiles   = [testFiles, allFiles(testIdx)];
    testLabels  = [testLabels, allLabels(testIdx)];
end

fprintf('Data Partitioning Summary:\n');
fprintf('  Training Set   : %d images (70.0%%)\n', numel(trainFiles));
fprintf('  Validation Set : %d images (15.0%%)\n', numel(valFiles));
fprintf('  Held-Out Test  : %d images (15.0%%)\n', numel(testFiles));

% Load or compute common 10 PCA bands using reduceHypercubePCA on training data
if isfile('Model/preprocessing_params.mat')
    prep = load('Model/preprocessing_params.mat');
    selectedBands = prep.selected_bands_1idx;
    targetH = prep.target_height;
    targetW = prep.target_width;
    normScale = prep.normalization_scale_factor;
    fprintf('Loaded common 10 PCA-selected bands from Model/preprocessing_params.mat: %s\n', ...
        mat2str(selectedBands));
else
    % Gather representative training spectra and fit reduceHypercubePCA
    fprintf('Extracting representative training spectra for PCA band selection...\n');
    spectraList = cell(1, numel(trainFiles));
    for i = 1:numel(trainFiles)
        c = readHypercubeFile(trainFiles{i});
        spectraList{i} = reshape(double(c), [], 168);
    end
    X_train = cat(1, spectraList{:});
    
    % Pass training data as 3D array into reduceHypercubePCA helper function
    trainCube3D = reshape(X_train, [], 1, 168);
    [~, selectedBands, ~, ~] = reduceHypercubePCA(trainCube3D, 10, 95.0);
    targetH = 80; targetW = 80; normScale = 16383.0;
end

%% TASK 3: Spatial Standardization Using Padding Only
fprintf('\n>>> TASK 3: Spatial Standardization Using Balanced Zero Padding Only...\n');
fprintf('Target Spatial Dimensions: %d x %d (Max in dataset: 68 x 73)\n', targetH, targetW);
fprintf('Padding Method: Balanced Zero-Padding (No cropping, resizing, or interpolation).\n');

%% TASK 4: Construct MATLAB Datastores
fprintf('\n>>> TASK 4: Constructing Datastores...\n');
trainDS = create_hyperspectral_datastore(trainFiles, trainLabels, selectedBands, targetH, targetW, normScale);
valDS   = create_hyperspectral_datastore(valFiles, valLabels, selectedBands, targetH, targetW, normScale);
testDS  = create_hyperspectral_datastore(testFiles, testLabels, selectedBands, targetH, targetW, normScale);

%% TASK 5: CNN Architecture Design & Model Training (MATLAB Deep Learning Toolbox)
fprintf('\n>>> TASK 5: CNN Architecture Design & Training (MATLAB Deep Learning Toolbox)...\n');
inputSize = [targetH, targetW, 10];
layers = build_cnn_architecture(inputSize, numClasses, 'baseline', 0.3);

% Configure training options for MATLAB Deep Learning Toolbox
trainOpts = struct();
trainOpts.maxEpochs = 45;
trainOpts.initialLearnRate = 0.001;
trainOpts.miniBatchSize = 16;
trainOpts.l2Regularization = 0.0001;
trainOpts.plots = 'none';
trainOpts.verbose = false;

% Execute training via train_cnn_model using Deep Learning Toolbox
[net, trainInfo] = train_cnn_model(layers, trainDS, valDS, trainOpts);

%% TASK 6: Controlled CNN Design Comparisons
fprintf('\n>>> TASK 6: Summary of Controlled CNN Design Comparisons...\n');
if isfile('Results/experimental_results.csv')
    resTable = readtable('Results/experimental_results.csv');
    disp(resTable(:, {'Configuration', 'TestOverallAcc', 'MacroF1_Score', 'FinalValAcc'}));
end

%% TASK 7: Evaluation on Held-Out Test Set
fprintf('\n>>> TASK 7: Evaluating Final CNN on Held-Out Test Images...\n');
testPredictions = cell(1, numel(testFiles));
testConfidences = zeros(1, numel(testFiles));

for i = 1:numel(testFiles)
    cube = readHypercubeFile(testFiles{i});
    cube10 = double(cube(:, :, selectedBands));
    padded = pad_hyperspectral_cube(cube10, targetH, targetW, 0.0);
    tensor = single(padded / normScale);
    
    [predCls, conf, ~] = predict_cnn_forward(net, tensor);
    testPredictions{i} = predCls;
    testConfidences(i) = conf;
end

metrics = evaluate_metrics(testLabels, testPredictions, classes);

%% TASK 8: Demonstration on Unseen Hyperspectral Image
fprintf('>>> TASK 8: Unseen Image Classification Demonstration...\n');
unseenSample = fullfile('Dataset', 'HDPE', 'scan0010_frag15.bil');
classify_unseen_image(unseenSample);

fprintf('====================================================================\n');
fprintf('   FULL PIPELINE EXECUTION COMPLETED SUCCESSFULLY!                 \n');
fprintf('====================================================================\n');

%% Local Helper Function: Read Hyperspectral Cube (readfile.m approach)
function cube = readHypercubeFile(bilPath)
    if exist('imhypercube', 'file') == 2
        hcube = imhypercube(bilPath);
        cube = gather(hcube);
    else
        [p, f] = fileparts(bilPath);
        hdrPath = fullfile(p, [f, '.hdr']);
        fid = fopen(hdrPath, 'r'); txt = fread(fid, '*char')'; fclose(fid);
        w = str2double(regexp(txt, 'samples\s*=\s*(\d+)', 'tokens', 'once'));
        h = str2double(regexp(txt, 'lines\s*=\s*(\d+)', 'tokens', 'once'));
        b = str2double(regexp(txt, 'bands\s*=\s*(\d+)', 'tokens', 'once'));
        fid = fopen(bilPath, 'r', 'ieee-le');
        raw = fread(fid, [w * b, h], '*uint16');
        fclose(fid);
        cube = permute(reshape(raw, [w, b, h]), [3, 1, 2]);
    end
end
