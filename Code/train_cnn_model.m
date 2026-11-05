function [net, trainInfo] = train_cnn_model(layers, X_train, Y_train, X_val, Y_val, opts)
%TRAIN_CNN_MODEL Trains a CNN model using the MATLAB Deep Learning Toolbox.
%
%   [net, trainInfo] = train_cnn_model(layers, X_train, Y_train, X_val, Y_val, opts)
%   [net, trainInfo] = train_cnn_model(layers, trainDS, valDS, opts)
%
% INPUTS:
%   layers    - Layer array created by build_cnn_architecture.
%   X_train   - 4D single array (H x W x C x N_train) OR custom datastore struct.
%   Y_train   - N_train x 1 categorical array of class labels (or valDS).
%   X_val     - 4D single array (H x W x C x N_val) for validation (or opts).
%   Y_val     - N_val x 1 categorical array of validation labels.
%   opts      - Struct of training options:
%                 .maxEpochs        (default: 45)
%                 .initialLearnRate (default: 0.001)
%                 .miniBatchSize    (default: 16)
%                 .l2Regularization (default: 0.0001)
%                 .plots            (default: 'none', or 'training-progress')
%                 .verbose          (default: true)
%
% OUTPUTS:
%   net       - Trained SeriesNetwork / DAGNetwork (or pre-trained fallback struct).
%   trainInfo - Training progress information struct.
%
% ME3101 - Mini-Project 2: Hyperspectral Polymer Classification

    % Flexible input handling for either datastore structs or 4D numeric arrays
    if isstruct(X_train) && isfield(X_train, 'read')
        trainDS = X_train;
        valDS = Y_train;
        if nargin >= 4 && isstruct(X_val)
            opts = X_val;
        else
            opts = struct();
        end

        fprintf('Extracting tensors from hyperspectral datastores for Deep Learning Toolbox...\n');
        N_tr = trainDS.numObservations;
        X_tr = zeros(trainDS.targetSize(1), trainDS.targetSize(2), trainDS.targetSize(3), N_tr, 'single');
        for i = 1:N_tr
            X_tr(:, :, :, i) = trainDS.read(i);
        end
        Y_tr = trainDS.labels;

        N_va = valDS.numObservations;
        X_va = zeros(valDS.targetSize(1), valDS.targetSize(2), valDS.targetSize(3), N_va, 'single');
        for i = 1:N_va
            X_va(:, :, :, i) = valDS.read(i);
        end
        Y_va = valDS.labels;

        X_train = X_tr;
        Y_train = Y_tr;
        X_val   = X_va;
        Y_val   = Y_va;
    else
        if nargin < 6 || isempty(opts)
            opts = struct();
        end
    end

    if ~isfield(opts, 'maxEpochs'), opts.maxEpochs = 45; end
    if ~isfield(opts, 'initialLearnRate'), opts.initialLearnRate = 0.001; end
    if ~isfield(opts, 'miniBatchSize'), opts.miniBatchSize = 16; end
    if ~isfield(opts, 'l2Regularization'), opts.l2Regularization = 0.0001; end
    if ~isfield(opts, 'plots'), opts.plots = 'none'; end
    if ~isfield(opts, 'verbose'), opts.verbose = true; end

    fprintf('====================================================================\n');
    fprintf('   CNN MODEL TRAINING (MATLAB DEEP LEARNING TOOLBOX)                \n');
    fprintf('====================================================================\n');
    fprintf('Architecture  : %d Convolutional Blocks with Batch Normalization\n', 3);
    fprintf('Optimizer     : Adam\n');
    fprintf('Max Epochs    : %d\n', opts.maxEpochs);
    fprintf('MiniBatchSize : %d\n', opts.miniBatchSize);
    fprintf('Initial LR    : %g\n', opts.initialLearnRate);
    fprintf('L2 Weight Dec.: %g\n', opts.l2Regularization);
    fprintf('Training Data : %d samples (%d x %d x %d)\n', size(X_train, 4), size(X_train, 1), size(X_train, 2), size(X_train, 3));
    fprintf('Val Data      : %d samples\n', size(X_val, 4));
    fprintf('--------------------------------------------------------------------\n');

    modelPath = fullfile('Model', 'trained_cnn.mat');

    % Check for MATLAB Deep Learning Toolbox
    if exist('trainNetwork', 'file') == 2
        fprintf('MATLAB Deep Learning Toolbox detected.\n');
        valFreq = max(1, floor(size(X_train, 4) / opts.miniBatchSize));

        options = trainingOptions('adam', ...
            'MaxEpochs', opts.maxEpochs, ...
            'MiniBatchSize', opts.miniBatchSize, ...
            'InitialLearnRate', opts.initialLearnRate, ...
            'L2Regularization', opts.l2Regularization, ...
            'ValidationData', {X_val, Y_val}, ...
            'ValidationFrequency', valFreq, ...
            'Shuffle', 'every-epoch', ...
            'Plots', opts.plots, ...
            'Verbose', opts.verbose);

        fprintf('Executing trainNetwork on MATLAB Deep Learning Toolbox...\n');
        [net, trainInfo] = trainNetwork(X_train, Y_train, layers, options);

        % Save trained model for offline/unseen deployment
        if ~isfolder('Model')
            mkdir('Model');
        end
        save(modelPath, 'net', 'trainInfo');
        fprintf('Training complete! Trained network saved to %s\n', modelPath);
    else
        fprintf('Deep Learning Toolbox is not installed in this MATLAB environment.\n');
        if isfile(modelPath)
            fprintf('Loading pre-trained CNN model from %s for execution...\n', modelPath);
            loadedData = load(modelPath);
            if isfield(loadedData, 'net')
                net = loadedData.net;
            else
                net = loadedData;
            end
            trainInfo = struct('status', 'Pre-trained weights loaded successfully');
            fprintf('Pre-trained model loaded successfully.\n');
        else
            error(['Deep Learning Toolbox is required to train from scratch in MATLAB. ' ...
                   'Please run on MATLAB Online (which includes the Deep Learning Toolbox) ' ...
                   'or provide the trained model at %s.'], modelPath);
        end
    end
    fprintf('====================================================================\n\n');
end
