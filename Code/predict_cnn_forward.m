function [predictedClass, confidence, probabilities] = predict_cnn_forward(modelData, inputTensor)
%PREDICT_CNN_FORWARD Performs forward-pass classification using the trained CNN.
%
%   [predictedClass, confidence, probabilities] = predict_cnn_forward(modelData, inputTensor)
%
% INPUTS:
%   modelData   - Trained network struct (loaded from Model/trained_cnn.mat) or DAGNetwork.
%   inputTensor - HT x WT x 10 single/double numeric array (e.g., 80 x 80 x 10).
%
% OUTPUTS:
%   predictedClass - String of the predicted polymer ('HDPE', 'PET', 'PP', or 'PS').
%   confidence     - Scalar confidence score in [0, 1].
%   probabilities  - 1 x 4 vector of softmax probabilities for [HDPE, PET, PP, PS].
%
% ME3101 - Mini-Project 2: Hyperspectral Polymer Classification

    if ischar(modelData) || isstring(modelData)
        modelData = load(modelData);
    end

    if isstruct(modelData) && isfield(modelData, 'net')
        modelData = modelData.net;
    end

    classNames = {'HDPE', 'PET', 'PP', 'PS'};
    if isstruct(modelData) && isfield(modelData, 'class_names')
        classNames = cellstr(modelData.class_names);
    end

    % Verify input shape
    [H, W, C] = size(inputTensor);
    if C ~= 10
        error('Input tensor must have exactly 10 spectral channels. Found: %d', C);
    end

    % If modelData is a MATLAB SeriesNetwork/DAGNetwork object:
    if isa(modelData, 'SeriesNetwork') || isa(modelData, 'DAGNetwork') || isa(modelData, 'dlnetwork')
        probs = predict(modelData, inputTensor);
        probabilities = double(probs(:))';
        [confidence, maxIdx] = max(probabilities);
        predictedClass = classNames{maxIdx};
        return;
    end

    % Pure MATLAB Standalone Forward Pass (Zero Toolbox Dependencies)
    % Extract trained weights and biases
    % Features Block 1
    w_conv1 = double(modelData.features_0_weight); % (3, 3, 10, 16)
    b_conv1 = double(modelData.features_0_bias(:));
    bn1_w   = double(modelData.features_1_weight(:));
    bn1_b   = double(modelData.features_1_bias(:));
    bn1_m   = double(modelData.features_1_running_mean(:));
    bn1_v   = double(modelData.features_1_running_var(:));

    % Features Block 2
    w_conv2 = double(modelData.features_4_weight); % (3, 3, 16, 32)
    b_conv2 = double(modelData.features_4_bias(:));
    bn2_w   = double(modelData.features_5_weight(:));
    bn2_b   = double(modelData.features_5_bias(:));
    bn2_m   = double(modelData.features_5_running_mean(:));
    bn2_v   = double(modelData.features_5_running_var(:));

    % Features Block 3
    w_conv3 = double(modelData.features_8_weight); % (3, 3, 32, 64)
    b_conv3 = double(modelData.features_8_bias(:));
    bn3_w   = double(modelData.features_9_weight(:));
    bn3_b   = double(modelData.features_9_bias(:));
    bn3_m   = double(modelData.features_9_running_mean(:));
    bn3_v   = double(modelData.features_9_running_var(:));

    % Classifier
    w_fc1   = double(modelData.classifier_2_weight); % (64, 64)
    b_fc1   = double(modelData.classifier_2_bias(:));
    w_fc2   = double(modelData.classifier_5_weight); % (4, 64)
    b_fc2   = double(modelData.classifier_5_bias(:));

    % --- Forward Execution ---
    x = double(inputTensor);

    % Block 1: Conv -> BN -> ReLU -> MaxPool (stride 2)
    x = conv2d_forward(x, w_conv1, b_conv1, 1);
    x = batchnorm_forward(x, bn1_w, bn1_b, bn1_m, bn1_v);
    x = max(0, x); % ReLU
    x = maxpool2d_forward(x, 2, 2); % 80x80 -> 40x40

    % Block 2: Conv -> BN -> ReLU -> MaxPool (stride 2)
    x = conv2d_forward(x, w_conv2, b_conv2, 1);
    x = batchnorm_forward(x, bn2_w, bn2_b, bn2_m, bn2_v);
    x = max(0, x); % ReLU
    x = maxpool2d_forward(x, 2, 2); % 40x40 -> 20x20

    % Block 3: Conv -> BN -> ReLU -> MaxPool (stride 2)
    x = conv2d_forward(x, w_conv3, b_conv3, 1);
    x = batchnorm_forward(x, bn3_w, bn3_b, bn3_m, bn3_v);
    x = max(0, x); % ReLU
    x = maxpool2d_forward(x, 2, 2); % 20x20 -> 10x10

    % Global Average Pooling across spatial dimensions: 10x10x64 -> 64x1
    gap = squeeze(mean(mean(x, 1), 2)); % 64x1

    % Classifier Dense Layer 1
    fc1_out = max(0, w_fc1 * gap + b_fc1); % 64x1

    % Classifier Dense Layer 2 (Logits)
    logits = w_fc2 * fc1_out + b_fc2; % 4x1

    % Softmax
    exp_logits = exp(logits - max(logits));
    probabilities = (exp_logits / sum(exp_logits))';

    [confidence, maxIdx] = max(probabilities);
    predictedClass = classNames{maxIdx};
end

% --- Helper Layer Implementations ---
function y = conv2d_forward(x, w, b, pad)
    [H, W, Cin] = size(x);
    [kH, kW, ~, Cout] = size(w);
    
    % Apply padding natively without padarray toolbox dependency
    x_pad = zeros(H + 2*pad, W + 2*pad, Cin);
    x_pad(pad+1:pad+H, pad+1:pad+W, :) = x;
    [Hpad, Wpad, ~] = size(x_pad);
    
    Hout = Hpad - kH + 1;
    Wout = Wpad - kW + 1;
    y = zeros(Hout, Wout, Cout);
    
    for c_out = 1:Cout
        val = b(c_out);
        for c_in = 1:Cin
            kernel = rot90(w(:, :, c_in, c_out), 2); % 2D convolution via filter2
            val = val + filter2(kernel, x_pad(:, :, c_in), 'valid');
        end
        y(:, :, c_out) = val;
    end
end

function y = batchnorm_forward(x, gamma, beta, mean_val, var_val, eps)
    if nargin < 6, eps = 1e-5; end
    [H, W, C] = size(x);
    y = zeros(H, W, C);
    for c = 1:C
        scale = gamma(c) / sqrt(var_val(c) + eps);
        shift = beta(c) - scale * mean_val(c);
        y(:, :, c) = x(:, :, c) * scale + shift;
    end
end

function y = maxpool2d_forward(x, poolSize, stride)
    [H, W, C] = size(x);
    Hout = floor((H - poolSize) / stride) + 1;
    Wout = floor((W - poolSize) / stride) + 1;
    y = zeros(Hout, Wout, C);
    for r = 1:Hout
        rStart = (r - 1) * stride + 1;
        rEnd   = rStart + poolSize - 1;
        for c = 1:Wout
            cStart = (c - 1) * stride + 1;
            cEnd   = cStart + poolSize - 1;
            patch = x(rStart:rEnd, cStart:cEnd, :);
            y(r, c, :) = max(max(patch, [], 1), [], 2);
        end
    end
end
