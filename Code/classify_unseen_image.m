function [predictedClass, confidence, probabilities] = classify_unseen_image(bilPath, hdrPath, modelPath, paramsPath, savePlotPath)
%CLASSIFY_UNSEEN_IMAGE Predicts the polymer class of an unseen hyperspectral image.
%
%   [predictedClass, confidence, probabilities] = ...
%       classify_unseen_image(bilPath, hdrPath, modelPath, paramsPath, savePlotPath)
%
%   Uses the imhypercube/enviinfo data ingestion and mat2gray pseudo-RGB visualization
%   demonstrated in readfile.m, applying the common PCA bands from reduceHypercubePCA.
%
% INPUTS:
%   bilPath      - (Optional) Path to unseen .bil file. Defaults to sample image.
%   hdrPath      - (Optional) Path to .hdr file. If omitted, uses same base name as bilPath.
%   modelPath    - (Optional) Path to trained model ('Model/trained_cnn.mat').
%   paramsPath   - (Optional) Path to preprocessing settings ('Model/preprocessing_params.mat').
%   savePlotPath - (Optional) Path to save the output demonstration figure.
%
% OUTPUTS:
%   predictedClass - Predicted polymer class ('HDPE', 'PET', 'PP', 'PS').
%   confidence     - Scalar probability score for top predicted class.
%   probabilities  - 1 x 4 vector of probabilities for [HDPE, PET, PP, PS].
%
% ME3101 - Mini-Project 2: Hyperspectral Polymer Classification (Task 8)

    % Default file paths if not provided
    if nargin < 1 || isempty(bilPath)
        bilPath = fullfile('Dataset', 'PP', 'scan0001_frag16.bil');
    end
    if nargin < 2 || isempty(hdrPath)
        [p, f, ~] = fileparts(bilPath);
        hdrPath = fullfile(p, [f, '.hdr']);
    end
    if nargin < 3 || isempty(modelPath)
        modelPath = fullfile('Model', 'trained_cnn.mat');
    end
    if nargin < 4 || isempty(paramsPath)
        paramsPath = fullfile('Model', 'preprocessing_params.mat');
    end
    if nargin < 5 || isempty(savePlotPath)
        savePlotPath = fullfile('Results', 'Plots', 'unseen_image_prediction_demonstration.png');
    end

    fprintf('================ TASK 8: UNSEEN IMAGE INFERENCE ================\n');
    fprintf('Input BIL file : %s\n', bilPath);
    fprintf('Input HDR file : %s\n', hdrPath);
    fprintf('Model path     : %s\n', modelPath);
    fprintf('Params path    : %s\n', paramsPath);

    if ~isfile(bilPath)
        error('Hyperspectral BIL file does not exist: %s', bilPath);
    end
    if ~isfile(hdrPath)
        error('Hyperspectral HDR file does not exist: %s', hdrPath);
    end

    % 1. Load Preprocessing Settings (Fixed parameters; NO refitting!)
    if ~isfile(paramsPath)
        error('Preprocessing parameters file not found: %s', paramsPath);
    end
    params = load(paramsPath);
    selectedBands = double(params.selected_bands_1idx(:)');
    targetH = double(params.target_height);
    targetW = double(params.target_width);
    normScale = double(params.normalization_scale_factor);
    classNames = cellstr(params.class_labels);

    % 2. Read BIL/HDR hyperspectral cube using readfile.m approach (enviinfo & imhypercube)
    if exist('enviinfo', 'file') == 2 && exist('imhypercube', 'file') == 2
        hdr = enviinfo(hdrPath);
        hcube = imhypercube(bilPath);
        cube = gather(hcube);
    else
        % Fallback reader for base MATLAB environments lacking Hyperspectral Library
        fid = fopen(hdrPath, 'r'); txt = fread(fid, '*char')'; fclose(fid);
        w = str2double(regexp(txt, 'samples\s*=\s*(\d+)', 'tokens', 'once'));
        h = str2double(regexp(txt, 'lines\s*=\s*(\d+)', 'tokens', 'once'));
        b = str2double(regexp(txt, 'bands\s*=\s*(\d+)', 'tokens', 'once'));
        fid = fopen(bilPath, 'r', 'ieee-le');
        raw = fread(fid, [w * b, h], '*uint16');
        fclose(fid);
        cube = permute(reshape(raw, [w, b, h]), [3, 1, 2]);
    end

    [H, W, B] = size(cube);
    fprintf('Loaded cube dimensions: %d rows x %d cols x %d spectral bands\n', H, W, B);

    % Verify bounds against target dimensions
    if H > targetH || W > targetW
        error('Image dimensions [%d x %d] exceed target dimensions [%d x %d]!', H, W, targetH, targetW);
    end

    % 3. Extract the 10 common PCA-selected bands
    cube10 = double(cube(:, :, selectedBands));

    % 4. Apply symmetric zero-padding to target size (80 x 80)
    paddedCube = pad_hyperspectral_cube(cube10, targetH, targetW, 0.0);

    % 5. Apply normalization scaling
    inputTensor = single(paddedCube / normScale);

    % 6. Forward pass inference
    [predictedClass, confidence, probabilities] = predict_cnn_forward(modelPath, inputTensor);

    fprintf('\n----------------- INFERENCE RESULTS -----------------\n');
    fprintf('PREDICTED CLASS : %s\n', predictedClass);
    fprintf('CONFIDENCE SCORE: %.2f%%\n', confidence * 100);
    fprintf('CLASS PROBABILITIES:\n');
    for k = 1:numel(classNames)
        fprintf('  %-6s: %6.2f%%\n', classNames{k}, probabilities(k) * 100);
    end
    fprintf('-----------------------------------------------------\n');

    % 7. Generate Visualization Figure (following readfile.m pseudo-RGB convention)
    fig = figure('Name', 'Unseen Image Classification Demonstration', ...
                 'Position', [100, 100, 950, 420], 'Visible', 'off');

    % Subplot 1: Pseudo-RGB image of the fragment using mat2gray as in readfile.m
    subplot(1, 2, 1);
    rCh = double(cube(:, :, selectedBands(3)));
    gCh = double(cube(:, :, selectedBands(6)));
    bCh = double(cube(:, :, selectedBands(10)));

    if exist('mat2gray', 'file') == 2
        rCh = mat2gray(rCh);
        gCh = mat2gray(gCh);
        bCh = mat2gray(bCh);
    else
        normCh = @(ch) (ch - min(ch(:))) / max(1e-6, max(ch(:)) - min(ch(:)));
        rCh = normCh(rCh); gCh = normCh(gCh); bCh = normCh(bCh);
    end
    rgbImg = cat(3, rCh, gCh, bCh);
    imshow(rgbImg);
    [~, f] = fileparts(bilPath);
    title(sprintf('Input Scan: %s\nRaw Dimensions: %d x %d x %d', ...
        f, H, W, B), 'FontWeight', 'bold', 'FontSize', 10);

    % Subplot 2: Horizontal Probability Bar Chart
    subplot(1, 2, 2);
    barH = barh(1:numel(classNames), probabilities * 100, 0.55);
    barH.FaceColor = 'flat';
    for k = 1:numel(classNames)
        if strcmp(classNames{k}, predictedClass)
            barH.CData(k, :) = [0.18, 0.80, 0.44]; % Green for predicted
        else
            barH.CData(k, :) = [0.20, 0.60, 0.86]; % Blue for others
        end
    end
    set(gca, 'YTick', 1:numel(classNames), 'YTickLabel', classNames, ...
             'XLim', [0, 105], 'FontSize', 10, 'FontWeight', 'bold');
    xlabel('Prediction Probability (%)', 'FontSize', 10);
    title(sprintf('Predicted Polymer: %s (%.1f%%)', predictedClass, confidence * 100), ...
          'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.1, 0.6, 0.2]);

    % Add percentage labels at the end of each bar
    for k = 1:numel(classNames)
        text(probabilities(k) * 100 + 2, k, sprintf('%.1f%%', probabilities(k) * 100), ...
            'VerticalAlignment', 'middle', 'FontWeight', 'bold', 'FontSize', 9);
    end

    % Save plot
    if ~isempty(savePlotPath)
        saveas(fig, savePlotPath);
        fprintf('Saved visualization plot to: %s\n', savePlotPath);
    end
    close(fig);
    fprintf('=================================================================\n\n');
end
