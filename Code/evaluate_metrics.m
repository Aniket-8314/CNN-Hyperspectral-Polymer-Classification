function metrics = evaluate_metrics(trueLabels, predLabels, classNames)
%EVALUATE_METRICS Computes multi-class classification metrics and confusion matrix.
%
%   metrics = evaluate_metrics(trueLabels, predLabels, classNames)
%
% INPUTS:
%   trueLabels - Vector or cell array of true ground-truth class labels.
%   predLabels - Vector or cell array of predicted class labels.
%   classNames - Cell array of unique class names (default {'HDPE','PET','PP','PS'}).
%
% OUTPUT:
%   metrics    - Struct containing:
%                .confusionMatrix  (4 x 4 matrix where rows=True, cols=Predicted)
%                .overallAccuracy  (scalar, total correct / total samples)
%                .precision        (1 x K vector of class-wise precisions)
%                .recall           (1 x K vector of class-wise recalls)
%                .f1Score          (1 x K vector of class-wise F1 scores)
%                .support          (1 x K vector of sample counts per class)
%                .macroPrecision   (unweighted mean precision)
%                .macroRecall      (unweighted mean recall / average class accuracy)
%                .macroF1          (unweighted mean F1-score)
%                .weightedF1       (support-weighted mean F1-score)
%                .metricsTable     (Formatted summary table)
%
% Note on undefined denominators:
% When TP + FP == 0 (no positive predictions), precision is defined as 0.0.
% When TP + FN == 0 (no positive true instances), recall is defined as 0.0.
% When Precision + Recall == 0, F1 is defined as 0.0.
%
% ME3101 - Mini-Project 2: Hyperspectral Polymer Classification

    if nargin < 3 || isempty(classNames)
        classNames = {'HDPE', 'PET', 'PP', 'PS'};
    end
    K = numel(classNames);

    % Convert to cellstr if categorical
    if iscategorical(trueLabels), trueLabels = cellstr(trueLabels); end
    if iscategorical(predLabels), predLabels = cellstr(predLabels); end

    % If numeric, convert to 1-based indexing
    if isnumeric(trueLabels)
        if min(trueLabels) == 0, trueLabels = trueLabels + 1; end
        trueLabels = classNames(trueLabels);
    end
    if isnumeric(predLabels)
        if min(predLabels) == 0, predLabels = predLabels + 1; end
        predLabels = classNames(predLabels);
    end

    % Construct 4 x 4 confusion matrix
    cm = zeros(K, K);
    for i = 1:numel(trueLabels)
        t_idx = find(strcmp(classNames, trueLabels{i}), 1);
        p_idx = find(strcmp(classNames, predLabels{i}), 1);
        if ~isempty(t_idx) && ~isempty(p_idx)
            cm(t_idx, p_idx) = cm(t_idx, p_idx) + 1;
        end
    end

    totalSamples = sum(cm(:));
    overallAccuracy = sum(diag(cm)) / totalSamples;

    precision = zeros(1, K);
    recall    = zeros(1, K);
    f1Score   = zeros(1, K);
    support   = zeros(1, K);

    for k = 1:K
        TP = cm(k, k);
        FP = sum(cm(:, k)) - TP;
        FN = sum(cm(k, :)) - TP;
        support(k) = sum(cm(k, :));

        % Precision = TP / (TP + FP)
        if (TP + FP) > 0
            precision(k) = TP / (TP + FP);
        else
            precision(k) = 0.0; % Handled undefined denominator
        end

        % Recall = TP / (TP + FN)
        if (TP + FN) > 0
            recall(k) = TP / (TP + FN);
        else
            recall(k) = 0.0; % Handled undefined denominator
        end

        % F1 = 2 * (P * R) / (P + R)
        if (precision(k) + recall(k)) > 0
            f1Score(k) = 2 * (precision(k) * recall(k)) / (precision(k) + recall(k));
        else
            f1Score(k) = 0.0;
        end
    end

    macroPrecision = mean(precision);
    macroRecall    = mean(recall); % Average class-wise accuracy
    macroF1        = mean(f1Score);
    weightedF1     = sum(f1Score .* support) / sum(support);

    % Build structured results table
    Class = classNames(:);
    Precision = precision(:);
    Recall = recall(:);
    F1 = f1Score(:);
    Support = support(:);
    metricsTable = table(Class, Precision, Recall, F1, Support);

    metrics.confusionMatrix = cm;
    metrics.classNames      = classNames;
    metrics.overallAccuracy = overallAccuracy;
    metrics.precision       = precision;
    metrics.recall          = recall;
    metrics.f1Score         = f1Score;
    metrics.support         = support;
    metrics.macroPrecision  = macroPrecision;
    metrics.macroRecall     = macroRecall;
    metrics.macroF1         = macroF1;
    metrics.weightedF1      = weightedF1;
    metrics.metricsTable    = metricsTable;

    % Display formatted output
    fprintf('\n================== EVALUATION METRICS REPORT ==================\n');
    fprintf('Overall Accuracy: %.2f%% (%d / %d)\n', overallAccuracy * 100, sum(diag(cm)), totalSamples);
    fprintf('Macro Precision : %.4f\n', macroPrecision);
    fprintf('Macro Recall (Mean Class Accuracy): %.4f\n', macroRecall);
    fprintf('Macro F1-Score  : %.4f\n', macroF1);
    fprintf('Weighted F1-Score: %.4f\n', weightedF1);
    fprintf('\nClass-Wise Metrics:\n');
    disp(metricsTable);

    fprintf('Confusion Matrix (Rows = True Class, Columns = Predicted Class):\n');
    fprintf('%8s', 'True\Pred');
    for k = 1:K, fprintf('%8s', classNames{k}); end
    fprintf('\n');
    for r = 1:K
        fprintf('%8s', classNames{r});
        for c = 1:K
            fprintf('%8d', cm(r, c));
        end
        fprintf('\n');
    end
    fprintf('===============================================================\n\n');
end
