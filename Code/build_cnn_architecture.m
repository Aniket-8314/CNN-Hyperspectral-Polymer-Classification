function layers = build_cnn_architecture(inputSize, numClasses, archType, dropoutProb)
%BUILD_CNN_ARCHITECTURE Defines 2D CNN architectures for hyperspectral polymer classification.
%
%   layers = build_cnn_architecture(inputSize, numClasses, archType, dropoutProb)
%
% INPUTS:
%   inputSize   - 1 x 3 vector [HT, WT, C], e.g., [80, 80, 10].
%   numClasses  - Number of target polymer classes (default 4).
%   archType    - 'baseline' (3 conv blocks), 'shallow' (2 blocks), or 'deep' (4 blocks).
%   dropoutProb - Dropout probability (default 0.3).
%
% OUTPUT:
%   layers      - Layer array compatible with MATLAB Deep Learning Toolbox.
%
% ME3101 - Mini-Project 2: Hyperspectral Polymer Classification

    if nargin < 1 || isempty(inputSize), inputSize = [80, 80, 10]; end
    if nargin < 2 || isempty(numClasses), numClasses = 4; end
    if nargin < 3 || isempty(archType), archType = 'baseline'; end
    if nargin < 4 || isempty(dropoutProb), dropoutProb = 0.3; end

    % If MATLAB Deep Learning Toolbox is installed, construct layer array
    if exist('imageInputLayer', 'file') == 2
        switch lower(archType)
        case 'shallow'
            % 2 Convolutional Blocks
            layers = [
                imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
                
                convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv1')
                batchNormalizationLayer('Name', 'bn1')
                reluLayer('Name', 'relu1')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1') % 40x40
                
                convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2')
                batchNormalizationLayer('Name', 'bn2')
                reluLayer('Name', 'relu2')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2') % 20x20
                
                globalAveragePooling2dLayer('Name', 'gap')
                dropoutLayer(dropoutProb, 'Name', 'drop1')
                fullyConnectedLayer(32, 'Name', 'fc1')
                reluLayer('Name', 'relu_fc')
                dropoutLayer(dropoutProb, 'Name', 'drop2')
                fullyConnectedLayer(numClasses, 'Name', 'fc_out')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'classoutput')
            ];

        case 'deep'
            % 4 Convolutional Blocks
            layers = [
                imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
                
                convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
                batchNormalizationLayer('Name', 'bn1')
                reluLayer('Name', 'relu1')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1') % 40x40
                
                convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
                batchNormalizationLayer('Name', 'bn2')
                reluLayer('Name', 'relu2')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2') % 20x20
                
                convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
                batchNormalizationLayer('Name', 'bn3')
                reluLayer('Name', 'relu3')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3') % 10x10
                
                convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4')
                batchNormalizationLayer('Name', 'bn4')
                reluLayer('Name', 'relu4')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool4') % 5x5
                
                globalAveragePooling2dLayer('Name', 'gap')
                dropoutLayer(dropoutProb, 'Name', 'drop1')
                fullyConnectedLayer(128, 'Name', 'fc1')
                reluLayer('Name', 'relu_fc')
                dropoutLayer(dropoutProb, 'Name', 'drop2')
                fullyConnectedLayer(numClasses, 'Name', 'fc_out')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'classoutput')
            ];

        otherwise % 'baseline' - 3 Convolutional Blocks
            layers = [
                imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
                
                convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
                batchNormalizationLayer('Name', 'bn1')
                reluLayer('Name', 'relu1')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1') % 40x40
                
                convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
                batchNormalizationLayer('Name', 'bn2')
                reluLayer('Name', 'relu2')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2') % 20x20
                
                convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
                batchNormalizationLayer('Name', 'bn3')
                reluLayer('Name', 'relu3')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3') % 10x10
                
                globalAveragePooling2dLayer('Name', 'gap')
                dropoutLayer(dropoutProb, 'Name', 'drop1')
                fullyConnectedLayer(64, 'Name', 'fc1')
                reluLayer('Name', 'relu_fc')
                dropoutLayer(dropoutProb, 'Name', 'drop2')
                fullyConnectedLayer(numClasses, 'Name', 'fc_out')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'classoutput')
            ];
        end
    else
        fprintf('Deep Learning Toolbox not found. Returning structured CNN architecture specification.\n');
        layers = struct(...
            'architecture', archType, ...
            'inputSize', inputSize, ...
            'numClasses', numClasses, ...
            'dropout', dropoutProb, ...
            'description', '2D CNN with Conv2D, BatchNorm, ReLU, MaxPool, GAP, FC, and Softmax');
    end
end
