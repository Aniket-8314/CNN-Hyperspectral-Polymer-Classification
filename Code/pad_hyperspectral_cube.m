function [paddedCube, padDetails] = pad_hyperspectral_cube(cube, targetH, targetW, padValue)
%PAD_HYPERSPECTRAL_CUBE Pads a hyperspectral cube to target dimensions without cropping.
%
%   [paddedCube, padDetails] = pad_hyperspectral_cube(cube, targetH, targetW, padValue)
%
% INPUTS:
%   cube        - H x W x C numeric array (e.g., C = 10 spectral channels).
%   targetH     - Target height (HT, default 80).
%   targetW     - Target width (WT, default 80).
%   padValue    - Scalar padding fill value (default 0). 
%
% OUTPUTS:
%   paddedCube  - targetH x targetW x C padded numeric array.
%   padDetails  - Struct with padding parameters [padTop, padBottom, padLeft, padRight].
%
% ME3101 - Mini-Project 2: Hyperspectral Polymer Classification

    if nargin < 2 || isempty(targetH)
        targetH = 80;
    end
    if nargin < 3 || isempty(targetW)
        targetW = 80;
    end
    if nargin < 4 || isempty(padValue)
        padValue = 0.0;
    end

    [H, W, C] = size(cube);

    % Safety Check: Do not permit cropping or downsampling
    if H > targetH || W > targetW
        error('Image dimensions [%d x %d] exceed target dimensions [%d x %d]. Cropping is strictly prohibited!', ...
            H, W, targetH, targetW);
    end

    % Balanced padding calculation across opposing edges
    deltaH = targetH - H;
    deltaW = targetW - W;

    padTop    = floor(deltaH / 2);
    padBottom = deltaH - padTop;
    padLeft   = floor(deltaW / 2);
    padRight  = deltaW - padLeft;

    % Allocate padded tensor
    paddedCube = cast(zeros(targetH, targetW, C) + padValue, class(cube));

    % Place original cube in center
    rowStart = padTop + 1;
    rowEnd   = padTop + H;
    colStart = padLeft + 1;
    colEnd   = padLeft + W;

    paddedCube(rowStart:rowEnd, colStart:colEnd, :) = cube;

    padDetails.padTop    = padTop;
    padDetails.padBottom = padBottom;
    padDetails.padLeft   = padLeft;
    padDetails.padRight  = padRight;
    padDetails.origSize  = [H, W, C];
    padDetails.paddedSize = [targetH, targetW, C];
end
