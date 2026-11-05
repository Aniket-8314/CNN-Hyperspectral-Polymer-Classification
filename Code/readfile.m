clc; clear; close all;

%% Initial Exploration Script (Helper Code: readfile.m)
% Course: ME3101 - Mini-Project 2
% Demonstrates:
%   1. Reading a BIL/HDR hyperspectral cube pair using enviinfo and imhypercube
%   2. Inspecting spatial and spectral dimensions
%   3. Displaying pseudo-RGB composite image (raw 168-band cube) using mat2gray
%   4. PCA-based band ranking and reduction using reduceHypercubePCA
%   5. Displaying pseudo-RGB composite image of the reduced 10-band cube

addpath('Code');

% Specify file paths (supports both root and subfolder execution)
if isfolder('Dataset')
    dataBase = 'Dataset';
elseif isfolder(fullfile('..', 'Dataset'))
    dataBase = fullfile('..', 'Dataset');
else
    dataBase = 'Dataset';
end

hdrFile = fullfile(dataBase, 'PET', 'scan0001_frag01.hdr');
bilFile = fullfile(dataBase, 'PET', 'scan0001_frag01.bil');

fprintf('Reading sample hyperspectral file: %s\n', bilFile);

% Read header and hyperspectral cube
% Uses Hyperspectral Imaging Library (enviinfo & imhypercube) when available,
% with self-contained fallback for base MATLAB environments.
if exist('enviinfo', 'file') == 2 && exist('imhypercube', 'file') == 2
    hdr = enviinfo(hdrFile);
    hcube = imhypercube(bilFile);
    cube = gather(hcube);
else
    fid = fopen(hdrFile, 'r'); txt = fread(fid, '*char')'; fclose(fid);
    w = str2double(regexp(txt, 'samples\s*=\s*(\d+)', 'tokens', 'once'));
    h = str2double(regexp(txt, 'lines\s*=\s*(\d+)', 'tokens', 'once'));
    b = str2double(regexp(txt, 'bands\s*=\s*(\d+)', 'tokens', 'once'));
    fid = fopen(bilFile, 'r', 'ieee-le');
    raw = fread(fid, [w * b, h], '*uint16');
    fclose(fid);
    cube = permute(reshape(raw, [w, b, h]), [3, 1, 2]);
    hdr = struct('samples', w, 'lines', h, 'bands', b);
end

% Check dimensions
[rows, cols, bands] = size(cube);

fprintf('Rows (Height)   : %d\n', rows);
fprintf('Columns (Width) : %d\n', cols);
fprintf('Spectral bands  : %d\n', bands);

%% 1. Display Pseudo-RGB of Original 168-Band Cube
rBand = 30; gBand = 20; bBand = 10;   % choose representative band numbers
R = double(cube(:,:,rBand));
G = double(cube(:,:,gBand));
B = double(cube(:,:,bBand));

% Scale each channel to [0,1] using mat2gray
if exist('mat2gray', 'file') == 2
    R = mat2gray(R);
    G = mat2gray(G);
    B = mat2gray(B);
else
    normCh = @(ch) (ch - min(ch(:))) / max(1e-6, max(ch(:)) - min(ch(:)));
    R = normCh(R);
    G = normCh(G);
    B = normCh(B);
end

rgb = cat(3, R, G, B);
figure('Name', 'Original 168-Band Pseudo-RGB (readfile.m)');
imshow(rgb);
title(sprintf('Original Cube (%d x %d x %d) - Pseudo-RGB [B%d, B%d, B%d]', ...
    rows, cols, bands, rBand, gBand, bBand), 'FontWeight', 'bold');

%% 2. PCA-Based Band Selection Using reduceHypercubePCA
numBands = 10;          % Retain 10 original spectral bands
varianceThreshold = 95; % Use PCs explaining >= 95% variance

[reducedCube, selectedBands, importance, explained] = ...
    reduceHypercubePCA(cube, numBands, varianceThreshold);

%% 3. Display Pseudo-RGB of Reduced 10-Band Cube
% Choose bands from the reduced 10-band cube (e.g. 2nd, 5th, and 10th selected bands)
rBandRed = 2; gBandRed = 5; bBandRed = 10;
R_red = double(reducedCube(:,:,rBandRed));
G_red = double(reducedCube(:,:,gBandRed));
B_red = double(reducedCube(:,:,bBandRed));

% Scale each channel to [0,1] using mat2gray
if exist('mat2gray', 'file') == 2
    R_red = mat2gray(R_red);
    G_red = mat2gray(G_red);
    B_red = mat2gray(B_red);
else
    normCh = @(ch) (ch - min(ch(:))) / max(1e-6, max(ch(:)) - min(ch(:)));
    R_red = normCh(R_red);
    G_red = normCh(G_red);
    B_red = normCh(B_red);
end

rgb_reduced = cat(3, R_red, G_red, B_red);
figure('Name', 'PCA-Reduced 10-Band Pseudo-RGB (readfile.m)');
imshow(rgb_reduced);
title(sprintf('Reduced Cube (%d x %d x %d) - Pseudo-RGB [Sel %d, %d, %d]', ...
    size(reducedCube, 1), size(reducedCube, 2), size(reducedCube, 3), ...
    rBandRed, gBandRed, bBandRed), 'FontWeight', 'bold');

fprintf('\nreadfile.m initial exploration completed successfully.\n');
