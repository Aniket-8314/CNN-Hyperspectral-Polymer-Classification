function ds = create_hyperspectral_datastore(filePaths, labels, selectedBands, targetH, targetW, normScale)
%CREATE_HYPERSPECTRAL_DATASTORE Creates a custom datastore structure for hyperspectral data.
%
%   ds = create_hyperspectral_datastore(filePaths, labels, selectedBands, targetH, targetW, normScale)
%
% INPUTS:
%   filePaths     - Cell array of strings/chars specifying .bil file paths.
%   labels        - Categorical array or cell array of class strings for each sample.
%   selectedBands - 1 x 10 vector of selected band indices (1-based).
%   targetH       - Target height (default 80).
%   targetW       - Target width (default 80).
%   normScale     - Reflectance scaling factor (default 16383.0).
%
% OUTPUT:
%   ds            - Custom datastore struct containing:
%                   .files, .labels, .read, .numObservations, .classCounts
%
% ME3101 - Mini-Project 2: Hyperspectral Polymer Classification

    if nargin < 4 || isempty(targetH), targetH = 80; end
    if nargin < 5 || isempty(targetW), targetW = 80; end
    if nargin < 6 || isempty(normScale), normScale = 16383.0; end

    if iscell(labels)
        labels = categorical(labels);
    end

    ds.files = filePaths(:);
    ds.labels = labels(:);
    ds.numObservations = numel(filePaths);
    ds.selectedBands = selectedBands;
    ds.targetSize = [targetH, targetW, numel(selectedBands)];
    ds.normScale = normScale;

    % Summary counts per class
    classes = categories(labels);
    counts = countcats(labels);
    classes = classes(:);
    counts = counts(:);
    ds.classCounts = table(classes, counts, 'VariableNames', {'Class', 'Count'});

    % Custom reader function handle using readfile.m imhypercube approach
    ds.read = @(idx) readSample(ds.files{idx}, ds.selectedBands, targetH, targetW, normScale);

    fprintf('Datastore created with %d observations across %d classes.\n', ...
        ds.numObservations, numel(classes));
    disp(ds.classCounts);
end

function paddedTensor = readSample(bilPath, selectedBands, targetH, targetW, normScale)
    % Read cube using imhypercube (readfile.m approach)
    if exist('imhypercube', 'file') == 2
        hcube = imhypercube(bilPath);
        cube = gather(hcube);
    else
        % Fallback for base MATLAB environments lacking Hyperspectral Library
        [p, f] = fileparts(bilPath);
        hdrPath = fullfile(p, [f, '.hdr']);
        fid = fopen(hdrPath, 'r');
        txt = fread(fid, '*char')';
        fclose(fid);
        w = str2double(regexp(txt, 'samples\s*=\s*(\d+)', 'tokens', 'once'));
        h = str2double(regexp(txt, 'lines\s*=\s*(\d+)', 'tokens', 'once'));
        b = str2double(regexp(txt, 'bands\s*=\s*(\d+)', 'tokens', 'once'));
        fid = fopen(bilPath, 'r', 'ieee-le');
        raw = fread(fid, [w * b, h], '*uint16');
        fclose(fid);
        cube = permute(reshape(raw, [w, b, h]), [3, 1, 2]);
    end

    cube10 = double(cube(:, :, selectedBands));
    paddedCube = pad_hyperspectral_cube(cube10, targetH, targetW, 0.0);
    paddedTensor = single(paddedCube / normScale);
end
