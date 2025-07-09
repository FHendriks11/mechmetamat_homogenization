%% setup
clc;
close all;
clear;

addpath('./fun');
addpath('./mex');

% folder with one subdirectory for each microstructure, which contains 
% (among other things) the .mat file with microstructure to process
folder_path = 'your_path_here';

% path to directory to put the results
results_dir = 'data_refD';

options = struct( ...
    'returnAllMacroResults', true, ... % return all macroscopic results, not just the last one
    'checkFullSystem', true, ...
    'defaultTimeStep',      0.1, ...
    'smallestTimeStep',     0.005, ...
    'timeStepShorteningCoeff', 1./pi(), ...
    'timeStepProlongationCoeff', 2.0, ...
    'nConstantTimeStepsToCoarsen', 2, ...
    'convergeUtol',         1e-06, ...  % 1e-4, ... %
    'convergeRtol',         1e-05, ...  % 1e-3, ... %
    'nIterLim',             20, ...
    'nIterLimLastresort',   70, ...
    'nQuadraturePoints',    3, ...
    'solverOptions',        struct( ...
            'direction', 'Newton', ...
            'lastResortNewton', true ...
            ), ...
    'useMappingU',          false, ...
    'verboseMode',          true, ...
    'storeSnapshots',       true, ...
    'initialGuessW',        [] ...
    );


%% Define settings/constants
Jmin = 'vol_frac';
Jmax = 1.5;
tmax = 0.5;

%% Find all geometries to simulate
% Find all folders in folder_path with a file in it that starts with 'info' and ends with '.txt'
folders = dir(folder_path);
folders = folders([folders.isdir]);  % exclude files
folders = folders(~ismember({folders.name},{'.','..'})); % exclude '.' and '..'
% exclude folders starting with 'old' or 'DEBUG'
folders = folders(~contains({folders.name},'old') & ~contains({folders.name},'DEBUG'));

% For each folder, see if there is an info file, which indicates a succesful geometry generation. Use this to create a list of geometries to simulate.
geometries = {};
for i = 1:length(folders)
    path = fullfile(folder_path, folders(i).name);
    files = dir(path);
    files = files(~[files.isdir]);
    files = files(contains({files.name},'info') & contains({files.name},'.txt'));
    if isempty(files)
        continue
    end
    geometries = [geometries; folders(i).name];
end

%Get a list of all items in the folder
items = dir(folder_path);

%Filter out only the folders
isFolder = [items.isdir];
folders = items(isFolder);

%Remove the '.' and '..' entries
folders = folders(~ismember({folders.name}, {'.', '..'}));

%Get the full paths of the folders
geometries = {folders.name};

%% Simulate all geometries
for i = 1:length(geometries)
    geometries{i}

    % new
    geometry = geometries{i};
    RVEnew = load(fullfile(folder_path, geometry, [geometry, '_00.mat']));
    RVEdata = convert_RVEdata(RVEnew);

    RVEmesh = RVEdata.RVEmesh;
    materials = [2,550,300,0,0,0,55000,1];

    nDOFs = 2*RVEmesh.nNodes;
    nQuadraturePoints = 3;

    RVEmesh4ORcode = struct( ...
        'p', RVEmesh.nodes(:,1:2)', ...
        't', [ RVEmesh.elements, RVEmesh.elemMats ]', ...
        'nGaussK', options.nQuadraturePoints, 'nGaussM', options.nQuadraturePoints, ...
        'V', RVEmesh.FE2.V );

    renumMap = create_renumbering_map(RVEmesh, [], true);
    nExtDOFs = 4;
    renumMapExt = [(1:nExtDOFs)'; renumMap+nExtDOFs];

    g0 = zeros(2,2);
    currentTransformer = transformer(g0 + eye(2));

    u = zeros(nDOFs, 1);

    [Wmacro, Ppseudo, Cpseudo] = upscale(RVEmesh4ORcode, ...
        materials, u, nExtDOFs, renumMapExt);
    [Pmacro, Cmacro] = currentTransformer.upscale(Ppseudo, ...
        Cpseudo);

    Wmacro
    Pmacro
    Cmacro

    save(fullfile(results_dir, [geometry, '.mat']), 'Cmacro');
end


    function [Wmacro, Pmacro, Cmacro] = upscale(mesh, materials, u, nExtDOFs, renumMapExt)

        [W, Fext, Kext] = build_extended_grad_hess_TLF2d_wOptimQuadrature(mesh.p, mesh.t, materials, mesh.nGaussK, u);

        % Include boundary conditions when upscaling
        [~, KextRenum] = renumber_F_and_K(Fext, Kext, renumMapExt);
        Kcond = full(KextRenum(1:nExtDOFs,1:nExtDOFs)) - full(KextRenum(1:nExtDOFs,nExtDOFs+1:end) * (KextRenum(nExtDOFs+1:end,nExtDOFs+1:end) \ KextRenum(nExtDOFs+1:end,1:nExtDOFs)));

        Wmacro = W ./ mesh.V;
        Pmacro = Fext(1:nExtDOFs) ./ mesh.V;
        Cmacro = Kcond ./ mesh.V;

        % Renumbering needed due to different ordering in OR's and MD's codes
        remapOR2MDindexing_ = [1,4,3,2];
        Pmacro = Pmacro(remapOR2MDindexing_);
        Cmacro = Cmacro(remapOR2MDindexing_, remapOR2MDindexing_);

    end

%%

