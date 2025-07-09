%% setup
clc;
close all;
clear;

fprintf('test\n')

addpath('./fun');
addpath('./mex');

% folder with one subdirectory for each microstructure, which contains
% (among other things) the .mat file with microstructure to process
folder_path = 'your_input_path_here';

% path to directory to put the results
results_dir = 'data';

% check if results_path directory exists, if not, create it
if exist(results_dir, 'dir') ~= 7
    mkdir(results_dir)
end


%% Define settings/constants
Jmin = 'vol_frac';
Jmax = 1.5;
tmax = 0.5;

% Get a list of all items in the folder
items = dir(folder_path);

% Filter out only the folders
isFolder = [items.isdir];
folders = items(isFolder);

% Remove the '.' and '..' entries
folders = folders(~ismember({folders.name}, {'.', '..'}));

% Get the full paths of the folders
geometries = {folders.name};

%% Iterate over all geometries
for i = 1:length(geometries)
    geometries{i}

    % check if folder_path/geometry/geometry.mat exists
    file_path = fullfile(folder_path, geometries{i}, [geometries{i}, '_00.mat'])
    if  ~(exist(file_path, 'file') == 2)
        fprintf(geometries{i})
        file_path
        fprintf('\nSimulation results not found!\n')
        continue
    end

    % check if results_dir/geometry_specialnodes.mat already exists
    path_temp = fullfile(results_dir, [geometries{i}, '_specialnodes.mat']);
    if  exist(path_temp, 'file') == 2
        disp(path_temp)
        fprintf(' already exists!\n')
        continue
    end

    % new
    geometry = geometries{i};
    RVEnew = load(fullfile(folder_path, geometry, [geometry, '_00.mat']));
    RVEdata = convert_RVEdata(RVEnew);

    source_nodes = RVEdata.RVEmesh.FE2.periodicSourceNodes;
    image_nodes = RVEdata.RVEmesh.FE2.periodicImageNodes;
    fixed_node = RVEdata.RVEmesh.FE2.fixedNode;

    path_temp = fullfile(results_dir, [geometry, '_specialnodes.mat']);
    save(path_temp, 'source_nodes', 'image_nodes', 'fixed_node');
    disp(path_temp)
    fprintf(' saved')
end


