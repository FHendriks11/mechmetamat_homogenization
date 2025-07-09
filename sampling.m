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
results_path = 'data';

% check if results folder exists, if not create it
if exist(results_path, 'dir') ~= 7
    mkdir(results_path)
end

microsolverOptions  = struct( ...
    'defaultTimeStep',      0.1, ...
    'smallestTimeStep',    0.005, ... % 5E-5, ... %
    'timeStepShorteningCoeff', 1./pi(), ... % 0.1, ... %
    'verboseMode', true, ...
    'checkFullSystem', true, ...
    'nIterLim', 20, ...
    'nIterLimLastresort',   70, ... %50, ... % 70, ...
    'solverOptions',        struct( ...
            'direction', 'Newton', ...
            'lastResortNewton', true ...
        ), ...
    'returnAllMacroResults', true ... % return all macroscopic results, not just the last one
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

% Get a list of all items in the folder
items = dir(folder_path);

% Filter out only the folders
isFolder = [items.isdir];
folders = items(isFolder);

% Remove the '.' and '..' entries
folders = folders(~ismember({folders.name}, {'.', '..'}));

% Get the full paths of the folders
geometries = {folders.name};

%% Simulate all geometries
for i = 1:length(geometries)
    geometries{i}

    % check if data/geometry.mat exists
    if exist(fullfile(results_path, [geometries{i}, '.mat']), 'file') == 2
        fprintf(geometries{i})
        fprintf('.mat already exists\n')
        continue
    end

    % new
    geometry = geometries{i};
    RVEnew = load(fullfile(folder_path, geometry, [geometry, '_00.mat']));
    RVEdata = convert_RVEdata(RVEnew);

    RVEdata.microMaterials = [2,550,300,0,0,0,55000,1];

    if strcmp(Jmin, 'vol_frac')
        Jmin = RVEnew.volume_fraction;
    end

    % Sample U
    [U_arr,J_arr,t_arr,phi_arr] = sample_U(Jmin, Jmax, tmax, 2, 2, 2)  %3,3,3);
    % nr of samples N = 2*Nt*Nphi + NJ*Nphi; -> 2,2,2: N=12
    data_sim.F_final   =    reshape(U_arr, [], 2, 2);
    data_sim.J_final   =    J_arr;
    data_sim.t_final   =    t_arr;
    data_sim.phi_final =  phi_arr;
    data_sim.computation_time = [];
    data_sim.errorFlag = [];

    % data per time step
    data_ts.F = [];
    data_ts.W = [];
    data_ts.P = [];
    data_ts.D = [];
    data_ts.Time = [];
    data_ts.traj = [];
    data_ts.bifurc = [];
    data_ts.microfluctuation = [];
    data_ts.bifurcMode = {};

    for j = 1:length(U_arr)
        geometries{i}
        fprintf('Trajectory nr: ')
        j

        Fmacro = reshape(U_arr(j, :), 2, 2);
        Gmacro = Fmacro - eye(2);

        tic
        [W, P, D, state, storedT, bifurc, bifurcMode, errorFlag] = ...
            get_FE2_internal_force_and_stiffness_NEW(Gmacro, RVEdata.RVEmesh, RVEdata.microMaterials, microsolverOptions);
        toc
        microfluctuation = reshape(state.snapshots, 2, [], length(storedT));
        microfluctuation = permute(microfluctuation, [3,2,1]);

        % add computation time to data.computation_time
        data_sim.computation_time = [data_sim.computation_time; toc];
        data_sim.errorFlag = [data_sim.errorFlag; errorFlag];

        Gmacros = repmat(reshape(Gmacro, 1, 2, 2), length(storedT), 1, 1) .* reshape(storedT, [], 1, 1);
%         size(Gmacros)
        data_ts.F = cat(1, data_ts.F, Gmacros+reshape(eye(2), 1, 2, 2));
        data_ts.W = cat(1, data_ts.W, W);
        data_ts.P = cat(1, data_ts.P, P);
        data_ts.D = cat(1, data_ts.D, D);
        data_ts.Time = cat(1, data_ts.Time, storedT);
        data_ts.traj = cat(1, data_ts.traj, repmat(j, length(storedT), 1));
        data_ts.bifurc = cat(2, data_ts.bifurc, bifurc);
        data_ts.bifurcMode = cat(2, data_ts.bifurcMode, bifurcMode);
        if size(data_ts.microfluctuation, 1) == 0
            % reshape to size (0, n_nodes, 2)
            data_ts.microfluctuation = reshape(data_ts.microfluctuation, 0, size(microfluctuation, 2), 2);
        end
        if size(microfluctuation, 1) == 0
            microfluctuation = reshape(microfluctuation, 0, size(data_ts.microfluctuation, 2), 2);
        end
        data_ts.microfluctuation = cat(1, data_ts.microfluctuation, microfluctuation);


        if length(storedT) > 1
            % Plot result RVE 2×2 (= unit cell 4×4)
            % Plot edges by count
            t = RVEdata.RVEmesh.elements;
            edges = [t(:, [1,4]); ...
                t(:, [4, 2]); ...
                t(:, [2, 5]); ...
                t(:, [5, 3]); ...
                t(:, [3, 6]); ...
                t(:, [6, 1])];
            edges2 = sort(edges, 2);
            [C,ia,ic] = unique(edges2, 'rows');
            a_counts = accumarray(ic,1);

            if isfield(RVEnew, 'lattice_vectors')
                tic
                figure;
                shifts = [[0,0];[1,0];[0,1];[1,1]];

                LV1 = transpose(RVEnew.lattice_vectors(1,:));
                LV2 = transpose(RVEnew.lattice_vectors(2,:));

                % Plot original in gray
                for k=1:4

                    temp = RVEnew.p + transpose(2*shifts(k,1)*LV1 + 2*shifts(k,2)*LV2);

                    temp_source2 = temp(C(:, 1), :);
                    temp_target2 = temp(C(:, 2), :);
                    x = transpose([temp_source2(:, 1), temp_target2(:, 1)]);
                    y = transpose([temp_source2(:, 2), temp_target2(:, 2)]);

%                     % Plot all edges
%                     plot(x,y, 'Color', "#cccccc");
%                     hold on;

                    % Plot only boundary edges
                    plot(x(:, a_counts~=2),y(:, a_counts~=2), 'Color', "#cccccc");
                    hold on;
                end

                ind = size(storedT, 1);  % index of time step to plot
                F_temp = ((Fmacro-eye(2))*storedT(ind)+eye(2));

                LV1 = F_temp*LV1;
                LV2 = F_temp*LV2;

                for k=1:4
                    temp = transpose(F_temp*transpose(RVEnew.p)) + transpose(reshape(state.snapshots(:,ind), 2, []) + 2*shifts(k,1)*LV1 + 2*shifts(k,2)*LV2);

                    temp_source2 = temp(C(:, 1), :);
                    temp_target2 = temp(C(:, 2), :);

                    x = transpose([temp_source2(:, 1), temp_target2(:, 1)]);
                    y = transpose([temp_source2(:, 2), temp_target2(:, 2)]);

%                     % Plot all edges
%                     plot(x,y, "blue");
%                     hold on;

                    % Plot only boundary edges
                    plot(x(:, a_counts~=2),y(:, a_counts~=2), "blue");
                    hold on;
                end

                set(gca,'DataAspectRatio',[1 1 1])
                title('Result')

                fprintf('Time for plotting:\n')
                toc

                % convert matrix Fmacro to string
                Fmacro_str = strrep(sprintf('%f ', Fmacro(:)),' ', '_');
                Fmacro_str = Fmacro_str(1:end-1);

                % create figure
                fig = gcf;
                fig.Position = [10 10 900 600];
                tic
                frame = getframe(fig); % fig is the figure handle to save
                [raster, raster_map] = frame2im(frame); % raster is the rasterized image, raster_map is the colormap
                fig_file = fullfile(results_path, [geometries{i}, '_', Fmacro_str, '.png']);
                if isempty(raster_map)
                    imwrite(raster, fig_file);
                else
                    imwrite(raster, raster_map, fig_file); % fig_file is the path to the image
                end
                fprintf('Time for saving figure, method 1:\n')
                toc

                info = rendererinfo(gca);
                info

                close(gcf);
            end  % isfield(RVEnew, 'lattice_vectors')
        end

    end
    data_ts.bifurc = transpose(data_ts.bifurc);

    lens = zeros(1, 9);
    lens(1) = size(data_ts.W, 1);
    lens(2) = size(data_ts.P, 1);
    lens(3) = size(data_ts.F, 1);
    lens(4) = size(data_ts.microfluctuation, 1);
    lens(5) = size(data_ts.Time, 1);
    lens(6) = size(data_ts.bifurc, 1);
    lens(7) = size(data_ts.bifurcMode, 2);
    lens(8) = size(data_ts.D, 1);
    lens(9) = size(data_ts.traj, 1);

    if ~(all(lens == lens(1)))
        lens
        error('not all quantities have the same nr of time steps!!\n')
    else
        lens
        fprintf('all quantities have the same nr of time steps\n')
    end

    % save to file
    tic
    save(fullfile(results_path, [geometry, '.mat']), 'data_sim', 'data_ts');
    fprintf('Time for saving .mat:\n')
    toc

end