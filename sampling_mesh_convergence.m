%% Define things

tic
addpath('./fun');
addpath('./mex');

microsolverOptions  = struct( ...
    'defaultTimeStep',      0.1, ...
    'smallestTimeStep',     1E-7, ... % 0.005, ... %
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

%% use first 4 outputs from samples, for [samples,J_arr,t_arr,phi_arr] = sample_U(0.75, 1.5, 0.5, 2, 2, 2)
% [0.796846017617814,0.0718391224105228,0.0718391224105228,0.947687310738255;
%     0.835612327574424,-0.107618477482941,-0.107618477482941,0.911405578357646;
%     1.12416197303503,0.131686056478878,0.131686056478878,0.682589551930208;
%     0.641873492899742,-0.0264735004315476,-0.0264735004315476,1.16954642079659]
%
geom = 'p4_square_2024-05-22_15-01-53.190117';
Fmacro = [0.796846017617814,0.0718391224105228;0.0718391224105228,0.947687310738255];

% geom = 'p4m_square_2024-05-22_15-08-35.472754';
% Fmacro = [0.835612327574424,-0.107618477482941;-0.107618477482941,0.911405578357646];

% geom = 'p6_hexagonal_2024-05-22_15-52-16.388326';
% Fmacro = [1.12416197303503,0.131686056478878;0.131686056478878,0.682589551930208];

% geom = 'pg_rectangular_2024-05-22_14-21-22.169238';
% Fmacro = [0.641873492899742,-0.0264735004315476;-0.0264735004315476,1.16954642079659];

%%
% directory with geometry
input_dir = ['your_input_path_here', geom];

% directory to put results
results_dir = ['your_result_path_here', geom];

input_dir
results_dir

% check if results_dir exists, if not, create it
if exist(results_dir, 'dir') ~= 7
    mkdir(results_dir)
end

%%
% Find all geometries to simulate
matFiles = dir(fullfile(input_dir, '*.mat')); % Get all .mat files in the folder
geometries = {matFiles.name}; % Extract the file names into a cell array

% Convert the cell array to a regular array
% geometries = char(fileNames);

for i = 1:length(geometries)
    temp = geometries{i};
    geometries{i} = temp(1:end-4); % remove '.mat'
end

%% Define settings/constants
for i = 1: length(geometries)
    geometries{i}
    geometry = geometries{i};

    % check if data/geometry.mat exists
    if exist(fullfile(results_dir, [geometry, '.mat']), 'file') == 2
        fprintf(geometries{i})
        fprintf('.mat already exists\n')
        continue
    end

    input_matfile = fullfile(input_dir, [geometry, '.mat']);

    disp(input_matfile)
    disp(geometry)

    % new
    RVEnew = load(input_matfile);
    RVEdata = convert_RVEdata(RVEnew);

    RVEdata.microMaterials = [2,550,300,0,0,0,55000,1];

    disp(Fmacro)

    % data per simulation

    data_sim.computation_time = [];
    data_sim.errorFlag = [];
    data_sim.F_final = Fmacro;

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

    fprintf('Time for preliminaries:\n')
    toc

    Gmacro = Fmacro - eye(2);

    %% Run simulation
    tic
    [W, P, D, state, storedT, bifurc, bifurcMode, errorFlag] = ...
        get_FE2_internal_force_and_stiffness_NEW(Gmacro, RVEdata.RVEmesh, RVEdata.microMaterials, microsolverOptions);
    toc
    microfluctuation = reshape(state.snapshots, 2, [], length(storedT));
    microfluctuation = permute(microfluctuation, [3,2,1]);

    % add computation time to data.computation_time
    data_sim.computation_time = [data_sim.computation_time; toc];
    data_sim.errorFlag = [data_sim.errorFlag; errorFlag];

    disp('errorFlag:');
    data_sim.errorFlag

    Gmacros = repmat(reshape(Gmacro, 1, 2, 2), length(storedT), 1, 1) .* reshape(storedT, [], 1, 1);
    %         size(Gmacros)
    data_ts.F = cat(1, data_ts.F, Gmacros+reshape(eye(2), 1, 2, 2));
    data_ts.W = cat(1, data_ts.W, W);
    data_ts.P = cat(1, data_ts.P, P);
    data_ts.D = cat(1, data_ts.D, D);
    data_ts.Time = cat(1, data_ts.Time, storedT);
    data_ts.traj = cat(1, data_ts.traj, repmat(1, length(storedT), 1));
    data_ts.microfluctuation = cat(1, data_ts.microfluctuation, microfluctuation);
    data_ts.bifurc = cat(2, data_ts.bifurc, bifurc);
    data_ts.bifurcMode = cat(2, data_ts.bifurcMode, bifurcMode);

    disp('errorFlag:');
    data_sim.errorFlag

    if length(data_ts.Time) > 1
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

            tic
            % convert matrix Fmacro to string
            Fmacro_str = strrep(sprintf('%f ', Fmacro(:)),' ', '_');
            Fmacro_str = Fmacro_str(1:end-1);

            % save figure
            saveas(gcf, fullfile(results_dir, [geometry, '_', Fmacro_str, '.png']));
            close(gcf);
            fprintf('Time for saving figure:\n')
            toc
        end  % isfield(RVEnew, 'lattice_vectors')
    end

    data_ts.bifurc = transpose(data_ts.bifurc);

    disp('errorFlag:');
    data_sim.errorFlag

    % save to file
    tic
    save(fullfile(results_dir, [geometry, '.mat']), 'data_sim', 'data_ts', 'Fmacro', 'microsolverOptions');
    fprintf('Time for saving .mat:\n')
    toc
end