
%% setup
clc;
close all;
clear;

addpath('./fun');
addpath('./mex');

microsolverOptions  = struct( ...
    'defaultTimeStep',      0.1, ...
    'smallestTimeStep',    0.005, ... % 5E-5, ... %
    'timeStepShorteningCoeff', 1./pi(), ... % 0.1, ... %
    'verboseMode', true, ...
    'checkFullSystem', true, ...
    'nIterLim', 20, ...
    'nIterLimLastresort',   50, ... % 70, ...
    'solverOptions',        struct( ...
            'direction', 'Newton', ...
            'lastResortNewton', true ...
            ), ...
    'returnAllMacroResults', true ... % return all macroscopic results, not just the last one
    );

% G = [-0.2, 0.0; 0.0, -0.2];  % equibiaxial compression
G = [-0.2966,    0.0296;    0.0296   -0.2469]; % biaxial (but not equibiaxial) compression + a bit of shear
Fmacro = eye(2) + G;

%% Convert data from Python to the right formats/shapes
RVEnew = load("cm_hexagonal1_2024-05-22_14-23-52.891252_00.mat");
RVEdata = convert_RVEdata(RVEnew);
RVEdata.microMaterials = [2,550,300,0,0,0,55000,1];

%% Or load data from .mat file, which is already in the right format
% RVEdata = load('./RVEdefinitionHEX.mat'); % hexonagally stacked circular holes
% RVEdata = load('./RVEdefinitionSQR.mat'); % square stacked circular holes

%% Plot indicating source and image node correspondence
figure();
scatter(RVEdata.RVEmesh.nodes(:, 1), RVEdata.RVEmesh.nodes(:, 2), 3, "filled");
hold on;

% Plot source & image nodes
temp_source_p = RVEdata.RVEmesh.nodes(RVEdata.RVEmesh.FE2.periodicSourceNodes, :);
scatter(temp_source_p(:, 1), temp_source_p(:, 2), 15, "filled", "red");
hold on;
temp_image_p = RVEdata.RVEmesh.nodes(RVEdata.RVEmesh.FE2.periodicImageNodes, :);
scatter(temp_image_p(:, 1), temp_image_p(:, 2), 15, "filled", "yellow");
hold on;
set(gca,'DataAspectRatio',[1 1 1])

% figure();
x = transpose([temp_source_p(:, 1), temp_image_p(:, 1)]);
y = transpose([temp_source_p(:, 2), temp_image_p(:, 2)]);
plot(x,y);
title('Mesh, showing periodic connectivity')

set(gca,'DataAspectRatio',[1 1 1])

%% Plot edges by count
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

figure();
temp_source2 = RVEdata.RVEmesh.nodes(C(:, 1), :);
temp_target2 = RVEdata.RVEmesh.nodes(C(:, 2), :);
x = transpose([temp_source2(:, 1), temp_target2(:, 1)]);
y = transpose([temp_source2(:, 2), temp_target2(:, 2)]);
plot(x(:, a_counts==2),y(:, a_counts==2), "blue");
hold on;
plot(x(:, a_counts~=2),y(:, a_counts~=2), "red");
set(gca,'DataAspectRatio',[1 1 1])
title('Mesh, edges colored by count')

%% Actual simulation

% ========================== ACTUAL SIMULATION ==========================
tic
[W, P, D, state, storedT, bifurc, errorFlag] = ...
    get_FE2_internal_force_and_stiffness_NEW(Fmacro - eye(2), RVEdata.RVEmesh, RVEdata.microMaterials, microsolverOptions);
toc
% ========================== END ACTUAL SIMULATION ==========================

%%
% Plot whole deformation
fprintf('size(state.u):');
size(state.u)
plot_initial_and_final_configuration(RVEdata.RVEmesh, state.u);
title('Final state')

% Plot only fluctuations
plot_initial_and_final_configuration(RVEdata.RVEmesh, state.snapshots(:,end));
title('Final fluctuation field')


%% Plot initial RVE 2×2 (= unit cell 4×4)

if isfield(RVEnew, 'lattice_vectors')
    figure;
    shifts = [[0,0];[1,0];[0,1];[1,1]];

    LV1 = transpose(RVEnew.lattice_vectors(1,:));
    LV2 = transpose(RVEnew.lattice_vectors(2,:));

    for i=1:4
        temp = RVEnew.p + transpose(2*shifts(i,1)*LV1 + 2*shifts(i,2)*LV2);

        temp_source2 = temp(C(:, 1), :);
        temp_target2 = temp(C(:, 2), :);

        x = transpose([temp_source2(:, 1), temp_target2(:, 1)]);
        y = transpose([temp_source2(:, 2), temp_target2(:, 2)]);
        plot(x,y, "blue");
        hold on;
    end

    set(gca,'DataAspectRatio',[1 1 1])
    title('Initial state')
end

%% Plot result RVE 2×2 (= unit cell 4×4)

if isfield(RVEnew, 'lattice_vectors')

    ind = size(storedT, 1);  % index of time step to plot
    F_temp = ((Fmacro-eye(2))*storedT(ind)+eye(2));

    figure;
    shifts = [[0,0];[1,0];[0,1];[1,1]];

    LV1 = transpose(RVEnew.lattice_vectors(1,:));
    LV2 = transpose(RVEnew.lattice_vectors(2,:));

    LV1 = F_temp*LV1;
    LV2 = F_temp*LV2;

    for i=1:4
        temp = transpose(F_temp*transpose(RVEnew.p)) + transpose(reshape(state.snapshots(:,ind), 2, []) + 2*shifts(i,1)*LV1 + 2*shifts(i,2)*LV2);

        temp_source2 = temp(C(:, 1), :);
        temp_target2 = temp(C(:, 2), :);

        x = transpose([temp_source2(:, 1), temp_target2(:, 1)]);
        y = transpose([temp_source2(:, 2), temp_target2(:, 2)]);
        plot(x,y, "blue");
        hold on;
    end

    set(gca,'DataAspectRatio',[1 1 1])
    title('Final state')
end