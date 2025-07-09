function [Wmacro, Pmacro, Cmacro, currentState, storedT, bifurc, ...
    bifurcMode, errorFlag] = get_FE2_internal_force_and_stiffness_NEW(g0, ...
    RVEmesh, materials, options, previousState)
% GET_FE2_INTERNAL_FORCE_AND_STIFFNESS serves as a material point for
% first-order computational homogenization of hyperelastic materials
%
%   [Wmacro, Fmacro, Kmacro, currentState, storedT, bifurc, bifurcMode, errorFlag] = get_FE2_internal_force_and_stiffness(g0, mesh, materials, options, previousState)
%
% Features: * Enable more iterations as a last-resort option before throwing too-short-time-step error
%           * Supports mapping of prescribed U (OFF by DEFAULT)
%
% Version:  2025-07-03
% Author:   Martin Doskar (MartinDoskar@gmail.com), modified by Fleur Hendriks


%% Parse solver options and set defaults

defaultOptions = struct( ...
    'defaultTimeStep',      0.1, ...
    'smallestTimeStep',     0.005, ...
    'timeStepShorteningCoeff', 1./pi(), ...
    'timeStepProlongationCoeff', 2.0, ...
    'nConstantTimeStepsToCoarsen', 2, ...
    'convergeUtol',         1e-06, ...  % 1e-4, ... %
    'convergeRtol',         1e-05, ...  % 1e-3, ... %
    'nIterLim',             20, ...
    'nIterLimLastresort',   50, ...
    'nQuadraturePoints',    3, ...
    'solverOptions',        struct( ...
            'direction', 'NewtonModified', ...
            'lastResortNewton', false ...
            ), ...
    'useMappingU',          false, ...
    'verboseMode',          true, ...
    'storeSnapshots',       true, ...
    'returnAllMacroResults', false, ...
    'initialGuessW',        [] ...
    );

if nargin < 4
    options = defaultOptions;
else
    options = add_default_options(options, defaultOptions);
end

x_diff_max = max(RVEmesh.nodes(:, 1)) - min(RVEmesh.nodes(:, 1));
y_diff_max = max(RVEmesh.nodes(:, 2)) - min(RVEmesh.nodes(:, 2));
L = max(x_diff_max, y_diff_max);  % an estimated length scale



%% Support starting from a previous state

nDOFs = 2*RVEmesh.nNodes;
if nargin < 5 || isempty(previousState)
    previousState = struct('u', zeros(nDOFs, 1), 'g0', zeros(2));
end
if isempty(previousState.u)
    previousState.u = zeros(nDOFs, 1);
end
if isempty(previousState.g0)
    previousState.g0 = zeros(2);
end


%% Convert data to comply with OR's code and established DOF's numbering

% Extract necessary boundary data if needed
if ~isfield(RVEmesh, 'FE2') || ~isfield(RVEmesh.FE2, 'V') || ~isfield(RVEmesh.FE2, 'periodicSourceNodes') || ~isfield(RVEmesh.FE2, 'periodicImageNodes') || ~isfield(RVEmesh.FE2, 'fixedNode')
    warning('Requested FE2 data are not provided in mesh. Extracting these data using extract_periodic_boundary_rectangular_domain function.');
    RVEmesh = extract_periodic_boundary_rectangular_domain(RVEmesh);
end

RVEmesh4ORcode = struct( ...
    'p', RVEmesh.nodes(:,1:2)', ...
    't', [ RVEmesh.elements, RVEmesh.elemMats ]', ...
    'nGaussK', options.nQuadraturePoints, 'nGaussM', options.nQuadraturePoints, ...
    'V', RVEmesh.FE2.V );

renumMap = create_renumbering_map(RVEmesh, [], true);
nExtDOFs = 4;
renumMapExt = [(1:nExtDOFs)'; renumMap+nExtDOFs];


%% Transform g0 into (potentially mapped) U

if options.useMappingU
    if (length(RVEmesh.phi) == 1)
        mappingU = @(x) map_U_onto_selected_section_SQR(x);
    elseif (length(RVEmesh.phi) == 3)
        mappingU = @(x) map_U_onto_selected_section_HEX(x);
    else
        error("Unsupported RVE type (based on the number of phi modes provided.");
    end
    currentTransformer = transformer(g0 + eye(2), mappingU);
    previousTransformer = transformer(previousState.g0 + eye(2), mappingU);
else
    currentTransformer = transformer(g0 + eye(2));
    previousTransformer = transformer(previousState.g0 + eye(2));
end

U = currentTransformer.get_U();
Uprev = previousTransformer.get_U();

if sum(abs(g0 + eye(2) - U) > 1e-7) ~= 0
    error('U does not match g0 + eye(2).');
end
if sum(abs(eye(2) - Uprev) > 1e-7) ~= 0
    error('Uprev does not match eye(2).');
end

%% Compute prescribed part of the solution and initialize state

vPrevious = zeros(nDOFs, 1);
vPrescribed = zeros(nDOFs, 1);
for iN = 1:RVEmesh.nNodes
    x = RVEmesh.nodes(iN,1:2)';
    vPrescribed([iN*2-1,iN*2]) = ((U - eye(2)) * x);
    vPrevious([iN*2-1,iN*2]) = ((Uprev - eye(2)) * x);
end

v = vPrevious;
w = (previousState.u - vPrevious);


%% Iterate until convergence

finalTime = 1.0;
loading_function = @(t) t;

storedU = zeros(nDOFs, ceil(finalTime/options.defaultTimeStep));
storedW = zeros(nDOFs, ceil(finalTime/options.defaultTimeStep));
storedF = zeros(nDOFs, ceil(finalTime/options.defaultTimeStep));
storedT = zeros(ceil(finalTime/options.defaultTimeStep), 1);

iTime = 1;

nSameTimeSteps = 0;
currentTimeStep = options.defaultTimeStep;
currentTime = options.defaultTimeStep;
lastPlotTime = 0;
errorFlag = false;
bifurc = [];
bifurcMode = {};
while currentTime <= finalTime

    if options.verboseMode
        fprintf('Time %f:\n', currentTime);
    end
    vPrev = v;
    wPrev = w;

    % Start with previously converged state
    [W, Fint, K] = my_grad_hess(RVEmesh4ORcode, materials, v + w);

    % Increment macroscopic loading
    v = vPrevious + loading_function(currentTime) * (vPrescribed - vPrevious);
    % for vPrevious = zeros and loading_function(t) = t (my use case):
    % v = t * vPrescribed; -> G = t*Gprescribed -> F = t*(Fprescribed-I) + I

    nIter = 0;
    uConverged = false;
    rConverged = false;
    shortenTime = false;

    storedInitialNorm = [];

    bifurc(iTime) = false;
    bifurcMode{iTime} = [];

    while ~uConverged || ~rConverged
        nIter = nIter + 1;

        % Use increment from previously converged state
        % ================= Initial guess =================
        if nIter == 1
            if isempty(options.initialGuessW)

                Fint = Fint + (K * (v - vPrev));

                [FintRenum, Krenum] = renumber_F_and_K(Fint, K, renumMap);

                dA = - Krenum \ FintRenum;
                dw = back_renumber_vector(dA, renumMap);
                w = w + dw;

                storedInitialNorm = norm(dw);

                [W, Fint, K] = my_grad_hess(RVEmesh4ORcode, materials, v + w);
                [FintRenum, Krenum] = renumber_F_and_K(Fint, K, renumMap);
                Krenum = 0.5 * (Krenum + Krenum');
                Rrenum = - FintRenum;

            else
                w = options.initialGuessW;
                [~, Fint, K] = my_grad_hess(RVEmesh4ORcode, materials, v + w);
                [FintRenum, Krenum] = renumber_F_and_K(Fint, K, renumMap);
                Rrenum = - FintRenum;

                storedInitialNorm = norm(w);
            end

        end
        % ================= End initial guess =================

        try
            [ddd, stepDir, stepLength, modificationFlag] = solve_nonlinear_system(FintRenum, Krenum, ...
                @(x) wrapper_renumbered_system(x, v, w, RVEmesh4ORcode, renumMap, materials), ...
                options.solverOptions );
        catch err
            fprintf('Shortening time step because of error in solve_nonlinear_system: %s\n', err.message);
            warning('Shortening time step because of error in solve_nonlinear_system: %s\n', err.message);
            % Shorten the time step
            shortenTime = true;
            break
        end

        dw = back_renumber_vector(ddd, renumMap);
        w = w + dw;

        [W, Fint, K] = my_grad_hess(RVEmesh4ORcode, materials, v + w);
        [FintRenum, Krenum] = renumber_F_and_K(Fint, K, renumMap);
        Krenum = 0.5 * (Krenum + Krenum');
        Rrenum = - FintRenum;

        % Check convergence and stability
        [uConverged, rConverged] = check_convergence(dw, w, Rrenum, currentTime, nIter, options);

        if uConverged && rConverged  %modificationFlag  %

            % ================= Stability test =================
            [~, Dtest, ~] = ldl(Krenum, 'vector');
            if any(diag(Dtest)< 0)
                bifurc(iTime) = true;

                fprintf('UNSTABLE configuration encountered in equilibrium.\n');

                % Try to perturb before error-emitting shorting
                if currentTimeStep * options.timeStepShorteningCoeff < options.smallestTimeStep
%                 if true

                    % fprintf('Shortening would emit an error, trying perturbing the solution at time %f ...\n', currentTimeStep);
                    fprintf('Trying perturbing the solution at time %f ...\n', currentTimeStep);

                    [eigVecs, eigVals, eigFlag] = eigs(Krenum, 16, 'smallestabs' );
                    eigVals

                    % find desired value in eigVals (currently just
                    % smallest)
                    [~, sortedIndices] = sort(diag(eigVals));
                    ChosenIndex = sortedIndices(1);

                    perturbation = eigVecs(:,ChosenIndex);
                    bifurcMode{iTime} = back_renumber_vector(perturbation, renumMap);
                    % max value of the perturbation is L/100
                    perturbation = perturbation / max(abs(perturbation)) * L/1000;
                    fprintf('size(perturbation):');
                    size(perturbation)
                    perturbation2 = back_renumber_vector(perturbation, renumMap);
                    fprintf('size(perturbation2):');
                    size(perturbation2)

%                     % Optionally, plot perturbation:
%                     plot_initial_and_final_configuration(RVEmesh, perturbation2);
%                     title('Perturbation')
%                     plot_initial_and_final_configuration(RVEmesh, 10*perturbation2);
%                     title('Perturbation (scaled ×10)')

                    try
                        pertStepLength = 1.0;
                        ddd = pertStepLength * perturbation;

                        dw = back_renumber_vector(ddd, renumMap);
                        w = w + dw;

                        [W, Fint, K] = my_grad_hess(RVEmesh4ORcode, materials, v + w);
                        [FintRenum, Krenum] = renumber_F_and_K(Fint, K, renumMap);
                        Krenum = 0.5 * (Krenum + Krenum');
                        Rrenum = - FintRenum;

                        [uConverged, rConverged] = check_convergence(dw, w, Rrenum, currentTime, nIter, options);
                        if uConverged && rConverged
                            [~, Dtest, ~] = ldl(Krenum, 'vector');
                            if any(diag(Dtest)<0)% || any(Dtest(Dtest~=0) < 0)
                                    warning('Perturbation did not result in stable solution, shortening time step\n');
                                shortenTime = true;
                                break
                            else
                                shortenTime = false;
                            end
                        end

                    catch err
                        warning('Perturbation failed %s\n', err.message );
                        shortenTime = true;
                        break
                    end


                else
                    % Shorten the time step without perturbation
                    warning('Shortening time step because of unstable configuration')
                    shortenTime = true;
                    break
                end

            end
            % ================= End stability test =================
        end

        if ~(uConverged && rConverged)
            if nIter >= options.nIterLim
                warning('Reached maximal number of iterations without converging.\n');

                % Shorten the time step
                shortenTime = true;
                break

            end
        end

    end  % end of 'while not converged'-loop

    % Support time shortening
    if shortenTime
        fprintf('FE2::Shortening time step from %e to %e at time %f\n', currentTimeStep, currentTimeStep * options.timeStepShorteningCoeff, currentTime);

        previousTime = currentTime - currentTimeStep;
        currentTimeStep = options.timeStepShorteningCoeff * currentTimeStep;
        currentTime = previousTime + currentTimeStep;
        nSameTimeSteps = 0;

        % Increase the number of iterations as a last-resort option if the next time shortening would throw an error
        if (options.timeStepShorteningCoeff * currentTimeStep < options.smallestTimeStep) && (options.nIterLim < options.nIterLimLastresort)
            fprintf('Increasing iteration limit for the shortest time step from %i to %i.\n', options.nIterLim, options.nIterLimLastresort);
            options.nIterLim = options.nIterLimLastresort;
        elseif (options.timeStepShorteningCoeff * currentTimeStep < options.smallestTimeStep) ...
            && (options.nIterLim >= options.nIterLimLastresort ...
            && options.solverOptions.lastResortNewton ...
            && strcmp(options.solverOptions.direction, 'NewtonModified'))
            options.solverOptions.direction = 'Newton';
            fprintf('Switching to Newton method as a last resort');
%             % undo time step shortening
            currentTimeStep = currentTimeStep/options.timeStepShorteningCoeff;
        end

        if currentTimeStep < options.smallestTimeStep
            warning('FE2:smallTimeStep', 'Time step %e shorten below options.smallestTimeStep %e.', currentTimeStep, options.smallestTimeStep);
            errorFlag = true;
            v = vPrev;
            w = wPrev;
            break
        end

        v = vPrev;
        w = wPrev;
        continue
    end

    % This part is only reached if the solution converged


    % Store solution
    storedU(:,iTime) = v + w;

    storedW(:,iTime) = w;
    storedF(:,iTime) = Fint;
    storedT(iTime) = currentTime;
    nSameTimeSteps = nSameTimeSteps + 1;

    % Try to increase time step if constant for 3 consecutive steps
    if nSameTimeSteps > options.nConstantTimeStepsToCoarsen && currentTimeStep < options.defaultTimeStep
        if options.verboseMode
            fprintf( '\tPROLONGING TIME STEP\n' );
        end
       currentTimeStep = options.timeStepProlongationCoeff * currentTimeStep;
       nSameTimeSteps = 0;
    end

    % Increase time
    if currentTime == finalTime
        break
    else
        currentTime = min(currentTime + currentTimeStep, finalTime);
        if abs(currentTime - finalTime) < (1e-15 * finalTime)
            currentTime = finalTime;
        end
        iTime = iTime + 1;
    end
end


%% Finalize and upscale

u = v + w;

if errorFlag % remove last failed time step
    storedW = storedW(:,1:iTime-1);
    storedF = storedF(:,1:iTime-1);
    storedT = storedT(1:iTime-1);
    storedU = storedU(:,1:iTime-1);
    bifurc = bifurc(1:iTime-1);
    bifurcMode = bifurcMode(1:iTime-1);
else
    storedW = storedW(:,1:iTime);
    storedF = storedF(:,1:iTime);
    storedT = storedT(1:iTime);
    storedU = storedU(:,1:iTime);
end


currentState = struct( 'u', u, 'g0', g0, 'snapshots', storedW, 'info', struct());

if options.returnAllMacroResults
    Wmacro = zeros(length(storedT), 1);
    Pmacro = zeros(length(storedT), 2, 2);
    Cmacro = zeros(length(storedT), 2, 2, 2, 2);
    for i = 1:length(storedT)
        v = vPrevious + loading_function(storedT(i)) * (vPrescribed - vPrevious);
        [Wmacro_temp, Ppseudo_temp, Cpseudo_temp] = upscale(RVEmesh4ORcode, materials, v + storedW(:,i), nExtDOFs, renumMapExt);
        [Pmacro_temp, Cmacro_temp] = currentTransformer.upscale(Ppseudo_temp, Cpseudo_temp);
        Pmacro_temp = reshape(Pmacro_temp, 2, 2);
        Cmacro_temp = reshape(Cmacro_temp, 2, 2, 2, 2);
        Wmacro(i) = Wmacro_temp;
        Pmacro(i, :, :) = Pmacro_temp;
        Cmacro(i, :, :, :, :) = Cmacro_temp;
    end

else
    [Wmacro, Ppseudo, Cpseudo] = upscale(RVEmesh4ORcode, materials, u, nExtDOFs, renumMapExt);
    [Pmacro, Cmacro] = currentTransformer.upscale(Ppseudo, Cpseudo);
end

[Wmacro_end, Ppseudo_end, Cpseudo_end] = upscale(RVEmesh4ORcode, materials, u, nExtDOFs, renumMapExt);
[Pmacro_end, Cmacro_end] = currentTransformer.upscale(Ppseudo_end, Cpseudo_end);

Wmacro_end
Pmacro_end
Cmacro_end

lens = zeros(1, 7);
lens(1) = size(Wmacro, 1);
lens(2) = size(Pmacro, 1);
lens(3) = size(Cmacro, 1);
lens(4) = size(currentState.snapshots, 2);
lens(5) = size(storedT, 1);
lens(6) = size(bifurc, 2);
lens(7) = size(bifurcMode, 2);

if ~(all(lens == lens(1)))
    lens
    error('not all quantities have the same nr of time steps!!\n')
else
    lens
    fprintf('all quantities have the same nr of time steps\n')
end

%% Auxiliary functions

    function [outW, outF, outK] = my_grad_hess(mesh, materials, u)

            [outW, outF, outK] = build_grad_hess_TLF2d_wOptimQuadrature(mesh.p, mesh.t, materials, mesh.nGaussK, u);

    end


    function [tempFun, tempGradRenum, tempHessRenum] = wrapper_renumbered_system(renumX, v, w, mesh, renumMap, materials)
        deltaW = back_renumber_vector(renumX, renumMap);
        [tempFun, tempGrad, tempHess] = my_grad_hess(mesh, materials, v+w+deltaW);
        [tempGradRenum, tempHessRenum] = renumber_F_and_K(tempGrad, tempHess, renumMap);
        tempHessRenum = 0.5 * (tempHessRenum + tempHessRenum');
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


    function [uConverged, rConverged] = check_convergence(dw, w, Rrenum, currentTime, nIter, options)

        % Check for zero solution
        uError = max( norm(dw), norm(dw) / norm(w));
        if norm(dw) < options.convergeUtol && norm(w) < options.convergeUtol
            uError = norm(dw);
        end
        rError = norm(Rrenum);

        uConverged = (uError < options.convergeUtol) || (norm(dw) < 1e-15);
        rConverged = rError < options.convergeRtol;

        if options.verboseMode
            fprintf('Time %f, iter %i: |dU| = %e, |R| = %e\n', currentTime, nIter, uError, rError);
        end

    end


end