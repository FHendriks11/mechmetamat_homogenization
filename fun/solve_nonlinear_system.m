function [dx, stepDir, stepLength, modificationFlag] = solve_nonlinear_system(grad, hess, systemHandle, options)
% SOLVE_NONLINEAR_SYSTEM yields a solution that minimizes a system
% described by systemHandle and the provided initial gradient `grad` and and
% Hessian `hess`.
%
% Features: * systemHandle must be provided in a format suitable for
%             perform_full_line_search() and can be undefined when
%             line-search algorithm is not used
%
% Version:  2025-07-03
% Author:   Martin Doskar (MartinDoskar@gmail.com), modified by Fleur Hendriks

modificationFlag = 0;

defaultOptions = struct( ...
    'direction',                            'Newton', ...       % Choose from { SteepestDescent, Newton, NewtonModified }
    'length',                               'LineSearch', ...   % Choose from { Fixed, LineSearch }
    'lineSearchOptions',                    struct( ...
        'stepInitial',                          1.0 ...
       ), ...
    'modificationThreshold',                1e-5, ...           % Threshold for value of D in LDLT below which it is mirrored in NewtonModified
    'modificationOffset',                   1e-3, ...           % Offset value of D in LDLT below which it is mirrored in NewtonModified ..
    'useNewtonLengthForSteepestDescent',    false, ...
    'verboseMode',                          false ...           % Set verbosity level
    );

if nargin < 4
    options = defaultOptions;
else
    options = add_default_options(options, defaultOptions);
end


%% Determine direction

modificationFlag = false;

if strcmpi( options.direction, 'SteepestDescent' )
    stepDir = -grad;

    if options.useNewtonLengthForSteepestDescent
        [L, D, perm] = ldl(hess, 'vector');
        foo = zeros(length(grad),1);
        foo(perm) = L'\(D\(L\(-grad(perm))));
        stepDir = stepDir ./ norm(stepDir) .* norm(foo);
    end

elseif strcmpi( options.direction, 'Newton' )
    [L, D, perm] = ldl(hess, 'vector');

    if any((D(D~=0))<0)
        fprintf('solve_nonlinear_system() uses Newton method, but matrix appears indefinite (based on LDL)\n');
    end

    stepDir = zeros(length(grad),1);
    stepDir(perm) = L'\(D\(L\(-grad(perm))));

elseif strcmpi( options.direction, 'NewtonModified' )

    [L, D, perm, ~, modificationFlag] = modchol_ldlt(hess, options.modificationThreshold);

    if options.verboseMode && modificationFlag
        fprintf('solve_nonlinear_system::Modification was used in ModifiedNewton direction.\n')
    end

    stepDir = zeros(length(grad),1);
    stepDir(perm) = L'\(D\(L\(-grad(perm))));

else
    error('Unsupported variant of direction definition.')
end

% Check for descent direction
dirGrad = grad' * stepDir;
if dirGrad > 0
   warning('Chosen direction is not a descent direction ((grad, dir) = %e. Flipping direction.)', dirGrad);
   stepDir = -stepDir;
end

%% Determine length

if strcmpi( options.length, 'Fixed' )
    stepLength = 1.0;

elseif strcmpi( options.length, 'LineSearch' )
    stepLength = perform_full_line_search(systemHandle, stepDir, options.lineSearchOptions);
%     plot_function_along_direction( systemHandle, stepDir, (-100:1:100) ./ 101 );

else
    error('Unsupported variant of length definition.')
end


%% Output quantities
dx = stepLength * stepDir;


end