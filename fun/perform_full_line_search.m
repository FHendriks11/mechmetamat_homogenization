function [stepFinal, stateFinal, errorFlag] = perform_full_line_search(systemHandle, dir, options)
% PERFORM_FULL_LINE_SEARCH uses line-search algorithm to find the minimizing step length in a given
% direction dir.
% 
%   [stepFinal, stateFinal, errorFlag] = perform_full_line_search(systemHandle, dir, options)
%
% Version:  0.3.1 (2023-06-09)
% Author:   Martin Doskar (MartinDoskar@gmail.com)

defaultOptions = struct( ...
    'verboseMode', false, ...
    'stepInitial', 1.0, ...
    'nIterLim', 20, ...
    'nIterLimExtension', 2, ...
    'returnAfterLim', false, ...            % When true it returns the best step (instead of throwing) after reaching maximal number of iterations(excluding initial)
    'gradientLimit', 1e-6, ...
    'relativeGradientForExtension', 0.1, ...
    'stepExtensionCoefficient', 1.5, ...
    'stepShorteningCoefficient', 0.25, ...  
    'maxStepLength', [], ...                % Limit the length of the step (checked only against Wolfe conditions on sufficient decrease)
    'c1', 0.1, ...
    'c2', 0.5, ...
    'convergenceCheck', 'StrongWolfe' ...   % { StrongWolfe, Armijo, absolute }
    );
if nargin < 3
    options = defaultOptions;
else
    options = add_default_options(options, defaultOptions);
end

% Initialize auxiliary variables
errorFlag = 0;
historyVals   = inf(options.nIterLim+1, 1);
historyStates = cell(options.nIterLim+1, 1);

stepCurrent = options.stepInitial;

stateInit = get_function_state(systemHandle, dir, 0);
stateLeft = stateInit;

errorThrown = false;
try
    stateRight = get_function_state(systemHandle, dir, stepCurrent);
catch
    errorThrown = true;
end

% Try to handle situations when the stateRight does not results in a valid state
iIter = 0;
while errorThrown || isnan(stateRight.fun)
    iIter = iIter + 1;
    stepCurrent = options.stepShorteningCoefficient * stepCurrent;
    
    errorThrown = false;
    try
        stateRight = get_function_state(systemHandle, dir, stepCurrent);
    catch
        errorThrown = true;
    end

    if iIter > options.nIterLim
        error('LS:TooManyIterations', 'Iteration limit %i reached without reaching non-NaN or non-throwing state.', options.nIterLim);
    end
end

% % Check for descend direction
% if stateInit.grad > 0
%     if (stateInit.grad < 1e-16) || (stateInit.grad < 1e-14 && stateInit.hess < 0) || (stateRight.fun < stateLeft.fun)
%         warning('The initial direction is not descent (%e)\nUsing ZERO descend direction instead\n', stateInit.grad);
%         stateInit.grad = 0;
%     else
%         if nargout == 3
%            errorFlag = 1;
%            stepFinal = [];
%            stateFinal = [];
%            return
%         else
%            warning('LS:InitDir', 'The initial direction is not descent ERROR (%e), flipping direction\n', stateInit.grad);
%            dir = -dir;
%         end
%     end
% end

% Handle the situation that a longer step can be taken
iterate = ~check_convergence(stateInit, stateRight, stepCurrent, options);
counterExtension = 0;
if ~iterate
    while ((stateRight.grad < options.relativeGradientForExtension * stateInit.grad) ...
            && (stateRight.fun < stateInit.fun) ...
            && (counterExtension < options.nIterLimExtension))
        stepCurrent = options.stepExtensionCoefficient * stepCurrent;
        stateRight = get_function_state(systemHandle, dir, stepCurrent);

        counterExtension = counterExtension + 1;
    end
    iterate = ~check_convergence(stateInit, stateRight, stepCurrent, options);
end

% Check for near-convergence situation (i.e. only numerical noise)
if (abs(stateInit.fun - stateRight.fun) / max(abs(stateInit.fun), abs(stateRight.fun)) < 1e-10) ...
        &&(abs(stateInit.grad) < 1e-15 && abs(stateRight.grad) < 1e-15)
    iterate = false;
end

% Handle satisfaction of the convergence criteria with the initialStep
if ~iterate
    stateFinal = stateRight;
    stepFinal = stepCurrent;
end

% Try to update step-length to meet criteria
flagLimitStep = false;
iIter = 0;
while iterate
    iIter = iIter + 1;
    
    % Predict the new step length
    stepCurrentOld = stepCurrent;
    stepCurrent = get_cubic_fit( ...
        stateLeft.a, stateLeft.fun, stateLeft.grad, ...
        stateRight.a, stateRight.fun, stateRight.grad );
    if ~isreal(stepCurrent)
        stepCurrent = 0.5 * (stateLeft.a + stateRight.a);
    end
    
    % Use safeguard
    if ~isempty(options.maxStepLength)
        if stepCurrent > options.maxStepLength
            stepCurrent = options.maxStepLength;
            flagLimitStep = true;
        end
    end
    
    % Compute new state
    errorThrown = false;
    try
        stateActual = get_function_state(systemHandle, dir, stepCurrent);
    catch
        errorThrown = true;
    end

    % Check for NaN 
    if errorThrown || isnan(stateActual.fun)
        if (stateRight.fun < stateInit.fun)            
            stepFinal = stateRight.a;
            stateFinal = stateRight;
            return;
        else
            stepCurrent = stepShorteningCoefficient * stateRight.a;
            stateActual = get_function_state(systemHandle, dir, stepCurrent);
        end
    end
    
    historyVals(iIter+1) = stateActual.fun;
    historyStates{iIter+1} = stateActual;

    % Check convergence
    if flagLimitStep
        tempOptions = options;
        tempOptions.convergenceCheck = 'Armijo';
        iterate = ~check_convergence(stateInit, stateActual, stepCurrent, tempOptions);
        if iterate
            error('LS:Runtime', 'Unexpected situation');
        end
    else
        % Avoid the situation of being stuck in one point due to small
        % gradient by switching to Armijo rule only
        if (norm(stepCurrentOld - stepCurrent) < 5e-2)    
            tempOptions = options;
            tempOptions.convergenceCheck = 'Armijo';
            iterate = ~check_convergence(stateInit, stateActual, stepCurrent, tempOptions);
        else
            iterate = ~check_convergence(stateInit, stateActual, stepCurrent, options);
        end    
    end
    if ~iterate
        stepFinal = stepCurrent;
        stateFinal = stateActual;
    end
    
    % Choose the next pair of left-right states
    if (stateLeft.a < stateActual.a) && (stateActual.a < stateRight.a)
        
        if (stateLeft.fun < stateActual.fun) && (stateActual.fun < stateRight.fun)
            stateRight = stateActual;
            
        elseif (stateLeft.fun > stateActual.fun) && (stateActual.fun > stateRight.fun)
            stateLeft = stateActual;
            
        else            
            if (stateActual.grad < 0) && (stateRight.grad > 0)
                stateLeft = stateActual;
            elseif (stateActual.grad > 0) && (stateLeft.grad <= 0)
                stateRight = stateActual;
            else
                if (stateLeft.fun > stateActual.fun) ...
                        && (stateActual.fun < stateRight.fun) ...
                        && (stateActual.grad < 0)
                    stateLeft = stateActual;
                else
%                     warning('Unable to decide which bracket to use');
                    if isinf(stateActual.fun)
                        stepCurrent = 0.1 * stepCurrent;
                        stateLeft = stateInit;
                        stateRight = get_function_state(systemHandle, dir, stepCurrent);
                        continue;
                    else
                        stepFinal = stepCurrent;
                        stateFinal = stateActual;
                        break;
                    end
                end
            end
        end
        
    else
        if (stateLeft.grad < 0) && (stateActual.grad > stateLeft.grad)
            stateRight = stateActual;
        else
            if (min([stateActual.fun, stateLeft.fun, stateRight.fun]) < stateInit.fun)
                [~, ind] = min([stateActual.fun, stateLeft.fun, stateRight.fun]);
                switch ind
                    case 1
                        stepFinal = stateActual.a;
                        stateFinal = stateActual;
                        break;
                    case 2
                        stepFinal = stateLeft.a;
                        stateFinal = stateLeft;
                        break;
                    case 3
                        stepFinal = stateRight.a;
                        stateFinal = stateRight;
                        break;
                end
            else
                error('LS:Runtime', 'Guess got out of the bracket but does not seem to provide new reasonable guess');
            end
        end
    end
    
%     fprintf('Line search converged in %i iterations\n', iIter);
    
    if stateLeft.fun > stateInit.fun
        error('LS:Runtime', 'Left state has higher function value than the initial solution.');
    end
    if iIter > options.nIterLim
        if options.returnAfterLim || any( historyVals <= stateInit.fun )
            iterate = false;
            [~,minInd] = min(historyVals);
            stepFinal =  historyStates{minInd}.a;
            stateFinal = historyStates{minInd};
        else
        error('LS:TooManyIterations', 'Iteration limit %i reached within the line-search procedure.', options.nIterLim); 
        end
    end
    
    if options.verboseMode
        fprintf('\tIter no. %i: c = %f: f = %e, g = %e, h = %e\n', ...
            iIter, stepCurrent, stateActual.fun, stateActual.grad, stateActual.hess );
    end
end

%% Auxiliary functions

    function [state] = get_function_state(systemHandle, dir, stepLength)
        state = struct( 'a', stepLength, 'fun', [], 'grad', [], 'hess', []);
        [state.fun, tempGrad, tempHess] = systemHandle(stepLength * dir);
        state.grad = tempGrad' * dir;
        state.hess = dir' * (tempHess * dir);
    end

    function converged = check_convergence(stateInit, stateActual, stepLength, options)
        
        if strcmpi(options.convergenceCheck, 'absolute')
            converged = (stateInit.fun > stateActual.fun) ...
                && ( (abs(stateActual.grad) < options.gradientLimit) );%...
            %                    || ( max(abs(a0 - aNew), abs(a1 - aNew)) / max(abs([a0, a1, aNew])) < 1-6));
            %                     dir' * (tempHessValue * dir) > 0
            
        elseif strcmpi(options.convergenceCheck, 'StrongWolfe')
            converged = (stateActual.fun < (stateInit.fun + options.c1 * stepLength * stateInit.grad)) ...
                && ((abs(stateActual.grad) < options.c2 * abs(stateInit.grad)) || (abs(stateActual.grad) < 1e-8) );
            
        elseif strcmpi(options.convergenceCheck, 'Armijo')
            converged = (stateActual.fun < (stateInit.fun + options.c1 * stepLength * stateInit.grad));
            
        else
            error('LS:Input', 'Unknown convergence criterion in line search');
        end
        
    end


end
