function [ iFig ] = plot_initial_and_final_configuration( mesh, solution, inMagnificationFactor, options )
% PLOT_INITIAL_AND_FINAL_DEFORMATION overlays final CONVERGED deformed state over
% initial discretization
%
%   [ iFig ] = plot_initial_and_final_configuration( mesh, solution, inMagnificationFactor, options )
%   [ iFig ] = plot_initial_and_final_configuration( mesh, solution, inMagnificationFactor, inFig )
%   [ iFig ] = plot_initial_and_final_configuration( mesh, plotNodeNumbers )
%
% Features: * Plot only converged solutions
%           * Enable just mesh plot (when called with one or two parameter)
%           * Support user-defined figure input
%           * Automatic scaling (when inMagnificationFactor == 0)
%           * Check for column and row input vector as a solution
%           * Support plot of element data (averaged) with options:
%               options.plotElementData = true
%               options.elementDataType = {'none', 'detF', 'rotF'}
%           * Min. and max. principal stretches over all quadrature points
%             can be printed by setting options.printDeformationStats = true
%
% Version:  1.6.1 (2020-12-20)
% Author:   Martin Doskar (MartinDoskar@gmail.com)

% Provide defaults and parse input options
defaultOptions = struct( ...
    'plotElementData', false, ...
    'elementDataType', 'none', ... % 'none', 'detF', 'rotF'
    'printDeformationStats', false ...
);

existingFig = [];
if nargin >= 4 && ishandle(options)
    existingFig = options;
end
    
if nargin < 4 || ishandle(options)
    options = struct();
end
if isfield(options, 'elementDataType') && (~isfield(options, 'plotElementData')) && ~strcmpi(options.elementDataType, 'none')
   options.plotElementData = true; 
end
for f = fieldnames(defaultOptions)'
  if ~isfield(options, f{1})
      options.(f{1}) = defaultOptions.(f{1});
  end
end

% Default magnification
if nargin > 2
    magnificationFactor = inMagnificationFactor;
else
    magnificationFactor = 1;
end

plotSolution = nargin >= 2 && ~isempty(solution) && ~islogical(solution);

if nargin == 2 && islogical(solution)
    if solution
        plotNodeNumbers = 'on';
    else
        plotNodeNumbers = 'off';
    end
else
    plotNodeNumbers = 'off';
end

% Enable plotting u directly (without the need for elaborate solution struct)
if plotSolution
    if isstruct(solution)   

        % Find last converged state
        iL = length(solution);
        iLastConverged = -1;
        while iLastConverged == -1 && iL > 0
           if solution(iL).stats.status
               iLastConverged = iL;
           else
               iL = iL - 1;
           end
        end
        assert( iLastConverged > 0, 'No converged solution found.' );
        Uplot = solution(iLastConverged).U;

    else    
        if size(solution, 1) ~= length(solution)
            Uplot = solution';
        else
            Uplot = solution;
        end
    end
end

% Automatically scale plot
if plotSolution && magnificationFactor == 0
    charSize = max(max(mesh.nodes)-min(mesh.nodes));
    magnificationFactor = 0.1 * charSize / max(Uplot);
end

if options.printDeformationStats
   minStretch = inf;
   maxStretch = -inf;
end

% Enable plotting of gauss-point data
if options.plotElementData || options.printDeformationStats
    
    quadrature.nQP = 3;
    quadrature.oQP = 2;
    quadrature.wQP = [1/3; 1/3; 1/3];
    quadrature.xQP = [1/6, 1/6; 2/3, 1/6; 1/6, 2/3];
    
    CDataValues = zeros(mesh.nElems, 1);
    for iE = 1:mesh.nElems
        elemInd = mesh.elements(iE,:);
        elemDOFs = reshape([ elemInd*2 - 1; elemInd*2 ], [], 1);   
            
        eVal = 0;
        for iQ = 1:quadrature.nQP

            parametricCoords = [ 1.0 - quadrature.xQP(iQ,1) - quadrature.xQP(iQ,2); quadrature.xQP(iQ,1); quadrature.xQP(iQ,2) ];
            [N,Bx,By,Jdet] = get_B_N_2d(length(elemInd), parametricCoords, mesh.nodes(elemInd,1), mesh.nodes(elemInd,2));

            dF = zeros(4, 2*length(elemInd));
            dF(1,1:2:end) = Bx;
            dF(2,2:2:end) = By;
            dF(3,1:2:end) = By;
            dF(4,2:2:end) = Bx;
            
            dFaux = magnificationFactor * dF * Uplot(elemDOFs);
            F = eye(2) + [dFaux(1), dFaux(3); dFaux(4), dFaux(2)]; 
            
            
            if options.printDeformationStats
                [U, ~] = perform_polar_decomposition(F);
                
                eigVals = eig(U);
                
                minStretch = min(minStretch, min(eigVals));
                maxStretch = max(maxStretch, max(eigVals));
            end
            
            if options.plotElementData
                switch options.elementDataType
                    case 'detF'
                        eVal = eVal + det(F);
                    case 'rotF'
                        [~, R] = perform_polar_decomposition(F);
                        eVal = eVal + atan2(R(2,1), R(1,1));
                    otherwise
                        error('Unsupported type for element data');
                end
            else
                eVal = 0;
            end
        end    
        CDataValues(iE) = eVal / quadrature.nQP;
    end
end

currentP = mesh.nodes(:,1:2);
if plotSolution
    currentP(:,1) = currentP(:,1) + Uplot(1:2:end) * magnificationFactor;
    currentP(:,2) = currentP(:,2) + Uplot(2:2:end) * magnificationFactor;
end

if isempty(existingFig)
    iFig = figure();
else    
    figure(existingFig);
    cla();
    iFig = existingFig;
end
pl0 = pdeplot( mesh.nodes(:,1:2)', mesh.elements', 'NodeLabels', plotNodeNumbers);
pl0.Color = 0.8*[1,1,1];
if plotSolution
    hold on;
    
    if options.plotElementData
        patch('Faces', mesh.elements(:,[1,4,2,5,3,6]), 'Vertices', currentP, 'FaceVertexCData', CDataValues, 'FaceColor', 'flat', 'EdgeColor', 'none');
    end
    
    pl1 = pdeplot( currentP', mesh.elements');
    hold off;
    if  isstruct(solution) && iL ~= length(solution)
       text(0,0,'Warning: Plotting last CONVERGED step'); 
    end
end
 
if options.plotElementData
    c = colorbar();
    switch options.elementDataType
        case 'detF'
            c.Label.String = 'det(F)';
        case 'rotF'
            c.Label.String = 'angle(F) [rad]';
    end
end

set( gca(), 'TickLabelInterpreter', 'latex', 'FontSize', 12 );
set( gca(), 'TickDir', 'in' );
axis equal;
grid on;
    

if options.printDeformationStats
    fprintf('Deformation gradient stats over all elements:\n');
    fprintf('\tmin stretch: %f\n', minStretch);
    fprintf('\tmax stretch: %f\n', maxStretch);
end

end