function [ mesh ] = extract_periodic_boundary_rectangular_domain( mesh )
% EXTRACT_PERIODIC_BOUNDARY_RECTANGULAR_DOMAIN parses provided mesh and
% fills in a boundary data structure assuming the mesh is rectangular
%
% Features: * Works only in 2D
%           * Refactor to reuse extract_boundary_rectangular_domain and add
%            periodicity and FE2-like data structure
%
% Version:  0.1.1 [2020-02-25]
% Author:   Martin Doskar (MartinDoskar@gmail.com)


GEOM_TOL_REL = 1e-12;

minBounds = min(mesh.nodes); 
maxBounds = max(mesh.nodes);

GEOM_TOL = GEOM_TOL_REL * min(maxBounds(1:2)-minBounds(1:2));

assert(minBounds(3) == maxBounds(3), 'Script is intended only for 2D problems');

if ~isfield(mesh, 'nNodes')
    mesh.nNodes = size(mesh.nodes, 1);
end
if ~isfield(mesh, 'nElems')
    mesh.nElems = size(mesh.elements, 1);
end

% Extract boundary data
mesh = extract_boundary_rectangular_domain(mesh);

% Ensure all edge nodes are sorted
[~, sortInd] = sort(mesh.nodes( mesh.boundary.edge{1}, 1 ), 'ascend' );
mesh.boundary.edge{1} = mesh.boundary.edge{1}(sortInd);
[~, sortInd] = sort(mesh.nodes( mesh.boundary.edge{2}, 1 ), 'ascend' );
mesh.boundary.edge{2} = mesh.boundary.edge{2}(sortInd);

[~, sortInd] = sort(mesh.nodes( mesh.boundary.edge{5}, 2 ), 'ascend' );
mesh.boundary.edge{5} = mesh.boundary.edge{5}(sortInd);
[~, sortInd] = sort(mesh.nodes( mesh.boundary.edge{6}, 2 ), 'ascend' );
mesh.boundary.edge{6} = mesh.boundary.edge{6}(sortInd);

% Check data correspondence
assert( ...
    all(abs(mesh.nodes( mesh.boundary.edge{1}, 1 ) - mesh.nodes( mesh.boundary.edge{2}, 1 )) < GEOM_TOL) ...
    && all(abs(mesh.nodes( mesh.boundary.edge{5}, 2 ) - mesh.nodes( mesh.boundary.edge{6}, 2 )) < GEOM_TOL), ...
    'Periodic nodes are not aligned' );

% Append FE2-related data
translationFixingNode = mesh.boundary.vertex{1};
if isempty(translationFixingNode)    
    translationFixingNode = mesh.boundary.edge{1}(1);
    assert( ~isempty(translationFixingNode), 'Domain discretization has no vertex nodes and empty edge structure');
end

mesh.FE2 = struct( ...
    'V', prod(maxBounds(1:2)-minBounds(1:2)), ...
    'periodicSourceNodes', [ mesh.boundary.vertex{1}; mesh.boundary.vertex{1}; mesh.boundary.vertex{1}; mesh.boundary.edge{1}; mesh.boundary.edge{5} ], ...
    'periodicImageNodes', [ mesh.boundary.vertex{2}; mesh.boundary.vertex{3}; mesh.boundary.vertex{4}; mesh.boundary.edge{2}; mesh.boundary.edge{6} ], ...
    'fixedNode', translationFixingNode );

end