    function [renumMap] = create_renumbering_map(RVEmesh, Q, fixNodes)
        
        if nargin <= 2
           fixNodes = true; 
        end
        
        nAllDOFs = 2*RVEmesh.nNodes;
        if isempty(Q)
            periodicMasterDOFs = reshape( [RVEmesh.FE2.periodicSourceNodes'*2 - 1, RVEmesh.FE2.periodicSourceNodes'*2], [], 1);
            periodicImageDOFs  = reshape( [RVEmesh.FE2.periodicImageNodes'*2 - 1, RVEmesh.FE2.periodicImageNodes'*2], [], 1);
            
            renumMapMask = true(nAllDOFs, 1);
            renumMapMask(periodicImageDOFs) = false;

            if fixNodes
                renumMapMask([RVEmesh.FE2.fixedNode*2 - 1; RVEmesh.FE2.fixedNode*2]) = false;
            end
            
            renumMap = cumsum(renumMapMask) .* renumMapMask;
            renumMap(periodicImageDOFs) = renumMap(periodicMasterDOFs);
        else
            renumMap = (1:nAllDOFs)';
        end
    end
