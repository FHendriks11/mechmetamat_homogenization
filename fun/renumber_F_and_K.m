    function [outF, outK] = renumber_F_and_K(inF, inK, map)
        if any(map ~= (1:size(inF,1))')
            [rInK, cInK, vInK] = find(inK);
            rRenum = map(rInK);
            cRenum = map(cInK);
            vRenum = vInK;
            removeMask = (rRenum == 0) | (cRenum == 0) | (vRenum == 0);
            rRenum(removeMask) = [];
            cRenum(removeMask) = [];
            vRenum(removeMask) = [];
            outK = sparse(rRenum, cRenum, vRenum);
            outK = 0.5 * (outK + outK');
            
            outF = zeros( size(outK, 1), 1);
            for a = 1:size(inF,1)
                if map(a) > 0
                    outF(map(a)) = outF(map(a)) + inF(a);
                end
            end
        else
            outF = inF;
            outK = 0.5 * (inK + inK');
        end
    end