function [outVec] = back_renumber_vector(inVec, map)
% BACK_RENUMBER_VECTOR performs renumbering and localisation of input
% vector following given renumbering map
%
%   [outVec] = back_renumber_vector(inVec, map)
%
% Version:  0.2.0 (2023-01-06)
% Author:   Martin Doskar (MartinDoskar@gmail.com)

if any(map ~= (1:length(map))')
    outVec = zeros(size(map));
    for a = 1:length(map)
        if map(a) > 0
            outVec(a) = inVec(map(a));
        end
    end
else
    outVec = inVec;
end

end