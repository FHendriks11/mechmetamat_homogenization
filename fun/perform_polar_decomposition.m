function [U, R] = perform_polar_decomposition(F)
% PERFORM_POLAR_DECOMPOSITION return stretch U and rotation R tensors such
% that F = R * U
%
%   [U, R] = perform_polar_decomposition(F)
%
% Version:  0.1 (2020-08-18)
% Author:   Martin Doskar (MartinDoskar@gmail.com)

C = F'*F;
U = sqrtm(C);
U = 0.5*(U+U');
R = F / U;

end