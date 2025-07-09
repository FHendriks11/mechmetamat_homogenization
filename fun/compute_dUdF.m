function [ dUdF, Ainv, B ] = compute_dUdF( F )
% COMPUTE_DUDF returns the total derivative of the right stretch tensor U
% with respect to the deformation gradient F as a fourth-order tensor in 
% a matrix form (using Voigt notation)
%
%   [ dUdF, auxU, auxF ] = compute_dUdF( F )
%
% Features: * Analytical expression from Rosati, L. (1999). Derivatives 
%             and Rates of the Stretch and Rotation Tensors. Journal of Elasticity, 
%             56(3), 213--230. https://doi.org/10.1023/A:1007663620943
%           * Returns auxiliary tensors dUdF = Ainv * B (e.g. used in compute_d2UdF2)
% 
% Version:  0.1.2 (2021-06-18)
% Author:   Martin Doskar (MartinDoskar@gmail.com)

[U, ~] = perform_polar_decomposition( F );

auxU = [ 2*U(1,1), 0, U(1,2), U(1,2); ...
         0, 2*U(2,2), U(2,1), U(2,1); ...
         U(2,1), U(1,2), U(1,1)+U(2,2), 0; ...
         U(2,1), U(1,2), 0, U(2,2)+U(1,1) ];

auxF = [ 2*F(1,1), 0, 0, 2*F(2,1); ...
         0, 2*F(2,2), 2*F(1,2), 0; ...
         F(1,2), F(2,1), F(1,1), F(2,2); ...
         F(1,2), F(2,1), F(1,1), F(2,2) ];
     
dUdF = auxU \ auxF;

Ainv = inv(auxU);
B = auxF;

end