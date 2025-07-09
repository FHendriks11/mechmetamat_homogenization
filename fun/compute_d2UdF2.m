function [ d2UdF2 ] = compute_d2UdF2( F )
% COMPUTE_D2UDF2 computes the Hessian, i.e. the second derivative of the 
% right stretch tensor U with respect to the deformation gradient F and returns it
% as a sixth-order tensor in a 3D array form (using Voigt notation)
%
%   [ d2UdF2 ] = compute_d2UdF2( F )
%
% Features: * Based on the analytical expression from Rosati, L. (1999). Derivatives 
%             and Rates of the Stretch and Rotation Tensors. Journal of Elasticity, 
%             56(3), 213--230. https://doi.org/10.1023/A:1007663620943
% 
% Version:  0.1.0 (2021-06-19)
% Author:   Martin Doskar (MartinDoskar@gmail.com)

[ dUdF, Ainv, B ] = compute_dUdF( F );

% dAdF is composed of two terms with dUdF and Kronecker delta
tempUFdelta = zeros(4,4,4);
tempUFdelta(1,1,:) = dUdF(1,:);
tempUFdelta(1,4,:) = dUdF(3,:);
tempUFdelta(2,2,:) = dUdF(2,:);
tempUFdelta(2,3,:) = dUdF(4,:);
tempUFdelta(3,2,:) = dUdF(3,:);
tempUFdelta(3,3,:) = dUdF(1,:);
tempUFdelta(4,1,:) = dUdF(4,:);
tempUFdelta(4,4,:) = dUdF(2,:);

tempdeltaUF = zeros(4,4,4);
tempdeltaUF(1,1,:) = dUdF(1,:);
tempdeltaUF(1,3,:) = dUdF(3,:);
tempdeltaUF(2,2,:) = dUdF(2,:);
tempdeltaUF(2,4,:) = dUdF(4,:);
tempdeltaUF(3,1,:) = dUdF(4,:);
tempdeltaUF(3,3,:) = dUdF(2,:);
tempdeltaUF(4,2,:) = dUdF(3,:);
tempdeltaUF(4,4,:) = dUdF(1,:);

dAdF = tempUFdelta + tempdeltaUF;
    
% dBdF formed by different Kronecker delta (see the commented auxiliary part below)
dBdF = zeros(4,4,4);
dBdF(:,:,1) = [ 2 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 1 0 ];
dBdF(:,:,2) = [ 0 0 0 0; 0 2 0 0; 0 0 0 1; 0 0 0 1 ];
dBdF(:,:,3) = [ 0 0 0 0; 0 0 2 0; 1 0 0 0; 1 0 0 0 ];
dBdF(:,:,4) = [ 0 0 0 2; 0 0 0 0; 0 1 0 0; 0 1 0 0 ];

d2UdF2 = zeros(4,4,4); 
for gamma = 1:4
    d2UdF2(:,:,gamma) = -Ainv * dAdF(:,:,gamma) * Ainv * B + Ainv * dBdF(:,:,gamma);
end
     
     
%% Auxiliary function and scripts for derivation
%  
%      for gamma = 1:4
%          [m,n] = map_Voigt_to_tensor(gamma);
%          for beta = 1:4
%              [k,l] = map_Voigt_to_tensor(beta);
%              for alpha = 1:4
%                  [a,b] = map_Voigt_to_tensor(alpha);
%                  
%                  D(alpha,beta,gamma) = (k==m)*(a==n)*(b==l) + (a==l)*(k==m)*(b==n);
%              end
%          end
%      end
% 
%     function [indOut1, indOut2] = map_Voigt_to_tensor(indIn)
%         switch indIn
%             case 1
%                 indOut1 = 1;
%                 indOut2 = 1;
%             case 2
%                 indOut1 = 2;
%                 indOut2 = 2;
%             case 3
%                 indOut1 = 1;
%                 indOut2 = 2;
%             case 4
%                 indOut1 = 2;
%                 indOut2 = 1;
%             otherwise
%                 error('Cannot map Voigt index higher than 4');
%         end            
%     end
     
end