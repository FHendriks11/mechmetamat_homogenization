function [L, Dmod, p, D, modificationFlag] = modchol_ldlt(A,delta)
% MODCHOL_LDLT performs the Modified Cholesky algorithm using the LDL' factorization.
%
%   [L, Dmod, p, D, modificationFlag] = modchol_ldlt(A, delta)
%
% This version builds on the original modchol_ldlt provided by Bobby Cheng
% and Nick Highem (revision 2015). Original description is provided below:
%
%     modchol_ldlt  Modified Cholesky algorithm based on LDL' factorization.
%       [L D,P,D0] = modchol_ldlt(A,delta) computes a modified
%       Cholesky factorization P*(A + E)*P' = L*D*L', where 
%       P is a permutation matrix, L is unit lower triangular,
%       and D is block diagonal and positive definite with 1-by-1 and 2-by-2 
%       diagonal blocks.  Thus A+E is symmetric positive definite, but E is
%       not explicitly computed.  Also returned is a block diagonal D0 such
%       that P*A*P' = L*D0*L'.  If A is sufficiently positive definite then 
%       E = 0 and D = D0.  
%       The algorithm sets the smallest eigenvalue of D to the tolerance
%       delta, which defaults to sqrt(eps)*norm(A,'fro').
%       The LDL' factorization is compute using a symmetric form of rook 
%       pivoting proposed by Ashcraft, Grimes and Lewis.
%     
%       Reference:
%       S. H. Cheng and N. J. Higham. A modified Cholesky algorithm based
%       on a symmetric indefinite factorization. SIAM J. Matrix Anal. Appl.,
%       19(4):1097-1110, 1998. doi:10.1137/S0895479896302898,
%     
%       Authors: Bobby Cheng and Nick Higham, 1996; revised 2015.
%
% Features: * The original script was rewritten to avoid for-loops and
%             maintain sparsity of data
%           * The script also returns flag whether modification was used
%
% Version:  0.1.0 (2023-04-28)
% Author:   Martin Doskar (MartinDoskar@gmail.com)

% Check inputs
if ~ishermitian(A)
    A = 0.5 * (A + A');
end

if nargin < 2
    delta = sqrt(eps) * norm(A, 'fro'); 
end

% Perform standard LDL' factorization
[L, D, p] = ldl(A, 'vector'); 

Dmod = D;
modificationFlag = 0;

% Change all diagonal entries below threshold
diagD = diag(D);
if any(diagD < delta)
    modificationFlag = 1;
    mask = diagD < delta;
    diagD(mask) = abs(diagD(mask));
    Dmod = diag(diagD);
end

% Handle 2x2 blocks on diagonal
offdiagEntries = find(diag(D,1) ~= 0);
if ~isempty(offdiagEntries)
    modificationFlag = 1;
    for ii = offdiagEntries'
        [U, S] = eig(full(D(ii:ii+1, ii:ii+1)));
        S = abs(S);
        Dmod(ii:ii+1, ii:ii+1) = U' * S * U;
    end
end

end
