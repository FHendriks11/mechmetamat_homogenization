function [aNew, fNew, fQuery] = get_cubic_fit(a0, f0, g0, a1, f1, g1, aQuery)
% GET_CUBIC_FIT returns a minimizer based on a cubic fit arising from the function values
% f0 and f1, its gradients f0 and f1 evaluated at a0 and a1 (with a0 < a1).
% 
%   [aNew, fNew, fQuery] = get_cubic_fit(a0, f0, g0, a1, f1, g1, aQuery)
%
% Features: * Fit is restricted to a >= 0
%           * Solve the fit and quadratic equation in parametric
%             coordinates \in [0, 1]
%
% Version:  0.3.0 (2023-01-06)
% Author:   Martin Doskar (MartinDoskar@gmail.com)

param_to_real = @(y) a0 + (a1 - a0) .* y;
real_to_param = @(a) (a - a0) / (a1 - a0);

% Compute cubic fit in parametric coordinates
coeffs = [ 0 0 0 1; 1 1 1 1; 0 0 1 0; 3 2 1 0] \ [f0; f1; g0; g1];

compute_param_cubic_fit = @(y) [y.^3, y.^2, y, 1] * coeffs;

% Auxiliary plotting
%plot( param_to_real([0:0.01:10]), [0:0.01:10].^3 .* coeffs(1) + [0:0.01:10].^2 .* coeffs(2)+  [0:0.01:10] .* coeffs(3)  + coeffs(4) )

% Solve quadratic equation to find stationary points of the cubic fit
a = 3*coeffs(1);
b = 2*coeffs(2);
c = coeffs(3);
D = b^2 - 4*a*c;

if D >= 0
    % Proceed with cubic fit
    candidate1 = (-b - sqrt(D)) / (2*a);
    candidate2 = (-b + sqrt(D)) / (2*a);
    
    fun1 = compute_param_cubic_fit(candidate1);
    fun2 = compute_param_cubic_fit(candidate1);
    
    if (0 < param_to_real(candidate1))
        if (0 < param_to_real(candidate2))
            if fun1 < fun2
                aNew = param_to_real(candidate1);
                fNew = fun1;
            else
                aNew = param_to_real(candidate2);
                fNew = fun2;
            end
        else
            aNew = param_to_real(candidate1);
            fNew = fun1;
        end
    else
        if (0 <= param_to_real(candidate2))
            aNew = param_to_real(candidate2);
            fNew = fun2;
        else
            %
            % No optimum from cubic fit found in interval [0,inf) -> using bracketing
            %
            if f1 <= f0
                if g1 < 0
                    aNew = a1;
                    fNew = f1;
                else
                    aNew = 0.5 * (a0 + a1);
                    fNew = compute_param_cubic_fit(real_to_param(aNew));
                end
            else
                aNew = 0.5 * (a0 + a1);
                fNew = compute_param_cubic_fit(real_to_param(aNew));
            end
        end
    end
    
else
    %
    % Unable to find stationary point of a cubic fit -> using bracketing
    %
    if f1 <= f0
        if g1 < 0
            aNew = a1;
            fNew = f1;
        else
            aNew = 0.5 * (a0 + a1);
            fNew = compute_param_cubic_fit(real_to_param(aNew));
        end
    else
        aNew = 0.5 * (a0 + a1);
        fNew = compute_param_cubic_fit(real_to_param(aNew));
    end

end

% Return also cubic-fit values are querry points
if nargout > 2
    ys = zeros(length(aQuery), 1);
    ys(:) = (aQuery(:) - a0) ./ (a1 - a0);
    
    fQuery = [ys.^3, ys.^2, ys, ones(size(ys))] * coeffs;
end

end