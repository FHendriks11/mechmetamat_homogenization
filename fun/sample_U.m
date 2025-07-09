function [samples,J_arr,t_arr,phi_arr] = sample_U(Jmin, Jmax, tmax, NJ, Nt, Nphi)

J_intervals = (0:NJ)/NJ*(Jmax-Jmin)+Jmin;
t_intervals = (0:Nt)/Nt*tmax;
phi_intervals = (0:Nphi)/Nphi*2*pi;

N = 2*Nt*Nphi + NJ*Nphi;

samples = zeros(N, 4);
J_arr = zeros(N, 1);
t_arr = zeros(N, 1);
phi_arr = zeros(N, 1);

% keep J constant at Jmin
for i = 1:Nt
    for j = 1:Nphi
        % sample t from ith interval
        t = rand_from(t_intervals(i),t_intervals(i+1));
        % sample t from jth interval
        phi = rand_from(phi_intervals(j),phi_intervals(j+1));

        sample = calculate_U(Jmin, t, phi);
        ind = Nphi*(i-1)+j;
        samples(ind,:) = reshape(sample, 1, 4);
        J_arr(ind) = Jmin;
        t_arr(ind) = t;
        phi_arr(ind) = phi;

    end
end

% keep J constant at Jmax
for i = 1:Nt
    for j = 1:Nphi
        % sample t from ith interval
        t = rand_from(t_intervals(i),t_intervals(i+1));
        % sample phi from jth interval
        phi = rand_from(phi_intervals(j),phi_intervals(j+1));

        sample = calculate_U(Jmax, t, phi);
        ind = Nt*Nphi + Nphi*(i-1)+j;
        samples(ind,:) = reshape(sample, 1, 4);
        J_arr(ind) = Jmax;
        t_arr(ind) = t;
        phi_arr(ind) = phi;
    end
end

% keep t constant at tmax
for i = 1:NJ
    for j = 1:Nphi
        % sample J from ith interval
        J = rand_from(J_intervals(i),J_intervals(i+1));
        % sample phi from jth interval
        phi = rand_from(phi_intervals(j),phi_intervals(j+1));

        sample = calculate_U(J, tmax, phi);
        ind = 2*Nt*Nphi + Nphi*(i-1)+j;
        samples(ind,:) = reshape(sample, 1, 4);
        J_arr(ind) = J;
        t_arr(ind) = tmax;
        phi_arr(ind) = phi;
    end
end
%
% % plot sample distribution in J, t, phi
% figure();
% scatter(J_arr,t_arr)
% xlabel('J');
% ylabel('t');
% set(gca, 'FontSize', 16)
%
% figure();
% scatter(J_arr,phi_arr)
% xlabel('J');
% ylabel('\phi');
% set(gca, 'FontSize', 16)
%
% figure();
% scatter(t_arr,phi_arr)
% xlabel('t');
% ylabel('\phi');
% set(gca, 'FontSize', 16)
%
% % plot sample distribution in U11, U12, U22
% figure();
% scatter(samples(:,1),samples(:,2))
% xlabel('U_{11}');
% ylabel('U_{12}');
% set(gca, 'FontSize', 16)
%
% figure();
% scatter(samples(:,1),samples(:,4))
% xlabel('U_{11}');
% ylabel('U_{22}');
% set(gca, 'FontSize', 16)
%
% figure();
% scatter(samples(:,4),samples(:,2))
% xlabel('U_{22}');
% ylabel('U_{12}');
% set(gca, 'FontSize', 16)

function x = rand_from(low, high)
    x = low + (high-low).*rand();
end

function U = calculate_U(J, t, phi)
    Y1 = sqrt(0.5) * [1 0; 0 -1];
    Y2 = sqrt(0.5) * [0 1; 1 0];
    c = cos(phi);
    s = sin(phi);
    U = sqrt(J) * expm( t * (c*Y1 + s*Y2));
end

end

