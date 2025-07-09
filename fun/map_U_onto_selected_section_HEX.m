function [Urot, Uref, Trot, Tref] = map_U_onto_selected_section_HEX(inU)
% MAP_U_ONTO_SELECTED_SECTION_HEX transforms macroscopic right stretch
% tensor U by rotation and reflection to a pre-selected part of the loading
% space of HEX arrangement of RVE
% 
%   [Urot, Uref, Trot, Tref] = map_U_onto_selected_section_HEX(inU)
%
% Version:  0.1.0 (2022-11-27)
% Author:   Martin Doskar (MartinDoskar@gmail.com)


angle = compute_extreme_stretch_angle(inU);

rotationAngle = 0;
if angle < 1/3*pi()
    rotationAngle = +pi/3;
elseif angle > 2/3*pi()
    rotationAngle = -pi/3;
end

useReflection = false;
if (angle + rotationAngle) < pi/2
    useReflection = true;
end

c = cos(rotationAngle);
s = sin(rotationAngle);

TrotMtrx = [c, -s; s, c];
Trot = [c*c, s*s, -c*s, -c*s; s*s, c*c, c*s, c*s; c*s, -c*s, c*c, -s*s; c*s, -c*s, -s*s, c*c];

if useReflection
    TrefMtrx = [-1, 0; 0, 1];
    Tref = diag([1,1,-1,-1]);
else
    TrefMtrx = [1, 0; 0, 1];
    Tref = diag([1,1,1,1]);
end

Urot = TrotMtrx * inU * TrotMtrx';
Uref = TrefMtrx * TrotMtrx * inU * TrotMtrx' * TrefMtrx';

end