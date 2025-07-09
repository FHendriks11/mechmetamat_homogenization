function [Urot, Uref, Trot, Tref] = map_U_onto_selected_section_SQR(inU)
% MAP_U_ONTO_SELECTED_SECTION_SQR transforms macroscopic right stretch
% tensor U by rotation and reflection to a pre-selected part of the loading
% space of SQR arrangement of RVE
% 
%   [Urot, Uref, Trot, Tref] = map_U_onto_selected_section_SQR(inU)
%
% Version:  0.1.0 (2022-11-27)
% Author:   Martin Doskar (MartinDoskar@gmail.com)

angle = compute_extreme_stretch_angle(inU);

rotationAngle = 0;
if (angle) > 3/4*pi()
    rotationAngle = -pi;
elseif angle > 1/4*pi()
    rotationAngle = -pi/2;
end

useReflection = false;
if (angle + rotationAngle) < 0
    useReflection = true;
end

c = cos(rotationAngle);
s = sin(rotationAngle);

TrotMtrx = [c, -s; s, c];
Trot = [c*c, s*s, -c*s, -c*s; s*s, c*c, c*s, c*s; c*s, -c*s, c*c, -s*s; c*s, -c*s, -s*s, c*c];

if useReflection
    TrefMtrx = [1, 0; 0, -1];
    Tref = diag([1,1,-1,-1]);
else
    TrefMtrx = [1, 0; 0, 1];
    Tref = diag([1,1,1,1]);
end

Urot = TrotMtrx * inU * TrotMtrx';
Uref = TrefMtrx * TrotMtrx * inU * TrotMtrx' * TrefMtrx';

end