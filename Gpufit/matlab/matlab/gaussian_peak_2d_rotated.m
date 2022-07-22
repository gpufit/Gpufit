function g = gaussian_peak_2d_rotated(x, y, p)
% Generates a 2D rotated elliptic Gaussian peak.
% http://gpufit.readthedocs.io/en/latest/fit_model_functions.html#d-gaussian-function-elliptical-rotated
%
% x,y - x and y grid position values
% p - parameters (amplitude, x,y center position, x,y-width, offset, rotation angle)

% cosine and sine of rotation angle
cp = cos(p(7));
sp = sin(p(7));

% Gaussian peak with two axes
arga = (x - p(2)) .* cp - (y - p(3)) .* sp;
argb = (x - p(2)) .* sp + (y - p(3)) .* cp;
ex = exp(-0.5 .* (((arga / p(4)) .* (arga / p(4))) + ((argb / p(5)) .* (argb / p(5)))));
g = p(1) .* ex + p(6);

end