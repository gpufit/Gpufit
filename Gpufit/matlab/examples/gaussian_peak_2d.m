function f = gaussian_peak_2d(x, y, p)
% Generates a 2D Gaussian peak.
% http://gpufit.readthedocs.io/en/latest/fit_model_functions.html#d-gaussian-function-cylindrical-symmetry
%
% x,y - x and y grid position values
% p - parameters (amplitude, x,y center position, width, offset)

assert(nargin == 3);

f = p(1) * exp(-((x - p(2)).^2 + (y - p(3)).^2) / (2 * p(4)^2)) + p(5);

end