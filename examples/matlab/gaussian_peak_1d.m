function f = gaussian_peak_1d(x, p)
% Generates a 1D Gaussian peak.
% http://gpufit.readthedocs.io/en/latest/fit_model_functions.html#d-gaussian-function
%
% x - x grid position values
% p - parameters (amplitude, center position, width, offset)

assert(nargin == 2);

f = p(1) * exp(-(x - p(2)).^2 / (2 * p(3)^2)) + p(4);

end