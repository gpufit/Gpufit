% Equivalent/similar to tests/Gauss_Fit_1D.cpp

assert(gpufit_cuda_available(), 'CUDA not available');

% constants
n_fits = 1;
n_points = 5;
n_parameters = 4;
true_parameters = single([4; 2; 0.5; 1]);

% data
x = single((1:n_points)' - 1);
y = gaussian_1d(true_parameters, x);
data = zeros(n_points, n_fits, 'single');
data(:, 1) = y;

% model
model_id = ModelID.GAUSS_1D;

% initial_parameters
initial_parameters = zeros(n_parameters, n_fits, 'single');
initial_parameters(:, 1) = [2, 1.5, 0.3, 0];

% call to gpufit
[parameters, states, chi_squares, n_iterations] = gpufit(data, [], model_id, initial_parameters);

%% Test results
assert(states == 0);
assert(n_iterations < 10);
assert(chi_squares < 1e-6);
assert(all(abs(parameters - true_parameters) < 1e-6));

function y = gaussian_1d(p, x)

y = p(1) * exp(-(x - p(2)).^2 ./ (2 * p(3).^2)) + p(4);

end