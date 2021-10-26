function gauss1d()
% Example of the Matlab binding of the Gpufit library implementing
% Levenberg Marquardt curve fitting in CUDA
% https://github.com/gpufit/Gpufit
%
% 1D fitting performance estimation
% http://gpufit.readthedocs.io/en/latest/bindings.html#matlab

if isempty(which('gpufit.m'))
    error('Gpufit library not found in Matlab path.');
end

assert(gpufit_cuda_available(), 'CUDA not available');
rng(0);

%% parameter
n_points = 20; % number of data points
true_parameters = single([30, 9, 8 / 2.35, 1].'); % true model parameters
tolerance = 1e-4;
max_n_iterations = 10;
model_id = ModelID.GAUSS_1D;
estimator_id = EstimatorID.LSE;
n_fits = 5e7;

%% generate data and weights
x = (0 : n_points-1).';
p = true_parameters;
m = p(1) * exp(-(x-p(2)).^2/(2*p(3)^2))+p(4);

data = repmat(m, 1, n_fits);
data = data + randn(size(data)) .* sqrt(data); 
weights = 1 ./ max(1, data);

%% generate initial parameters
initial_parameters = repmat(true_parameters, 1, n_fits);
initial_parameters(1, :) = initial_parameters(1, :)  + rand(1, n_fits) * 3 - 1.5;
initial_parameters(2, :) = initial_parameters(2, :)  + rand(1, n_fits) * 3 - 1.5;
initial_parameters(3, :) = initial_parameters(3, :)  + rand(1, n_fits) * 1 - 0.5;
initial_parameters(4, :) = initial_parameters(4, :)  + rand(1, n_fits) * 1 - 0.5;

%% run Gpufit
[parameters, states, chi_squares, n_iterations, time] = gpufit(data, weights, ...
        model_id, initial_parameters, tolerance, max_n_iterations, [], estimator_id);

for i = 1 : 4
    fprintf('p%d: %.2f est: %.2f (%.2f)\n', i, p(i), mean(parameters(i, :)), std(parameters(i, :), [], 2));
end

fprintf('evaluation of %d million fits took %.1fs (Mfits/s = %.2f)\n', n_fits / 1e6, time, n_fits / 1e6 / time);

end