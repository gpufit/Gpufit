function gauss2d_fixed_parameters()
% Example of the Matlab binding of the Gpufit library implementing
% Levenberg Marquardt curve fitting in CUDA
% https://github.com/gpufit/Gpufit
%
% Multiple fits of a 2D Gaussian peak function ones without fixed
% parameters and once with fixed parameters, measures performance
% Longest times is spent in preparing the data
% http://gpufit.readthedocs.io/en/latest/bindings.html#matlab

if isempty(which('gpufit.m'))
    error('Gpufit library not found in Matlab path.');
end

assert(gpufit_cuda_available(), 'CUDA not available');

% perform some 2D Gaussian peak fits with an assymetrical Gaussian peak
fit_gauss2d_elliptic();

% perform some 2D Gaussian peak fits with an asymmetrical, rotated Gaussian
% peak (but with fixed rotation angle)
fit_gauss2d_rotated_fixed();

end
function fit_gauss2d_elliptic()

%% number of fits and fit points
number_fits = 1e4;
size_x = 20;
number_parameters = 6;

%% set input arguments

% true parameters
true_parameters = single([20, 9.5, 9.5, 3, 4, 10]);

% initialize random number generator
rng(0);

% initial parameters (randomized)
initial_parameters = repmat(single(true_parameters'), [1, number_fits]);
% randomize relative to width for positions
initial_parameters(2, :) = initial_parameters(2, :) + true_parameters(4) * (-0.2 + 0.4 * rand(1, number_fits));
initial_parameters(3, :) = initial_parameters(3, :) + true_parameters(5) * (-0.2 + 0.4 * rand(1, number_fits));
% randomize relative for other parameters
initial_parameters([1,4,5,6], :) = initial_parameters([1,4,5,6], :) .* (0.8 + 0.4 * rand(4, number_fits));

% generate x and y values
g = single(0 : size_x - 1);
[x, y] = ndgrid(g, g);

% generate data with Poisson noise
data = gaussian_2d_rotated(x, y, [true_parameters, 0]);
data = repmat(data(:), [1, number_fits]);
data = poissrnd(data);

% tolerance
tolerance = 1e-3;

% maximum number of iterations
max_n_iterations = 20;

% estimator id
estimator_id = EstimatorID.MLE;

% model ID
model_id = ModelID.GAUSS_2D_ELLIPTIC;

%% run Gpufit
[parameters, states, chi_squares, n_iterations, time] = gpufit(data, [], ...
    model_id, initial_parameters, tolerance, max_n_iterations, [], estimator_id, []);

%% displaying results
display_results('2D elliptic Gaussian peak', model_id, number_fits, number_parameters, size_x, time, true_parameters, parameters, states, chi_squares, n_iterations);

end

function fit_gauss2d_rotated_fixed()
% with the rotation as fixed parameter

%% number of fits and fit points
number_fits = 1e4;
size_x = 20;
number_parameters = 7;

%% set input arguments

% true parameters
true_parameters = single([20, 9.5, 9.5, 3, 4, 10, 0]);

% initialize random number generator
rng(0);

% initial parameters (randomized)
initial_parameters = repmat(single(true_parameters'), [1, number_fits]);
% randomize relative to width for positions
initial_parameters(2, :) = initial_parameters(2, :) + true_parameters(4) * (-0.2 + 0.4 * rand(1, number_fits));
initial_parameters(3, :) = initial_parameters(3, :) + true_parameters(5) * (-0.2 + 0.4 * rand(1, number_fits));
% randomize relative for other parameters
initial_parameters([1,4,5,6], :) = initial_parameters([1,4,5,6], :) .* (0.8 + 0.4 * rand(4, number_fits));

% set fixed parameters
parameters_to_fit = ones([number_parameters, 1], 'int32');
parameters_to_fit(7) = 0;

% generate x and y values
g = single(0 : size_x - 1);
[x, y] = ndgrid(g, g);

% generate data with Poisson noise
data = gaussian_2d_rotated(x, y, true_parameters);
data = repmat(data(:), [1, number_fits]);
data = poissrnd(data);

% tolerance
tolerance = 1e-3;

% maximum number of iterations
max_n_iterations = 20;

% estimator id
estimator_id = EstimatorID.MLE;

% model ID
model_id = ModelID.GAUSS_2D_ROTATED;

%% run Gpufit
[parameters, states, chi_squares, n_iterations, time] = gpufit(data, [], ...
    model_id, initial_parameters, tolerance, max_n_iterations, parameters_to_fit, estimator_id, []);

%% displaying results
display_results('2D rotated Gaussian peak (fixed parameter)', model_id, number_fits, number_parameters, size_x, time, true_parameters, parameters, states, chi_squares, n_iterations);


end

function g = gaussian_2d(x, y, p)
% Generates a 2D Gaussian peak.
% http://gpufit.readthedocs.io/en/latest/api.html#gauss-2d
%
% x,y - x and y grid position values
% p - parameters (amplitude, x,y center position, width, offset)

g = p(1) * exp(-((x - p(2)).^2 + (y - p(3)).^2) / (2 * p(4)^2)) + p(5);

end

function g = gaussian_2d_rotated(x, y, p)
% Generates a 2D rotated elliptic Gaussian peak.
% http://gpufit.readthedocs.io/en/latest/api.html#d-rotated-elliptic-gaussian-peak
%
% x,y - x and y grid position values
% p - parameters (amplitude, x,y center position, width, offset)

% cosine and sine of rotation angle
cp = cos(p(7));
sp = sin(p(7));

% Gaussian peak with two axes
arga = (x - p(2)) .* cp - (y - p(3)) .* sp;
argb = (x - p(2)) .* sp + (y - p(3)) .* cp;
ex = exp(-0.5 .* (((arga / p(4)) .* (arga / p(4))) + ((argb / p(5)) .* (argb / p(5)))));
g = p(1) .* ex + p(6);

end

function display_results(name, model_id, number_fits, number_parameters, size_x, time, true_parameters, parameters, states, chi_squares, n_iterations)

%% displaying results
converged = states == 0;
fprintf('\nGpufit of %s\n', name);

% print summary
fprintf('\nmodel ID:        %d\n', model_id);
fprintf('number of fits:  %d\n', number_fits);
fprintf('fit size:        %d x %d\n', size_x, size_x);
fprintf('mean chi-square: %6.2f\n', mean(chi_squares(converged)));
fprintf('mean iterations: %6.2f\n', mean(n_iterations(converged)));
fprintf('time:            %6.2f s\n', time);

% get fit states
number_converged = sum(converged);
fprintf('\nratio converged         %6.2f %%\n', number_converged / number_fits * 100);
fprintf('ratio max it. exceeded  %6.2f %%\n', sum(states == 1) / number_fits * 100);
fprintf('ratio singular hessian  %6.2f %%\n', sum(states == 2) / number_fits * 100);
fprintf('ratio neg curvature MLE %6.2f %%\n', sum(states == 3) / number_fits * 100);

% mean and std of fitted parameters
converged_parameters = parameters(:, converged);
converged_parameters_mean = mean(converged_parameters, 2);
converged_parameters_std  = std(converged_parameters, [], 2);
fprintf('\nparameters of %s\n', name);
for i = 1 : number_parameters
    fprintf('p%d true %6.2f mean %6.2f std %6.2f\n', i, true_parameters(i), converged_parameters_mean(i), converged_parameters_std(i));
end

end