function gauss2d_constrained()
% Example of the Matlab binding of the Gpufit library implementing
% Levenberg Marquardt curve fitting in CUDA
% https://github.com/gpufit/Gpufit
%
% Multiple fits of a 2D symmetric Gaussian peak function
% Comparison of unconstrained vs. constrained fit
% http://gpufit.readthedocs.io/en/latest/bindings.html#matlab

if isempty(which('gpufit.m'))
    error('Gpufit library not found in Matlab path.');
end

assert(gpufit_cuda_available(), 'CUDA not available');

%% number of fits and fit points
number_fits = 2e5;
size_x = 20;
number_parameters = 5;

%% set input arguments

% true parameters (amplitude, center, center, sigma, background)
true_parameters = single([2, 9.5, 9.5, 3, 10]);

% initialize random number generator
rng(0);

% initial parameters (randomized)
initial_parameters = repmat(single(true_parameters'), [1, number_fits]);
% randomize relative to width for positions
initial_parameters([2,3], :) = initial_parameters([2,3], :) + true_parameters(4) * (-0.2 + 0.4 * rand(2, number_fits));
% randomize relative for other parameters
initial_parameters([1,4,5], :) = initial_parameters([1,4,5], :) .* (0.8 + 0.4 * rand(3, number_fits));

% generate x and y values
g = single(0 : size_x - 1);
[x, y] = ndgrid(g, g);

% generate data with Poisson noise
data = gaussian_2d(x, y, true_parameters);
data = repmat(data(:), [1, number_fits]);
data = data + randn(size(data)).* sqrt(data);

% tolerance
tolerance = 1e-3;

% maximum number of iterations
max_n_iterations = 20;

% estimator id
estimator_id = EstimatorID.LSE;

% model ID
model_id = ModelID.GAUSS_2D;

%% run unconstrained Gpufit
[parameters, states, chi_squares, n_iterations, time] = gpufit(data, [], ...
    model_id, initial_parameters, tolerance, max_n_iterations, [], estimator_id, []);

%% displaying results
display_results('unconstrained 2D Gaussian peak', model_id, number_fits, number_parameters, size_x, time, true_parameters, parameters, states, chi_squares, n_iterations);

%% set constraints
constraints = zeros([2*number_parameters, number_fits], 'single');
constraints(7, :) = 2.9;
constraints(8, :) = 3.1;
constraint_types = int32([ConstraintType.LOWER, ConstraintType.FREE, ConstraintType.FREE, ConstraintType.LOWER_UPPER, ConstraintType.LOWER]);

%% run constrained Gpufit
[parameters, states, chi_squares, n_iterations, time] = gpufit_constrained(data, [], ...
    model_id, initial_parameters, constraints, constraint_types, tolerance, max_n_iterations, [], estimator_id, []);

%% displaying results
display_results('constrained 2D Gaussian peak', model_id, number_fits, number_parameters, size_x, time, true_parameters, parameters, states, chi_squares, n_iterations);

% gist: if I know the width of a peak really well before-hand, I can estimate its position better

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