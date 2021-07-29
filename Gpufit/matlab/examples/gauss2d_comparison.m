function gauss2d_comparison()
% Example of the Matlab binding of the Gpufit library implementing
% Levenberg Marquardt curve fitting in CUDA
% https://github.com/gpufit/Gpufit
%
% Multiple fits of a 2D Gaussian peak function with Poisson distributed noise
% compared to a generic Matlab implementation using fminunc and supplying
% the gradient by the user (uses quasi-newton as algorithm)
% http://gpufit.readthedocs.io/en/latest/bindings.html#matlab

if isempty(which('gpufit.m'))
    error('Gpufit library not found in Matlab path.');
end

assert(gpufit_cuda_available(), 'CUDA not available');

%% number of fits and fit points
number_fits = 1e3;
size_x = 20;
number_parameters = 5;

%% set input arguments

% true parameters
true_parameters = single([10, 9.5, 9.5, 3, 10]);

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
data = poissrnd(data);

% tolerance
tolerance = 1e-4;

% maximum number of iterations
max_n_iterations = 20;

% estimator id
estimator_id = EstimatorID.MLE;

% model ID
model_id = ModelID.GAUSS_2D; % Gaussian peak in 2D

%% run Gpufit
fprintf('run Gpufit\n');
[gf_parameters, gf_states, gf_chi_squares, gf_n_iterations, time] = gpufit(data, [], ...
    model_id, initial_parameters, tolerance, max_n_iterations, [], estimator_id, []);

% display results
display_results('Gpufit', gf_parameters, gf_states, gf_chi_squares, gf_n_iterations, time, true_parameters);

% store parameters

%% run Matlab

% convert data and initial_parameters to double (otherwise causes an error
% in fminunc)
data = double(data);
initial_parameters = double(initial_parameters);
xi = double(x(:)');
yi = double(y(:)');

% set fit options
options = optimoptions(@fminunc,'Display', 'off', 'MaxIter', max_n_iterations, 'Algorithm', 'quasi-newton', 'TolFun', tolerance, 'GradObj', 'on', 'DerivativeCheck', 'off', 'Diagnostics', 'off');

% initialize output arrays
m_parameters = zeros(number_parameters, number_fits);
m_states = zeros(1, number_fits);
m_chi_squares = zeros(1, number_fits);
m_n_iterations = zeros(1, number_fits);

% loop over each fit
fprintf('\n')
progress = 0;
L = 50; % length of progressbar
tic;
for i = 1 : number_fits
    
    % get data and initial_parameters
    d = data(:, i)';
    p0 = initial_parameters(:, i);
    
    % define minimizer function (give grid and data as implicit parameters)
    fun = @(p) minimizer(p, xi, yi, d);
    
    % call to fminunc
    [p, fval, exitflag, output] = fminunc(fun, p0, options);
    
    % copy to output
    m_parameters(:, i) = p;
    m_chi_squares(i) = fval;
    m_states(i) = exitflag - 1;
    m_n_iterations(i) = output.iterations;
    
    progress = progress + 1;
    if progress >= number_fits / L
        progress = 0;
        fprintf('|');
    end
end
time = toc;
fprintf(repmat('\b', [1, L]));

% display results
display_results('Matlab (one CPU kernel)', m_parameters, m_states, m_chi_squares, m_n_iterations, time, true_parameters);

end

function [f, g] = minimizer(p, xi, yi, d)
% calls the model with the current parameters, then the likelihood function
% and returns value and derivatives of the likelihood function
%
% p - current parameters
% xi, yi - grid positions
% d - current data

if nargout > 1
    [m, mg] = gaussian_2d_with_gradient(xi, yi, p);
    [f, g] = poisson_likelihood(m, mg, d);
else
    m = gaussian_2d(xi, yi, p);
    f = poisson_likelihood(m, [], d);
end

end

function [f, g] = poisson_likelihood(m, mg, d)
% Calculates value and derivatives of the poisson likelihood function for
% given model and model derivatives

h = d > 0;
f = 2 * (sum(m-d) - sum(d(h) .* log(m(h) ./ d(h))));

if nargout > 1 % gradient required
    h = 2 * (1 - d ./ max(m, 1e-6));
    h = repmat(h, [size(mg, 1), 1]);
    g = h .* mg;
    g = sum(g, 2);
end

end


function display_results(name, parameters, ~, chi_squares, n_iterations, time, true_parameters)
% displaying results

fprintf('*%s*\n', name);
number_parameters = size(parameters, 1);
number_fits = size(parameters, 2);

% print summary
fprintf('\nnumber of fits: %d\n', number_fits);
fprintf('mean chi-square: %6.2f\n', mean(chi_squares));
fprintf('mean iterations: %6.2f\n', mean(n_iterations));
fprintf('time:            %6.2f s\n', time);
fprintf('fits per second: %.0f\n', number_fits / time);

% mean and std of fitted parameters
parameters_mean = mean(parameters, 2);
parameters_std  = std(parameters, [], 2);
fprintf('\nparameters of 2D Gaussian peak\n');
for i = 1 : number_parameters
    fprintf('p%d true %6.2f mean %6.2f std %6.2f\n', i, true_parameters(i), parameters_mean(i), parameters_std(i));
end

end

function f = gaussian_2d(x, y, p)
% Generates a 2D Gaussian peak.
% http://gpufit.readthedocs.io/en/latest/api.html#gauss-2d
%
% x,y - x and y grid position values
% p - parameters (amplitude, x,y center position, width, offset)

f = p(1) * exp(-((x - p(2)).^2 + (y - p(3)).^2) / (2 * p(4)^2)) + p(5);

end

function [f, g] = gaussian_2d_with_gradient(x, y, p)
% Computes the gradient for a 2D Gaussian peak with respect to parameters.

dx = x - p(2);
dy = y - p(3);
p42 = p(4)^2;
arg = (dx.^2 + dy.^2) / p42;
exp_f = exp(-0.5 * arg);
p1_exp_f = p(1) * exp_f;

f = p1_exp_f + p(5);

g1 = exp_f;
g2 = p1_exp_f .* dx / p42;
g3 = p1_exp_f .* dy / p42;
g4 = p1_exp_f .* arg / p(4);
g5 = ones(size(x));
g = [g1; g2; g3; g4; g5];

end
