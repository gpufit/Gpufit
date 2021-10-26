function splinefit_1d()
% Spline fit 1D example
%
% 
%
% Requires Gpuspline (https://github.com/gpufit/Gpuspline) additionally

% there are two coordinate systems
% the spline coordinate system: 0:1:size-1
% the user coordinate system: whatever
% be careful to not mix them up (fit coordinates must be in the spline coordinate system!)

if isempty(which('gpufit.m'))
    error('Gpufit library not found in Matlab path.');
end
if isempty(which('spline_coefficients.m'))
    error('Gpuspline library not found in Matlab path.');
end

% initialize random number generator
rng(0);

% data size
size_x = 25;

% tolerances
tolerance = 1e-30;
max_n_iterations = 100;
estimator_id = EstimatorID.LSE;
model_id = ModelID.SPLINE_1D;

% derived values
x = (0 : size_x - 1)';
SF = 2; % scaling factor
x_spline = (0 : SF * (size_x - 1))';
x2 = single(x_spline / SF); % 0, 0.5, .., size_x -1

%% generate PSF (two Gaussians)
psf_parameters = single([100, (size_x-1)/2, 1.5, 10]);
psf = calculate_non_gaussian_psf(x2, psf_parameters);
psf_normalized = (psf - psf_parameters(4)) / psf_parameters(1);
psf_normalized = psf_normalized(1:end-5); % splines do not cover all the data

%% calculate spline coefficients of the PSF template
coefficients = spline_coefficients(psf_normalized);
n_intervals = size(psf_normalized, 1) - 1;

%% add noise to PSF data (no shift)
snr = 10;
amplitude = psf_parameters(1);
noise_std_dev = amplitude ./ (snr * log(10.0));
noise = noise_std_dev * randn(size(psf));
noisy_psf = psf + noise;

%% set user info
user_info = [n_intervals, reshape(coefficients,1,numel(coefficients))];

%% true fit parameters
true_fit_parameters = zeros(3, 1, 'single');
true_fit_parameters(1) = psf_parameters(1); % amplitude
true_fit_parameters(2) = 0;                 % center shift
true_fit_parameters(3) = psf_parameters(4); % offset

%% set initial fit parameters
pos_shift = -1.2 * SF;
amp_shift = 30;
off_shift = 20;

fit_initial_parameters = true_fit_parameters + [amp_shift; pos_shift; off_shift]; % small deviation

gauss_fit_initial_parameters = psf_parameters' + [amp_shift, pos_shift, 0, off_shift]';

%% call to gpufit with spline fit
[parameters_gpufit_spline, states_gpufit_spline, chi_squares_gpufit_spline, n_iterations_gpufit_spline, time_gpufit_spline]...
    = gpufit(noisy_psf, [], model_id, fit_initial_parameters, tolerance, max_n_iterations, [], estimator_id, user_info);
assert(all(states_gpufit_spline == 0));

%% call to cpufit with spline fit
[parameters_cpufit_spline, states_cpufit_spline, chi_squares_cpufit_spline, n_iterations_cpufit_spline, time_cpufit_spline]...
    = cpufit(noisy_psf, [], model_id, fit_initial_parameters, tolerance, max_n_iterations, [], estimator_id, user_info);
assert(all(states_cpufit_spline == 0));

%% call to gpufit with gauss1d fit
[parameters_gpufit_gauss, states_gpufit_gauss, chi_squares_gpufit_gauss, n_iterations_gpufit_gauss, time_gpufit_gauss]...
    = gpufit(noisy_psf, [], ModelID.GAUSS_1D, gauss_fit_initial_parameters, tolerance, max_n_iterations, [], estimator_id, x2);
assert(all(states_gpufit_gauss == 0));

%% get data to plot
a = true_fit_parameters(1);
x = x_spline-true_fit_parameters(2);
b = true_fit_parameters(3);
spline_model  = a * spline_values(coefficients, x) + b;
a = fit_initial_parameters(1);
x = x_spline-fit_initial_parameters(2);
b = fit_initial_parameters(3);
initial_spline_fit = a * spline_values(coefficients, x) + b;
a = parameters_gpufit_spline(1);
x = x_spline-parameters_gpufit_spline(2);
b = parameters_gpufit_spline(3);
final_fit_gpufit = a * spline_values(coefficients, x) + b;
a = parameters_cpufit_spline(1);
x = x_spline-parameters_cpufit_spline(2);
b = parameters_cpufit_spline(3);
final_fit_cpufit = a * spline_values(coefficients, x) + b;
gauss_final_fit = gaussian_peak_1d(x2, parameters_gpufit_gauss(:, 1));

%% make a figure of function values
figure(1);
hold off;
plot(x2, gauss_final_fit, '--.y', 'MarkerSize', 8, 'LineWidth', 2);
hold on;
plot(x2, noisy_psf(:, 1), 'ks', 'MarkerSize', 8, 'LineWidth', 2);
plot(x2, initial_spline_fit,'--sg', 'MarkerSize', 8, 'LineWidth', 2);
plot(x2, final_fit_cpufit,'-xc', 'MarkerSize', 8, 'LineWidth', 2);
plot(x2, final_fit_gpufit,'--+b', 'MarkerSize', 8, 'LineWidth', 2);
plot(x2, spline_model, ':r', 'MarkerSize', 8, 'LineWidth', 1.5);
ylim([0, max(initial_spline_fit)]);
legend(...
    'final gauss fit',...
    'noisy data',...
    'initial spline fit',...
    'final spline fit cpu',...
    'final spline fit gpu',...
    'true parameters spline');

end

function y = calculate_non_gaussian_psf(x, p)
% Test PSF consists of sum of two Gaussians with different centers and
% width
% TODO define center of PSF (for now we assume that center is at middle)

% p(1) - amplitude first Gaussian
% p(2) - center (both Gaussians shifted a bit towards left and right
% p(3) - Standard deviation (second Gaussian is a bit wider)
% p(4) - constant background

assert(nargin == 2);

distance = x(end) / 12;
arg1 = ((x - p(2)-distance).^2) / (2 * p(3)^2);
arg2 = ((x - p(2)+distance).^2) / (2 * p(3)^2);
ex = exp(-arg1) + 0.5 * exp(-arg2);

y = ex / max(ex); % normalized to [0,1]
y = p(1) * y + p(4); % scale with amplitude and background

end