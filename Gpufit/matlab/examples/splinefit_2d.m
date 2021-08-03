function splinefit_2d()
% spline fit 2D rectangular
%
% Requires Gpuspline (https://github.com/gpufit/Gpuspline) additionally

if isempty(which('gpufit.m'))
    error('Gpufit library not found in Matlab path.');
end
if isempty(which('spline_coefficients.m'))
    error('Gpuspline library not found in Matlab path.');
end

% initialize random number generator
rng(0);

% data size
size_x = 12;
size_y = 14;

% tolerances
tolerance = 1e-30;
max_n_iterations = 100;
estimator_id = EstimatorID.LSE;

% derived values
x = single(0 : size_x - 1)';
y = single(0 : size_y - 1);

SF = 2; % scaling factor
x_spline = single(0 : SF * (size_x - 1))';
y_spline = single(0 : SF * (size_y - 1));

x2 = x_spline / SF;
y2 = y_spline / SF;

%% generate PSF
psf_parameters = single([100, (size_x-1)/2, (size_y-1)/2, 1, 10]);

%% calculate PSF template
% calculate PSF on fine grid
psf = calculate_psf(x2, y2, psf_parameters);
% PSF template (normalized to minimum = 0 and maximum = 1)
psf_normalized = (psf - psf_parameters(5)) / psf_parameters(1);

%% calculate spline coefficients of the PSF template
coefficients = spline_coefficients(psf_normalized);
n_intervals = size(psf_normalized) - 1;

%% add noise to PSF data
snr = 50;
amplitude = psf_parameters(1);
noise_std_dev = amplitude ./ (snr * log(10.0));
noise = noise_std_dev * randn(size(psf));
noisy_psf = psf + noise;

%% set user info
user_info = [...
    numel(x2); numel(y2);...
    n_intervals';...
    coefficients(:)];

%% true fit parameters
true_fit_parameters(1,:) = psf_parameters(1); % amplitude
true_fit_parameters(2,:) = 0;                 % center x shift
true_fit_parameters(3,:) = 0;                 % center y shift
true_fit_parameters(4,:) = psf_parameters(5); % offset

%% set initial fit parameters
pos_shift_x = 1.;
pos_shift_y = -2.1;
amp_shift = 20;
off_shift = 13;

spline_fit_initial_parameters = true_fit_parameters + [amp_shift; pos_shift_x * SF; pos_shift_y * SF; off_shift];

gauss_fit_initial_parameters(1,:) = (psf_parameters(1) + amp_shift);
gauss_fit_initial_parameters(2,:) = (psf_parameters(2) + pos_shift_x) * SF;
gauss_fit_initial_parameters(3,:) = (psf_parameters(3) + pos_shift_y) * SF;
gauss_fit_initial_parameters(4,:) = (psf_parameters(4) + 0)           * SF;
gauss_fit_initial_parameters(5,:) = (psf_parameters(5) + off_shift);

%% reshape data
linear_noisy_psf = reshape(noisy_psf, numel(noisy_psf), 1);
linear_noisy_psf_gauss = reshape(noisy_psf(:, 3:end-2), 23^2, 1); % this will only work if size_x,y aren't changed above (try to cut off data symmetrical to make a square data array for gauss fit)

%% call to cpufit with spline fit
[parameters_cpufit_spline, states_cpufit_spline, chi_squares_cpufit_spline, n_iterations_cpufit_spline, time_cpufit_spline]...
    = cpufit(linear_noisy_psf, [], ModelID.SPLINE_2D, spline_fit_initial_parameters, tolerance, max_n_iterations, [], estimator_id, user_info);
assert(all(states_cpufit_spline == 0));

%% call to gpufit with spline fit
[parameters_gpufit_spline, states_gpufit_spline, chi_squares_gpufit_spline, n_iterations_gpufit_spline, time_gpufit_spline]...
    = gpufit(linear_noisy_psf, [], ModelID.SPLINE_2D, spline_fit_initial_parameters, tolerance, max_n_iterations, [], estimator_id, user_info);
assert(all(states_gpufit_spline == 0));

%% call to gpufit with gauss1d fit
[parameters_gpufit_gauss, states_gpufit_gauss, chi_squares_gpufit_gauss, n_iterations_gpufit_gauss, time_gpufit_gauss]...
    = gpufit(linear_noisy_psf_gauss, [], ModelID.GAUSS_2D, gauss_fit_initial_parameters, tolerance, max_n_iterations, [], estimator_id);
assert(all(states_gpufit_gauss == 0));

%% get data to plot
a = spline_fit_initial_parameters(1);
xx = x_spline - spline_fit_initial_parameters(2);
yy = y_spline - spline_fit_initial_parameters(3);
b = spline_fit_initial_parameters(4);
initial_spline_fit = a * spline_values(coefficients, xx, yy) + b;
a = parameters_cpufit_spline(1);
xx = x_spline - parameters_cpufit_spline(2);
yy = y_spline - parameters_cpufit_spline(3);
b = parameters_cpufit_spline(4);
final_spline_fit = a * spline_values(coefficients, xx, yy) + b;
initial_gauss_fit = gaussian_peak(x_spline, y_spline, gauss_fit_initial_parameters);
final_gauss_fit = gaussian_peak(x_spline, y_spline, parameters_gpufit_gauss);

%% make a figure of psf, psf template, initial and final gauss fit, initial and final spline fit
figure(1);
min_noisy_psf = min(min(noisy_psf));
max_noisy_psf = max(max(noisy_psf));
min_temp = min(min([initial_gauss_fit initial_spline_fit psf_normalized final_gauss_fit final_spline_fit]));
max_temp = max(max([initial_gauss_fit initial_spline_fit psf_normalized final_gauss_fit final_spline_fit]));
min_value = min(min_noisy_psf, min_temp);
max_value = max(max_noisy_psf, max_temp);
clims = [min_value max_value];
subplot(231); imagesc(x, y, noisy_psf, clims);            colorbar; title('noisy PSF'); axis square;
subplot(232); imagesc(x2, y2, initial_gauss_fit, clims);  colorbar; title('initial Gaussian fit'); axis square;
subplot(233); imagesc(x2, y2, initial_spline_fit, clims); colorbar; title('initial spline fit'); axis square;
subplot(234); imagesc(x2, y2, psf_normalized);            colorbar; title('PSF template'); axis square;
subplot(235); imagesc(x2, y2, final_gauss_fit, clims);    colorbar; title('final Gaussian fit'); axis square;
subplot(236); imagesc(x2, y2, final_spline_fit, clims);   colorbar; title('final spline fit'); axis square;
colormap('hot');

end

function z = calculate_psf(x, y, p)
% Test PSF consists of an elliptic 2D Gaussian

% p(1) - amplitude
% p(2) - center x
% p(3) - center y
% p(4) - Standard deviation
% p(5) - constant background
assert(nargin == 3);

sx = p(4) + 0.1;
sy = p(4) - 0.2;

arg_ex = exp(-1/2*((x-p(2))/sx).^2-1/2*((y-p(3))/sy).^2);

z = p(1) .* arg_ex + p(5); % scale with amplitude and background

end

function f = gaussian_peak(x, y, p)

% x,y - x and y grid position values
% p(1) - amplitude
% p(2) - center x
% p(3) - center y
% p(4) - Standard deviation
% p(5) - constant background

assert(nargin == 3);

f = p(1) * exp(-((x - p(2)).^2 + (y - p(3)).^2) / (2 * p(4)^2)) + p(5);

end