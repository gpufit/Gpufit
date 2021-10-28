function splinefit_3d()
% spline fit 3D
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
size_x = 18;
size_y = 13;
size_z = 100;

% tolerances
tolerance = 1e-30;
max_n_iterations = 100;
estimator_id = EstimatorID.LSE;

% derived values
x = single(0 : size_x - 1);
y = single(0 : size_y - 1);
z = single(0 : size_z - 1);

%% generate PSF
psf_parameters = single([100, (size_x-1)/2, (size_y-1)/2, (size_z-1)/2+1, 1, 10]);
psf = calculate_psf(x, y, z, psf_parameters);
z_slice_index = 61;

%% add noise
snr = 10;
amplitude = psf_parameters(1);
noise_std_dev = amplitude ./ (snr * log(10.0));
noise = noise_std_dev * randn(size(psf));
noisy_psf = psf + noise;

%% calculate PSF template
psf_template = (psf - psf_parameters(6)) / psf_parameters(1);

%% calculate spline coefficients of the PSF template
coefficients = spline_coefficients(psf_template);
n_intervals = size(psf_template) - 1;
coefficients = reshape(coefficients, 64, n_intervals(1), n_intervals(2), n_intervals(3));

%% set user info
user_info = [...
    size_x; size_y; 1;...
    n_intervals';...
    coefficients(:)];

%% true fit parameters
true_fit_parameters(1,:) = psf_parameters(1); % amplitude
true_fit_parameters(2,:) = 0;                 % center x shift
true_fit_parameters(3,:) = 0;                 % center y shift
true_fit_parameters(4,:) = 1 - z_slice_index;     % center z shift
true_fit_parameters(5,:) = psf_parameters(6); % offset

%% set initial fit parameters
pos_shift_x = 1.2;
pos_shift_y = -0.3;
pos_shift_z = -20;
amp_shift = -15;
off_shift = 2;

spline_fit_initial_parameters = true_fit_parameters + [amp_shift; pos_shift_x; pos_shift_y; pos_shift_z; off_shift];

%% reshape data
linear_psf = reshape(noisy_psf(:,:,z_slice_index), numel(noisy_psf(:,:,z_slice_index)), 1);

%% call to gpufit with spline fit
[parameters_gpufit_spline, states_gpufit_spline, chi_squares_gpufit_spline, n_iterations_gpufit_spline, time_gpufit_spline]...
    = gpufit(linear_psf, [], ModelID.SPLINE_3D, spline_fit_initial_parameters, tolerance, max_n_iterations, [], estimator_id, user_info);
assert(all(states_gpufit_spline == 0));

%% get data to plot
% spline with true parameters
a_true = true_fit_parameters(1);
x_true = x-true_fit_parameters(2);
y_true = y-true_fit_parameters(3);
z_true = -true_fit_parameters(4);
b_true = true_fit_parameters(5);
true_spline_fit = a_true * spline_values(coefficients, x_true, y_true, z_true) + b_true;

% spline with initial fit parameters
a_init = spline_fit_initial_parameters(1);
x_init = x-spline_fit_initial_parameters(2);
y_init = y-spline_fit_initial_parameters(3);
z_init = -spline_fit_initial_parameters(4);
b_init = spline_fit_initial_parameters(5);
initial_spline_fit = a_init * spline_values(coefficients, x_init, y_init, z_init) + b_init;

% spline with fit parameters
a_fit = parameters_gpufit_spline(1);
x_fit = x-parameters_gpufit_spline(2);
y_fit = y-parameters_gpufit_spline(3);
z_fit = -parameters_gpufit_spline(4);
b_fit = parameters_gpufit_spline(5);
final_spline_gpufit = a_fit * spline_values(coefficients, x_fit, y_fit, z_fit) + b_fit;

%% figure
figure(2);
current_slice = noisy_psf(:,:,z_slice_index);
min_noisy_psf = min(current_slice(:));
max_noisy_psf = max(current_slice(:));
min_temp = min(min([initial_spline_fit final_spline_gpufit]));
max_temp = max(max([initial_spline_fit final_spline_gpufit]));
min_value = min(min_noisy_psf, min_temp);
max_value = max(max_noisy_psf, max_temp);
clims = [min_value max_value];
subplot(2,2,1); imagesc(x, y', true_spline_fit.', clims);     title(sprintf('true spline z=%.2f', true_fit_parameters(4))); axis image;
subplot(2,2,2); imagesc(x, y', current_slice.', clims);       title(sprintf('noisy psf z=%.2f', true_fit_parameters(4))); axis image;
subplot(2,2,3); imagesc(x, y', initial_spline_fit.', clims);  title(sprintf('initial spline fit z=%.2f', spline_fit_initial_parameters(4))); axis image;
subplot(2,2,4); imagesc(x, y', final_spline_gpufit.', clims); title(sprintf('final spline gpufit z=%.2f', parameters_gpufit_spline(4))); axis image;
colormap('hot');

end

function f = calculate_psf(x, y, z, p)

size_x = numel(x);
size_y = numel(y);
size_z = numel(z);

s_max = p(5) * 5;
s_min = p(5) / 5;

sx = linspace(s_max, s_min, numel(z));
sy = linspace(s_min, s_max, numel(z));
sz = p(5) * 10;

f = zeros(size_x, size_y, size_z, 'single');

for x = 0 : size_x-1
    for y = 0 : size_y-1
        for z = 0 : size_z-1
            
            arg_x = exp(-1/2*((x+1-p(2))/sx(z+1)).^2);
            arg_y = exp(-1/2*((y-p(3))/sy(z+1)).^2);
            arg_z = exp(-1/2*((z-p(4))/sz     ).^2);
            
            f(x+1,y+1,z+1) = p(1) * arg_x * arg_y * arg_z + p(6);
            
        end
    end
end

end
