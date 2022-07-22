function splinefit_3d_multichannel()
% Spline fit 3D 4 channel example. Can for example be used in 4Pi-Storm microscopy.
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
size_x = 19;
size_y = 25;
size_z = 50;
n_channels = 4;

% tolerances
tolerance = 1e-30;
max_n_iterations = 200;
estimator_id = EstimatorID.LSE;

% derived values
x = single(0 : size_x - 1);
y = single(0 : size_y - 1);
z = single(0 : size_z - 1);

%% generate PSF
psf_parameters = single([100, (size_x-1)/2+1, (size_y-1)/2-1, (size_z-1)/2, 1, 10]);
psf = calculate_psf(size_x, size_y, size_z, psf_parameters, n_channels);
z_slice_index = 25;

%% add noise
rng(0);
snr = 5;
amplitude = psf_parameters(1);
noise_std_dev = amplitude ./ (snr * log(10.0));
noise = noise_std_dev * randn(size(psf));
noisy_psf = psf + noise;

%% calculate PSF template
psf_normalized = (psf - psf_parameters(6)) / psf_parameters(1);

%% calculate spline coefficients of the PSF template
coefficients = zeros([64 size(psf(:,:,:,1))-1 n_channels],'single');
for ch=1:n_channels; coefficients(:,:,:,:,ch) = spline_coefficients(psf_normalized(:,:,:,ch)); end
n_intervals = size(psf_normalized) - 1;
n_intervals = n_intervals(1:3);

%% set user info
user_info = [...
    n_channels,...
    size_x, size_y, 1,...
    n_intervals,...
    reshape(coefficients, 1, numel(coefficients))];

%% true fit parameters
true_fit_parameters(1,:) = psf_parameters(1); % amplitude
true_fit_parameters(2,:) = 0;                 % center x shift
true_fit_parameters(3,:) = 0;                 % center y shift
true_fit_parameters(4,:) = 1 - z_slice_index;     % z index
true_fit_parameters(5,:) = psf_parameters(6); % offset

%% set initial fit parameters
pos_shift_x = -0.9;
pos_shift_y = 0.7;
pos_shift_z = 10;
amp_shift = -20;
off_shift = 5;

spline_fit_initial_parameters = true_fit_parameters + [amp_shift; pos_shift_x; pos_shift_y; pos_shift_z; off_shift];

%% reshape data
linear_psf = reshape(noisy_psf(:,:,z_slice_index,:), size_x * size_y * n_channels, 1);

%% call to gpufit with spline fit
[parameters_gpufit_spline, states_gpufit_spline, chi_squares_gpufit_spline, n_iterations_gpufit_spline, time_gpufit_spline]...
    = gpufit(linear_psf, [], ModelID.SPLINE_3D_MULTICHANNEL, spline_fit_initial_parameters, tolerance, max_n_iterations, [], estimator_id, user_info);

% check if gpufit succeeded
assert(all(states_gpufit_spline == 0));

%% get data to plot
merge_channels = @(x) [x(:, :, 1), x(:, :, 2); x(:, :, 3), x(:, :, 4)];

psf_4ch = merge_channels(psf(:,:,z_slice_index,:));

noisy_psf_4ch = merge_channels(noisy_psf(:,:,z_slice_index,:));

initial_spline_fit = spline_values_3d_multichannel(coefficients, n_intervals, n_channels, spline_fit_initial_parameters);
initial_spline_fit_4ch = merge_channels(initial_spline_fit);

final_spline_gpufit = spline_values_3d_multichannel(coefficients, n_intervals, n_channels, parameters_gpufit_spline);
final_spline_gpufit_4ch = merge_channels(final_spline_gpufit);

%% plot
figure(5);
min_value = min([psf(:); noisy_psf(:); initial_spline_fit(:); final_spline_gpufit(:)]);
max_value = max([psf(:); noisy_psf(:); initial_spline_fit(:); final_spline_gpufit(:)]);
clims = [min_value max_value];

subplot(2,2,1); imagesc(x, y', psf_4ch, clims); title(sprintf('psf z=%.2f', true_fit_parameters(4))); axis image;
subplot(2,2,2); imagesc(x, y', noisy_psf_4ch, clims); title(sprintf('noisy psf z=%.2f', true_fit_parameters(4))); axis image;
subplot(2,2,3); imagesc(x, y', initial_spline_fit_4ch, clims); title(sprintf('initial spline fit z=%.2f', spline_fit_initial_parameters(4))); axis image;
subplot(2,2,4); imagesc(x, y', final_spline_gpufit_4ch, clims); title(sprintf('final gpufit z=%.2f', parameters_gpufit_spline(4))); axis image;
colormap('hot');

end


%% calculate psf
function f = calculate_psf(size_x, size_y, size_z, p, n_channels)

s_max = p(5) * 5;
s_min = p(5) / 5;

sx = linspace(s_max, s_min, size_z);
sy = linspace(s_min, s_max, size_z);
sz = 8;

delta_s = sx(1) - sx(2);
sx = [sx ,s_min - 1*delta_s, s_min - 2*delta_s, s_min - 3*delta_s];
sy = [sy ,s_max + 1*delta_s, s_max + 2*delta_s, s_max + 3*delta_s];

x = linspace(0,pi,size_z);
a = zeros(numel(x), 4);
a(:,1) = sin(x + pi * 0/4) * 0.5 + 0.5;
a(:,2) = sin(x + pi * 1/4) * 0.5 + 0.5;
a(:,3) = sin(x + pi * 2/4) * 0.5 + 0.5;
a(:,4) = sin(x + pi * 3/4) * 0.5 + 0.5;

f = zeros(size_x, size_y, size_z, n_channels, 'single');

for ch = 0 : n_channels-1
    for xi = 0 : size_x-1
        for yi = 0 : size_y-1
            for zi = 0 : size_z-1
                
                arg_x = exp(-1/2*((xi-p(2))/sx(zi+1+ch)).^2);
                arg_y = exp(-1/2*((yi-p(3))/sy(zi+1+ch)).^2);
                arg_z = exp(-1/2*((zi+ch-p(4))/sz     ).^2);
                
                f(xi+1,yi+1,zi+1,ch+1) = a(zi+1,ch+1) * p(1) * arg_x * arg_y * arg_z + p(6);
                
            end
        end
    end
end

end

function f = spline_values_3d_multichannel(coefficients, n_intervals, n_channels, p)

coefficients = reshape(coefficients, 64, n_intervals(1), n_intervals(2), n_intervals(3), n_channels);
x = (0:n_intervals(1))-p(2);
y = (0:n_intervals(2))-p(3);
z = -p(4);
f = p(1) * spline_values(coefficients, x, y, z) + p(5);

end