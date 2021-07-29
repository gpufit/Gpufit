function gauss2d_plot()
% Example of the Matlab binding of the Gpufit library implementing
% Levenberg Marquardt curve fitting in CUDA
% https://github.com/gpufit/Gpufit
%
% Multiple fits of a 2D Gaussian peak function with Poisson distributed noise
% repeated for a different total number of fits each time and plotting the
% results
% http://gpufit.readthedocs.io/en/latest/bindings.html#matlab

if isempty(which('gpufit.m'))
    error('Gpufit library not found in Matlab path.');
end

assert(gpufit_cuda_available(), 'CUDA not available');

%% number of fit points
size_x = 5;
n_points = size_x * size_x;

%% set input arguments

% mean true parameters
mean_true_parameters = single([100, 3, 3, 1, 10]);

% average noise level
average_noise_level = 10;

% initialize random number generator
rng(0);

% tolerance
tolerance = 1e-4;

% max number of itetations
max_n_iterations = 10;

% model id
model_id = ModelID.GAUSS_2D;

%% loop over different number of fits
n_fits_all = round(logspace(2, 6, 20));

% generate x and y values
g = single(0 : size_x - 1);
[x, y] = ndgrid(g, g);

% loop
speed = zeros(length(n_fits_all), 1);
for i = 1:length(n_fits_all)
    n_fits = n_fits_all(i);
    
    % vary positions of 2D Gaussians peaks slightly
    test_parameters = repmat(mean_true_parameters', [1, n_fits]);
    test_parameters([2,3], :) = test_parameters([2,3], :) + mean_true_parameters(4) * (-0.2 + 0.4 * rand(2, n_fits));
    
    % generate data
    data = gaussians_2d(x, y, test_parameters);
    data = reshape(data, [n_points, n_fits]);
    
    % add noise
    data = data + average_noise_level * randn(size(data), 'single');
    
    % initial parameters (randomized)
    initial_parameters = repmat(mean_true_parameters', [1, n_fits]);
    % randomize relative to width for positions
    initial_parameters([2,3], :) = initial_parameters([2,3], :) + mean_true_parameters(4) * (-0.2 + 0.4 * rand(2, n_fits));
    % randomize relative for other parameters
    initial_parameters([1,4,5], :) = initial_parameters([1,4,5], :) .* (0.8 + 0.4 * rand(3, n_fits));
    
    % run Gpufit
    [parameters, states, chi_squares, n_iterations, time] = gpufit(data, [], ...
        model_id, initial_parameters, tolerance, max_n_iterations);

    % analyze result
    converged = states == 0;
    speed(i) = n_fits / time;
    precision_x0 = std(parameters(2, converged) - test_parameters(2, converged));
    
    % display result
    fprintf('     iterations: %.2f | time: %.3f s | speed: %8.0f fits/s\n', ...
        mean(n_iterations(converged)), time, speed(i));
end

%% plot
figure();
semilogx(n_fits_all, speed, 'bo-')
xlabel('number of fits per function call')
ylabel('fits per second')
legend('Gpufit', 'Location', 'NorthWest')
grid on;
xlim(n_fits_all([1,end]));

end

function g = gaussians_2d(x, y, p)
% Generates many 2D Gaussians peaks for a given set of parameters

n_fits = size(p, 2);
msg = sprintf('generating %d fits ', n_fits);
fprintf(msg);

g = zeros([size(x), n_fits], 'single');

progress = 0;
L = 50; % length of progressbar
l = 0;
for i = 1 : n_fits
    
    pi = p(:, i);
    g(:, :, i) = pi(1) * exp(-((x - pi(2)).^2 + (y - pi(3)).^2) / (2 * pi(4)^2)) + pi(5);
    
    progress = progress + 1;
    if progress >= n_fits / L
        progress = 0;
        fprintf('|');
        l = l + 1;
    end
end
fprintf(repmat('\b', [1, length(msg) + l]));
fprintf('%7d fits', n_fits);

end
