function simple()
% Example of the Matlab binding of the Gpufit library implementing
% Levenberg Marquardt curve fitting in CUDA
% https://github.com/gpufit/Gpufit
%
% Simple example demonstrating a minimal call of all needed parameters for the Matlab interface
% http://gpufit.readthedocs.io/en/latest/bindings.html#matlab

if isempty(which('gpufit.m'))
    error('Gpufit library not found in Matlab path.');
end

assert(gpufit_cuda_available(), 'CUDA not available');

% number of fits, number of points per fit
number_fits = 10;
number_points = 10;

% model ID and number of parameter
model_id = ModelID.GAUSS_1D;
number_parameter = 4;

% initial parameters
initial_parameters = zeros(number_parameter, number_fits, 'single');

% data
data = zeros(number_points, number_fits, 'single');

% run Gpufit
[parameters, states, chi_squares, number_iterations, execution_time] = gpufit(data, [], model_id, initial_parameters);

end