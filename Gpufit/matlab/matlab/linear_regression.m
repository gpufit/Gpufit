function linear_regression()
% Example of the Matlab binding of the Gpufit library implementing
% Levenberg Marquardt curve fitting in CUDA
% https://github.com/gpufit/Gpufit
%
% 1D linear regression with custom x values given as user information
% http://gpufit.readthedocs.io/en/latest/bindings.html#matlab

% x values for the 1D linear regression model must be single
x = single(0:10);
parameters = single([0;1]);
y = single(parameters(1)+parameters(2)*x');

% fit parameter
tolerance = 1e-9;
max_n_iterations = 1000;
estimator_id = EstimatorID.LSE;
model_id = ModelID.LINEAR_1D;
initial_parameters = single([0;1]); % should result in correct parameters with only 1 iteration

% with user info
[parameters, states, chi_squares, n_iterations, time] = gpufit(y, [], ...
    model_id, initial_parameters, tolerance, max_n_iterations, [], estimator_id, x);
fprintf('first fit with user info: fitted parameters = [%.2f, %.2f]\n', parameters);

% without user info
[parameters, states, chi_squares, n_iterations, time] = gpufit(y, [], ...
    model_id, initial_parameters, tolerance, max_n_iterations, [], estimator_id);
fprintf('first fit without user info: fitted parameters = [%.2f, %.2f]\n', parameters);

% now more meaningful
x = single([1,3,4,5,5.5]);
y = single(parameters(1)+parameters(2)*x');
[parameters, states, chi_squares, n_iterations, time] = gpufit(y, [], ...
    model_id, initial_parameters, tolerance, max_n_iterations, [], estimator_id, x);
fprintf('second fit with user info: fitted parameters = [%.2f, %.2f]\n', parameters);

end