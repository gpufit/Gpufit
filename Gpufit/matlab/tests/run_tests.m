function run_tests()
% Runs all test scripts in this folder.
% See also: http://www.mathworks.com/help/matlab/script-based-unit-tests.html

suite = testsuite();
result = run(suite);
disp(result);
end