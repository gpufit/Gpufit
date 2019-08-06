"""
    Example of the Python binding of the Gpufit library which implements
    Levenberg Marquardt curve fitting in CUDA
    https://github.com/gpufit/Gpufit


"""

import numpy as np
import matplotlib.pyplot as plt
import pygpufit.gpufit as gf
import cmath


if __name__ == '__main__':

    # cuda available checks
    print('CUDA available: {}'.format(gf.cuda_available()))
    if not gf.cuda_available():
        raise RuntimeError(gf.get_last_error())
    print('CUDA versions runtime: {}, driver: {}'.format(*gf.get_cuda_version()))
    
    #Signal to Noise Ratio
    snr = 5000

    # number of fits and fit points
    number_fits = 100
    number_points = 6
    number_parameters = 3
    
    # Echo Times
    TEn = np.array((1.23, 2.48, 3.65, 4.84, 6.03, 7.22), dtype=np.float32)
    
    # User_info set up   
    #user_info = TEn

    # set input arguments
    
    # true parameters
    true_parameters = np.array((210, 100, .06), dtype=np.float32)
    
    sigma = (true_parameters[0] + true_parameters[1]) / snr

    # initialize random number generator
    np.random.seed(0)
    
    initial_parameters = []
    for n in range (0, number_fits):
        #initial parameters
        initial_parameters.append(true_parameters[0] * (0.8 + 0.4 * np.random.uniform()))
        initial_parameters.append(true_parameters[1] * (0.8 + 0.4 * np.random.uniform()))
        initial_parameters.append(true_parameters[2] * (0.8 + 0.4 * np.random.uniform()))
        
        print("parameter 0 + noise ", initial_parameters[n * number_parameters + 0])
        print("parameter 1 + noise ", initial_parameters[n * number_parameters + 1])
        print("parameter 2 + noise ", initial_parameters[n * number_parameters + 2], "\n")
    
    # Creating complex number
    ppm_list=[-0.4764702, -0.4253742, -0.3883296, -0.332124, -0.3040212, -0.2375964, 0.0868632]
    i=complex(0,1)
    C_n = 0
    weight_list=[0.08,0.63,0.07,0.09,0.07,0.02,0.04]
      
    # generating data
    data = np.zeros((100,6))  
    for l in range(0, number_fits):
        for m in range(0, number_points):
            #k = l % number_points
        
            C_n = 0
            for n in range(0, 7):
                C_n += weight_list[n] * np.exp(i * 2 * cmath.pi * ppm_list[n] * TEn[m])
            y = abs((true_parameters[0] + C_n * true_parameters[1]) * np.exp(-1 * true_parameters[2] * TEn[m]))
            rician_noise = cmath.sqrt(np.random.normal(0,sigma) ** 2 + np.random.normal(0,sigma) ** 2)
            data[l,m] = (abs(y + rician_noise))
            print("y            ", y)
            print("rician noise ", rician_noise)
            print("y with noise ", data[l,m])
            print()  
    
    # use this to check how data is being collected
    print(data)
     
    # tolerance
    tolerance = 10e-15
    
    # maximum number of iterations
    max_number_iterations = 200

    # model ID
    model_id = gf.ModelID.LIVER_FAT_THREE

    # run Gpufit
    parameters, states, chi_squares, number_iterations, execution_time = gf.fit(data, None, model_id, initial_parameters, \
                                                        tolerance, max_number_iterations, None, None, TEn)

    # print fit results
    converged = states == 0
    print('*Gpufit*')

    # print summary
    print('\nmodel ID:        {}'.format(model_id))
    print('number of fits:  {}'.format(number_fits))
    print('fit size:        {} x {}'.format(size_x, size_x))
    print('mean chi_square: {:.2f}'.format(np.mean(chi_squares[converged])))
    print('iterations:      {:.2f}'.format(np.mean(number_iterations[converged])))
    print('time:            {:.2f} s'.format(execution_time))

    # get fit states
    number_converged = np.sum(converged)
    print('\nratio converged         {:6.2f} %'.format(number_converged / number_fits * 100))
    print('ratio max it. exceeded  {:6.2f} %'.format(np.sum(states == 1) / number_fits * 100))
    print('ratio singular hessian  {:6.2f} %'.format(np.sum(states == 2) / number_fits * 100))
    print('ratio neg curvature MLE {:6.2f} %'.format(np.sum(states == 3) / number_fits * 100))

    # mean, std of fitted parameters
    converged_parameters = parameters[converged, :]
    converged_parameters_mean = np.mean(converged_parameters, axis=0)
    converged_parameters_std = np.std(converged_parameters, axis=0)
    print('\nparameters of Liver Fat 3')
    for i in range(number_parameters):
        print('p{} true {:6.2f} mean {:6.2f} std {:6.2f}'.format(i, true_parameters[i], converged_parameters_mean[i], converged_parameters_std[i]))
