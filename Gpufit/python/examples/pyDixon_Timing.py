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

    Model_id_pick = input("select a method (3 or 4): ")
    if Model_id_pick == "3":
        # cuda available checks
        print('CUDA available: {}'.format(gf.cuda_available()))
        if not gf.cuda_available():
            raise RuntimeError(gf.get_last_error())
        print('CUDA versions runtime: {}, driver: {}'.format(*gf.get_cuda_version()))
        
        #Signal to Noise Ratio
        snr = 50
    
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
        true_parameters = np.array((210, 20, 0.1), dtype=np.float32)
        
        sigma = (true_parameters[0] + true_parameters[1]) / snr
    
        # initialize random number generator
        np.random.seed(0)
        
        initial_parameters = np.zeros((number_fits,number_parameters), dtype=np.float32)#np.tile(true_parameters, (number_fits, 1))
        for n in range (0, number_fits):
            #initial parameters
            initial_parameters[n,0] = true_parameters[0] * (0.5 + 1 * np.random.uniform())
            initial_parameters[n,1] = true_parameters[1] * (0.5 + 1 * np.random.uniform())
            initial_parameters[n,2] = true_parameters[2] * (0.5 + 1 * np.random.uniform())
            #print("initial guess 0 ", initial_parameters[n, 0])
            #print("initial guess 1 ", initial_parameters[n, 1])
            #print("initial guess 2 ", initial_parameters[n, 2], "\n")    
        #print(initial_parameters.shape)
        
        
        # Creating complex number
        ppm_list=[-0.4764702, -0.4253742, -0.3883296, -0.332124, -0.3040212, -0.2375964, 0.0868632]
        i=complex(0,1)
        C_n = 0
        weight_list=[0.08,0.63,0.07,0.09,0.07,0.02,0.04]
         
          
        # generating data
        data = np.zeros((100,6), dtype=np.float32)  
        for l in range(0, number_fits):
            for m in range(0, number_points):
                #k = l % number_points
            
                C_n = 0
                for n in range(0, 7):
                    C_n += weight_list[n] * np.exp(i * 2 * cmath.pi * ppm_list[n] * TEn[m])
                y = abs((true_parameters[0] + C_n * true_parameters[1]) * np.exp(-1 * true_parameters[2] * TEn[m]))
                rician_noise = cmath.sqrt(np.random.normal(0,sigma) ** 2 + np.random.normal(0,sigma) ** 2)
                data[l,m] = (abs(y + rician_noise))
                
                if l==number_fits-1:
                    print("y            ", y)
                    print("rician noise ", rician_noise)
                    print("y with noise ", data[l,m])
                    print()  
        
        # use this to check how data is being collected
        signal = np.mean(data[:,0])
        noise = np.std(data[:,0])
        SNR_actual = signal/noise
        #print(data)
        #print(data.shape)
        print("input SNR ",snr)
        print("actual SNR ", SNR_actual)
         
        # tolerance
        tolerance = 10e-3
        
        # maximum number of iterations
        max_number_iterations = 200
    
        # model ID
        model_id = gf.ModelID.LIVER_FAT_THREE
    
        # run Gpufit
        parameters, states, chi_squares, number_iterations, execution_time = gf.fit(data, None, model_id, initial_parameters, \
                                                            tolerance, max_number_iterations, None, None, TEn)
    
        # print fit results
        converged = states == 0
        print('\n\n*Gpufit*')
        
        # checking how parameters are made
        print('\n\nParameters\n', parameters)
    
        # print summary
        print('\nmodel ID:        {}'.format(model_id))
        print('number of fits:  {}'.format(number_fits))
        # print('fit size:        {} x {}'.format(size_x, size_x))
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
 
#########################################################################################################################
    
    if Model_id_pick == "4":
                # cuda available checks
        print('CUDA available: {}'.format(gf.cuda_available()))
        if not gf.cuda_available():
            raise RuntimeError(gf.get_last_error())
        print('CUDA versions runtime: {}, driver: {}'.format(*gf.get_cuda_version()))
        
        #Signal to Noise Ratio
        snr = 50
    
        # number of fits and fit points
        number_fits = 100
        number_points = 6
        number_parameters = 4
        
        # Echo Times
        TEn = np.array((1.23, 2.48, 3.65, 4.84, 6.03, 7.22), dtype=np.float32)
        
        # User_info set up   
        #user_info = TEn
    
        # set input arguments
        
        # true parameters
        true_parameters = np.array((290.21, 44.11, 0.05, 0.11), dtype=np.float32)
        
        sigma = (true_parameters[0] + true_parameters[1]) / snr
    
        # initialize random number generator
        np.random.seed(0)
        
        initial_parameters = np.zeros((number_fits,number_parameters), dtype=np.float32)#np.tile(true_parameters, (number_fits, 1))
        for n in range (0, number_fits):
            #initial parameters
            initial_parameters[n,0] = true_parameters[0] * (0.5 + 1 * np.random.uniform())
            initial_parameters[n,1] = true_parameters[1] * (0.5 + 1 * np.random.uniform())
            initial_parameters[n,2] = true_parameters[2] * (0.5 + 1 * np.random.uniform())
            initial_parameters[n,3] = true_parameters[3] * (0.5 + 1 * np.random.uniform())
            #print("initial guess 0 ", initial_parameters[n, 0])
            #print("initial guess 1 ", initial_parameters[n, 1])
            #print("initial guess 2 ", initial_parameters[n, 2], "\n")    
        #print(initial_parameters.shape)
        
        
        # Creating complex number
        ppm_list=[-0.4764702, -0.4253742, -0.3883296, -0.332124, -0.3040212, -0.2375964, 0.0868632]
        i=complex(0,1)
        C_n = 0
        weight_list=[0.08,0.63,0.07,0.09,0.07,0.02,0.04]
         
          
        # generating data
        data = np.zeros((100,6), dtype=np.float32)  
        for l in range(0, number_fits):
            for m in range(0, number_points):
                #k = l % number_points
            
                C_n = 0
                for n in range(0, 7):
                    C_n += weight_list[n] * np.exp(i * 2 * cmath.pi * ppm_list[n] * TEn[m])
                y = abs(true_parameters[0] * np.exp(-1 * true_parameters[2] * TEn[m]) + C_n * true_parameters[1]* np.exp(-1 * true_parameters[3] * TEn[m])) 
                rician_noise = cmath.sqrt(np.random.normal(0,sigma) ** 2 + np.random.normal(0,sigma) ** 2)
                data[l,m] = (abs(y + rician_noise))
                
                if l==number_fits-1:
                    print("y            ", y)
                    print("rician noise ", rician_noise)
                    print("y with noise ", data[l,m])
                    print()  
        
        # use this to check how data is being collected
        signal = np.mean(data[:,0])
        noise = np.std(data[:,0])
        SNR_actual = signal/noise
        #print(data)
        #print(data.shape)
        print("input SNR ",snr)
        print("actual SNR ", SNR_actual)
         
        # tolerance
        tolerance = 10e-5
        
        # maximum number of iterations
        max_number_iterations = 200
    
        # model ID
        model_id = gf.ModelID.LIVER_FAT_FOUR
    
        # run Gpufit
        parameters, states, chi_squares, number_iterations, execution_time = gf.fit(data, None, model_id, initial_parameters, \
                                                            tolerance, max_number_iterations, None, None, TEn)
        
    
        # print fit results
        converged = states == 0
        print('\n\n*Gpufit*')
        
        # checking how parameters are made
        print('\n\nParameters\n', parameters)
    
        # print summary
        print('\nmodel ID:        {}'.format(model_id))
        print('number of fits:  {}'.format(number_fits))
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
        print('\nparameters of Liver Fat 4')
        for i in range(number_parameters):
            print('p{} true {:6.2f} mean {:6.2f} std {:6.2f}'.format(i, true_parameters[i], converged_parameters_mean[i], converged_parameters_std[i]))
            
    else:
        print("ending program now")