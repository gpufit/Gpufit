package com.github.gpufit.examples;

import com.github.gpufit.*;

/**
 * Example of the Java binding of the Gpufit library which implements
 * Levenberg Marquardt curve fitting in CUDA
 * https://github.com/gpufit/Gpufit
 *
 * Simple example demonstrating a minimal call of all needed parameters for the Java interface
 * http://gpufit.readthedocs.io/en/latest/bindings.html#java
 *
 * Note: The path of compiled Gpufit and GpufitJNI libraries must be in the Java library path.
 */
public class SimpleExample {

    public static void main(String [] args)
    {
        System.out.println(String.format("Gpufit version: %s", Gpufit.VERSION));

        // print general CUDA information
        System.out.println(String.format("CUDA available: %b", Gpufit.isCudaAvailable()));
        CudaVersion cudaVersion = Gpufit.getCudaVersion();
        System.out.println(String.format("CUDA versions runtime: %s, driver: %s", cudaVersion.runtime, cudaVersion.driver));

        // number of fits, number of points per fit
        int numberFits = 10;
        int numberPoints = 10;

        // model and estimator
        Model model = Model.GAUSS_1D;
        Estimator estimator = Estimator.LSE;

        // create fit model
        FitModel fitModel = new FitModel(numberFits, numberPoints, false, model, null, null, null, estimator, 0);

        // run Gpufit
        FitResult fitResult = Gpufit.fit(fitModel);
    }
}
