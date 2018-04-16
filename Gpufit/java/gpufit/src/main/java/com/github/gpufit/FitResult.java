package com.github.gpufit;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

/**
 * Java binding for Gpufit, a Levenberg Marquardt curve fitting library written in CUDA
 * See https://github.com/gpufit/Gpufit, http://gpufit.readthedocs.io/en/latest/bindings.html#java
 *
 * Holds the fit results in buffers. The buffers are pre-allocated in the constructor. An instance can be reused for
 * consecutive runs with the same number of fits and the same number of model parameters.
 */
public class FitResult {

    /**
     * Values of resulting fit parameters for each data set.
     */
    public final FloatBuffer parameters;
    /**
     * Ids of the fit results (@see FitResult)
     */
    public final IntBuffer states;
    /**
     * Final Chi² for each fit
     */
    public final FloatBuffer chiSquares;
    /**
     * Used number of iterations for each fit
     */
    public final IntBuffer numberIterations;
    /**
     * Duration of fit in seconds.
     */
    public float fitDuration;

    /**
     * Given the number of fits and the number of parameters of the fit model, pre-allocates memory for the fit results.
     *
     * @param numberFits       Number of fits in the call to Gpufit.fit
     * @param numberParameters Number of parameters of the model
     */
    public FitResult(int numberFits, int numberParameters) {
        parameters = GpufitUtils.allocateDirectFloatBuffer(numberFits * numberParameters);
        states = GpufitUtils.allocateDirectIntBuffer(numberFits);
        chiSquares = GpufitUtils.allocateDirectFloatBuffer(numberFits);
        numberIterations = GpufitUtils.allocateDirectIntBuffer(numberFits);
    }

    /**
     * Checks where the sizes of the existing member variables is consistent with a given number of fits and number of
     * parameters of the fit model. Required when re-using an instance between different runs of the fit.
     *
     * @param numberFits       Number of fits in the call to Gpufit.fit
     * @param numberParameters Number of parameters of the model
     */
    public void isCompatible(int numberFits, int numberParameters) {
        GpufitUtils.assertTrue(parameters.capacity() == numberFits * numberParameters, "Expected different size of parameters");
        GpufitUtils.assertTrue(states.capacity() == numberFits, "Expected different size of states");
        GpufitUtils.assertTrue(chiSquares.capacity() == numberFits, "Expected different size of chiSquares");
        GpufitUtils.assertTrue(numberIterations.capacity() == numberFits, "Expected different size of numberIterations");
    }
}
