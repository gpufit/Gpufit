package com.github.gpufit;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

/**
 *
 */
public class FitResults {

    public final FloatBuffer parameters;
    public final IntBuffer states;
    public final FloatBuffer chiSquares;
    public final IntBuffer numberIterations;
    /**
     * Duration of fit in seconds.
     */
    public float fitDuration;

    /**
     *
     * @param numberFits
     * @param numberParameters
     */
    public FitResults(int numberFits, int numberParameters) {
        parameters = GpufitUtils.allocateDirectFloatBuffer(numberFits * numberParameters);
        states = GpufitUtils.allocateDirectIntBuffer(numberFits);
        chiSquares = GpufitUtils.allocateDirectFloatBuffer(numberFits);
        numberIterations = GpufitUtils.allocateDirectIntBuffer(numberFits);
    }

    /**
     *
     * @param numberFits
     * @param numberParameters
     */
    public void isCompatible(int numberFits, int numberParameters) {
        GpufitUtils.assertTrue(parameters.capacity() == numberFits * numberParameters, "Expected different size of parameters");
        GpufitUtils.assertTrue(states.capacity() == numberFits, "Expected different size of states");
        GpufitUtils.assertTrue(chiSquares.capacity() == numberFits, "Expected different size of chiSquares");
        GpufitUtils.assertTrue(numberIterations.capacity() == numberFits, "Expected different size of numberIterations");
    }
}
