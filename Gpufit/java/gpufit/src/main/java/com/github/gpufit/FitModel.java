package com.github.gpufit;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

/**
 * Optional variables are weights, initialParameters, tolerance, maxNumberIterations, parametersToFit.
 */
public class FitModel {

    public final int numberFits;
    public final int numberPoints;
    public final FloatBuffer data;
    public final FloatBuffer weights;
    public final Model model;
    public final FloatBuffer initialParameters;
    public final float tolerance;
    public final int maxNumberIterations;
    public final IntBuffer parametersToFit;
    public final Estimator estimator;
    public final ByteBuffer userInfo;

    /**
     *
     * Indicate if weights or userInfo is needed by passing withWeights true and userInfoSize > 0.
     *
     * Sets default values for tolerance, maxNumberIterations and parametersToFit if these parameters are passed
     * as null.
     *
     * Fill data, weights (if needed), initialParameters and userInfo (if needed) afterwards.
     *
     * @param numberFits
     * @param numberPoints
     * @param withWeights
     * @param model
     * @param tolerance
     * @param maxNumberIterations
     * @param parametersToFit
     * @param estimator
     * @param userInfoSize
     */
    public FitModel(int numberFits, int numberPoints, boolean withWeights, Model model, Float tolerance, Integer maxNumberIterations, Boolean[] parametersToFit, Estimator estimator, int userInfoSize) {

        this.numberFits = numberFits;
        this.numberPoints = numberPoints;
        this.data = GpufitUtils.allocateDirectFloatBuffer(numberFits * numberPoints);
        this.weights = withWeights ? GpufitUtils.allocateDirectFloatBuffer(numberFits * numberPoints) : null;
        this.model = model;
        this.initialParameters = GpufitUtils.allocateDirectFloatBuffer(numberFits * model.numberParameters);
        this.tolerance = tolerance == null ? 1e-4f : tolerance;
        this.maxNumberIterations = maxNumberIterations == null ? 25 : maxNumberIterations;
        this.parametersToFit = GpufitUtils.allocateDirectIntBuffer(model.numberParameters);
        if (null == parametersToFit) {
            // fill with ones
            for (int i = 0; i < model.numberParameters; i++) {
                this.parametersToFit.put(1);
            }
        } else {
            // fill with given values
             for (int i = 0; i < model.numberParameters; i++) {
                this.parametersToFit.put(parametersToFit[i] ? 1 : 0);
            }
        }
        this.estimator = estimator;
        this.userInfo = GpufitUtils.allocateDirectByteBuffer(Math.max(0, userInfoSize));
    }
}
