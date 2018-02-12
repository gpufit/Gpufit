package com.github.gpufit;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

/**
 * Java binding for Gpufit, a Levenberg Marquardt curve fitting library written in CUDA
 * See https://github.com/gpufit/Gpufit, http://gpufit.readthedocs.io/en/latest/bindings.html#java
 *
 * Holds all the input arguments to a single fit execution.
 *
 * A number of fits with the same model and estimator and fit data size are executed in parallel.
 *
 * Some parameters are given in the constructor, buffers are pre-allocated there and must be filled with data
 * afterwards.
 *
 * Optional variables are weights, initialParameters, tolerance, maxNumberIterations, parametersToFit.
 */
public class FitModel {

    /**
     * Number of fits, i.e. number of independent data sets
     */
    public final int numberFits;
    /**
     * Number of data points per fit
     */
    public final int numberPoints;
    /**
     * Buffer holding the data
     */
    public final FloatBuffer data;
    /**
     * Buffer holding the data weights (or null)
     */
    public final FloatBuffer weights;
    /**
     * Fit function model enum
     */
    public final Model model;
    /**
     * Initial value of parameters for each data set
     */
    public final FloatBuffer initialParameters;
    /**
     * Minimal fit tolerance.
     */
    public final float tolerance;
    /**
     * Maximal number of iterations per data set.
     */
    public final int maxNumberIterations;
    /**
     * Indication which parameters should be fitted (value 1) and which should be kept constant (value 0).
     */
    public final IntBuffer parametersToFit;
    /**
     * Fit estimator enum
     */
    public final Estimator estimator;
    /**
     * Additional user info (optional).
     */
    public final ByteBuffer userInfo;

    /**
     * Provide a number of input arguments for the fit.
     *
     * Indicate if weights or userInfo is needed by passing withWeights true and userInfoSize is larger than zero.
     *
     * Sets default values for tolerance, maxNumberIterations, parametersToFit and estimator if these parameters are passed
     * as null.
     *
     * Fill data, weights (if needed), initialParameters and userInfo (if needed) afterwards with values.
     *
     * @param numberFits          Number of fits
     * @param numberPoints        Number of data points per fit
     * @param withWeights         If True, weights will be pre-allocated, otherwise not
     * @param model               Fit model enum
     * @param tolerance           Fit tolerance (if null, a default value is chosen)
     * @param maxNumberIterations Maximal number of iterations per fit (if null, a default value is chosen)
     * @param parametersToFit     For each parameter indicates if it should be fit (true) or kept constant (false).
     *                            If null all parameters are fit by default.
     * @param estimator           Fit estimator enum (if null, a default value is chosen)
     * @param userInfoSize        If positive, userInfo is pre-allocated with userInfoSize as capacity, otherwise not
     */
    public FitModel(int numberFits, int numberPoints, boolean withWeights, Model model, Float tolerance, Integer maxNumberIterations, Boolean[] parametersToFit, Estimator estimator, int userInfoSize) {

        this.numberFits = numberFits;
        this.numberPoints = numberPoints;
        this.data = GpufitUtils.allocateDirectFloatBuffer(numberFits * numberPoints);
        this.weights = withWeights ? GpufitUtils.allocateDirectFloatBuffer(numberFits * numberPoints) : null;
        this.model = GpufitUtils.verifyNotNull(model);
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
        this.estimator = estimator == null ? Estimator.LSE : estimator;
        this.userInfo = GpufitUtils.allocateDirectByteBuffer(Math.max(0, userInfoSize));
    }
}
