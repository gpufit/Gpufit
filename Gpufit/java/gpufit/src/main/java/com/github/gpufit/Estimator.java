package com.github.gpufit;

/**
 * Java binding for Gpufit, a Levenberg Marquardt curve fitting library written in CUDA
 * See https://github.com/gpufit/Gpufit, http://gpufit.readthedocs.io/en/latest/bindings.html#java
 *
 * The available estimators. See the documentation for details.
 */
public enum Estimator {

    /**
     * Least-squares estimator
     */
    LSE(0),

    /**
     * Poisson maximum likelihood estimator
     */
    MLE(1);

    public final int id;

    Estimator(int id) {
        this.id = id;
    }

}
