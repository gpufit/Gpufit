package com.github.gpufit;

/**
 * Java binding for Gpufit, a Levenberg Marquardt curve fitting library written in CUDA
 * See https://github.com/gpufit/Gpufit, http://gpufit.readthedocs.io/en/latest/bindings.html#java
 */
public enum Model {

    GAUSS_1D(0, 4), GAUSS_2D(1, 5), GAUSS_2D_ELLIPTIC(2, 6), GAUSS_2D_ROTATED(3, 7), CAUCHY_2D_ELLIPTIC(4, 6), LINEAR_1D(5, 2);

    public final int id, numberParameters;

    Model(int id, int numberParameters) {
        this.id = id;
        this.numberParameters = numberParameters;
    }
}
