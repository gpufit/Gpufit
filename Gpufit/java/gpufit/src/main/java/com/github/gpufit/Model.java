package com.github.gpufit;

/**
 * Java binding for Gpufit, a Levenberg Marquardt curve fitting library written in CUDA
 * See https://github.com/gpufit/Gpufit, http://gpufit.readthedocs.io/en/latest/bindings.html#java
 */
public enum Model {

    GAUSS_1D(0, 4), GAUSS_2D(1, 5), GAUSS_2D_ELLIPTIC(2, 6), GAUSS_2D_ROTATED(3, 7), CAUCHY_2D_ELLIPTIC(4, 6), LINEAR_1D(5, 2), FLETCHER_POWELL_HELIX(6, 3), BROWN_DENNIS(7, 4), SPLINE_1D(8, 3), SPLINE_2D(9, 4), SPLINE_3D(10, 5), SPLINE_3D_MULTICHANNEL(11, 5), SPLINE_3D_PHASE_MULTICHANNEL(12, 6), SPLINE_4D(13, 6), SPLINE_5D(15, 7);

    public final int id, numberParameters;

    Model(int id, int numberParameters) {
        this.id = id;
        this.numberParameters = numberParameters;
    }
}
