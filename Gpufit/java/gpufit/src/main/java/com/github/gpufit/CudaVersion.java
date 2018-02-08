package com.github.gpufit;

/**
 * Java binding for Gpufit, a Levenberg Marquardt curve fitting library written in CUDA
 * See https://github.com/gpufit/Gpufit, http://gpufit.readthedocs.io/en/latest/bindings.html#java
 *
 * Holds the runtime version (version of CUDA toolkit used in the creation of the used version of Gpufit) and the
 * driver version (installed Nvidia driver version).
 *
 * Driver version must be equal or greater than the runtime version.
 */
public class CudaVersion {

    public final String runtime, driver;

    CudaVersion(String runtime, String driver) {
        this.runtime = runtime;
        this.driver = driver;
    }
}
