package com.github.gpufit;

/**
 *
 */
public class CudaVersion {

    public final String runtime, driver;

    CudaVersion(String runtime, String driver) {
        this.runtime = runtime;
        this.driver = driver;
    }
}
