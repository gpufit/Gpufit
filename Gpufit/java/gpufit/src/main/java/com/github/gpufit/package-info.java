package com.github.gpufit;

/**
 * Java binding for Gpufit, a Levenberg Marquardt curve fitting library written in CUDA
 * See https://github.com/gpufit/Gpufit, http://gpufit.readthedocs.io/en/latest/bindings.html#java
 *
 * Java binding of Gpufit.
 *
 * The Gpufit class uses JNI and a small adapter library GpufitJNI to connect Java and Gpufit. On the Java side buffers
 * are used to hold the data. The model and estimator can be selected using Java enums.
 *
 * See the examples for how to use this Java binding and the documentation of Gpufit.
 */