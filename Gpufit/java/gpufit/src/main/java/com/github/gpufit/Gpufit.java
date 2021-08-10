package com.github.gpufit;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

/**
 * Java binding for Gpufit, a Levenberg Marquardt curve fitting library written in CUDA
 * See https://github.com/gpufit/Gpufit, http://gpufit.readthedocs.io/en/latest/bindings.html#java
 *
 * Mirror of the C interface of Gpufit. Loads the Gpufit library and calls it via JNI. On the Java side the arguments
 * to Gpufit are bundled in a FitModel, the results of the fit are bundled in a FitResult. The data arrays in FitModel
 * and FitResult are pre-allocated by us and need to be filled by the user of this library accordingly.
 *
 * See the documentation of FitModel and FitResult as well as the examples.
 */
public class Gpufit {

    /**
     * Version of the used Gpufit library.
     */
    public static final String VERSION = "1.2.0";

    static {
        /*
         * Need to load Gpufit first, otherwise the call to loadLibrary with GpufitJNI will throw an exception
         * on Windows.
         * java.lang.UnsatisfiedLinkError: GpufitJNI.dll: Can't find dependent libraries
         */
        System.loadLibrary("Gpufit");
        System.loadLibrary("GpufitJNI");
    }

    /**
     * Native method. More of less calls gpufit() in the gpufit C interface directly. Used only internally.
     */
    private static native int fit(int numberFits, int numberPoints, FloatBuffer data, FloatBuffer weights, int model_id, FloatBuffer initialParameters, float tolerance, int maxNumberIterations, IntBuffer parametersToFit, int estimatorID, int userInfoSize, ByteBuffer userInfo, FloatBuffer outputParameters, IntBuffer outputStates, FloatBuffer outputChiSquares, IntBuffer outputNumberIterations);

    /**
     * Use this method to perform a parallel fit of many single fits of the same Function model and the same
     * fit data size in parallel. The input is given as a {@link FitModel}, the output as a {@link FitResult}.
     * If null is passed for fitResult, a new output structure is created, otherwise the results overwrite the
     * existent content (size is checked for compatibility before).
     *
     * A call to Gpufit is performed. If an error is encountered, a runtime exception is thrown.
     *
     * @param fitModel  Fit data including the model
     * @param fitResult Fit result (could be old one which is reused) or null
     * @return Fit result
     */
    public static FitResult fit(FitModel fitModel, FitResult fitResult) {

        // Should we reuse fitResult?
        if (null == fitResult) {
            fitResult = new FitResult(fitModel.numberFits, fitModel.model.numberParameters);
        } else {
            // check sizes
            fitResult.isCompatible(fitModel.numberFits, fitModel.model.numberParameters);
        }

        // call into native code to perform a fit
        long t0 = System.currentTimeMillis();
        int status = Gpufit.fit(fitModel.numberFits, fitModel.numberPoints, fitModel.data, fitModel.weights, fitModel.model.id, fitModel.initialParameters, fitModel.tolerance, fitModel.maxNumberIterations, fitModel.parametersToFit, fitModel.estimator.id, fitModel.userInfo.capacity(), fitModel.userInfo, fitResult.parameters, fitResult.states, fitResult.chiSquares, fitResult.numberIterations);
        long t1 = System.currentTimeMillis();
        fitResult.fitDuration = (float) (t1 - t0) / 1000;

        // check status
        if (status != Status.OK.id) {
            String message = getLastError();
            throw new RuntimeException(String.format("status = %s, message = %s", status, message));
        }

        // return results
        return fitResult;
    }

    /**
     * Convenience method. Calls fit with only a FitModel. Returns the result.
     *
     * @param fitModel Fit data including the model
     * @return Fit result
     */
    public static FitResult fit(FitModel fitModel) {
        return fit(fitModel, null);
    }

    /**
     * Native method. Returns a string representing the last error message from Gpufit.
     *
     * @return The last error message from Gpufit.
     */
    public static native String getLastError();

    /**
     * Native method. Indicates if CUDA capability is available.
     *
     * @return True if available, false otherwise.
     */
    public static native boolean isCudaAvailable();

    /**
     * Native method. Gets the CUDA runtime and driver version as ints. Used only internally.
     */
    private static native int[] getCudaVersionAsArray();

    /**
     * Gets the CUDA runtime and driver version as strings.
     *
     * Throws an exception if CUDA is not available.
     *
     * @return The CUDA version.
     */
    public static CudaVersion getCudaVersion() {
        int[] version = getCudaVersionAsArray();
        if (null == version) {
            String message = getLastError();
            throw new RuntimeException(message);
        }
        return new CudaVersion(versionAsString(version[0]), versionAsString(version[1]));
    }

    /**
     * Special conversion for our versions from Integer to String. The convention is that the integer is
     * major version * 1000 + minor version * 10.
     *
     * @param version An integer version.
     * @return A version string.
     */
    private static String versionAsString(int version) {
        return String.format("%d.%d", version / 1000, (version % 1000) / 10);
    }

    /**
     * The status of a call to the Gpufit routines (see Gpufit documentation for details). Used only internally.
     */
    private enum Status {

        OK(0), ERROR(-1);

        final int id;

        Status(int id) {
            this.id = id;
        }
    }
}
