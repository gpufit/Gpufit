package com.github.gpufit;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

/**
 *
 */
public class Gpufit {

    /**
     * Version of the used Gpufit library.
     */
    public static final String VERSION = "1.0.2";

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
    private static native int fit(int numberFits, int numberPoints, FloatBuffer data, FloatBuffer weights, int model_id,
                                  FloatBuffer initialParameters, float tolerance, int maxNumberIterations,
                                  IntBuffer parametersToFit, int estimatorID, int userInfoSize, ByteBuffer userInfo,
                                  FloatBuffer outputParameters, IntBuffer outputStates, FloatBuffer outputChiSquares,
                                  IntBuffer outputNumberIterations);

    /**
     *
     * @param fitModel
     * @param fitResults
     */
    public static FitResults fit(FitModel fitModel, FitResults fitResults) {

        // Should we reuse fitResults?
        if (null == fitResults) {
            fitResults = new FitResults(fitModel.numberFits, fitModel.model.numberParameters);
        } else {
            // check sizes
            fitResults.isCompatible(fitModel.numberFits, fitModel.model.numberParameters);
        }

        // call into native code to perform a fit
        long t0 = System.currentTimeMillis();
        int status = Gpufit.fit(fitModel.numberFits, fitModel.numberPoints, fitModel.data, fitModel.weights, fitModel.model.id, fitModel.initialParameters, fitModel.tolerance, fitModel.maxNumberIterations, fitModel.parametersToFit,
                fitModel.estimator.id, fitModel.userInfo.capacity(), fitModel.userInfo, fitResults.parameters, fitResults.states, fitResults.chiSquares, fitResults.numberIterations);
        long t1 = System.currentTimeMillis();
        fitResults.fitDuration = (float) (t1 - t0) / 1000;

        // check status
        if (status != Status.OK.id) {
            String message = getLastError();
            throw new RuntimeException(String.format("status = %s, message = %s", status, message));
        }

        // return results
        return fitResults;
    }

    /**
     * Convenience method.
     *
     * @param fitModel
     * @return
     */
    public static FitResults fit(FitModel fitModel) {
        return fit(fitModel, null);
    }

    /**
     * Native method. Returns a string representing the last error message.
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
     *
     * @return
     */
    private static native int[] getCudaVersionAsArray();

    /**
     *
     * @return
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
     *
     * @param version
     * @return
     */
    private static String versionAsString(int version) {
        return String.format("%d.%d", version / 1000, (version % 1000) / 10);
    }

    /**
     *
     */
    private enum Status {

        OK(0), ERROR(-1);

        public final int id;

        Status(int id) {
            this.id = id;
        }
    }
}
