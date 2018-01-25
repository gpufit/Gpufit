package com.github.gpufit;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class Gpufit {

    static {
        System.loadLibrary("GpufitJNI");
    }

    public static native int add(int a, int b);

    private static native int fit(int numberFits, int numberPoints, ByteBuffer data);

    public static void test() {
        ByteBuffer b = ByteBuffer.allocateDirect(Float.BYTES * 5);
        b.order(ByteOrder.nativeOrder());
        FloatBuffer f = b.asFloatBuffer();
        f.put(1.234f);
        f.rewind();
        while (f.hasRemaining()) {
            System.out.println(f.get());
        }
        Gpufit.fit(5, 3, b);
    }

    public static native String getLastError();

    public static native boolean isCudaAvailable();

    public static native int[] getCudaVersion();

    public static class CudaVersion {

        public final int runtime, driver;

        private CudaVersion(int runtime, int driver) {
            this.runtime = runtime;
            this.driver = driver;
        }
    }

    public enum ModelID {

        GAUSS_1D(0),
        GAUSS_2D(1),
        GAUSS_2D_ELLIPTIC(2),
        GAUSS_2D_ROTATED(3),
        CAUCHY_2D_ELLIPTIC(4),
        LINEAR_1D(5);

        public final int id;

        ModelID(int id) {
            this.id = id;
        }
    }

    public enum EstimatorID {

        LSE(0), MLE(1);

        public final int id;

        EstimatorID(int id) {
            this.id = id;
        }

    }
}
