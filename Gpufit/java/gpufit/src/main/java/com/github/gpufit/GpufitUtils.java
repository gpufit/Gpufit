package com.github.gpufit;

import java.io.File;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

/**
 * Java binding for Gpufit, a Levenberg Marquardt curve fitting library written in CUDA
 * See https://github.com/gpufit/Gpufit, http://gpufit.readthedocs.io/en/latest/bindings.html#java
 *
 * Helper utilities.
 */
public class GpufitUtils {

    private GpufitUtils() {
    }

    /**
     * Creates a direct ByteBuffer with the native Byte order, which is exactly what we need to send and receive data
     * arrays via JNI.
     *
     * @param length Desired number of elements in ByteBuffer
     * @return Direct ByteBuffer
     */
    public static ByteBuffer allocateDirectByteBuffer(int length) {
        assertNotNegative(length, "Parameter length must be non-negative");
        ByteBuffer buffer = ByteBuffer.allocateDirect(length);
        buffer.order(ByteOrder.nativeOrder());
        return buffer;
    }

    /**
     * Creates a FloatBuffer of a certain length backed up by a direct ByteBuffer.
     *
     * @param length Desired number of elements in FloatBuffer
     * @return FloatBuffer backed by direct ByteBuffer
     */
    public static FloatBuffer allocateDirectFloatBuffer(int length) {
        return allocateDirectByteBuffer(Float.BYTES * length).asFloatBuffer();
    }

    /**
     * Creates an IntBuffer of a certain length backed up by a direct ByteBuffer.
     *
     * @param length Desired number of elements in IntBuffer
     * @return IntBuffer backed by direct ByteBuffer
     */
    public static IntBuffer allocateDirectIntBuffer(int length) {
        return allocateDirectByteBuffer(Integer.BYTES * length).asIntBuffer();
    }

    /**
     * Ensures that an integer value is not negative. Throws a runtime exception otherwise.
     *
     * @param value   Integer value
     * @param message Error message
     */
    public static void assertNotNegative(int value, String message) {
        if (!(value >= 0)) {
            throw new RuntimeException(message);
        }
    }

    /**
     * Ensures that a boolean is true. Throws a runtime exception otherwise.
     *
     * @param value   Boolean value
     * @param message Error message
     */
    public static void assertTrue(boolean value, String message) {
        if (!value) {
            throw new RuntimeException(message);
        }
    }

    /**
     * Ensures that an object reference passed as a parameter to the calling method is not null.
     *
     * @param <T>       Type of the reference to be checked
     * @param reference Object reference
     * @return Non-null reference that was checked
     */
    public static <T> T verifyNotNull(T reference) throws NullPointerException {
        if (null == reference) {
            throw new NullPointerException();
        }
        return reference;
    }

    /**
     * Adds a path to the java.library.path programmatically.
     *
     * See also: https://stackoverflow.com/questions/11783632/how-do-i-load-and-use-native-library-in-java
     *
     * @param path Path String to add to the java.library.path
     */
    public static void addPathToJavaLibraryPath(String path) {

        // add path to system property
        String libraryPath = System.getProperty("java.library.path");
        libraryPath += File.pathSeparator + path;
        System.setProperty("java.library.path", libraryPath);

        // clear field in class loader
        try {
            Field fieldSysPath = ClassLoader.class.getDeclaredField("sys_paths");
            fieldSysPath.setAccessible(true);
            fieldSysPath.set(null, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException("Java Library Path addition failed.");
        }
    }
}
