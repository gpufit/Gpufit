package com.github.gpufit;

import java.io.File;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

/**
 *
 */
public class GpufitUtils {

    private GpufitUtils() {}

    /**
     *
     * @param length
     * @return
     */
    public static ByteBuffer allocateDirectByteBuffer(int length) {
        assertNotNegative(length, "Parameter length must be non-negative");
        ByteBuffer buffer = ByteBuffer.allocateDirect(length);
        buffer.order(ByteOrder.nativeOrder());
        return buffer;
    }

    /**
     *
     * @param length
     * @return
     */
    public static FloatBuffer allocateDirectFloatBuffer(int length) {
        return allocateDirectByteBuffer(Float.BYTES * length).asFloatBuffer();
    }

    /**
     *
     * @param length
     * @return
     */
    public static IntBuffer allocateDirectIntBuffer(int length) {
        return allocateDirectByteBuffer(Integer.BYTES * length).asIntBuffer();
    }

    /**
     *
     * @param value
     * @param message
     */
    public static void assertNotNegative(int value, String message) {
        if (!(value >= 0)) {
            throw new RuntimeException(message);
        }
    }

    /**
     *
     * @param value
     * @param message
     */
    public static void assertNotNegative(float value, String message) {
        if (!(value >= 0)) {
            throw new RuntimeException(message);
        }
    }

    /**
     *
     * @param value
     * @param message
     */
    public static void assertTrue(boolean value, String message) {
        if (!value) {
            throw new RuntimeException(message);
        }
    }

    /**
     * Ensures that an object reference passed as a parameter to the calling method is not null.
     *
     * @param reference an object reference
     * @param <T>
     * @return the non-null reference that was checked
     */
    public static <T> T verifyNotNull(T reference) throws NullPointerException {
        if (null == reference) {
            throw new NullPointerException();
        }
        return reference;
    }

    /**
     * See also: https://stackoverflow.com/questions/11783632/how-do-i-load-and-use-native-library-in-java
     * @param path
     */
    public static void addPathToJavaLibraryPath(String path) {

        // add path to system property
        String libraryPath = System.getProperty("java.library.path");
        libraryPath += File.pathSeparator + path;
        System.setProperty("java.library.path", libraryPath);

        // clear field in class loader
        try {
            Field fieldSysPath = ClassLoader.class.getDeclaredField("sys_paths");
            fieldSysPath.setAccessible( true );
            fieldSysPath.set( null, null );
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException("Java Library Path addition failed.");
        }
    }


}
