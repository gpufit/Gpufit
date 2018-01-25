package com.github.gpufit;

import java.io.File;
import java.lang.reflect.Field;

public class RunIt {

    public static void main(String[] args) throws NoSuchFieldException, IllegalAccessException {

        // fix library path
        String libraryPath = System.getProperty("java.library.path");
        libraryPath += File.pathSeparator + "C:\\00_Jan\\00_Sources\\10_Gpufit.jkfindeisen.git-build\\VC14x64-9.1\\RelWithDebInfo";
        System.setProperty("java.library.path", libraryPath);
        Field fieldSysPath = ClassLoader.class.getDeclaredField("sys_paths");
        fieldSysPath.setAccessible( true );
        fieldSysPath.set( null, null );

        // some tests
        System.out.println(Gpufit.add(1, 2));
        Gpufit.test();
    }
}
