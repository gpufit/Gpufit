package com.github.gpufit;

/**
 *
 */
public enum Estimator {

    LSE(0), MLE(1);

    public final int id;

    Estimator(int id) {
        this.id = id;
    }

}
