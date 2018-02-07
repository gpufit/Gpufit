package com.github.gpufit;

/**
 *
 */
public enum FitState {

    CONVERGED(0), MAX_ITERATIONS(1), SINGULAR_HESSIAN(2), NEG_CURVATURE_MLE(3), GPU_NOT_READY(4);

    public final int id;

    FitState(int id) {
        this.id = id;
    }

    /**
     *
     * @param id
     * @return
     */
    public static FitState fromID(int id) {
       for (FitState fitState : values()) {
           if (fitState.id == id) {
               return fitState;
           }
       }
       throw new RuntimeException("Unknown id");
    }
}
