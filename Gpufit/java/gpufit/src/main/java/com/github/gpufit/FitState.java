package com.github.gpufit;

/**
 * Java binding for Gpufit, a Levenberg Marquardt curve fitting library written in CUDA
 * See https://github.com/gpufit/Gpufit, http://gpufit.readthedocs.io/en/latest/bindings.html#java
 *
 * Possible outcomes of a fit. See also the documentation of Gpufit.
 */
public enum FitState {

    /**
     * Converged within the given tolerance
     */
    CONVERGED(0),

    /**
     * Maximum number of iterations exceeded
     */
    MAX_ITERATIONS(1),

    /**
     * Singular Hessian
     */
    SINGULAR_HESSIAN(2),

    /**
     * Negative curvature in the MLE
     */
    NEG_CURVATURE_MLE(3),

    /**
     * GPU not ready
     */
    GPU_NOT_READY(4);

    /**
     * Id is the same as the output of the Gpufit fit.
     */
    private final int id;

    FitState(int id) {
        this.id = id;
    }

    /**
     * Retrieves the enum corresponding to a certain id.
     *
     * Throws a runtime exception if the id is unknown.
     *
     * @param id An id of a FitState (for example the output of the fit routines).
     * @return An FitState enum member as defined above.
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
