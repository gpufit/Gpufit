#ifndef GPUFIT_CONSTANTS_H_INCLUDED
#define GPUFIT_CONSTANTS_H_INCLUDED

// fitting model ID

enum ModelID {
    GAUSS_1D = 0,
    GAUSS_2D = 1,
    GAUSS_2D_ELLIPTIC = 2,
    GAUSS_2D_ROTATED = 3,
    CAUCHY_2D_ELLIPTIC = 4,
    LINEAR_1D = 5,
    FLETCHER_POWELL_HELIX = 6,
    BROWN_DENNIS = 7,
	SPLINE_1D = 8,
    SPLINE_2D = 9,
    SPLINE_3D = 10,
    SPLINE_3D_MULTICHANNEL = 11,
    SPLINE_3D_PHASE_MULTICHANNEL = 12
    ESR3111 = 13,
	ESR2111 = 14,
    DIPBIPROJY_1_2D = 15,
	DIPBIPROJY_2_2D = 16,
	DIPBIPROJY_3_2D = 17,  // NOT USED
	DIPPROJ111YZ_1_2D = 18,  // NOT USED
	DIPPROJ111U4_PU4_1_2D = 19,
	DIPPROJ111U4_PU4_2_2D = 20,
	DIPPROJ111U4_PU4_3_2D = 21,
	DIPPROJ111U4_PZ_1_2D = 22,
	DIPPROJ111U4_PZ_2_2D = 23,
	DIPPROJ111U4_PZ_3_2D = 24
};

// estimator ID
enum EstimatorID { LSE = 0, MLE = 1 };

// fit state
enum FitState { CONVERGED = 0, MAX_ITERATION = 1, SINGULAR_HESSIAN = 2, NEG_CURVATURE_MLE = 3, GPU_NOT_READY = 4 };

// return state
enum ReturnState { OK = 0, ERROR = -1 };

// input/output data location
enum DataLocation { HOST = 0, DEVICE = 1 };

// bounds
enum Bound { LOWER_BOUND = 0, UPPER_BOUND = 1 };

// constraint type
enum ConstraintType { NONE = 0, LOWER = 1, UPPER = 2, LOWER_UPPER = 3 };

#endif
