classdef ModelID
    properties (Constant = true)
        GAUSS_1D = 0
        GAUSS_2D = 1
        GAUSS_2D_ELLIPTIC = 2
        GAUSS_2D_ROTATED = 3
        CAUCHY_2D_ELLIPTIC = 4
        LINEAR_1D = 5
        FLETCHER_POWELL = 6
        BROWN_DENNIS = 7
        PATLAK = 12
	TOFTS = 13
	TOFTS_EXTENDED = 14
	TISSUE_UPTAKE = 15
	TWO_COMPARTMENT_EXCHANGE = 16
    end
end
