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
        SPLINE_1D = 8
        SPLINE_2D = 9
        SPLINE_3D = 10
        SPLINE_3D_MULTICHANNEL = 11
        SPLINE_3D_PHASE_MULTICHANNEL = 12
    end
    methods (Static)
        function v = validID(id)
            meta = ?ModelID;
            v = false;
            for i = 1 : numel(meta.PropertyList)
                property_value = meta.PropertyList(i).DefaultValue;
                if id == property_value
                    v = true;
                end
            end
        end
    end
end