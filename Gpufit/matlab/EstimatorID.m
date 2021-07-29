classdef EstimatorID
    properties (Constant = true)
        LSE = 0
        MLE = 1
    end
    methods (Static)
        function v = validID(id)
            meta = ?EstimatorID;
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