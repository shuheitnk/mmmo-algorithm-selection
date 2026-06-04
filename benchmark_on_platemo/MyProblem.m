classdef MyProblem < PROBLEM
    properties
        multi_msg
        device
    end

    methods
        function obj = MyProblem(multi_msg, device, D)

            obj.multi_msg = multi_msg;
            obj.device    = device;

            obj.D = D;

            obj.lower    = zeros(1, D);
            obj.upper    = ones(1, D);
            obj.encoding = ones(1, D);
        end

        function PopObj = CalObj(obj, PopDec)
            PopObj = msg_wrapper_batch(PopDec, obj.multi_msg, obj.device);
        end
    end
end