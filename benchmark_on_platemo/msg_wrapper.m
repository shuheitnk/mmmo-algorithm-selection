function y = msg_wrapper(X, multi_msg, device)

    persistent torch_py device_py
    if isempty(torch_py)
        torch_py = py.importlib.import_module('torch');
    end
    if isempty(device_py) || ~strcmp(char(device_py.type), device)
        device_py = torch_py.device(device);
    end
    X = double(X);
    if isvector(X)
        X = reshape(X, 1, []);
    end
    x_tensor = torch_py.tensor( ...
        X, ...
        pyargs('dtype', torch_py.float32, 'device', device_py) ...
    );
    if int64(x_tensor.ndim) == 1
        x_tensor = x_tensor.unsqueeze(int32(0));
    end
    y_tensor = multi_msg(x_tensor);
    y = double(y_tensor.detach().cpu().numpy());
    
end