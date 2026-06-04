
function Y = msg_wrapper_batch(X, multi_msg, device)

    persistent torch_py device_py
    if isempty(torch_py)
        torch_py = py.importlib.import_module('torch');
    end
    if isempty(device_py) || ~strcmp(char(device_py.type), device)
        device_py = torch_py.device(device);
    end
    X = single(X); % Convert to single precision
    x_tensor = torch_py.from_numpy(X).to(device_py);
    if x_tensor.ndim == 1 % If input is 1D, add a batch dimension
        x_tensor = x_tensor.unsqueeze(int32(0));
    end
    y_tensor = multi_msg(x_tensor);
    Y = double(y_tensor.detach().cpu().numpy());
    
end