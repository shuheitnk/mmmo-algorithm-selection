function X = sample_pareto_set(optima_msg, m, num_samples_per_axis)
% SAMPLE_PARETO_SET  Combine 2D local optima with an (m-1)-dim grid.
%
% Inputs
%   optima_msg : cell whose first element is a PyTorch tensor of shape
%                (N_opt x 2). Each row is a local optimum; the two
%                columns are paired coordinates.
%   m          : number of objectives (k = m-1 free dimensions are added).
%
% Output
%   X : (N_opt * 10^k) x (k+2) matrix. Each row is one decision-space
%       sample. Layout: [samples_1, ..., samples_k, x1*, x2*].

    interval      = 1 / num_samples_per_axis;
    half_interval = interval / 2;
    samples = (half_interval:interval:1-half_interval)';   % num_samples_per_axis x 1, cell centers
    k = m - 1;

    % PyTorch tensor -> MATLAB double matrix
    opt_mat = double(optima_msg{1}.detach().cpu().numpy()); % N_opt x 2
    N_opt   = size(opt_mat, 1);

    % Grid over the k free dimensions only (NOT over the optima axes)
    if k == 0
        S = zeros(1, 0);                                    % no free axis
    elseif k == 1
        S = samples;                                        % num_samples_per_axis x 1
    else
        grids = repmat({samples}, 1, k);
        [grid_arrays{1:k}] = ndgrid(grids{:});
        % Flatten each axis to a column, then concat horizontally
        S = cell2mat(cellfun(@(g) g(:), grid_arrays, ...
                             'UniformOutput', false));      % num_samples_per_axis^k x k
    end

    % Pair each optimum with every grid tuple, keeping (x1*, x2*) together
    nS = size(S, 1);
    X  = [repmat(S, N_opt, 1), repelem(opt_mat, nS, 1)];    % (N_opt*10^k) x (k+2)
end