function [pop_filt, F_filt] = get_epsilon_local_optima(pop, F_pop, F_global, epsilon)
    N = size(F_pop, 1);
    dominated = false(N, 1);

    % Apply epsilon shift
    F_eps = F_pop - epsilon;

    for k = 1:size(F_global, 1)
        dominated = dominated | ...
            (all(F_global(k,:) <= F_eps, 2) & ...
             any(F_global(k,:) <  F_eps, 2));
    end

    pop_filt = pop(~dominated, :);
    F_filt   = F_pop(~dominated, :);
end