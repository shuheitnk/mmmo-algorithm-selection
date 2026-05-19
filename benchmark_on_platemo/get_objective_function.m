function val = get_objective_function(x, f_msg, idx)
    y   = f_msg(x);
    val = y(1, idx);
end