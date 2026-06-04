import torch

def make_multi_objective_msg(
    m: int,
    dim_msg: int,
    function_g,
    pf_shape: str = "convex",
    k: float = 0.0,
    A: float = 1.0
):

    if m < 2 or dim_msg < 1:
        raise ValueError("m must be >=2 and dim_msg >=1")

    dim_h = m - 1
    dim_multi = dim_h + dim_msg

    def multi_objective_MSG(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != dim_multi:
            raise ValueError(f"Expected input dim {dim_multi}, got {x.shape[-1]}")

        x_h = x[..., :dim_h] 
        x_g = x[..., dim_h:]

        phi = 0.5 * torch.pi * x_h + torch.pi * (pf_shape == "convex")
        r = 1.0 + A * torch.sum(torch.sin(k * phi), dim=-1, keepdim=True) 
        *batch, n = phi.shape
        h = torch.zeros(*batch, m, device=x.device)
        sin_prod = torch.ones(*batch, device=x.device)
        for i in range(n):
            h[..., i] = r.squeeze(-1) * sin_prod * torch.cos(phi[..., i])
            sin_prod = sin_prod * torch.sin(phi[..., i])
        h[..., n] = r.squeeze(-1) * sin_prod  # last component
  
        g_val = function_g(x_g)  # (...,) or (...,1)
        g_val = g_val.unsqueeze(-1) 
        g_vec = g_val.repeat(1, m)

        F = h + g_vec
        
        return F

    return multi_objective_MSG
