import torch

class MSGLandscape(torch.nn.Module):
    """
    Max-Set of Gaussians (MSG) landscape.

    Attributes:
        means (torch.Tensor): Gaussian centers (M, D)
        theta (torch.nn.Parameter): alpha (height) and sigma (width) parameters (2*M,)
    """
    def __init__(self, means: torch.Tensor, theta: torch.Tensor):
        super().__init__()
        self.register_buffer("means", means)      # fixed centers
        self.theta = torch.nn.Parameter(theta)    # tunable alpha and sigma
        self.num_functions, self.dim = means.shape

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate MSG landscape.

        Args:
            X: (N, D) input points

        Returns:
            (N,) negative max of Gaussian contributions
        """
        alphas, sigmas = self.theta[:self.num_functions], self.theta[self.num_functions:]
        sq_dist = (X**2).sum(dim=1, keepdim=True) + (self.means**2).sum(dim=1).unsqueeze(0) - 2*X @ self.means.T
        vals = alphas.unsqueeze(0) * torch.exp(-0.5 * sq_dist / sigmas**2)
        return -vals.max(dim=1).values

    @torch.no_grad()
    def find_optima_exact(self, atol: float = 0.0):

        alphas, sigmas = self.theta[:self.num_functions], self.theta[self.num_functions:]
        sq_dist = ((self.means[:, None, :] - self.means[None, :, :])**2).sum(dim=-1)
        vals = alphas.unsqueeze(0) * torch.exp(-0.5 * sq_dist / sigmas**2)
        candidate_values = vals.max(dim=1).values

        is_local = torch.isclose(candidate_values, alphas, atol=atol)
        local_optima, local_fitness = self.means[is_local], candidate_values[is_local]

        global_idx = torch.argmax(local_fitness)
        mask = torch.ones(len(local_fitness), dtype=torch.bool, device=local_fitness.device)
        mask[global_idx] = True

        local_optima_only, local_fitness_only = local_optima[mask], local_fitness[mask]
        global_optima_all, global_fitness_all = local_optima[global_idx].unsqueeze(0), local_fitness[global_idx].unsqueeze(0)

        return local_optima_only, local_fitness_only, global_optima_all, global_fitness_all
