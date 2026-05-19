import torch
from typing import Callable, Dict, Optional
import random
import numpy as np

# =====================================================
# Reproducibility Utilities
# =====================================================
def set_seed(seed: int, device="cuda") -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dev_str = str(device) if isinstance(device, torch.device) else device

    if dev_str.startswith("cuda"):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EvolutionStrategy:
    def __init__(
        self,
        dim_theta: int,
        population_size: int,
        mutation_std: float,
        fitness_fn: Callable[[torch.Tensor], torch.Tensor],
        D: int,
        device: str = "cuda",
        seed: Optional[int] = None,
    ):
        self.dim_theta = dim_theta
        self.population_size = population_size
        self.mutation_std = mutation_std
        self.fitness_fn = fitness_fn
        self.D = D
        self.device = device

        if seed is not None:
            set_seed(seed, device)

    # ---------------- Vanilla ES ----------------
    def run_vanilla_es(self, theta_init: torch.Tensor, generations: int) -> Dict[str, torch.Tensor]:
        """Vanilla Evolution Strategy (minimization)"""
        theta_parent = theta_init.clone().to(self.device)
        dim_theta = theta_parent.numel()
        M = dim_theta // 2

        best_fitness = float('inf')
        best_theta = None
        fitness_history = []
        theta_history = []
        best_fitness_history = []

        half = self.population_size // 2

        for gen in range(generations):
            eps_half = torch.randn(half, dim_theta, device=self.device)
            eps = torch.cat([eps_half, -eps_half], dim=0)
            theta_children = theta_parent + self.mutation_std * eps
            alphas= torch.clamp(theta_children[:, :M], min=0)
            sigmas = torch.clamp(theta_children[:, M:], min=0.05*(self.D**0.5), max=self.D**0.5)
            thetas_clipped = torch.cat([alphas, sigmas], dim=1)

            with torch.no_grad():
                fitness_scores = self.fitness_fn(thetas_clipped)
                best_idx = torch.argmin(fitness_scores)
                min_fitness = fitness_scores[best_idx].item()

            best_theta = thetas_clipped[best_idx].detach().clone()
            theta_parent = thetas_clipped[best_idx].detach().clone()

            if min_fitness < best_fitness:
                best_fitness = min_fitness
                best_theta = best_theta

            fitness_history.append(min_fitness)
            theta_history.append(best_theta)
            best_fitness_history.append(best_fitness)

        return {
            "theta_best": best_theta,
            "fitness_history": torch.tensor(fitness_history, device=self.device),
            "theta_history": torch.stack(theta_history),
            "best_fitness_history": torch.tensor(best_fitness_history, device=self.device),
        }
    
    # ---------------- Vanilla ES Max ----------------
    def run_vanilla_es_maximize(self, theta_init: torch.Tensor, generations: int) -> Dict[str, torch.Tensor]:
        """Vanilla Evolution Strategy (maximization)"""
        theta_parent = theta_init.clone().to(self.device)
        dim_theta = theta_parent.numel()
        M = dim_theta // 2

        best_fitness = -float('inf')
        best_theta = None
        fitness_history = []
        theta_history = []
        best_fitness_history = []
        half = self.population_size // 2

        for gen in range(generations):
            eps_half = torch.randn(half, dim_theta, device=self.device)
            eps = torch.cat([eps_half, -eps_half], dim=0)
            theta_children = theta_parent + self.mutation_std * eps
            alphas= torch.clamp(theta_children[:, :M], min=0)
            sigmas = torch.clamp(theta_children[:, M:], min=0.05*(self.D**0.5), max=self.D**0.5)
            thetas_clipped = torch.cat([alphas, sigmas], dim=1)

            with torch.no_grad():
                fitness_scores = self.fitness_fn(thetas_clipped)
                best_idx = torch.argmax(fitness_scores)
                max_fitness = fitness_scores[best_idx].item()

            best_theta = thetas_clipped[best_idx].detach().clone()
            theta_parent = best_theta

            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_theta = best_theta

            fitness_history.append(max_fitness)
            theta_history.append(best_theta)
            best_fitness_history.append(best_fitness)

        return {
            "theta_best": best_theta,
            "fitness_history": torch.tensor(fitness_history, device=self.device),
            "theta_history": torch.stack(theta_history),
            "best_fitness_history": torch.tensor(best_fitness_history, device=self.device),
        }

    # ---------------- Random Search ----------------
    def run_random_search(self, generations: int) -> Dict[str, torch.Tensor]:
        """Random search (minimization)"""
        dim_theta = self.dim_theta
        M = dim_theta // 2
        fitness_history = []
        theta_history = []
        best_fitness_history = []

        best_fitness = float('inf')
        best_theta = None

        for gen in range(generations):
            theta_random = torch.rand(self.population_size, dim_theta, device=self.device) * (self.D**0.5)
            alphas= torch.clamp(theta_random[:, :M], min=0, max=1)
            sigmas = torch.clamp(theta_random[:, M:], min=0.05*(self.D**0.5), max=self.D**0.5)
            thetas_clipped = torch.cat([alphas, sigmas], dim=1)

            with torch.no_grad():
                fitness_scores = self.fitness_fn(thetas_clipped)
                min_idx = torch.argmin(fitness_scores)
                min_fitness = fitness_scores[min_idx].item()
                best_theta = thetas_clipped[min_idx].reshape(-1).clone()

            if min_fitness < best_fitness:
                best_fitness = min_fitness
                best_theta = best_theta

            fitness_history.append(min_fitness)
            theta_history.append(best_theta)
            best_fitness_history.append(best_fitness)

        return {
            "theta_best": best_theta,
            "fitness_history": torch.tensor(fitness_history, device=self.device),
            "theta_history": torch.stack(theta_history),
            "best_fitness_history": torch.tensor(best_fitness_history, device=self.device),
        }

    # ---------------- Random Search Max ----------------
    def run_random_search_maximize(self, generations: int) -> Dict[str, torch.Tensor]:
        """Random search (maximization)"""
        dim_theta = self.dim_theta
        M = dim_theta // 2
        fitness_history = []
        theta_history = []
        best_fitness_history = []

        best_fitness = -float('inf')
        best_theta = None

        for gen in range(generations):
            theta_random = torch.rand(self.population_size, dim_theta, device=self.device) * (self.D**0.5)
            alphas= torch.clamp(theta_random[:, :M], min=0, max=1)
            sigmas = torch.clamp(theta_random[:, M:], min=0.05*(self.D**0.5), max=self.D**0.5)
            thetas_clipped = torch.cat([alphas, sigmas], dim=1)

            with torch.no_grad():
                fitness_scores = self.fitness_fn(thetas_clipped)
                max_idx = torch.argmax(fitness_scores)
                max_fitness = fitness_scores[max_idx].item()
                best_theta = thetas_eval[max_idx].reshape(-1).clone()

            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_theta = best_theta

            fitness_history.append(max_fitness)
            theta_history.append(best_theta)
            best_fitness_history.append(best_fitness)

        return {
            "theta_best": best_theta,
            "fitness_history": torch.tensor(fitness_history, device=self.device),
            "theta_history": torch.stack(theta_history),
            "best_fitness_history": torch.tensor(best_fitness_history, device=self.device),
        }

    
