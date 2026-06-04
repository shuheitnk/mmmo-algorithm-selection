import torch
from typing import List
from x_msg.extract_features import compute_features

def make_fitness_function(
    num_samples: int,
    means: torch.Tensor,
    features_to_optimize: List[str],
    device: str = "cuda",
    feature_type: List[str] = ["optima_feature", "fdc_feature", "disp_feature"],
    ps: List[float] = [0.10],
    seed: int | None = None,
):
    def fitness_fn(theta_batch: torch.Tensor) -> torch.Tensor:
        features = compute_features(
            num_samples,
            theta_batch,
            means,
            ps=ps,
            feature_type=feature_type,
            device=device, 
            seed=seed
        )
        return features[features_to_optimize[0]]

    return fitness_fn
