import torch
from typing import List, Dict
from x_msg.extract_features import compute_features

def make_loss_function(
    num_samples: int,
    means: torch.Tensor,
    features_to_optimize: List[str],
    targets: Dict[str, float],
    weights: Dict[str, float],
    device: str = "cuda",
    feature_type: List[str] = ["optima_feature", "fdc_feature", "disp_feature"],
    ps: List[float] = [0.10], 
    seed: int | None = None
):
    def loss_fn(theta_batch: torch.Tensor) -> torch.Tensor:

        features = compute_features(
            int(num_samples),
            theta_batch,
            means,
            ps=ps,
            feature_type=feature_type,
            device=device, 
            seed=seed
        )

        total_loss = torch.zeros(theta_batch.shape[0], device=device)

        for feat in features_to_optimize:
            if feat not in features:
                raise KeyError(
                    f"[ERROR] Feature '{feat}' not found. "
                    f"Available keys: {list(features.keys())}"
                )
            feat_val = features[feat]
            target_val = targets.get(feat, 0.0)
            weight = weights.get(feat, 1.0)
            total_loss += (weight * (feat_val - target_val)) ** 2

        return total_loss

    return loss_fn
