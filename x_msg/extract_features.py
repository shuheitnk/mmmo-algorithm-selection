import torch
from typing import List, Dict, Optional
from x_msg.sampling import sobol_sampling

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


@torch.no_grad()
def batch_forward(
    X: torch.Tensor,
    thetas: torch.Tensor,
    means: torch.Tensor,
    chunk_size: int = 100
) -> torch.Tensor:

    M, D = means.shape
    B, _ = thetas.shape
    N = X.shape[0]

    # Expand for batch computation
    X_expand = X.unsqueeze(0)            # (1, N, D)
    means_expand = means.unsqueeze(0)    # (1, M, D)

    # Squared distances
    X_norm2 = (X_expand ** 2).sum(dim=2, keepdim=True)
    M_norm2 = (means_expand ** 2).sum(dim=2).unsqueeze(1)
    sq_dist_base = X_norm2 + M_norm2 - 2 * (
        X_expand @ means_expand.transpose(1, 2)
    )  # (1, N, M)

    outputs = []

    for i in range(0, B, chunk_size):

        theta_chunk = thetas[i:i+chunk_size]
        b = theta_chunk.shape[0]

        # Split alpha and sigma
        alphas = theta_chunk[:, :M].unsqueeze(1)  # (b,1,M)
        sigmas = theta_chunk[:, M:].unsqueeze(1)  # (b,1,M)

        sq_dist = sq_dist_base.expand(b, -1, -1)

        # Gaussian values
        vals = alphas * torch.exp(-0.5 * sq_dist / (sigmas ** 2))
        Y = -vals.max(dim=2).values

        outputs.append(Y)

    return torch.cat(outputs, dim=0)


@torch.no_grad()
def minmax_normalization(Y: torch.Tensor) -> torch.Tensor:

    Y_min = Y.min(dim=1, keepdim=True).values
    Y_max = Y.max(dim=1, keepdim=True).values
    denom = Y_max - Y_min
    denom[denom == 0] = 1.0
    return (Y - Y_min) / denom


@torch.no_grad()
def compute_basic_components(thetas: torch.Tensor, means: torch.Tensor, device: str = None):

    if device is None:
        device = thetas.device

    thetas = thetas.to(device)
    means = means.to(device)

    B, M = thetas.shape[0], means.shape[0]
    alphas = thetas[:, :M]
    sigmas = thetas[:, M:]

    # Distance matrix
    dist2 = ((means[:, None, :] - means[None, :, :]) ** 2).sum(-1).unsqueeze(0).expand(B, -1, -1)
    sigma_sq = (2 * sigmas ** 2).unsqueeze(1)
    alpha_expand = alphas.unsqueeze(1)

    results = torch.exp(-dist2 / sigma_sq) * alpha_expand
    row_max_values, max_idx = results.max(dim=2)
    indicators = (max_idx == torch.arange(M, device=device)[None, :]).float()

    optima_alphas = indicators * alphas
    max_idx_global = optima_alphas.argmax(dim=1)
    global_optima_coords = means[max_idx_global]

    return indicators, global_optima_coords


@torch.no_grad()
def compute_distances_to_global(X: torch.Tensor, global_optima_coords: torch.Tensor) -> torch.Tensor:

    D = X.shape[1]
    diff = X[None, :, :] - global_optima_coords[:, None, :]
    return torch.linalg.norm(diff, dim=-1) / (D ** 0.5)


@torch.no_grad()
def extract_optima_features(indicators: torch.Tensor, device: str = 'cuda') -> Dict[str, torch.Tensor]:

    indicators = indicators.to(device)
    num_local_optima = indicators.sum(dim=1)
    return {"num_local_optima": num_local_optima}


@torch.no_grad()
def extract_fdc_feature(distances: torch.Tensor, Y: torch.Tensor) -> Dict[str, torch.Tensor]:

    x = minmax_normalization(distances)
    y_norm = minmax_normalization(Y)
    vx = x - x.mean(dim=1, keepdim=True)
    vy = y_norm - y_norm.mean(dim=1, keepdim=True)
    cov = (vx * vy).mean(dim=1)
    std_x = vx.std(dim=1)
    std_y = vy.std(dim=1)
    corr = cov / (std_x * std_y)
    return {"fdc": corr}


@torch.no_grad()
def compute_dispersion_features(
    X: torch.Tensor, Y: torch.Tensor, ps: List[float] = [0.10], use_fp16: bool = False
) -> Dict[str, torch.Tensor]:

    B, M = Y.shape
    D = X.shape[1]
    X_batch = X.unsqueeze(0).expand(B, -1, -1)
    ranks_norm = Y

    if use_fp16 and X_batch.is_cuda:
        X_batch = X_batch.half()
        ranks_norm = ranks_norm.half()

    dispersion_dict = {}
    for p in ps:
        top_k = max(int(M * p), 1)
        top_indices = torch.topk(-ranks_norm, top_k, dim=1).indices
        top_points = torch.gather(X_batch, 1, top_indices.unsqueeze(-1).expand(-1, -1, D))
        pairwise_dist = torch.cdist(top_points, top_points, p=2)
        triu_rows, triu_cols = torch.triu_indices(top_k, top_k, offset=1)
        disp_values = pairwise_dist[:, triu_rows, triu_cols].mean(dim=1)
        dispersion_dict[f"disp_{int(p*100)}pct"] = disp_values

    return dispersion_dict


@torch.no_grad()
def compute_all_features(num_samples: int, thetas: torch.Tensor, means: torch.Tensor, ps: List[float] = [0.10], device: str = 'cuda') -> Dict[str, torch.Tensor]:

    M, D = means.shape
    sampled_solutions = sobol_sampling(D, num_samples=num_samples - M, device=device)
    X = torch.cat([sampled_solutions, means], dim=0)
    Y = batch_forward(X, thetas, means)

    indicators, global_optima_coords = compute_basic_components(thetas, means, device=device)
    distances = compute_distances_to_global(X, global_optima_coords)
    optima_features = extract_optima_features(indicators, device=device)
    fdc_features = extract_fdc_feature(distances, Y)
    disp_features = compute_dispersion_features(X, Y, ps=ps)

    all_features = {}
    all_features.update(optima_features)
    all_features.update(fdc_features)
    all_features.update(disp_features)
    return all_features


@torch.no_grad()
def compute_features(
    num_samples: int,
    thetas: torch.Tensor,
    means: torch.Tensor,
    ps: List[float] = [0.1],
    device: str = 'cuda',
    feature_type: List[str] = ["optima_feature", "fdc_feature", "disp_feature", "r2_feature"], 
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:

    M, D = means.shape

    if seed is not None:
        set_seed(seed, device)

    thetas = thetas.reshape(1, -1) if thetas.ndim == 1 else thetas

    sampled_solutions = sobol_sampling(D, num_samples=num_samples - M, device=device, seed=seed)
    X = torch.cat([sampled_solutions, means], dim=0)
    features = {}

    if "optima_feature" in feature_type or "fdc_feature" in feature_type:
        indicators, global_optima_coords = compute_basic_components(thetas, means, device=device)

    if "optima_feature" in feature_type:
        features.update(extract_optima_features(indicators, device=device))

    if any(f in feature_type for f in ["fdc_feature", "disp_feature", "r2_feature"]):
        Y = batch_forward(X, thetas, means)

    if "fdc_feature" in feature_type:
        features.update(extract_fdc_feature(compute_distances_to_global(X, global_optima_coords), Y))

    if "disp_feature" in feature_type:
        features.update(compute_dispersion_features(X, Y, ps=ps))

    return features
