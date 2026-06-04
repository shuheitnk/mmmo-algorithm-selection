import torch
import pandas as pd
from pflacco import sampling

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from x_msg.make_multi_objective_msg import make_multi_objective_msg
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


# -------------------------
# Min–max normalization
# -------------------------
def minmax_norm(tensor):
    min_val = tensor.min(dim=0, keepdim=True)[0]
    max_val = tensor.max(dim=0, keepdim=True)[0]
    return (tensor - min_val) / (max_val - min_val)  


# -------------------------
# Remove duplicate theta
# -------------------------
def unique_theta_list(theta_list, device=None):
    """
    Remove duplicate theta tensors and optionally move to a specific device.
    """
    seen = set()
    unique_thetas = []

    for t in theta_list:
        if device is not None:
                t = t.to(device)
        unique_thetas.append(t)

    return unique_thetas


# -------------------------
# Remove duplicate objective outputs
# -------------------------
def list_to_tensor(tensor_list):
    """
    Convert a list of tensors to a single stacked tensor.
    
    Args:
        tensor_list (List[torch.Tensor]): List of tensors with the same shape
    
    Returns:
        torch.Tensor: Stacked tensor of shape (len(tensor_list), ...)
    """
    if not tensor_list:
        return torch.empty((0, 0))
    
    return torch.stack(tensor_list)



# -------------------------
# Single-objective MSG sampling
# -------------------------
def create_msg_samples(
        result,
        means,
        MSGLandscape,
        D,
        sampling_factor=500,
        device='cuda',
        seed=None,
        m=1):
    """
    Generate normalized MSG sample data for single-objective landscapes.
    """
    # -------------------------
    # Seed configuration
    # -------------------------
    if seed is not None:
        set_seed(seed, device=device)

    # -------------------------
    # LHS sampling
    # -------------------------
    x_samples_np = sampling.create_initial_sample(
        dim=D,
        sample_coefficient=sampling_factor,
        sample_type='lhs',
        seed=seed
    )
    x_samples = torch.tensor(x_samples_np.values, dtype=torch.float32, device=device)

    # -------------------------
    # Remove duplicate theta
    # -------------------------
    theta_history = result.get("theta_history", [])
    unique_thetas = unique_theta_list(theta_history)

    # -------------------------
    # Evaluate MSG landscapes
    # -------------------------
    y_list = []
    with torch.no_grad():
        for theta in unique_thetas:
            msg = MSGLandscape(means, theta)
            if m == 1:
                y = msg.forward(X=x_samples)
                y_list.append(y.cpu())

    y_unique = list_to_tensor(y_list)

    # -------------------------
    # Transpose and normalize
    # -------------------------
    y_T = y_unique.T  # (n_samples, num_unique)
    y_norm = minmax_norm(y_T)

    # -------------------------
    # Convert to DataFrame
    # -------------------------
    X_df = pd.DataFrame(x_samples.cpu().numpy(), columns=[f"x{i+1}" for i in range(D)])
    Y_df = pd.DataFrame(y_norm.cpu().numpy(), columns=[f"gen{i+1}" for i in range(y_norm.shape[1])])

    return X_df, Y_df


# -------------------------
# Bi-objective MSG sampling
# -------------------------
def create_bi_msg_samples(
        result,
        means,
        MSGLandscape,
        D,
        sampling_factor=500,
        device='cuda',
        seed=None):
    """
    Generate normalized sample data for bi-objective MSG landscapes.
    """
    # -------------------------
    # Seed configuration
    # -------------------------
    if seed is not None:
        set_seed(seed, device=device)

    # -------------------------
    # LHS sampling
    # -------------------------
    x_samples_df = sampling.create_initial_sample(
        dim=D + 1,
        sample_coefficient=sampling_factor,
        sample_type='lhs',
        seed=seed
    )
    x_samples = torch.tensor(x_samples_df.values, dtype=torch.float32, device=device)

    # -------------------------
    # Remove duplicate theta
    # -------------------------
    theta_history = result.get("theta_history", [])
    unique_thetas = unique_theta_list(theta_history, device=device)

    if len(unique_thetas) == 0:
        raise ValueError("theta_history is empty.")

    y1_list, y2_list = [], []

    # -------------------------
    # Evaluate bi-objective MSG
    # -------------------------
    with torch.no_grad():
        for theta in unique_thetas:
            distance_function = MSGLandscape(means.to(device), theta)
            msg = make_multi_objective_msg(
                m=2,
                dim_msg=D,
                function_g=distance_function
            )
            y_vec = msg(x=x_samples)
            y1_list.append(y_vec[:, 0].cpu())
            y2_list.append(y_vec[:, 1].cpu())

    # -------------------------
    # Remove duplicate outputs
    # -------------------------

    y1_unique = list_to_tensor(y1_list)
    y2_unique = list_to_tensor(y2_list)

    print("y1_list", len(y1_unique))
    print("y2_list", len(y2_unique))

    # -------------------------
    # Transpose and normalize
    # -------------------------
    y1_T = y1_unique.T
    y2_T = y2_unique.T

    y1_norm = minmax_norm(y1_T)
    y2_norm = minmax_norm(y2_T)

    # -------------------------
    # Convert to DataFrame
    # -------------------------
    X_df = pd.DataFrame(x_samples.cpu().numpy(), columns=[f"x{i+1}" for i in range(x_samples.shape[1])])
    Y1_df = pd.DataFrame(y1_norm.cpu().numpy(), columns=[f"gen{i+1}" for i in range(y1_norm.shape[1])])
    Y2_df = pd.DataFrame(y2_norm.cpu().numpy(), columns=[f"gen{i+1}" for i in range(y2_norm.shape[1])])

    return X_df, Y1_df, Y2_df

