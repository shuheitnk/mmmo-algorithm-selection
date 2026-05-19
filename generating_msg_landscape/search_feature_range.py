import os
import time
import argparse
import torch
import random
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from x_msg.sampling import sobol_sampling
from x_msg.evolution_strategy import EvolutionStrategy
from x_msg.search_minmax_featurevalue import make_fitness_function


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

# =====================================================
# Utility: aggregate histories and save
# =====================================================
def aggregate_and_save(es_histories, save_path):
    es_stack = torch.stack(es_histories)

    stats = {
        "es_min": es_stack.min(dim=0).values,
        "es_max": es_stack.max(dim=0).values,
    }

    torch.save(stats, save_path)
    print("[Saved]", save_path)

# =====================================================
# Main experiment
# =====================================================
def run_experiment(D, num_gaussians, population_size, generations, num_runs, out_dir):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_base = 42
    mutation_std = 0.1 * (D**0.5) / (num_gaussians**0.5)
    num_samples = 50 * D

    feature_targets = ["num_local_optima", "fdc", "disp_10pct"]
    feature_types = ["optima_feature", "fdc_feature", "disp_feature"]
    output_names = ["num_local_optima", "fdc", "dispersion"]
    search_modes = ["min", "max"]

    save_dir = os.path.join(out_dir, f"feature_{D}d")
    os.makedirs(save_dir, exist_ok=True)

    for feat_idx in range(len(feature_types)):
        feature_key = feature_targets[feat_idx]
        feature_type = feature_types[feat_idx]
        output_name = output_names[feat_idx]

        for mode in search_modes:

            es_histories = []

            for run in range(num_runs):
                print(f"[{output_name}] mode={mode} run={run}")

                start_t = time.time()
                seed = seed_base + run

                # Means sampling
                torch.manual_seed(seed)
                means = sobol_sampling(D, num_gaussians, device=device, seed=seed)

                # Fitness
                loss_fn = make_fitness_function(
                    num_samples=num_samples,
                    means=means,
                    features_to_optimize=[feature_key],
                    feature_type=[feature_type],
                    seed=seed
                )

                # Manually specified initialization
                theta_init = torch.cat([
                    torch.full((num_gaussians,), 0.5, device=device),              # First half: heights = 0.5
                    torch.full((num_gaussians,), 0.05 * (D**0.5), device=device) # Second half: sigmas = 0.05 * sqrt(D)
                ])

                theta_init = theta_init.unsqueeze(0)

                # -------------------------------------------------
                # ES execution
                # -------------------------------------------------
                es = EvolutionStrategy(
                    dim_theta=2 * num_gaussians,
                    population_size=population_size,
                    mutation_std=mutation_std,
                    fitness_fn=loss_fn,
                    D=D,
                    device=device,
                    seed=seed,
                )

                if mode == "min":
                    es_res = es.run_vanilla_es(theta_init, generations=generations)
                else:
                    es_res = es.run_vanilla_es_maximize(theta_init, generations=generations)

                es_histories.append(es_res["fitness_history"].cpu())

                print("Elapsed:", time.time() - start_t, "sec")

            # Save results
            save_path = os.path.join(save_dir, f"{output_name}_{mode}_{D}d.pt")
            aggregate_and_save(es_histories, save_path)


# =====================================================
# Entry point
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="EvoMSG Feature Optimization Experiment")

    parser.add_argument("--D", type=int, required=True, help="Dimensionality of the search space")
    parser.add_argument("--out_dir", type=str, default="results", help="Relative path to save results")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--pop_size", type=int, default=200)
    parser.add_argument("--num_runs", type=int, default=11)
    

    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    run_experiment(
        D=args.D,
        num_gaussians=args.D * 50,
        population_size=args.pop_size,
        generations=args.generations,
        num_runs=args.num_runs,
        out_dir=out_dir,   
    )


if __name__ == "__main__":
    main()



