import argparse
import numpy as np
import torch
import pandas as pd
from pflacco import classical_ela_features, misc_features, sampling
import cocoex
import random
import os

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

def compute_bbob_ela(
    D=10,
    sampling_factor=500,
    out_dir=".",
    seed=42,
    device="cpu",
    functions=range(1, 25),
    instances=range(1, 11),
    runs=range(1, 12)
):
    set_seed(seed, device=device)
    rows_median, rows_allruns = [], []

    for f in functions:
        for inst in instances:
            run_features = []
            print(f"\n=== Function {f}, Instance {inst} ===")
            for run in runs:
                print(f"Run {run} / {len(runs)} ...", end="")
                # Sample solutions using LHS
                X_np = sampling.create_initial_sample(
                    dim=D,
                    sample_coefficient=sampling_factor,
                    sample_type='lhs',
                    seed=seed + run  
                )
                X_np = np.asarray(X_np, dtype=np.float64)
                X_scaled = X_np * 10 - 5  # scale to BBOB domain [-5,5]^D

                problem = cocoex.BareProblem("bbob", function=f, dimension=D, instance=inst)
                y = np.array([problem(x) for x in X_scaled])
                y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
                y_tensor = torch.tensor(y_norm, dtype=torch.float32, device=device)

                X_df = pd.DataFrame(X_np, columns=[f"x{i+1}" for i in range(D)])
                y_series = pd.Series(y_tensor.cpu().numpy().ravel(), name="y")

                features = {}
                features.update(classical_ela_features.calculate_ela_distribution(X_df, y_series))
                features.update(classical_ela_features.calculate_ela_level(X_df, y_series))
                features.update(classical_ela_features.calculate_ela_meta(X_df, y_series))
                features.update(classical_ela_features.calculate_pca(X_df, y_series))
                features.update(classical_ela_features.calculate_nbc(X_df, y_series))
                features.update(classical_ela_features.calculate_dispersion(X_df, y_series))
                features.update(classical_ela_features.calculate_information_content(X_df, y_series))
                features.update(misc_features.calculate_fitness_distance_correlation(X_df, y_series))

                features = {k: v for k, v in features.items() if not k.endswith("costs_runtime")}

                all_features = {"function_id": f, "instance_id": inst, "seed": seed + run, **features}
                rows_allruns.append(all_features)
                run_features.append(features)

                print(" done")


            df_runs = pd.DataFrame(run_features)
            median_features = df_runs.median(axis=0).to_dict()
            rows_median.append({"function_id": f, "instance_id": inst, **median_features})


    df_all = pd.DataFrame(rows_allruns)
    df_median = pd.DataFrame(rows_median)

    id_cols_all = ["function_id", "instance_id", "seed"]
    id_cols_median = ["function_id", "instance_id"]

    feature_cols = [c for c in df_all.columns if c not in id_cols_all]

    df_all = df_all[id_cols_all + feature_cols]
    df_median = df_median[id_cols_median + feature_cols]


    os.makedirs(out_dir, exist_ok=True)

    df_all.to_csv(os.path.join(out_dir, f"ela_bbob_{D}d_all.csv"), index=False)
    df_median.to_csv(os.path.join(out_dir, f"ela_bbob_{D}d_median.csv"), index=False)


    return pd.DataFrame(rows_median), pd.DataFrame(rows_allruns)


def main():
    parser = argparse.ArgumentParser(description="BBOB ELA feature computation")
    parser.add_argument("--D", type=int, default=10, help="Dimension")
    parser.add_argument("--sampling_factor", type=int, default=500, help="LHS factor")
    parser.add_argument("--out_dir", type=str, default=".", help="Output folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_runs", type=int, default=11, help="Number of runs per function/instance")
    parser.add_argument("--max_functions", type=int, default=24, help="maximum BBOB function IDs")
    parser.add_argument("--max_instances", type=int, default=10, help="maximum BBOB instance IDs")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device")
    args = parser.parse_args()

    compute_bbob_ela(D=args.D, sampling_factor=args.sampling_factor, out_dir=args.out_dir, seed=args.seed, device=args.device, functions=range(1, args.max_functions + 1), instances=range(1, args.max_instances + 1), runs=range(1, args.num_runs + 1))


if __name__ == "__main__":
    main()
