import argparse
import os
import time
from collections import defaultdict
import torch
import pandas as pd
import create_msg_samples
import random
import numpy as np
import re
import sys
from pathlib import Path
from x_msg.construct_msg_landscape import MSGLandscape
from x_msg.sampling import sobol_sampling
from x_msg.make_loss_function import make_loss_function
from x_msg.evolution_strategy import EvolutionStrategy
from pflacco import classical_ela_features, misc_features

sys.path.append(str(Path(__file__).resolve().parent.parent))


def set_seed(seed: int, device: str = "cuda") -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_ela_features(X, y):
    """
    Compute ELA features for a given sample set X and objective values y.
    """
    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    y_min, y_max = y_np.min(), y_np.max()
    y_norm = (y_np - y_min) / (y_max - y_min) if y_max - y_min > 0 else y_np * 0.0

    feats = {
        **classical_ela_features.calculate_ela_distribution(X, y_norm),
        **classical_ela_features.calculate_ela_level(X, y_norm),
        **classical_ela_features.calculate_ela_meta(X, y_norm),
        **classical_ela_features.calculate_pca(X, y_norm),
        **classical_ela_features.calculate_nbc(X, y_norm),
        **classical_ela_features.calculate_dispersion(X, y_norm),
        **classical_ela_features.calculate_information_content(X, y_norm),
        **misc_features.calculate_fitness_distance_correlation(X, y_norm),
    }
    # Remove runtime costs
    return {k: v for k, v in feats.items() if not k.endswith("costs_runtime")}

def generate_tag(num_opt, fdc, disp, D, suffix=""):
    tag = f"{num_opt}_{fdc}_{disp}_{D}d"
    if suffix:
        tag += f"_{suffix}"
    return tag

def load_feature_range_files(base_path, dims, features):
    data = {}
    for dim in dims:
        data[dim] = {}
        for feat in features:
            fname_max = f"{feat}_max_{dim}.pt"
            fname_min = f"{feat}_min_{dim}.pt"
            path_max = os.path.join(base_path, f"feature_{dim}", fname_max)
            path_min = os.path.join(base_path, f"feature_{dim}", fname_min)

            if not os.path.isfile(path_max) or not os.path.isfile(path_min):
                raise FileNotFoundError(f"Missing feature file(s) for {feat} in {dim}")

            val_max = torch.load(path_max, map_location="cpu", weights_only=True)["es_max"][-1]
            val_min = torch.load(path_min, map_location="cpu", weights_only=True)["es_min"][-1]
            val_range = val_max - val_min
            data[dim][feat] = {"max": float(val_max), "min": float(val_min), "range": float(val_range)}

            print(f"[DEBUG] Loaded {feat} range in {dim}: min={val_min:.4f}, max={val_max:.4f}, range={val_range:.4f}")

    return data


def parse_args():
    parser = argparse.ArgumentParser(description="EvoMSG Optimization Pipeline")
    parser.add_argument("--D", type=int, default=2, help="Dimensionality (e.g., 2,5,10)")
    parser.add_argument("--num_gaussians", type=int, default=100, help="Number of Gaussians in MSG")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_path", type=str, default=".", help="Base path where feature_* folders are found")
    parser.add_argument("--out_dir", type=str, default="out_ela", help="Directory to save results")
    parser.add_argument("--pop_size", type=int, default=200, help="ES population size")
    parser.add_argument("--generations", type=int, default=200, help="ES generations")
    parser.add_argument("--sampling_factor", type=int, default=500, help="Samples for feature computation")
    parser.add_argument("--num_runs", type=int, default=11, help="Number of repetitions for ELA aggregation")
    return parser.parse_args()



def _type_to_int(x: str) -> int:
    return 1 if x == "min" else 2

def make_function_id(opt_type, fdc_type, disp_type) -> int:
    return (
        100 * _type_to_int(opt_type)
        + 10 * _type_to_int(fdc_type)
        + _type_to_int(disp_type)
    )

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed, device=device)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Random seed: {args.seed}")

    # Create output directory
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Results will be saved in: {out_dir}")

    # Load feature ranges
    dims = ["2d", "5d", "10d"]
    features = ["dispersion", "fdc", "num_local_optima"]
    print(f"[INFO] Loading feature range files from {args.base_path} ...")
    try:
        data = load_feature_range_files(args.base_path, dims, features)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    print("[INFO] Feature ranges loaded successfully.")

    # Generate Gaussian centers
    means = sobol_sampling(args.D, args.num_gaussians, device=device, seed=args.seed)
    print(f"[DEBUG] Generated {means.shape[0]} Gaussian centers in {args.D}D space.")

    # Manually specified initialization
    theta_init = torch.cat([
        torch.full((args.num_gaussians,), 0.5, device=device),              # First half: heights = 0.5
        torch.full((args.num_gaussians,), 0.05 * (args.D**0.5), device=device) # Second half: sigmas = 0.05 * sqrt(D)
    ])

    theta_init = theta_init.unsqueeze(0)
    

    feature_types = ["optima_feature", "fdc_feature", "disp_feature"]
    search_types = ["min", "max"]

    df_all_msg = []
    df_median_msg = []

    for opt_type in search_types:
        for fdc_type in search_types:
            for disp_type in search_types:
                print(f"\n[RUN] num_local_optima={opt_type} | fdc={fdc_type} | dispersion={disp_type}")
                dim_key = f"{args.D}d"


                tag = generate_tag(opt_type, fdc_type, disp_type, args.D)
                out_pt = os.path.join(out_dir, f"results_{tag}.pt")
                out_csv_all = os.path.join(out_dir, f"ela_features_{tag}_all.csv")
                out_csv_median = os.path.join(out_dir, f"ela_features_{tag}_median.csv")

                if os.path.isfile(out_pt) and os.path.isfile(out_csv_all) and os.path.isfile(out_csv_median):
                    print(f"[SKIP] already exists: {tag}")
                    continue

                try:
                    targets = {
                        "num_local_optima": data[dim_key]["num_local_optima"][opt_type]+1,
                        "fdc": data[dim_key]["fdc"][fdc_type],
                        "disp_10pct": data[dim_key]["dispersion"][disp_type],
                    }
                    weights = {
                        "num_local_optima": 1.0 / data[dim_key]["num_local_optima"]["range"],
                        "fdc": 1.0 / data[dim_key]["fdc"]["range"],
                        "disp_10pct": 1.0 / data[dim_key]["dispersion"]["range"],
                    }
                    print(f"[DEBUG] Targets: {targets}")
                    print(f"[DEBUG] Weights: {weights}")
                except KeyError as e:
                    print(f"[ERROR] Missing key in loaded data: {e}", file=sys.stderr)
                    continue

                # Create ES loss function
                loss_fn = make_loss_function(
                    num_samples=args.sampling_factor * args.D,
                    means=means,
                    features_to_optimize=list(targets.keys()),
                    targets=targets,
                    weights=weights,
                    device=device,
                    feature_type=feature_types,
                    seed=args.seed,
                )

                # Run Evolution Strategy
                mutation_std = 0.1 * (args.D ** 0.5) / (args.num_gaussians ** 0.5)
                es = EvolutionStrategy(
                    dim_theta=2 * args.num_gaussians,
                    population_size=args.pop_size,
                    mutation_std=mutation_std,
                    fitness_fn=loss_fn,
                    D=args.D,
                    device=device,
                    seed=args.seed,
                )
                print(f"[INFO] Starting Evolution Strategy: pop_size={args.pop_size}, gen={args.generations}, mutation_std={mutation_std:.4f}")
                start_es = time.time()
                result = es.run_vanilla_es(theta_init, generations=args.generations)
                print(f"[INFO] ES finished in {time.time() - start_es:.1f}s")

                df_all_reps = pd.DataFrame()
                features_by_col = defaultdict(list)
                for rep in range(1, args.num_runs + 1):
                    start_rep = time.time()
                    rep_seed = args.seed + rep
                    set_seed(rep_seed, device=device)
                    print(f"[INFO] Computing ELA features for repetition {rep}/{args.num_runs} with seed {rep_seed} ...")
                    X, Y = create_msg_samples.create_msg_samples(
                        result, means, MSGLandscape, args.D,
                        sampling_factor=args.sampling_factor,
                        device=device, seed=rep_seed
                    )
                    print(f"[DEBUG] Generated sample matrix X shape: {X.shape}, Y shape: {Y.shape}")

                    for col_idx, col in enumerate(Y.columns, start=1):
                        start_col = time.time()
                        
                        function_id = make_function_id(opt_type, fdc_type, disp_type)
                        feat = compute_ela_features(X, Y[col])
                        
                        feat["function_id"] = function_id

                        feat["instance_id"] = int(re.search(r'\d+', col).group())  # gen → instance_id
                        feat["seed"] = int(rep_seed)           # seed → run

                        cols = ["function_id", "instance_id", "seed"] + [
                            c for c in feat.keys()
                            if c not in ["function_id", "instance_id", "seed"]
                        ]

                        df_all_reps = pd.concat(
                            [df_all_reps, pd.DataFrame([feat], columns=cols)],
                            axis=0,
                            ignore_index=True
                        )
                        features_by_col[col].append(feat)


                        elapsed_col = time.time() - start_col
                        print(f"[DEBUG] Rep {rep}/{args.num_runs}, Col {col_idx}/{len(Y.columns)} ('{col}') processed in {elapsed_col:.2f}s")

                    print(f"[INFO] Completed rep {rep}/{args.num_runs} ({time.time() - start_rep:.2f}s)")

                tag_all = generate_tag(opt_type, fdc_type, disp_type, args.D, suffix="all")
                out_csv_all = os.path.join(out_dir, f"ela_features_{tag_all}.csv")
                df_all_reps.to_csv(out_csv_all, index=False)
                print(f"[SAVED] All reps features: {out_csv_all}")

                df_all_msg.append(df_all_reps)

                df_median = (
                    df_all_reps
                    .groupby(["function_id", "instance_id"], as_index=False)
                    .median(numeric_only=True)
                )

                tag_median = generate_tag(opt_type, fdc_type, disp_type, args.D, suffix="median")
                out_csv_median = os.path.join(out_dir, f"ela_features_{tag_median}.csv")
                df_median.to_csv(out_csv_median, index=False)
                print(f"[SAVED] Median features: {out_csv_median}")

                df_median_msg.append(df_median)
                

                out_pt = os.path.join(out_dir, f"results_{generate_tag(opt_type, fdc_type, disp_type, args.D)}.pt")
                torch.save(result, out_pt)
                print(f"[SAVED] ES result: {out_pt}")

    df_all_merged = pd.concat(df_all_msg, ignore_index=True)
    df_median_merged = pd.concat(df_median_msg, ignore_index=True)

    id_cols_all = ["function_id", "instance_id", "seed"]
    id_cols_median = ["function_id", "instance_id"]

    feature_cols = [c for c in df_all_merged.columns if c not in id_cols_all]

    df_all_merged = df_all_merged[id_cols_all + feature_cols]


    feature_cols_median = [
        c for c in df_median_merged.columns
        if c not in id_cols_median
    ]

    df_median_merged = df_median_merged[id_cols_median + feature_cols_median]


    df_all_merged.to_csv(
        os.path.join(out_dir, f"ela_msg_{args.D}d_all.csv"),
        index=False
    )
    df_median_merged.to_csv(
        os.path.join(out_dir, f"ela_msg_{args.D}d_median.csv"),
        index=False
    )

    print("[SAVED] BBOB-style merged MSG files:")
    print(f"  - ela_msg_{args.D}d_all.csv")
    print(f"  - ela_msg_{args.D}d_median.csv")


    print("[ALL DONE]")

if __name__ == "__main__":
    main()

