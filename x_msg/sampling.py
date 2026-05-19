import torch
from torch.quasirandom import SobolEngine
from concurrent.futures import ThreadPoolExecutor
import queue
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



# =====================================================
# Sobol Sampling
# =====================================================
def sobol_sampling(
    dim: int,
    num_samples: int,
    max_workers: int = 4,
    device: str = "cuda",
    vram_gb: float = 6.0,
    max_batch_points: int = 50_000,
    seed: int | None = None,
) -> torch.Tensor:

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but device='cuda' was specified.")

    device = torch.device(device)
    results = torch.empty((num_samples, dim), device=device, dtype=torch.float32)

    bytes_per_point = dim * 4  # float32
    batch_size = min(max_batch_points, int(vram_gb * 1e9 * 0.2 // bytes_per_point))
    total_batches = (num_samples + batch_size - 1) // batch_size

    q = queue.Queue(maxsize=max_workers * 2)

    if seed is not None:
        set_seed(seed, device=device)

    def cpu_producer():
        """Generate Sobol points on CPU and enqueue batches."""
        engine = SobolEngine(dim, scramble=True, seed=seed)
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            current_size = min(batch_size, num_samples - start)
            engine.fast_forward(start)
            q.put((start, engine.draw(current_size, dtype=torch.float32)))
        # Signal consumers to stop
        for _ in range(max_workers):
            q.put(None)

    def gpu_consumer():
        while True:
            item = q.get()
            if item is None:
                break
            start, cpu_batch = item
            results[start:start + cpu_batch.size(0)] = cpu_batch.to(device, non_blocking=True)
            del cpu_batch

    with ThreadPoolExecutor(max_workers=max_workers + 1) as executor:
        executor.submit(cpu_producer)
        futures = [executor.submit(gpu_consumer) for _ in range(max_workers)]
        for f in futures:
            f.result()

    return results
