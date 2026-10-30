"""CPU threading helpers (NumPy replacement for the GPU / torchrun helpers of the PyTorch build).

hoh runs on the CPU only: there is no CUDA, no DDP and no torchrun. The speed knob is the number of BLAS threads used by NumPy
matrix products. `apply_threads` uses threadpoolctl when installed; otherwise set OMP_NUM_THREADS / OPENBLAS_NUM_THREADS /
MKL_NUM_THREADS in the environment BEFORE starting Python (NumPy reads them at import time)."""
import os


def resolve_threads(requested):
    """'auto' | None | 0 -> all CPUs; int / str(int) -> that many (at least 1)."""
    n_cpu = os.cpu_count() or 1
    if requested in (None, "", "auto", 0, "0"):
        return n_cpu
    return max(1, min(int(requested), n_cpu))


def apply_threads(n):
    """Best effort. Returns True if the limit was applied in this process."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return False
    threadpool_limits(limits=int(n))
    return True


def run_train(train_args, threads=None, runner=None):
    """Train in-process (the only mode there is). `threads` limits BLAS threads when threadpoolctl is available."""
    from .train import train
    if threads not in (None, "", "auto", 0, "0"):
        apply_threads(resolve_threads(threads))
    return train(list(train_args))
