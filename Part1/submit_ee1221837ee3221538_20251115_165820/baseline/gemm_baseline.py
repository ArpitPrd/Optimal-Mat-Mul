# Usage: python3 gemm_baseline.py N num_procs
import sys
import numpy as np
from multiprocessing import Pool

def worker_multiply(args):
    A_block, B = args
    return A_block.dot(B)

def run_gemm(P, blocks, B):
    """Encapsulates the core multiplication logic."""
    with Pool(P) as p:
        C_blocks = p.map(worker_multiply, [(blk, B) for blk in blocks])
    # vstack is part of the operation
    C = np.vstack(C_blocks) 
    return C

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 gemm_baseline.py N num_procs')
        sys.exit(1)
    N = int(sys.argv[1])
    P = int(sys.argv[2])

    # reproducible random
    rng = np.random.default_rng(12345)
    A = rng.standard_normal((N, N)).astype(np.float64)
    B = rng.standard_normal((N, N)).astype(np.float64)

    # simple blocked approach: split A into P row-blocks
    blocks = [A[i::P, :] for i in range(P)]

    # --- Run the multiplication one time ---
    _ = run_gemm(P, blocks, B)

    # All timing and reporting has been removed.