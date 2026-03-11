"""
Binary search over --alpha to find the smallest value that causes
NLL to change by more than `nll_threshold` over `min_iters` iterations.

Usage:
    python search_alpha.py [--cutoff 0.5] [--min_iters 2000] [--threshold 0.005]
"""

import subprocess
import sys
import os
import argparse
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────
# Fixed training flags (same across all runs)
FIXED_FLAGS = [
    "--task",       "torder",
    "--model",      "rnn",
    "--init",       "smart_tanh",
    "--nhid",       "200",
    "--lr",         "0.0001",
    "--bs",         "20",
    "--min_length", "50",
    "--max_length", "200",
    "--ebs",        "10000",
    "--cbs",        "1000",
    "--checkFreq",  "20",
    "--seed",       "52",
    "--valid_seed", "12345",
]

NLL_WINDOW = 50   # average this many iters at start/end to smooth noise


# ── Core helpers ───────────────────────────────────────────────────────────────

def run_training(alpha: float, cutoff: float, clipstyle: str, n_iters: int, name: str) -> str:
    """Launch the training script and return path to the saved npz."""
    npz = f"{name}_final_state.npz"
    clip_flags = ["--clipstyle", clipstyle]
    if clipstyle == "rescale":
        clip_flags += ["--cutoff", str(cutoff)]
    cmd = [
        sys.executable, "-m", "trainingRNNs_torch.train",
        *FIXED_FLAGS,
        "--alpha",    str(alpha),
        *clip_flags,
        "--maxiters", str(n_iters),
        "--name",     name,
    ]
    print(f"\n  Running: alpha={alpha:.4f}  clipstyle={clipstyle}  cutoff={cutoff if clipstyle == 'rescale' else 'N/A'}  iters={n_iters}")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for alpha={alpha}")
    return npz


def nll_change(npz_path: str, window: int = NLL_WINDOW) -> float:
    """
    Load train_nll from npz and return:
        mean(first `window` iters) - mean(last `window` iters)
    Positive value = NLL decreased = some learning happened.
    """
    with np.load(npz_path) as z:
        nll = z["train_nll"].copy()

    valid = nll[nll != -1.0]
    if len(valid) < 2 * window:
        window = max(1, len(valid) // 4)
    start_nll = float(valid[:window].mean())
    end_nll   = float(valid[-window:].mean())
    change    = start_nll - end_nll          # positive = improvement
    print(f"  NLL: start={start_nll:.5f}  end={end_nll:.5f}  change={change:.5f}")
    os.remove(npz_path)
    print(f"  Deleted: {npz_path}")
    return change


def is_learning(npz_path: str, threshold: float, window: int = NLL_WINDOW) -> bool:
    return nll_change(npz_path, window) > threshold


# ── Search ─────────────────────────────────────────────────────────────────────

def exponential_search(cutoff, clipstyle, n_iters, threshold, alpha_max=32.0):
    """
    Phase 1: starting from alpha=0, double until we find an alpha that works.
    Returns (lo, hi) bracket where lo doesn't work and hi does,
    or (None, None) if nothing works up to alpha_max.
    """
    print("\n══ Phase 1: Exponential search to find working bracket ══")
    alpha = 0.0
    prev_alpha = None

    # First check alpha=0 (no Omega)
    name = f"search_alpha_{alpha:.4f}".replace(".", "p")
    npz  = run_training(alpha, cutoff, clipstyle, n_iters, name)
    if is_learning(npz, threshold):
        print(f"  alpha=0 already works (NLL changes enough without Omega).")
        return None, alpha      # no search needed

    prev_alpha = alpha
    alpha = 0.5

    while alpha <= alpha_max:
        name = f"search_alpha_{alpha:.4f}".replace(".", "p")
        npz  = run_training(alpha, cutoff, clipstyle, n_iters, name)
        if is_learning(npz, threshold):
            print(f"\n  ✓ Bracket found: lo={prev_alpha}  hi={alpha}")
            return prev_alpha, alpha
        prev_alpha = alpha
        alpha *= 2.0

    print(f"\n  ✗ No alpha up to {alpha_max} caused sufficient NLL change.")
    return None, None


def binary_search(lo, hi, cutoff, clipstyle, n_iters, threshold, tol=0.25, max_rounds=8):
    """
    Phase 2: binary search in [lo, hi] to find the smallest alpha that works.
    Stops when hi - lo < tol or max_rounds reached.
    """
    print(f"\n══ Phase 2: Binary search in [{lo:.4f}, {hi:.4f}] ══")
    best_alpha = hi

    for rnd in range(max_rounds):
        if hi - lo < tol:
            print(f"  Bracket too narrow ({hi-lo:.4f} < tol={tol}), stopping.")
            break

        mid = (lo + hi) / 2.0
        print(f"\n  Round {rnd+1}: testing alpha={mid:.4f}  (bracket [{lo:.4f}, {hi:.4f}])")
        name = f"search_alpha_{mid:.4f}".replace(".", "p")
        npz  = run_training(mid, cutoff, clipstyle, n_iters, name)

        if is_learning(npz, threshold):
            best_alpha = mid
            hi = mid
            print(f"  ✓ Works — narrowing upper bound to {hi:.4f}")
        else:
            lo = mid
            print(f"  ✗ Doesn't work — raising lower bound to {lo:.4f}")

    return best_alpha


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cutoff",    type=float, default=5,
                   help="Gradient clip cutoff (fixed across search). Ignored if --clipstyle nothing.")
    p.add_argument("--clipstyle", type=str,   default="nothing", choices=["rescale", "nothing"],
                   help="Clipping style: rescale (use --cutoff) or nothing (no clipping).")
    p.add_argument("--min_iters", type=int,   default=200,
                   help="Training iterations per trial.")
    p.add_argument("--threshold", type=float, default=0.005,
                   help="Minimum NLL drop to count as 'learning'.")
    p.add_argument("--alpha_max", type=float, default=256.0,
                   help="Maximum alpha to try in exponential search.")
    p.add_argument("--tol",       type=float, default=0.25,
                   help="Binary search stops when bracket < tol.")
    p.add_argument("--max_rounds",type=int,   default=8,
                   help="Maximum binary search rounds.")
    args = p.parse_args()

    print(f"""
════════════════════════════════════════
  Alpha Binary Search
  clipstyle = {args.clipstyle}
  cutoff    = {args.cutoff if args.clipstyle == "rescale" else "N/A"}
  min_iters = {args.min_iters}
  threshold = {args.threshold}
  alpha_max = {args.alpha_max}
════════════════════════════════════════""")

    # Phase 1
    lo, hi = exponential_search(
        cutoff=args.cutoff,
        clipstyle=args.clipstyle,
        n_iters=args.min_iters,
        threshold=args.threshold,
        alpha_max=args.alpha_max,
    )

    if lo is None and hi is None:
        print("\n✗ Search failed: no alpha found. Try raising --alpha_max or --cutoff.")
        sys.exit(1)

    if lo is None:
        # alpha=0 already works
        print(f"\n✓ Best alpha = {hi} (no Omega needed)")
        sys.exit(0)

    # Phase 2
    best = binary_search(
        lo=lo, hi=hi,
        cutoff=args.cutoff,
        clipstyle=args.clipstyle,
        n_iters=args.min_iters,
        threshold=args.threshold,
        tol=args.tol,
        max_rounds=args.max_rounds,
    )

    clip_flags = f"--clipstyle rescale --cutoff {args.cutoff}" if args.clipstyle == "rescale" else "--clipstyle nothing"
    print(f"""
════════════════════════════════════════
  ✓ Search complete
  Best alpha found : {best:.4f}
  clipstyle        : {args.clipstyle}
  cutoff           : {args.cutoff if args.clipstyle == "rescale" else "N/A"}
  NLL threshold    : {args.threshold}
════════════════════════════════════════

  Suggested full run:
  python -m trainingRNNs_torch.train \\
    --task torder --model rnn --init smart_tanh \\
    --alpha {best:.4f} {clip_flags} \\
    --nhid 50 --lr 0.01 --bs 20 \\
    --min_length 50 --max_length 200 \\
    --maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 \\
    --seed 52 --valid_seed 12345 \\
    --collectDiags --diagBins 60 --satThresh 0.05 \\
    --name EC_torder_best
""")


if __name__ == "__main__":
    main()