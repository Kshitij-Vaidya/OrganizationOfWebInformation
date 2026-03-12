import os

import numpy as np

# Use a non-interactive backend for headless environments.
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


RUNS = [
    # Memorization (classification)
    dict(name="A1_mem_rnn_tanh_noclip", task="mem", model="rnn", cutoff=None),
    dict(name="A2_mem_rnn_tanh_clip005", task="mem", model="rnn", cutoff=0.05),
    dict(name="A3_mem_rnn_tanh_clip001", task="mem", model="rnn", cutoff=0.01),
    dict(name="A4_mem_gru_noclip", task="mem", model="gru", cutoff=None),
    dict(name="A5_mem_gru_clip005", task="mem", model="gru", cutoff=0.05),
    # Multiplication (regression)
    dict(name="B1_mul_rnn_tanh_noclip", task="mul", model="rnn", cutoff=None),
    dict(name="B2_mul_gru_noclip", task="mul", model="gru", cutoff=None),
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _finite_prefix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    m = np.isfinite(x) & (x >= 0)
    if not np.any(m):
        return np.array([], dtype=float)
    idx = np.where(m)[0]
    return x[idx[0] : idx[-1] + 1]


def _last_nonempty_row(a: np.ndarray) -> int:
    if a.ndim != 2:
        raise ValueError(a.shape)
    rows = np.where(np.isfinite(a).any(axis=1))[0]
    return int(rows[-1]) if len(rows) else 0


def save_rho_curve(z: dict, out_png: str, title: str) -> None:
    rho = _finite_prefix(z["rho_Whh"])
    check_freq = int(z.get("checkFreq", 1))
    if len(rho) == 0:
        return

    x = np.arange(len(rho)) * check_freq
    plt.figure(figsize=(6.4, 4.0))
    plt.plot(x, rho, lw=1.5)
    plt.xlabel("iteration")
    plt.ylabel(r"$\rho(W_{hh})$ (spectral radius)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_gradnorm_curve(z: dict, out_png: str, title: str, cutoff: float | None) -> None:
    gn = np.asarray(z["gradient_norm"], dtype=float)
    m = np.isfinite(gn) & (gn >= 0)
    if not np.any(m):
        return
    gn = gn.copy()
    gn[~m] = np.nan

    plt.figure(figsize=(6.4, 4.0))
    plt.plot(gn, lw=0.8, label="pre-clip (stored)")
    if cutoff is not None:
        # Reconstruct post-clip norm for rescale clipping:
        # if ||g||>cutoff, grads are scaled so global norm == cutoff.
        gn_post = np.minimum(gn, cutoff)
        plt.plot(gn_post, lw=1.2, label=f"post-clip (recon, cutoff={cutoff:g})")
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel(r"$\|\nabla_\theta L\|_2$ (log scale)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_gate_hist(z: dict, out_png: str, title: str) -> None:
    zsat = np.asarray(z.get("gate_z_sat_time"))
    rsat = np.asarray(z.get("gate_r_sat_time"))
    if zsat.ndim != 2 or rsat.ndim != 2:
        return

    zi = _last_nonempty_row(zsat)
    ri = _last_nonempty_row(rsat)
    zv = zsat[zi]
    rv = rsat[ri]
    zv = zv[np.isfinite(zv)]
    rv = rv[np.isfinite(rv)]
    if len(zv) == 0 or len(rv) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    axes[0].hist(zv, bins=60, range=(0.0, 0.5))
    axes[0].set_title("update gate $d(z_t)$")
    axes[0].set_xlabel("distance to saturation")
    axes[0].set_ylabel("count (timesteps)")
    axes[1].hist(rv, bins=60, range=(0.0, 0.5))
    axes[1].set_title("reset gate $d(r_t)$")
    axes[1].set_xlabel("distance to saturation")
    fig.suptitle(title)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_compare_curve(
    items: list[dict],
    out_png: str,
    title: str,
    key: str,
    y_label: str,
    log_y: bool = False,
) -> None:
    plt.figure(figsize=(6.8, 4.0))
    for it in items:
        z = np.load(f"{it['name']}_final_state.npz")
        y = _finite_prefix(z[key])
        if len(y) == 0:
            continue
        x = np.arange(len(y)) * int(z.get("checkFreq", 1))
        plt.plot(x, y, lw=1.4, label=it["name"])
    if log_y:
        plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    out_dir = os.path.join("report", "images")
    _ensure_dir(out_dir)

    # Per-run extra plots.
    for it in RUNS:
        name = it["name"]
        z = np.load(f"{name}_final_state.npz")
        save_rho_curve(
            z,
            os.path.join(out_dir, f"{name}_rho.png"),
            f"{name}: $\\rho(W_{{hh}})$ over training",
        )
        save_gradnorm_curve(
            z,
            os.path.join(out_dir, f"{name}_grad_norm.png"),
            f"{name}: global grad norm over training",
            cutoff=it["cutoff"],
        )
        if it["model"] == "gru":
            save_gate_hist(
                z,
                os.path.join(out_dir, f"{name}_gate_sat_hist.png"),
                f"{name}: GRU gate saturation distances",
            )

    # Comparison plots used for Q3/Q5 discussion.
    save_compare_curve(
        [r for r in RUNS if r["task"] == "mem"],
        os.path.join(out_dir, "mem_rho_compare.png"),
        "Memorization: $\\rho(W_{hh})$ comparison",
        key="rho_Whh",
        y_label=r"$\rho(W_{hh})$",
    )
    save_compare_curve(
        [r for r in RUNS if r["task"] == "mul"],
        os.path.join(out_dir, "mul_rho_compare.png"),
        "Multiplication: $\\rho(W_{hh})$ comparison",
        key="rho_Whh",
        y_label=r"$\rho(W_{hh})$",
    )


if __name__ == "__main__":
    main()
