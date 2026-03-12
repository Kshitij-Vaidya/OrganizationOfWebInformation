import numpy as np


RUNS = [
    dict(run="A1", name="A1_mem_rnn_tanh_noclip", task="mem", model="RNN", clip="none"),
    dict(run="A2", name="A2_mem_rnn_tanh_clip005", task="mem", model="RNN", clip="0.05"),
    dict(run="A3", name="A3_mem_rnn_tanh_clip001", task="mem", model="RNN", clip="0.01"),
    dict(run="A4", name="A4_mem_gru_noclip", task="mem", model="GRU", clip="none"),
    dict(run="A5", name="A5_mem_gru_clip005", task="mem", model="GRU", clip="0.05"),
    dict(run="B1", name="B1_mul_rnn_tanh_noclip", task="mul", model="RNN", clip="none"),
    dict(run="B2", name="B2_mul_gru_noclip", task="mul", model="GRU", clip="none"),
]


def _finite_prefix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x) & (x >= 0)
    if not np.any(m):
        return np.array([], dtype=float)
    idx = np.where(m)[0]
    return x[idx[0] : idx[-1] + 1]


def _last_nonempty_row(a: np.ndarray) -> int:
    rows = np.where(np.isfinite(a).any(axis=1))[0]
    return int(rows[-1]) if len(rows) else 0


def _fmt(x: float, digits: int) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def summarize_npz(path: str, has_gates: bool) -> dict:
    z = np.load(path)

    grad_time = z["grad_time"]
    sat_time = z["sat_time"]

    gi = _last_nonempty_row(grad_time)
    si = _last_nonempty_row(sat_time)

    g = grad_time[gi]
    s = sat_time[si]
    g = g[np.isfinite(g)]
    s = s[np.isfinite(s)]

    eps = 1e-12
    gl = np.log10(g + eps) if len(g) else np.array([np.nan])

    rho = _finite_prefix(z["rho_Whh"])
    valid = _finite_prefix(z["valid_error"])

    out = dict(
        g_med=float(np.nanmedian(gl)),
        s_mean=float(np.nanmean(s)),
        s_sat_frac=float(np.mean(s < 0.05) * 100.0) if len(s) else float("nan"),
        rho_last=float(rho[-1]) if len(rho) else float("nan"),
        valid_best=float(np.min(valid)) if len(valid) else float("nan"),
    )

    if has_gates:
        zt = z["gate_z_sat_time"]
        rt = z["gate_r_sat_time"]
        zi = _last_nonempty_row(zt)
        ri = _last_nonempty_row(rt)
        zv = zt[zi]
        rv = rt[ri]
        zv = zv[np.isfinite(zv)]
        rv = rv[np.isfinite(rv)]
        out["z_mean"] = float(np.mean(zv)) if len(zv) else float("nan")
        out["r_mean"] = float(np.mean(rv)) if len(rv) else float("nan")
    else:
        out["z_mean"] = float("nan")
        out["r_mean"] = float("nan")

    return out


def main() -> None:
    lines = []
    lines.append(r"\begin{tabular}{lllcccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Run & Task & Model & Clip & $\mathrm{median}[\log_{10}\|dL/dh_t\|]$ & mean(sat) & sat<0.05 (\%) & $\rho(W_{hh})$ (last) & best valid err (\%) \\"
    )
    lines.append(r"\midrule")

    for r in RUNS:
        s = summarize_npz(f"{r['name']}_final_state.npz", has_gates=(r["model"] == "GRU"))
        lines.append(
            " & ".join(
                [
                    r["run"],
                    r["task"],
                    r["model"],
                    r["clip"],
                    _fmt(s["g_med"], 2),
                    _fmt(s["s_mean"], 3),
                    _fmt(s["s_sat_frac"], 1),
                    _fmt(s["rho_last"], 2),
                    _fmt(s["valid_best"], 2),
                ]
            )
            + r" \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_path = "report/summary_table.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()

