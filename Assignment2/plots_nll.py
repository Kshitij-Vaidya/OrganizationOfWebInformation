# Run from assignment2 directory

import numpy as np
import matplotlib.pyplot as plt
save_dir = "plots"
filename = "EC_torder_alpha8_noclip_final_state.npz"
z = np.load(f"npz_files/{filename}")

valid_err = z["valid_error"] # (num_checkpoints,)
nll = z["train_nll"] # (num_checkpoints, Tstore), NaN padded

smooth_window = 100

nll_smooth = np.convolve(nll, np.ones(smooth_window)/smooth_window, mode='valid')
iters = np.arange(len(nll_smooth))

# plot validation error
plt.figure(); plt.plot(valid_err); plt.title("validation error (%)")
plt.grid(True); plt.xlabel("checkpoint"); plt.ylabel("validation error (%)")
plt.savefig(f"{save_dir}/{filename}_val_err.png")

# plot training nll
plt.figure(); plt.plot(nll); plt.title("train NLL")
plt.grid(True); plt.xlabel("checkpoint"); plt.ylabel("train NLL")
plt.savefig(f"{save_dir}/{filename}_train_nll.png")

# plot smoothed training nll
plt.figure(); plt.plot(iters, nll_smooth); plt.title("smoothed train NLL")
plt.grid(True);  plt.xlabel("checkpoint"); plt.ylabel("smoothed train NLL")
plt.savefig(f"{save_dir}/{filename}_train_nll_smooth.png")
