# creates comparison plots of asb and dymos runs, shows convergence 

# need
# TODO check with different initial guesses (sensitivity)
# TODO check IPOPT iterations/evaluations

# maybe
# TODO quantify difference in trajectories (L2 error?) see how that changes as mesh increases
# TODO print derivatives/trajectories, tabulate collocation points to show difference
# TODO visualize design space? can we do two points? dymos needs N>=1, O>=3, asb needs N>=11

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

comparison_dir = Path(__file__).resolve().parent

# --- Helper functions ---
def load_npz_files(prefix):
    files = sorted(comparison_dir.glob(f"{prefix}*.npz"))
    data = []
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        d["filename"] = f.name
        data.append(d)
    return data

def resample_to_common_time(t_ref, t, y):
    t_ref = np.asarray(t_ref).squeeze()
    t = np.asarray(t).squeeze()
    y = np.asarray(y).squeeze()
    return np.interp(t_ref, t, y)

def trajectory_l2_error(ref, test):
    dx = ref["x"] - test["x"]
    dy = ref["y"] - test["y"]
    return np.sqrt(np.mean(dx**2 + dy**2))

# --- Load data ---
asb_data = load_npz_files("aerosandbox_N")
dymos_data = load_npz_files("dymos_seg")

# Sort by mesh density
asb_data.sort(key=lambda d: d["N"])
dymos_data.sort(key=lambda d: d.get("num_segments", 0))

# Reference solutions (largest mesh)
asb_ref = asb_data[-1]
dymos_ref = asb_data[4]

# --- Compute convergence metrics ---
asb_N, asb_tf, asb_err, asb_wall, asb_iter, asb_mass, asb_constr = [], [], [], [], [], [], []
t_ref_asb = asb_ref["time"]
x_ref_asb = asb_ref["x"]
y_ref_asb = asb_ref["y"]

for d in asb_data[:-1]:
    x_i = resample_to_common_time(t_ref_asb, d["time"], d["x"])
    y_i = resample_to_common_time(t_ref_asb, d["time"], d["y"])

    asb_N.append(d["N"])
    asb_tf.append(d["tf"])
    asb_err.append(trajectory_l2_error({"x": x_ref_asb, "y": y_ref_asb}, {"x": x_i, "y": y_i}))
    asb_wall.append(d["wall_time"])
    asb_iter.append(d.get("iters", np.nan))
    asb_mass.append(d["m"][-1] if "m" in d else np.nan)
    
    # max constraint violation: try multiple keys for backwards compatibility
    if "constr_violation" in d:
        asb_constr.append(np.max(np.abs(d["constr_violation"])))
    else:
        asb_constr.append(np.nan)

dymos_seg, dymos_tf, dymos_err, dymos_wall, dymos_iter, dymos_mass, dymos_constr = [], [], [], [], [], [], []
t_ref_dym = dymos_ref["time"]
x_ref_dym = dymos_ref["x"]
y_ref_dym = dymos_ref["y"]

for d in dymos_data[:-1]:
    x_i = resample_to_common_time(t_ref_dym, d["time"], d["x"])
    y_i = resample_to_common_time(t_ref_dym, d["time"], d["y"])

    dymos_seg.append(d.get("num_segments", np.nan))
    dymos_tf.append(d["tf"])
    dymos_err.append(trajectory_l2_error({"x": x_ref_dym, "y": y_ref_dym}, {"x": x_i, "y": y_i}))
    dymos_wall.append(d["wall_time"])
    dymos_iter.append(d.get("iter")[-1] if "iter" in d else np.nan)
    dymos_mass.append(d["m"][-1] if "m" in d else np.nan)
    dymos_constr.append(np.max(np.abs(d.get("constr_violation", [np.nan]))))

# --- Plotting ---
asb_DOF = [6 * N + 1 for N in asb_N]
d_DOF = [13 * seg - 4 for seg in dymos_seg]

plt.figure()
plt.loglog(asb_DOF, asb_err, 'o-', label="AeroSandbox")
plt.loglog(d_DOF, dymos_err, 'o-', label="Dymos")
plt.xlabel("Design Variables")
plt.ylabel("L2 trajectory error (m)")
#plt.title("Trajectory Convergence")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.semilogx(asb_DOF, asb_tf, 'o-', label="AeroSandbox")
plt.semilogx(d_DOF, dymos_tf, 'o-', label="Dymos")
plt.xlabel("Design Variables")
plt.ylabel("Optimal final time (s)")
#plt.title("Optimal Time vs Mesh Resolution")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.loglog(asb_DOF, asb_wall, 'o-', label="AeroSandbox")
plt.loglog(d_DOF, dymos_wall, 'o-', label="Dymos")
plt.xlabel("Design Variables")
plt.ylabel("Wall time (s)")
#plt.title("Computational Cost Scaling")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
dymos_dofs = [61, 100, 204, 412, 828, 1660]
dymos_iterations = [9, 42, 48, 50, 65, 76]
plt.semilogx(asb_DOF, asb_iter, 'o-', label="AeroSandbox")
plt.semilogx(dymos_dofs, dymos_iterations, 'o-', label="Dymos")
plt.xlabel("Design Variables")
plt.ylabel("Optimizer iterations")
#plt.title("IPOPT Iteration Count vs Mesh")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.semilogx(asb_DOF, asb_mass, 'o-', label="AeroSandbox")
plt.semilogx(d_DOF, dymos_mass, 'o-', label="Dymos")
plt.xlabel("Design Variables")
plt.ylabel("Final mass (kg)")
#plt.title("Final Mass vs Mesh Resolution")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Trajectories (fine mesh)
plt.figure()
plt.plot(asb_ref["x"], asb_ref["y"], '-', label="AeroSandbox", markersize=2)
plt.plot(dymos_ref["x"], dymos_ref["y"], '--', label="Dymos", markersize=2)
plt.xlabel("Range (m)")
plt.ylabel("Altitude (m)")
#plt.title("Optimized Trajectory (Finest Mesh)")
plt.legend()
plt.tight_layout()
plt.show()

# Control history (fine mesh)
plt.figure()
plt.plot(asb_ref["time"], asb_ref["theta"] * 180 / np.pi, '-', label="AeroSandbox")
plt.plot(dymos_ref["time"], dymos_ref["theta"] * 180 / np.pi, '--', label="Dymos")
plt.xlabel("Time (s)")
plt.ylabel("Pitch angle (deg)")
#plt.title("Optimized Control History (Finest Mesh)")
plt.legend()
plt.tight_layout()
plt.show()

# --- AeroSandbox trajectories ---
asb_files = sorted(comparison_dir.glob("aerosandbox_N*.npz"), key=lambda f: int(f.stem.split("N")[-1]))

plt.figure(figsize=(10, 6))
for f in asb_files:
    data = np.load(f)
    N = data['N']
    plt.plot(data['x'], data['y'], label=f"N={N}")
plt.xlabel("Range (m)")
plt.ylabel("Altitude (m)")
#plt.title("AeroSandbox Trajectories Across Mesh Sweep")
plt.legend()
plt.tight_layout()
plt.show()


# --- Dymos trajectories ---
dym_files = sorted(comparison_dir.glob("dymos_seg*_ord3.npz"))

# Sort files by number of segments
def get_seg_num(f):
    stem = f.stem  # e.g., "dymos_seg12_ord3"
    return int(stem.split("seg")[1].split("_")[0])

dym_files_sorted = sorted(dym_files, key=get_seg_num)

plt.figure(figsize=(10, 6))
for f in dym_files_sorted:
    data = np.load(f)
    seg_num = get_seg_num(f)
    plt.plot(data['x'], data['y'], label=f"Segments={seg_num}")

plt.xlabel("Range (m)")
plt.ylabel("Altitude (m)")
#plt.title("Dymos Trajectories Across Mesh Sweep")
plt.legend()
plt.tight_layout()
plt.show()