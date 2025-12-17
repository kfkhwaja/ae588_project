# Compare rocket trajectory optimizations from Dymos and AeroSandbox
# Plots trajectory, pitch angle, objective convergence, constraint violation

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

def get_guess_id_from_filename(path):
    """
    Extract guess ID from filenames like:
    aerosandbox_N69_guess3.npz
    dymos_seg32_ord3_guess2.npz
    """
    m = re.search(r"guess(\d+)", path.stem)
    if m is None:
        raise ValueError(f"Could not parse guess ID from {path.name}")
    return int(m.group(1))

# Paths and load Dymos data
base_dir = Path(__file__).resolve().parent

dymos_file = base_dir / "dymos_seg5_ord3.npz"
dymos = np.load(dymos_file)

# Dymos trajectory solution
t_d_sol = dymos["time_sol"]
x_d_sol = dymos["x_sol"]
y_d_sol = dymos["y_sol"]
theta_d_sol = dymos["theta_sol"]

# Dymos convergence data
iter_d = dymos["iter"] - dymos["iter"][0]  # start from zero
obj_d = dymos["obj"]
constr_d = dymos["constr_violation"]

# Load AeroSandbox data
N = 10  # change if needed
asb_deriv_file = base_dir / f"aerosandbox_derivatives_N{N}.npz"
asb_sol_file = base_dir / f"aerosandbox_N{N}.npz"

# Iteration/convergence data
asb_deriv = np.load(asb_deriv_file, allow_pickle=True)
iter_asb = asb_deriv["iter"].tolist()
obj_asb = asb_deriv["obj"].tolist()
constr_asb = asb_deriv["constr_violation"].tolist()

# Final solution data
asb_sol = np.load(asb_sol_file)
t_a = asb_sol["time"]
x_a = asb_sol["x"]
y_a = asb_sol["y"]
theta_a = asb_sol["theta"]

# Trajectory comparison
# plt.figure(figsize=(8, 6))
# plt.plot(x_d_sol, y_d_sol, "o-", color='blue', label="Dymos", markersize=2)
# plt.plot(x_a, y_a, "o-", color='red', label="AeroSandbox", markersize=2)
# plt.xlabel("Range (m)")
# plt.ylabel("Altitude (m)")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.axis("equal")
# plt.tight_layout()
# plt.savefig("trajectory_comp_simple.pdf")
# plt.show()
plt.figure(figsize=(8, 6))

files = sorted(base_dir.glob("aerosandbox_N69_guess*.npz"))

for i, f in enumerate(files):
    d = np.load(f)
    guess_id = get_guess_id_from_filename(f)
    print(f'final time asb guess {guess_id} = {d["tf"]}')

    linestyle = '-' if i % 2 == 0 else '--'

    plt.plot(
        d["x"],
        d["y"],
        linestyle=linestyle,
        alpha=0.6,
        linewidth=1.8,
        label=f"guess {guess_id}",
    )

plt.xlabel("Range (m)")
plt.ylabel("Altitude (m)")
#plt.title("AeroSandbox Robustness (N = 69)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
for f in sorted(base_dir.glob("dymos_seg32_ord3_guess*.npz")):
    d = np.load(f)
    guess_id = get_guess_id_from_filename(f)
    print(f'final time dymos guess {guess_id} = {d["tf"]}')
    plt.plot(d["x"], d["y"], alpha=0.6, label=f"guess {guess_id}")
plt.xlabel("Range (m)")
plt.ylabel("Altitude (m)")
#plt.title("Dymos Robustness (N=32)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis("equal")
plt.show()

plt.figure(figsize=(8,6))
files = sorted(base_dir.glob("aerosandbox_N69_guess*.npz"))

for i, f in enumerate(files):
    d = np.load(f)
    guess_id = get_guess_id_from_filename(f)

    styles = ['-', '--', ':']
    linestyle = styles[i % len(styles)]

    plt.plot(
        d["time"],
        d["theta"] * 180 / np.pi,
        linestyle=linestyle,
        alpha=0.6,
        linewidth=1.8,
        label=f"guess {guess_id}",
    )

plt.xlabel("Time (s)")
plt.ylabel("Pitch Angle (deg)")
#plt.title("AeroSandbox Pitch Robustness (N = 69)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("asb_pitch_robustness.pdf")
plt.show()


plt.figure(figsize=(8, 6))

for f in sorted(base_dir.glob("dymos_seg32_ord3_guess*.npz")):
    d = np.load(f)
    guess_id = get_guess_id_from_filename(f)
    plt.plot(
        d["time_sol"],
        d["theta_sol"] * 180 / np.pi,
        alpha=0.6,
        label=f"guess {guess_id}"
    )

plt.xlabel("Time (s)")
plt.ylabel("Pitch Angle (deg)")
#plt.title("Dymos Pitch Robustness (32 segments)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("dymos_pitch_robustness.pdf")
plt.show()

# Pitch angle comparison
# plt.figure(figsize=(8, 6))
# plt.plot(t_d_sol, theta_d_sol * 180 / np.pi, "o-", color='blue', label="Dymos", markersize=2)
# plt.plot(t_a, theta_a * 180 / np.pi, "o-", color='red', label="AeroSandbox", markersize=2)
# plt.xlabel("Time (s)")
# plt.ylabel("Pitch Angle (deg)")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("pitch_comp_simple.pdf")
# plt.show()

# # Objective convergence
# plt.figure(figsize=(8, 6))
# plt.semilogy(iter_d, obj_d*100, "o-", color='blue', label="Dymos", markersize=2)
# plt.semilogy(iter_asb, obj_asb, "o-", color='red', label="AeroSandbox", markersize=2)
# plt.xlabel("IPOPT Iteration")
# plt.ylabel("Objective")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("objective_convergence_simple.pdf")
# plt.show()

# Constraint violation convergence
# plt.figure(figsize=(8, 6))
# plt.semilogy(iter_d, constr_d, "o-", color='blue', label="Dymos", markersize=2)
# plt.semilogy(iter_asb, constr_asb, "o-", color='red', label="AeroSandbox", markersize=2)
# plt.xlabel("IPOPT Iteration")
# plt.ylabel("Max Constraint Violation")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("constraint_convergence.pdf")
# plt.show()

# altitude vs speed, plot contours of energy
# energy = 1/2 v^2 + gh
# overlay (altitude, velocity magnitude) for both trajectories
# velocity magnitude = sqrt(vx^2 + vy^2)

# g = 9.80665
# h_target = 185_000.0        # m
# vx_target = 7796.6961       # m/s
# vy_target = 0.0             # m/s
# E_target = 0.5 * (vx_target**2 + vy_target**2) + g * h_target

# # velocity magnitudes
# vx_a = np.gradient(asb_sol["x"], asb_sol["time"])
# vy_a = np.gradient(asb_sol["y"], asb_sol["time"])
# v_a = np.sqrt(vx_a**2 + vy_a**2)

# t_d_sol = np.array(t_d_sol).flatten()
# x_d_sol = np.array(x_d_sol).flatten()
# y_d_sol = np.array(y_d_sol).flatten()

# vx_d = np.gradient(x_d_sol, t_d_sol)
# vy_d = np.gradient(y_d_sol, t_d_sol)
# v_d = np.sqrt(vx_d**2 + vy_d**2)

# # Create grid for energy contours
# h_min = 0
# h_max = max(max(y_a), max(y_d_sol))*1.2
# v_min = 0
# v_max = max(max(v_a), max(v_d))*1.2

# h_grid, v_grid = np.meshgrid(
#     np.linspace(h_min, h_max, 100),
#     np.linspace(v_min, v_max, 100)
# )

# energy_grid = np.abs(E_target - (0.5 * v_grid**2 + g * h_grid))  # specific energy (J/kg)

# plt.figure(figsize=(8, 6))
# # Contours
# contours = plt.contour(v_grid, h_grid, energy_grid, levels=20, cmap='viridis')
# plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

# # Trajectories overlay
# plt.plot(v_d, y_d_sol, "o-", color='blue', label="Dymos", markersize=2)
# plt.plot(v_a, y_a, "o-", color='red', label="AeroSandbox", markersize=2)

# plt.xlabel("Velocity magnitude (m/s)")
# plt.ylabel("Altitude (m)")
# plt.title("Rocket Trajectories on Energy Contours")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("trajectory_energy_contours.pdf")
# plt.show()
