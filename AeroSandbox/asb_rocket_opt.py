# rocket trajectory optimization with aerosandbox

import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time

GUESS_ID = int(os.environ.get("ASB_GUESS_ID", 0))

# Optimization environment
opti = asb.Opti()

# TODO N = 3*[4, 8, 16, 32, 64]
# points on plot = N
# DOF = 6*N + 1
N = int(os.environ.get("ASB_N", 4096))
t_final = opti.variable(init_guess=150)
t = np.linspace(0, t_final, N)
dt = np.diff(t)

if GUESS_ID == 0:
    x_init = np.linspace(0, 1.15e5, N)
    y_init = np.linspace(0, 1.85e5, N)
    theta_init = np.linspace(1.5, -0.76, N)

elif GUESS_ID == 1:  # shallow
    x_init = np.linspace(0, 1.4e5, N)
    y_init = np.linspace(0, 1.2e5, N)
    theta_init = np.linspace(0.3, -0.3, N)

elif GUESS_ID == 2:  # steep
    x_init = np.linspace(0, 8e4, N)
    y_init = np.linspace(0, 2.2e5, N)
    theta_init = np.linspace(1.4, 0.2, N)

elif GUESS_ID == 3:  # low energy
    x_init = np.linspace(0, 7e4, N)
    y_init = np.linspace(0, 8e4, N)
    theta_init = np.zeros(N)

elif GUESS_ID == 4:  # aggressive
    x_init = np.linspace(0, 2e5, N)
    y_init = np.linspace(0, 2.5e5, N)
    theta_init = np.linspace(1.2, -1.0, N)

# Constants (match Dymos)
g = 9.80665
rho_ref = 1.225
h_scale = 8.44e3
CD = 0.5
S = 7.069
CDA = CD * S

thrust = 2.1e6 # N
Isp = 265.2 # s

# States
x = opti.variable(init_guess=x_init)
y = opti.variable(init_guess=y_init)

vx = opti.variable(init_guess=1000, n_vars=N)
vy = opti.variable(init_guess=1000, n_vars=N)

m = opti.variable(init_guess=np.linspace(117000, 1163, N))

# Control
theta = opti.variable(init_guess=theta_init)

# Atmosphere
rho = rho_ref * np.exp(-y / h_scale)

# Forces
Fx = thrust * np.cos(theta) - 0.5 * CDA * rho * vx**2
Fy = thrust * np.sin(theta) - 0.5 * CDA * rho * vy**2 - m * g
ax = Fx / m
ay = Fy / m
mdot = -thrust / (g * Isp)

# Dynamics (via derivative constraints)
opti.constrain_derivative(
    derivative=vx,
    variable=x,
    with_respect_to=t,
)

opti.constrain_derivative(
    derivative=vy,
    variable=y,
    with_respect_to=t,
)

opti.constrain_derivative(
    derivative=ax,
    variable=vx,
    with_respect_to=t,
)

opti.constrain_derivative(
    derivative=ay,
    variable=vy,
    with_respect_to=t,
)

opti.constrain_derivative(
    derivative=mdot,
    variable=m,
    with_respect_to=t,
)

# Boundary conditions (match Dymos)
opti.subject_to([
    x[0] == 0,
    y[0] == 0,
    vx[0] == 0,
    vy[0] == 0,
    m[0] == 117000,
])

opti.subject_to([
    y[-1] >= 1.85e5,
    vx[-1] >= 7796.6961,
    vy[-1] == 0,
])

# Path constraints
opti.subject_to([
    m > 0,
    theta >= -1.57,
    theta <= 1.57,
])

# Objective (minimize final time)
opti.minimize(t_final)

# track wall time
t0 = time.time()

# Prepare logging lists
xdot_iter, ydot_iter, vxdot_iter, vydot_iter, mdot_iter = [], [], [], [], []
x_iter, y_iter, vx_iter, vy_iter, m_iter, theta_iter = [], [], [], [], [], []
obj_hist = []
constr_violation_hist = []
iter_hist = []

def log_convergence(iteration):
    # Objective
    f_val = float(opti.debug.value(opti.f))

    # Evaluate all constraints numerically
    try:
        g_val = opti.debug.value(opti.g)
        # For inequality constraints, compute violation only if negative
        g_val_numeric = np.array(g_val, dtype=float)  # convert MX to numeric array
        violation = np.max(np.maximum(0, np.abs(g_val_numeric)))  # max absolute violation
    except Exception as e:
        print(f"Warning: could not evaluate constraints at iteration {iteration}: {e}")
        violation = np.nan

    obj_hist.append(f_val)
    constr_violation_hist.append(violation)
    iter_hist.append(iteration)

LOG_EVERY = 1

def log_derivatives(iteration):
    if iteration % LOG_EVERY != 0:
        return

    x_val = opti.debug.value(x)
    y_val = opti.debug.value(y)
    vx_val = opti.debug.value(vx)
    vy_val = opti.debug.value(vy)
    m_val = opti.debug.value(m)
    theta_val = opti.debug.value(theta)

    rho_val = rho_ref * np.exp(-y_val / h_scale)

    xdot_iter.append(vx_val)
    ydot_iter.append(vy_val)
    vxdot_iter.append(
        (thrust * np.cos(theta_val) - 0.5 * CDA * rho_val * vx_val**2) / m_val
    )
    vydot_iter.append(
        (thrust * np.sin(theta_val) - 0.5 * CDA * rho_val * vy_val**2) / m_val - g
    )
    mdot_iter.append(-thrust / (g * Isp))

def callback(iteration):
    log_convergence(iteration)
    log_derivatives(iteration)

# Solve
options_ipopt = {
    'ipopt.mu_init': 1e-1,
    'ipopt.constr_viol_tol': 1e-4,
    'ipopt.compl_inf_tol': 1e-4,
    'ipopt.tol': 1e-4,
    'ipopt.nlp_scaling_method': 'gradient-based',
    'ipopt.alpha_for_y': 'safer-min-dual-infeas',
    'ipopt.mu_strategy': 'monotone',
    'ipopt.bound_mult_init_method': 'mu-based',
    'ipopt.print_frequency_iter':1,
    'ipopt.print_level':4
}
sol = opti.solve(verbose=True, max_iter=5000, options=options_ipopt, callback=callback)
wall_time = time.time() - t0

# Save everything to a file
output_dir = Path(__file__).resolve().parents[1] / "comparison"
output_dir.mkdir(exist_ok=True)
time_sol = sol.value(t)
np.savez(
    output_dir / f"aerosandbox_derivatives_N{N}_guess{GUESS_ID}.npz",
    time=time_sol,
    x_iter=x_iter,
    y_iter=y_iter,
    vx_iter=vx_iter,
    vy_iter=vy_iter,
    m_iter=m_iter,
    theta_iter=theta_iter,
    xdot_iter=xdot_iter,
    ydot_iter=ydot_iter,
    vxdot_iter=vxdot_iter,
    vydot_iter=vydot_iter,
    mdot_iter=mdot_iter,
    obj=np.array(obj_hist),
    constr_violation=np.array(constr_violation_hist),
    iter=np.array(iter_hist),
)

print(f"Solved in {sol.stats()['iter_count']} iterations.")
print(f"Final time: {sol.value(t_final):f} s")

h_final = sol.value(y)[-1]
v_final = sol.value(vx)[-1]
m_final = sol.value(m)[-1]
print(f"Final altitude: {h_final} m")
print(f"Final speed: {v_final} m/s")
print(f'Final mass: {m_final}')
print(f'wall time = {wall_time}')

# Plotting
x_sol = sol.value(x)
y_sol = sol.value(y)
theta_sol = sol.value(theta)
time_sol = sol.value(t)

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Trajectory plot
axs[0].plot(x_sol, y_sol, '-o')
axs[0].set_xlabel("Range (m)")
axs[0].set_ylabel("Altitude (m)")
axs[0].set_aspect("equal", adjustable="box")
axs[0].grid(True)

# Control history
axs[1].plot(time_sol, theta_sol * 180/np.pi, '-o')
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Pitch angle (deg)")
axs[1].grid(True)

plt.suptitle("Rocket Ascent Trajectory Optimization (AeroSandbox)")
plt.tight_layout()
plt.show()
# if os.environ.get("ASB_PLOT", "0") == "1":
#     plt.show()
# else:
#     plt.close(fig)

# extract data
time_sol = sol.value(t)
x_sol = sol.value(x)
y_sol = sol.value(y)
theta_sol = sol.value(theta)

# save to comparison folder
output_dir = Path(__file__).resolve().parents[1] / "comparison"
output_dir.mkdir(exist_ok=True)

np.savez(
    output_dir / f"aerosandbox_N{N}_guess{GUESS_ID}.npz",
    time=time_sol,
    x=x_sol,
    y=y_sol,
    theta=theta_sol,
    tf=sol.value(t_final),
    iters=sol.stats()["iter_count"],
    wall_time=wall_time,
    N=N,
    m=sol.value(m),
)

print(f"AeroSandbox solution saved to {output_dir / f'aerosandbox_N{N}.npz'}")