# rocket trajectory optimization with dymos

import matplotlib.pyplot as plt
import openmdao.api as om
import dymos as dm
from dymos_rocket_class import LaunchVehicleODE
from pathlib import Path
import numpy as np
import os
import time
import re

# helper function read ipopt file output
import numpy as np
import re

def parse_ipopt_output(filepath):
    iters = []
    obj = []
    inf_pr = []
    inf_du = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Match iteration lines: start with integer iteration count
            if re.match(r"^\d+\s", line):
                parts = line.split()

                # Expected format:
                # iter objective inf_pr inf_du lg(mu) ||d|| ...
                try:
                    iters.append(int(parts[0]))
                    obj.append(float(parts[1]))
                    inf_pr.append(float(parts[2]))
                    inf_du.append(float(parts[3]))
                except ValueError:
                    pass

    iters = np.array(iters)
    obj = np.array(obj)
    inf_pr = np.array(inf_pr)
    inf_du = np.array(inf_du)

    return {
        "iter": iters,
        "obj": obj,
        "constr_violation": inf_pr,
        "nlp_error": np.maximum(inf_pr, inf_du),
        "inf_du": inf_du,
    }

def apply_initial_guess(phase, guess_id):
    """
    Apply different structured initial guesses for robustness testing.
    """
    if guess_id == 0:
        # Baseline (your current guess)
        phase.set_time_val(initial=0.0, duration=150.0)
        phase.set_state_val('x', [0, 1.15E5])
        phase.set_state_val('y', [0, 1.85E5])
        phase.set_state_val('vx', [0.0, 7796.0])
        phase.set_state_val('vy', [1e-6, 0.0])
        phase.set_state_val('m', [117000, 1163])
        phase.set_control_val('theta', [1.5, -0.76])

    elif guess_id == 1:
        # Shallow pitch, longer burn
        phase.set_time_val(initial=0.0, duration=220.0)
        phase.set_state_val('x', [0, 1.8E5])
        phase.set_state_val('y', [0, 1.6E5])
        phase.set_state_val('vx', [0.0, 7500.0])
        phase.set_state_val('vy', [0.0, -100.0])
        phase.set_state_val('m', [117000, 2000])
        phase.set_control_val('theta', [0.8, -0.2])

    elif guess_id == 2:
        # Aggressive vertical climb
        phase.set_time_val(initial=0.0, duration=120.0)
        phase.set_state_val('x', [0, 5.0E4])
        phase.set_state_val('y', [0, 2.2E5])
        phase.set_state_val('vx', [0.0, 8200.0])
        phase.set_state_val('vy', [500.0, -50.0])
        phase.set_state_val('m', [117000, 1000])
        phase.set_control_val('theta', [1.4, -1.2])

    elif guess_id == 3:
        # Nearly horizontal flight
        phase.set_time_val(initial=0.0, duration=180.0)
        phase.set_state_val('x', [0, 2.2E5])
        phase.set_state_val('y', [0, 1.2E5])
        phase.set_state_val('vx', [500.0, 7800.0])
        phase.set_state_val('vy', [0.0, -200.0])
        phase.set_state_val('m', [117000, 3000])
        phase.set_control_val('theta', [0.2, -0.1])

    elif guess_id == 4:
        # Poor but feasible guess (stress test)
        phase.set_time_val(initial=0.0, duration=300.0)
        phase.set_state_val('x', [0, 1.0E5])
        phase.set_state_val('y', [0, 1.0E5])
        phase.set_state_val('vx', [0.0, 6000.0])
        phase.set_state_val('vy', [0.0, 0.0])
        phase.set_state_val('m', [117000, 5000])
        phase.set_control_val('theta', [0.0, 0.0])

    else:
        raise ValueError(f"Unknown GUESS_ID = {guess_id}")


# Setup and solve the optimal control problem
p = om.Problem(model=om.Group())
p.driver = om.pyOptSparseDriver()

# ipopt options - TODO check against ASB
# TODO try lowering tolerance way down
p.driver.options["optimizer"] = "IPOPT"
p.driver.opt_settings['mu_init'] = 1e-1
p.driver.opt_settings['max_iter'] = 5000
p.driver.opt_settings['constr_viol_tol'] = 1e-12
p.driver.opt_settings['compl_inf_tol'] = 1e-12
p.driver.opt_settings['tol'] = 1e-10
p.driver.opt_settings['print_level'] = 3
p.driver.opt_settings['file_print_level'] = 5
p.driver.opt_settings['output_file'] = 'ipopt_dymos.out'
p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
p.driver.opt_settings['mu_strategy'] = 'monotone'
p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
p.driver.options['print_results'] = True

p.driver.declare_coloring(tol=1.0E-12)

# Initialize our Trajectory and Phase
traj = dm.Trajectory()

# TODO try increasing num_segments: [1, 2, 4, 8, 16, 32, 64] and compare convergence
# points on plot = 2*seg + 1
# DOF: 13*seg - 4 (for order = 3)
# 8*18 = 144

# 
# seg = 5, ord = 3 --> 61 vars
# 4, 3 --> 48
# 3, 3 --> 35
# 2, 3 --> 22 (5 points)
# 1, 3 --> 9 (3 points)
NUM_SEG = int(os.environ.get("DYMOS_SEG", 128))
ORDER = int(os.environ.get("DYMOS_ORDER", 3))
GUESS_ID = int(os.environ.get("DYMOS_GUESS_ID", 0))

transcription=dm.GaussLobatto(
    num_segments=NUM_SEG,
    order=ORDER,
    compressed=False
)
phase = dm.Phase(ode_class=LaunchVehicleODE, transcription=transcription)

traj.add_phase('phase0', phase)
p.model.add_subsystem('traj', traj)

# Set the options for the variables
phase.set_time_options(fix_initial=True, duration_bounds=(10, 500))

phase.add_state('x', fix_initial=True, ref=1.0E5, defect_ref=10000.0,
                rate_source='xdot')
phase.add_state('y', fix_initial=True, ref=1.0E5, defect_ref=10000.0,
                rate_source='ydot')
phase.add_state('vx', fix_initial=True, ref=1.0E3, defect_ref=1000.0,
                rate_source='vxdot')
phase.add_state('vy', fix_initial=True, ref=1.0E3, defect_ref=1000.0,
                rate_source='vydot')
phase.add_state('m', fix_initial=True, ref=1.0E3, defect_ref=100.0,
                rate_source='mdot')

phase.add_control('theta', units='rad', lower=-1.57, upper=1.57, targets=['theta'])
phase.add_parameter('thrust', units='N', opt=False, val=2100000.0, targets=['thrust'])

# Set the options for our constraints and objective
phase.add_boundary_constraint('y', loc='final', equals=1.85E5, linear=True)
phase.add_boundary_constraint('vx', loc='final', equals=7796.6961)
phase.add_boundary_constraint('vy', loc='final', equals=0)

phase.add_objective('time', loc='final', scaler=0.01)

p.model.linear_solver = om.DirectSolver()

# Setup and set initial values
p.setup(check=True)

apply_initial_guess(phase, GUESS_ID)
phase.set_parameter_val('thrust', 2.1, units='MN')

# Solve the Problem
t0 = time.time()
dm.run_problem(p, simulate=True)
wall_time = time.time() - t0

# save data to comparison folder
# extract data
sol = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')
sim = om.CaseReader(traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db').get_case('final')
time_series = sim.get_val('traj.phase0.timeseries.time')
x = sim.get_val('traj.phase0.timeseries.x')
y = sim.get_val('traj.phase0.timeseries.y')
theta = sim.get_val('traj.phase0.timeseries.theta')
vx = sim.get_val('traj.phase0.timeseries.vx')
vy = sim.get_val('traj.phase0.timeseries.vy')
m = sim.get_val('traj.phase0.timeseries.m')

time_series_sol = sol.get_val('traj.phase0.timeseries.time')
x_sol = sol.get_val('traj.phase0.timeseries.x')
y_sol = sol.get_val('traj.phase0.timeseries.y')
theta_sol = sol.get_val('traj.phase0.timeseries.theta')
vx_sol = sol.get_val('traj.phase0.timeseries.vx')
vy_sol = sol.get_val('traj.phase0.timeseries.vy')
m_sol = sol.get_val('traj.phase0.timeseries.m')

# save to that directory
output_dir = Path(__file__).resolve().parents[1] / "comparison"
output_dir.mkdir(exist_ok=True)

ipopt_file = Path(__file__).resolve().parent / "dymos_rocket_opt_out" / "IPOPT.out"

ipopt_data = parse_ipopt_output(ipopt_file)

np.savez(
    output_dir / f"dymos_seg{NUM_SEG}_ord{ORDER}_guess{GUESS_ID}.npz",
    time=time_series,
    time_sol=time_series_sol,
    x=x,
    x_sol=x_sol,
    y=y,
    y_sol=y_sol,
    theta=theta,
    m=m,
    theta_sol=theta_sol,
    tf=time_series[-1],
    wall_time=wall_time,
    num_segments=NUM_SEG,
    order=ORDER,
    obj=ipopt_data["obj"],
    constr_violation=ipopt_data["constr_violation"],
    nlp_error=ipopt_data["nlp_error"],
    iter=ipopt_data["iter"],
)

print(f"Dymos solution saved to {output_dir / 'dymos_solution.npz'}")

# print optimal time
opt_time = time_series[-1]
h_final = y_sol[-1]
v_final = vx_sol[-1]
m_final = m_sol[-1]
print(f'Optimal final time: {opt_time}')
print(f'Final altitude: {h_final}')
print(f'Final speed: {v_final}')
print(f'Final mass: {m_final}')
print(f'wall time = {wall_time}')

# Extract rates
# vx = sim.get_val('traj.phase0.timeseries.vx')
# vy = sim.get_val('traj.phase0.timeseries.vy')
# m = sim.get_val('traj.phase0.timeseries.m')
# xdot = sim.get_val('traj.phase0.timeseries.xdot')
# ydot = sim.get_val('traj.phase0.timeseries.ydot')
# vxdot = sim.get_val('traj.phase0.timeseries.vxdot')
# vydot = sim.get_val('traj.phase0.timeseries.vydot')
# mdot = sim.get_val('traj.phase0.timeseries.mdot')
# rho = sim.get_val('traj.phase0.timeseries.rho')

# Save to file
output_dir = Path(__file__).resolve().parents[1] / "comparison"
output_dir.mkdir(exist_ok=True)

# np.savez(
#     output_dir / "dymos_derivatives.npz",
#     time=time,
#     # xdot=xdot,
#     # ydot=ydot,
#     # vxdot=vxdot,
#     # vydot=vydot,
#     # mdot=mdot,
#     # rho=rho,
#     # x=x, y=y, vx=vx, vy=vy, m=m, theta=theta
# )

# plotting
fig, [traj_ax, control_ax] = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

traj_ax.plot(sol.get_val('traj.phase0.timeseries.x'),
             sol.get_val('traj.phase0.timeseries.y'),
             marker='o',
             ms=4,
             linestyle='None',
             label='solution')

traj_ax.plot(sim.get_val('traj.phase0.timeseries.x'),
             sim.get_val('traj.phase0.timeseries.y'),
             marker='o',
             linestyle='-',
             label='simulation')

traj_ax.set_xlabel('range (m)')
traj_ax.set_ylabel('altitude (m)')
traj_ax.set_aspect('equal')
traj_ax.grid(True)

control_ax.plot(sol.get_val('traj.phase0.timeseries.time'),
             sol.get_val('traj.phase0.timeseries.theta'),
             marker='None',
             ms=4,
             linestyle='None')

control_ax.plot(sim.get_val('traj.phase0.timeseries.time'),
             sim.get_val('traj.phase0.timeseries.theta'),
             linestyle='-',
             marker=None)

control_ax.set_xlabel('time (s)')
control_ax.set_ylabel('theta (deg)')
control_ax.grid(True)

plt.suptitle('Single Stage to Orbit Solution Using A Dynamic Control')
fig.legend(loc='lower center', ncol=2)

plt.show()

# 1.4285252017523817e+00
# 1.4285252017523704e+00
# 1.4285252017523800e+00