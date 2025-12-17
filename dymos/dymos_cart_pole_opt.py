# Dymos test script
# cart pole optimization example

# TODO configure IPOPT to run with that instead of scipy SLSQP

"""
Cart-pole optimization runscript
"""

import openmdao.api as om
om.display_source("dymos.examples.cart_pole.cartpole_dynamics")

import numpy as np
import openmdao.api as om
import dymos as dm
from dymos.examples.plotting import plot_results
from dymos_cart_pole_class import CartPoleDynamics

p = om.Problem()

# --- instantiate trajectory and phase, setup transcription ---
traj = dm.Trajectory()
p.model.add_subsystem('traj', traj)
phase = dm.Phase(transcription=dm.GaussLobatto(num_segments=40, order=3, compressed=True, solve_segments=False), ode_class=CartPoleDynamics)
# NOTE: set solve_segments=True to do solver-based shooting
traj.add_phase('phase', phase)

# --- set state and control variables ---
phase.set_time_options(fix_initial=True, fix_duration=True, duration_val=2., units='s')
# declare state variables. You can also set lower/upper bounds and scalings here.
phase.add_state('x', fix_initial=True, lower=-2, upper=2, rate_source='x_dot', shape=(1,), ref=1, defect_ref=1, units='m')
phase.add_state('x_dot', fix_initial=True, rate_source='x_dotdot', shape=(1,), ref=1, defect_ref=1, units='m/s')
phase.add_state('theta', fix_initial=True, rate_source='theta_dot', shape=(1,), ref=1, defect_ref=1, units='rad')
phase.add_state('theta_dot', fix_initial=True, rate_source='theta_dotdot', shape=(1,), ref=1, defect_ref=1, units='rad/s')
phase.add_state('energy', fix_initial=True, rate_source='e_dot', shape=(1,), ref=1, defect_ref=1, units='N**2*s')  # integration of force**2. This does not have the energy unit, but I call it "energy" anyway.

# declare control inputs
phase.add_control('f', fix_initial=False, rate_continuity=False, lower=-20, upper=20, shape=(1,), ref=0.01, units='N')

# add cart-pole parameters (set static_target=True because these params are not time-depencent)
phase.add_parameter('m_cart', val=1., units='kg', static_target=True)
phase.add_parameter('m_pole', val=0.3, units='kg', static_target=True)
phase.add_parameter('l_pole', val=0.5, units='m', static_target=True)

# --- set terminal constraint ---
# alternatively, you can impose those by setting `fix_final=True` in phase.add_state()
phase.add_boundary_constraint('x', loc='final', equals=1, ref=1., units='m')  # final horizontal displacement
phase.add_boundary_constraint('theta', loc='final', equals=np.pi, ref=1., units='rad')  # final pole angle
phase.add_boundary_constraint('x_dot', loc='final', equals=0, ref=1., units='m/s')  # 0 velocity at the and
phase.add_boundary_constraint('theta_dot', loc='final', equals=0, ref=0.1, units='rad/s')  # 0 angular velocity at the end
phase.add_boundary_constraint('f', loc='final', equals=0, ref=1., units='N')  # 0 force at the end

# --- set objective function ---
# we minimize the integral of force**2.
phase.add_objective('energy', loc='final', ref=100)

# --- configure optimizer ---
# SCIPY options
# p.driver = om.ScipyOptimizeDriver()
# p.driver.options["optimizer"] = "SLSQP"
# p.driver.options['tol'] = 1e-9
# p.driver.options['maxiter'] = 500

# IPOPT options
p.driver = om.pyOptSparseDriver()
p.driver.options["optimizer"] = "IPOPT"
p.driver.opt_settings['mu_init'] = 1e-1
p.driver.opt_settings['max_iter'] = 600
p.driver.opt_settings['constr_viol_tol'] = 1e-6
p.driver.opt_settings['compl_inf_tol'] = 1e-6
p.driver.opt_settings['tol'] = 1e-5
p.driver.opt_settings['print_level'] = 0
p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
p.driver.opt_settings['mu_strategy'] = 'monotone'
p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
p.driver.options['print_results'] = False

# declare total derivative coloring to accelerate the UDE linear solves
p.driver.declare_coloring()

p.setup(check=False)

# --- set initial guess ---
# The initial condition of cart-pole (i.e., state values at time 0) is set here because we set `fix_initial=True` when declaring the states.
phase.set_time_val(initial=0.0)  # set initial time to 0.
phase.set_state_val('x', vals=[0, 1, 1], time_vals=[0, 1, 2], units='m')
phase.set_state_val('x_dot', vals=[0, 0.1, 0], time_vals=[0, 1, 2], units='m/s')
phase.set_state_val('theta', vals=[0, np.pi/2, np.pi], time_vals=[0, 1, 2], units='rad')
phase.set_state_val('theta_dot', vals=[0, 1, 0], time_vals=[0, 1, 2], units='rad/s')
phase.set_state_val('energy', vals=[0, 30, 60], time_vals=[0, 1, 2])
phase.set_control_val('f', vals=[3, -1, 0], time_vals=[0, 1, 2], units='N')

# --- run optimization ---
dm.run_problem(p, run_driver=True, simulate=True, simulate_kwargs={'method': 'Radau', 'times_per_seg': 10})
# NOTE: with Simulate=True, dymos will call scipy.integrate.solve_ivp and simulate the trajectory using the optimized control inputs.

# --- get results and plot ---
# objective value
obj = p.get_val('traj.phase.states:energy', units='N**2*s')[-1]
print('objective value:', obj)

# get optimization solution and simulation (time integration) results
sol = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')
sim = om.CaseReader(traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db').get_case('final')

# plot time histories of x, x_dot, theta, theta_dot
plot_results([('traj.phase.timeseries.time', 'traj.phase.timeseries.x', 'time (s)', 'x (m)'),
              ('traj.phase.timeseries.time', 'traj.phase.timeseries.x_dot', 'time (s)', 'vx (m/s)'),
              ('traj.phase.timeseries.time', 'traj.phase.timeseries.theta', 'time (s)', 'theta (rad)'),
              ('traj.phase.timeseries.time', 'traj.phase.timeseries.theta_dot', 'time (s)', 'theta_dot (rad/s)'),
              ('traj.phase.timeseries.time', 'traj.phase.timeseries.f', 'time (s)', 'control (N)')],
             title='Cart-Pole Problem', p_sol=sol, p_sim=sim)

# uncomment the following lines to show the cart-pole animation
x = sol.get_val('traj.phase.timeseries.x', units='m')
theta = sol.get_val('traj.phase.timeseries.theta', units='rad')
force = sol.get_val('traj.phase.timeseries.f', units='N')
npts = len(x)

from dymos.examples.cart_pole.animate_cartpole import animate_cartpole
animate_cartpole(x.reshape(npts), theta.reshape(npts), force.reshape(npts), interval=20, force_scaler=0.02)