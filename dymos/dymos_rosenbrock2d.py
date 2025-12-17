# test openmdao on rosenbrock
# try increasing dimensions
# compare iterations, wall time, function evaluations, etc.

print('-'*60)
print('Running OpenMDAO optimization...')
print('-'*60)
import openmdao.api as om

# build the model
prob = om.Problem()

prob.model.add_subsystem('rosenbrock', om.ExecComp('f = (1 - x) ** 2 + 100 * (y - x**2) ** 2'))

# setup the optimization
# prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options['optimizer'] = 'SLSQP'

prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = "IPOPT"
prob.driver.opt_settings['mu_init'] = 1e-1
prob.driver.opt_settings['max_iter'] = 600
prob.driver.opt_settings['constr_viol_tol'] = 1e-6
prob.driver.opt_settings['compl_inf_tol'] = 1e-6
prob.driver.opt_settings['tol'] = 1e-6
prob.driver.opt_settings['print_level'] = 0
prob.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
prob.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
prob.driver.opt_settings['mu_strategy'] = 'monotone'
prob.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
prob.driver.options['print_results'] = False

prob.model.add_design_var('rosenbrock.x', lower=-50, upper=50)
prob.model.add_design_var('rosenbrock.y', lower=-50, upper=50)
prob.model.add_objective('rosenbrock.f')

prob.setup()

# Set initial values.
prob.set_val('rosenbrock.x', 3.0)
prob.set_val('rosenbrock.y', -4.0)

# run the optimization
prob.run_driver()

# Print values from OpenMDAO
print('Optimal x (OpenMDAO): ', prob.get_val('rosenbrock.x'))
print('Optimal y (OpenMDAO): ', prob.get_val('rosenbrock.y'))
print('Objective value (OpenMDAO): ', prob.get_val('rosenbrock.f'))