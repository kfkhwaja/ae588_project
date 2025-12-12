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
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

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