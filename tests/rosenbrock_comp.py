# test script to compare OpenMDAO vs ASB performance on rosenbrock problem
# def rosenbrock(x, y):
#     return (1 - x) ** 2 + 100 * (y - x**2) ** 2

# TODO
# compare with both using IPOPT
# try increasing dimensions
# compare iterations, wall time, function evaluations, etc.
# maybe compare them to SciPy?

#--------------------------------------------------
# AEROSANDBOX IMPLEMENTATION 
print('-'*60)
print('Running AeroSandbox optimization...')
print('-'*60)
import aerosandbox as asb

opti = (asb.Opti())  # Initialize a new optimization environment; convention is to name it 'opti'

# Define your optimization variables
x = opti.variable(init_guess=0)  # must provide initial guesses
y = opti.variable(init_guess=0)

# Define your objective
f = (1 - x) ** 2 + 100 * (y - x**2) ** 2  # You can construct nonlinear functions of variables...
opti.minimize(f)  # ...and then optimize them.

# Optimize
sol = opti.solve()  # This is the conventional syntax to solve the optimization problem

# Extract values at the optimum
x_opt = sol(x)  # Evaluates x at the point where the solver converged
y_opt = sol(y)
f_opt = sol(f) 

#--------------------------------------------------
# OPENMDAO IMPLEMENTATION
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

#--------------------------------------------------
# PRINTING RESULTS
print('-'*60)
print('Optimization Results:')
print('-'*60)

# Print values from ASB
print(f"Optimal x (ASB)= {x_opt}")
print(f"Optimal y (ASB) = {y_opt}")
print(f"Objective value (ASB) = {f_opt}")

# Print values from OpenMDAO
print('Optimal x (OpenMDAO): ', prob.get_val('rosenbrock.x'))
print('Optimal y (OpenMDAO): ', prob.get_val('rosenbrock.y'))
print('Objective value (OpenMDAO): ', prob.get_val('rosenbrock.f'))
