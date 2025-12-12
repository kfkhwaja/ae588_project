# AeroSandbox test script
# rosenbrock 2d

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

# Print values from ASB
print(f"Optimal x (ASB)= {x_opt}")
print(f"Optimal y (ASB) = {y_opt}")
print(f"Objective value (ASB) = {f_opt}")