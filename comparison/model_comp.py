# compare aerosandbox model on dymos trajectory
# using dymos optimized theta values over time, recreate the trajectory using forward euler integration
# if ASB forward euler trajectory is the same as dymos trajectory, then ASB model is same

import aerosandbox as asb
import aerosandbox.numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# dymos trajectory
dymos_file = "dymos_seg128_ord3.npz"  # finest mesh so far
dymos_data = np.load(dymos_file)

time_d = dymos_data["time"]
theta_d = dymos_data["theta"]
x_d = dymos_data["x"]
y_d = dymos_data["y"]

# parameters
g = 9.80665
thrust = 2.1e6
Isp = 265.2
rho_ref = 1.225
h_scale = 8.44e3
CD = 0.5
S = 7.069
CDA = CD*S
m0 = 117000

# initialize states
N = len(time_d)
x = np.zeros(N)
y = np.zeros(N)
vx = np.zeros(N)
vy = np.zeros(N)
m = np.zeros(N)
m[0] = m0

# time integration over times
for i in range(N-1):
    dt = time_d[i+1] - time_d[i]

    rho = rho_ref * np.exp(-y[i]/h_scale)
    Fx = thrust * np.cos(theta_d[i]) - 0.5 * CDA * rho * vx[i]**2
    Fy = thrust * np.sin(theta_d[i]) - 0.5 * CDA * rho * vy[i]**2 - m[i]*g

    ax = Fx / m[i]
    ay = Fy / m[i]
    mdot = -thrust / (Isp * g)

    # forward euler integration
    vx[i+1] = vx[i] + ax*dt
    vy[i+1] = vy[i] + ay*dt
    x[i+1] = x[i] + vx[i]*dt
    y[i+1] = y[i] + vy[i]*dt
    m[i+1] = m[i] + mdot*dt

# final time (check if this is same)
t_final_sim = time_d[-1]

# Save results
np.savez(
    "asb_comp_dymos_sim.npz",
    time=time_d,
    x=x,
    y=y,
    vx=vx,
    vy=vy,
    m=m,
    theta=theta_d,
    t_final=t_final_sim
)

# plot trajectory
plt.figure()
plt.plot(x_d, y_d, label='Dymos Optimized')
plt.plot(x, y, '--', label="ASB Forward Euler Model")
plt.xlabel("Range (m)")
plt.ylabel("Altitude (m)")
#plt.title("asb dymos trajectory comp")
plt.grid(True)
plt.legend()
plt.show()
