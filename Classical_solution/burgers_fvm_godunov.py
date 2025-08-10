# ========================
# One Time Step Simulation
# ========================



import numpy as np
import matplotlib.pyplot as plt

# Parameters

nu = 0.01                 # Viscosity
L = 1.0                   # Domain length
N = 200                   # Number of cells
dx = L / N                # Cell size
x = np.linspace(dx/2, L - dx/2, N)  # Cell centers

# Time step based on CFL (both convection and diffusion)
CFL = 0.2
u_max = 1.0               # Max velocity for stability estimate
dt_conv = CFL * dx / u_max
dt_diff = 0.5 * dx**2 / nu
dt = min(dt_conv, dt_diff)
t_final = 0.1
nt = int(t_final / dt)

# Initial Condition (Riemann Step)

def initial_condition(x):
    return np.where(x <= 0.5, 1.0, 0.0)

u = initial_condition(x)

#  Apply Dirichlet BCs using ghost cells

def apply_boundary_conditions(u):
    u_bc = np.zeros(len(u) + 2)
    u_bc[1:-1] = u
    u_bc[0] = 1.0      # Left boundary: u(0,t) = 1
    u_bc[-1] = 0.0     # Right boundary: u(1,t) = 0
    return u_bc

# Godunov Flux for Burgers Equation:

def godunov_flux(uL, uR):
    flux = np.zeros_like(uL)
    for i in range(len(uL)):
        ul = uL[i]
        ur = uR[i]
        if ul <= ur:
            if ul >= 0:
                flux[i] = 0.5 * ul**2
            elif ur <= 0:
                flux[i] = 0.5 * ur**2
            else:
                flux[i] = 0.0
        else:
            flux[i] = 0.5 * max(ul**2, ur**2)
    return flux

# Time Integration Loop:

for n in range(nt):
    u_bc = apply_boundary_conditions(u)

    # Convective flux at interfaces:
    uL = u_bc[:-1]
    uR = u_bc[1:]
    F = godunov_flux(uL, uR)

    # Diffusion term (central difference):
    diffusion = nu * (u_bc[0:-2] - 2*u_bc[1:-1] + u_bc[2:]) / dx**2

    # Finite volume update (explicit Euler)
    u = u + dt * (-(F[1:] - F[:-1]) / dx + diffusion)

# Plot the Final Result:

plt.figure(figsize=(8,5))
plt.plot(x, u, label=f'Numerical solution at t={t_final:.3f}', linewidth=2)
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('1D Viscous Burgers Equation\nFinite Volume + Godunov + Central Diff')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



# ========================
# Multiple Time Step Simulation
# ========================

import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.01                 # viscosity
L = 1.0                   # domain length
N = 200                   # number of cells
dx = L / N
x = np.linspace(dx/2, L - dx/2, N)  # cell centers

# Time step based on CFL (convection + diffusion)
CFL = 0.2
u_max = 1.0
dt_conv = CFL * dx / u_max
dt_diff = 0.5 * dx**2 / nu
dt = min(dt_conv, dt_diff)

t_final = 0.1
nt = int(t_final / dt)

## Selected specific time snapshots to save
snapshot_times = [0.0, 0.01, 0.03, 0.05, 0.07, 0.1]
snapshot_indices = [int(t / dt) for t in snapshot_times]
snapshots = []

# Initial condition (Riemann step):
def initial_condition(x):
    return np.where(x <= 0.5, 1.0, 0.0)

u = initial_condition(x)

# save initial condition snapshot (t=0)
if 0 in snapshot_indices:
    snapshots.append(u.copy())

# Dirichlet BCs with ghost cells:
def apply_boundary_conditions(u):
    u_bc = np.zeros(len(u) + 2)
    u_bc[1:-1] = u
    u_bc[0] = 1.0       # u(0,t) = 1
    u_bc[-1] = 0.0      # u(1,t) = 0
    return u_bc

# Godunov flux for Burgers' equation:
def godunov_flux(uL, uR):
    flux = np.zeros_like(uL)
    for i in range(len(uL)):
        ul = uL[i]
        ur = uR[i]
        if ul <= ur:
            if ul >= 0:
                flux[i] = 0.5 * ul**2
            elif ur <= 0:
                flux[i] = 0.5 * ur**2
            else:
                flux[i] = 0.0
        else:
            flux[i] = 0.5 * max(ul**2, ur**2)
    return flux

# Time-stepping loop (start from n=1, since n=0 initial condition already saved)
for n in range(1, nt + 1):
    u_bc = apply_boundary_conditions(u)

    # flux calculation at interfaces
    uL = u_bc[:-1]
    uR = u_bc[1:]
    F = godunov_flux(uL, uR)

    # Diffusion term (central difference):
    diffusion = nu * (u_bc[0:-2] - 2*u_bc[1:-1] + u_bc[2:]) / dx**2

    # Update solution:
    u = u + dt * (-(F[1:] - F[:-1]) / dx + diffusion)

    # Save snapshot:
    if n in snapshot_indices:
        snapshots.append(u.copy())

# Plot snapshots:
plt.figure(figsize=(10, 6))

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

for i, u_snap in enumerate(snapshots):
    if snapshot_times[i] == 0.0:
        # initial condition: blue dotted line
        plt.plot(x, u_snap, color='blue', linestyle='--', label=f't = {snapshot_times[i]:.2f}')
    else:
        color_index = (i - 1) % len(colors)
        plt.plot(x, u_snap, color=colors[color_index], linestyle='-', label=f't = {snapshot_times[i]:.2f}')

plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Burgers Equation — Evolution Over Time (Including Initial Condition)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
