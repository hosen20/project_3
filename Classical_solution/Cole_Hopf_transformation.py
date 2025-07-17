import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Parameters
nu = 0.01
L = 1.0
nx = 501
dx = L / (nx - 1)
x = np.linspace(0, L, nx)

dt = 0.0005
nt = 700

# Times we’ll use for plotting later
plot_indices = [0, int(nt*0.1), int(nt*0.2), int(nt*0.4), int(nt*0.7)]
plot_labels = [f"t={i*dt:.3f}" for i in plot_indices]

# Initial condition for phi using Cole–Hopf
phi0 = np.where(x <= 0.5, np.exp(-x/(2*nu)), np.exp(-0.5/(2*nu)))

# Setup storage
phi = np.zeros((nx, nt+1))
phi[:, 0] = phi0

# Crank-Nicolson coefficients
r = nu * dt / dx**2

A_diag = (1 + r) * np.ones(nx-2)
A_lower = (-0.5*r) * np.ones(nx-3)
A_upper = (-0.5*r) * np.ones(nx-3)

B_diag = (1 - r) * np.ones(nx-2)
B_lower = (0.5*r) * np.ones(nx-3)
B_upper = (0.5*r) * np.ones(nx-3)

# Time-stepping loop
for n in range(nt):
    b = B_diag * phi[1:-1, n]
    b[1:] += B_lower * phi[2:-1, n]
    b[:-1] += B_upper * phi[0:-3, n]

    # Robin BC at x=0
    robin_factor = 1 / (1 + dx / (2*nu))
    b[0] += 0.5 * r * robin_factor * phi[1, n]

    # Solve tridiagonal system
    ab = np.zeros((3, nx-2))
    ab[0, 1:] = A_upper
    ab[1] = A_diag
    ab[2, :-1] = A_lower

    phi_inner = solve_banded((1, 1), ab, b)

    phi[1:-1, n+1] = phi_inner
    phi[0, n+1] = robin_factor * phi[1, n+1]
    phi[-1, n+1] = phi[-2, n+1]  # Neumann BC at x=1

# Recover u from phi
u = np.zeros_like(phi)

for n in range(nt+1):
    u[:, n] = -2 * nu * np.gradient(phi[:, n], dx)
    with np.errstate(divide='ignore', invalid='ignore'):
        u[:, n] /= phi[:, n]
        u[phi[:, n] == 0, n] = 0  # Just in case

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, np.where(x <= 0.5, 1, 0), 'k--', label='Initial $u(x,0)$', linewidth=2)

for idx, lbl in zip(plot_indices[1:], plot_labels[1:]):
    plt.plot(x, u[:, idx], label=lbl)

plt.xlabel('$x$')
plt.ylabel('$u(x,t)$')
plt.title("Viscous Burgers’ Equation via Cole–Hopf")
plt.legend()
plt.grid(True)
plt.xlim([0, L])
plt.tight_layout()
plt.show()
