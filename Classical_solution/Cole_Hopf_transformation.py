import numpy as np
import matplotlib.pyplot as plt

def burgers_tanh_solution(x, t, u_L, u_R, x0, nu):
    """
    Analytical solution of the 1D viscous Burgers' equation for
    a Riemann step initial condition, using the Cole–Hopf transform.

    Parameters
    ----------
    x : ndarray
        Spatial grid points.
    t : float
        Time at which to evaluate the solution.
    u_L : float
        Left state (x < x0) velocity.
    u_R : float
        Right state (x >= x0) velocity.
    x0 : float
        Position of the initial step.
    nu : float
        Viscosity (kinematic), controls shock thickness.

    Returns
    -------
    u : ndarray
        Velocity profile u(x, t).
    """
    if t == 0:
        # Exact step initial condition
        return np.where(x < x0, u_L, u_R)

    # Shock center moves at average velocity
    shock_center = x0 + 0.5 * (u_L + u_R) * t

    # Argument of tanh function from Cole–Hopf solution
    arg = (u_L - u_R) * (x - shock_center) / (4 * nu)

    # Smooth shock profile
    u = 0.5 * (u_L + u_R) - 0.5 * (u_L - u_R) * np.tanh(arg)
    return u


# Parameters
u_L = 1.0
u_R = 0.0
x0 = 0.5
nu = 0.01

# Spatial and temporal grids
x = np.linspace(0, 1, 500)
time_values = [0.0, 0.01, 0.03, 0.05, 0.07, 0.10]

# Plot
plt.figure(figsize=(10, 6))

for idx, t in enumerate(time_values):
    u_profile = burgers_tanh_solution(x, t, u_L, u_R, x0, nu)
    if t == 0:
        plt.plot(x, u_profile, '--', color='gray', lw=2, label='t = 0.00 (Initial)')
    else:
        plt.plot(x, u_profile, lw=2, label=f't = {t:.2f}')

plt.title("Analytical Viscous Burgers Solution (Cole–Hopf, tanh profile)", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("u(x, t)", fontsize=12)
plt.grid(ls=':', alpha=0.7)
plt.legend()
plt.ylim(-0.1, 1.1)
plt.tight_layout()
plt.show()
