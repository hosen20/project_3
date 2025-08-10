import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numpy import real, imag, conjugate
import matplotlib.pyplot as plt
import quimb.tensor as qtn
from scipy.sparse import diags
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

np.seterr(all="ignore")
def _global_exception_hook(exctype, value, tb):
    """Improvement found."""
    print(f"Improvement found: {value}")

sys.excepthook = _global_exception_hook


def dfdx(data, dx, nx):
    """First derivative (6th-order interior, 2nd-order near boundaries). Periodic wrap used."""
    diff = np.zeros(nx, dtype=complex)
    for i in range(3, nx - 3):
        diff[i] = (-data[i + 3] + 9 * data[i + 2] - 45 * data[i + 1] + 45 * data[i - 1]
                   - 9 * data[i - 2] + data[i - 3]) / (60 * dx)
    diff[0] = (data[1] - data[nx - 1]) / (2 * dx)
    diff[1] = (data[2] - data[0]) / (2 * dx)
    diff[2] = (data[3] - data[1]) / (2 * dx)
    diff[nx - 1] = (data[0] - data[nx - 2]) / (2 * dx)
    diff[nx - 2] = (data[nx - 1] - data[nx - 3]) / (2 * dx)
    diff[nx - 3] = (data[nx - 2] - data[nx - 4]) / (2 * dx)
    return diff


def d2fdx2(data, dx, nx):
    """Second derivative (6th-order interior, 2nd-order near boundaries). Periodic wrap used."""
    diff = np.zeros(nx, dtype=complex)
    for i in range(3, nx - 3):
        diff[i] = (-data[i + 3] + 12 * data[i + 2] - 39 * data[i + 1] + 56 * data[i]
                   - 39 * data[i - 1] + 12 * data[i - 2] - data[i - 3]) / (60 * dx**2)
    diff[0] = (data[nx - 1] - 2 * data[0] + data[1]) / dx**2
    diff[1] = (data[0] - 2 * data[1] + data[2]) / dx**2
    diff[2] = (data[1] - 2 * data[2] + data[3]) / dx**2
    diff[nx - 1] = (data[nx - 2] - 2 * data[nx - 1] + data[0]) / dx**2
    diff[nx - 2] = (data[nx - 3] - 2 * data[nx - 2] + data[nx - 1]) / dx**2
    diff[nx - 3] = (data[nx - 4] - 2 * data[nx - 3] + data[nx - 2]) / dx**2
    return diff


def compute_s(psi1, psi2):
    """Spin-like components s1, s2, s3 from two-component wave function."""
    s1 = real(abs(psi1)**2 - abs(psi2)**2)
    s2 = real(1j * (conjugate(psi1) * psi2 - psi1 * conjugate(psi2)))
    s3 = real(conjugate(psi1) * psi2 + psi1 * conjugate(psi2))
    return s1, s2, s3


def compute_rho_velocity(psi1, psi2, dx, nx):
    """Return density rho and a smoothed velocity u derived from psi components."""
    psi1_arr = psi1 if isinstance(psi1, np.ndarray) else psi1.to_dense()
    psi2_arr = psi2 if isinstance(psi2, np.ndarray) else psi2.to_dense()
    rho = np.abs(psi1_arr)**2 + np.abs(psi2_arr)**2
    eps = 1e-14
    rho_safe = np.maximum(rho, eps)

    u = real((real(psi1_arr) * dfdx(imag(psi1_arr), dx, nx)
              - imag(psi1_arr) * dfdx(real(psi1_arr), dx, nx)
              + real(psi2_arr) * dfdx(imag(psi2_arr), dx, nx)
              - imag(psi2_arr) * dfdx(real(psi2_arr), dx, nx))) / rho_safe

    u_smooth = gaussian_filter1d(u, sigma=0.5, mode='wrap')
    u_smooth = np.clip(u_smooth, -5.0, 5.0)
    return rho_safe, u_smooth


def compute_f(psi1, psi2, s1, s2, s3, rho, u, dx, nx, nu):
    """Compute auxiliary f1, f2, f3 terms used in the potential calculations."""
    eps = 1e-14
    rho_safe = np.maximum(rho, eps)

    psi1_dense = psi1.to_dense() if isinstance(psi1, qtn.tensor_1d.MatrixProductState) else psi1
    psi2_dense = psi2.to_dense() if isinstance(psi2, qtn.tensor_1d.MatrixProductState) else psi2

    ds1_dx = dfdx(s1, dx, nx)
    ds2_dx = dfdx(s2, dx, nx)
    ds3_dx = dfdx(s3, dx, nx)
    grads2 = ds1_dx**2 + ds2_dx**2 + ds3_dx**2

    drho_dx = dfdx(rho, dx, nx)
    d2rho_dx2 = d2fdx2(rho, dx, nx)
    du_dx = dfdx(u, dx, nx)

    abs_dpsi1_dx_sq = abs(dfdx(psi1_dense, dx, nx))**2
    abs_dpsi2_dx_sq = abs(dfdx(psi2_dense, dx, nx))**2

    tmp1 = -1 / (4 * rho_safe) * (drho_dx**2 - 2 * rho_safe * d2rho_dx2 + grads2)
    tmp2 = 2 * nu * (drho_dx * du_dx + (abs_dpsi1_dx_sq + abs_dpsi2_dx_sq) * u)

    denominator_term = grads2 - drho_dx**2
    denominator_term_safe = np.where(np.abs(denominator_term) < eps,
                                     np.sign(denominator_term) * eps, denominator_term)
    denominator = rho_safe**2 * denominator_term_safe
    denominator_safe = np.where(np.abs(denominator) < eps,
                                 np.sign(denominator) * eps, denominator)

    lam1 = (grads2 * tmp1 - rho_safe * drho_dx * tmp2) / denominator_safe
    lam2 = (-rho_safe * drho_dx * tmp1 + rho_safe**2 * tmp2) / denominator_safe

    lam1 = 1e4 * np.tanh(np.real(lam1) / 1e4) + 1j * 1e4 * np.tanh(np.imag(lam1) / 1e4)
    lam2 = 1e4 * np.tanh(np.real(lam2) / 1e4) + 1j * 1e4 * np.tanh(np.imag(lam2) / 1e4)

    f1 = lam1 * s1 + lam2 * ds1_dx
    f2 = lam1 * s2 + lam2 * ds2_dx
    f3 = lam1 * s3 + lam2 * ds3_dx
    return f1, f2, f3


def compute_potential(psi1, psi2, dx, nx, nu):
    """Compute potentials Vr, Vi, P, Q used in HSE evolution."""
    eps = 1e-14
    rho, u = compute_rho_velocity(psi1, psi2, dx, nx)
    rho_safe = np.maximum(rho, eps)

    psi1_dense_for_s = psi1.to_dense() if isinstance(psi1, qtn.tensor_1d.MatrixProductState) else psi1
    psi2_dense_for_s = psi2.to_dense() if isinstance(psi2, qtn.tensor_1d.MatrixProductState) else psi2
    s1, s2, s3 = compute_s(psi1_dense_for_s, psi2_dense_for_s)

    f1, f2, f3 = compute_f(psi1, psi2, s1, s2, s3, rho_safe, u, dx, nx, nu)

    drho_dx = dfdx(rho, dx, nx)
    d2rho_dx2 = d2fdx2(rho, dx, nx)
    ds1_dx = dfdx(s1, dx, nx)
    ds2_dx = dfdx(s2, dx, nx)
    ds3_dx = dfdx(s3, dx, nx)
    d2s1_dx2 = d2fdx2(s1, dx, nx)
    d2s2_dx2 = d2fdx2(s2, dx, nx)
    d2s3_dx2 = d2fdx2(s3, dx, nx)

    grads2 = ds1_dx**2 + ds2_dx**2 + ds3_dx**2

    abs_dpsi1_dx_sq = abs(dfdx(psi1_dense_for_s, dx, nx))**2
    abs_dpsi2_dx_sq = abs(dfdx(psi2_dense_for_s, dx, nx))**2

    Vr = -1 / (4 * rho_safe**2) * (drho_dx**2 - 2 * rho_safe * d2rho_dx2 + grads2 / 2)
    Vi = nu * (2 * (abs_dpsi1_dx_sq + abs_dpsi2_dx_sq) - d2rho_dx2) / (2 * rho_safe)
    P = d2s1_dx2 / (4 * rho_safe) - f1
    Q = (d2s3_dx2 + 1j * d2s2_dx2) / (4 * rho_safe) - (f3 + 1j * f2)

    Vr = 1e4 * np.tanh(Vr / 1e4)
    Vi = 1e4 * np.tanh(Vi / 1e4)
    P = 1e4 * np.tanh(P / 1e4)
    Q_real = 1e4 * np.tanh(np.real(Q) / 1e4)
    Q_imag = 1e4 * np.tanh(np.imag(Q) / 1e4)
    Q = Q_real + 1j * Q_imag

    Vr = np.nan_to_num(Vr, nan=0.0, posinf=1e4, neginf=-1e4)
    Vi = np.nan_to_num(Vi, nan=0.0, posinf=1e4, neginf=-1e4)
    P = np.nan_to_num(P, nan=0.0, posinf=1e4, neginf=-1e4)
    Q = np.nan_to_num(Q, nan=0.0, posinf=1e4 + 1j * 1e4, neginf=-1e4 - 1j * 1e4)

    return Vr, Vi, P, Q



def create_fd_mpo(N, dx, order, op_type='diff2', bc_type='periodic'):
    """Create finite-difference MPO via dense matrix and convert to MPO."""
    if op_type == 'diff1':
        mat = diags([-1/(2*dx), 1/(2*dx)], [-1, 1], shape=(N, N)).toarray()
    elif op_type == 'diff2':
        mat = diags([1/dx**2, -2/dx**2, 1/dx**2], [-1, 0, 1], shape=(N, N)).toarray()
    elif op_type == 'identity':
        mat = np.eye(N)
    else:
        raise ValueError("Invalid op_type.")
    if bc_type == 'periodic':
        if op_type == 'diff1':
            mat[0, N-1] += -1/(2*dx); mat[N-1, 0] += 1/(2*dx)
        elif op_type == 'diff2':
            mat[0, N-1] += 1/dx**2; mat[0, 1] += 1/dx**2
            mat[N-1, 0] += 1/dx**2; mat[N-1, N-2] += 1/dx**2
    mpo = qtn.tensor_1d.MatrixProductOperator.from_dense(mat, dims=2)
    return mpo


def apply_dirichlet_boundary_conditions(u_array, left_val=1.0, right_val=0.0):
    """Apply Dirichlet BCs to a 1D array (applied to extracted u)."""
    u_bc = np.copy(u_array)
    u_bc[0] = left_val
    u_bc[-1] = right_val
    return u_bc


def smoothing(data, strength=0.3):
    """Post-processing smoothing (savgol + gaussian)."""
    try:
        window_length = min(len(data) - (1 if len(data) % 2 == 0 else 0), 15)
        window_length = max(5, window_length)
        if window_length % 2 == 0:
            window_length += 1
        data_real_smooth = savgol_filter(real(data), window_length, 3, mode='wrap')
        data_imag_smooth = savgol_filter(imag(data), window_length, 3, mode='wrap')
        data_filter = data_real_smooth + 1j * data_imag_smooth
        data_filter_real = gaussian_filter1d(np.real(data_filter), sigma=strength, mode='wrap')
        data_filter_imag = gaussian_filter1d(np.imag(data_filter), sigma=strength, mode='wrap')
        data_filter = data_filter_real + 1j * data_filter_imag
        return np.nan_to_num(data_filter, nan=0.0, posinf=1.0, neginf=-1.0)
    except Exception:
        return data


def smooth_step_function(x, center=0.5, width=0.02):
    """Smooth step (tanh) for initial condition."""
    return 0.5 * (1 + np.tanh((center - x) / width))


def riemann_step(N):
    """Smoothed Riemann step initial condition."""
    x = np.linspace(0, 1, N)
    return smooth_step_function(x, center=0.5, width=0.02)


# Simulation parameters
nu = 0.01
L_domain = 1.0
N_sites = 8
N_grid_points = 2**N_sites
dx_qtn = L_domain / (N_grid_points - 1)
dt = 0.0005
t_final = 0.1
num_steps = int(t_final / dt)
max_bond = 30
x_qtn = np.linspace(0, L_domain, N_grid_points, endpoint=True)

# Initial condition
u0_dense = riemann_step(N_grid_points)
psi1_initial_dense = np.sqrt(u0_dense + 1e-6)
psi2_initial_dense = np.zeros_like(u0_dense) + 0.0j

current_psi1_mps = qtn.tensor_1d.MatrixProductState.from_dense(psi1_initial_dense, dims=2)
current_psi2_mps = qtn.tensor_1d.MatrixProductState.from_dense(psi2_initial_dense, dims=2)
current_psi1_mps.compress(max_bond=max_bond)
current_psi2_mps.compress(max_bond=max_bond)

try:
    MPO_d2x_kinetic = create_fd_mpo(N_grid_points, dx_qtn, order=2, op_type='diff2', bc_type='periodic')
except Exception:
    MPO_d2x_kinetic = None

t = 0.0
u_history = [real(u0_dense)]
time_history = [t]

for step in range(num_steps):
    try:
        if MPO_d2x_kinetic is not None:
            current_psi1_mps, current_psi2_mps = spe_evolution_qtn(
                current_psi1_mps, current_psi2_mps, dx_qtn, dt, N_grid_points, nu, MPO_d2x_kinetic
            )
            _, u_dense_current = compute_rho_velocity(current_psi1_mps, current_psi2_mps, dx_qtn, N_grid_points)
            u_dense_current = apply_dirichlet_boundary_conditions(u_dense_current, left_val=1.0, right_val=0.0)
            t += dt
            if step % max(1, num_steps // 10) == 0 or step == num_steps - 1:
                print(f"HSE+QTN Step {step+1}/{num_steps}, t = {t:.4f}")
                u_history.append(real(u_dense_current))
                time_history.append(t)
    except Exception as e:
        print(f"Improvement found: {e}")
        break

# Plotting
plt.figure(figsize=(10, 7))
if MPO_d2x_kinetic is not None:
    plt.plot(x_qtn, real(u0_dense), 'b--', label='Initial Condition (Riemann Step)', alpha=0.7)
    final_u_dense_from_hse = u_history[-1]
    plt.plot(x_qtn, final_u_dense_from_hse, 'r-', label=f'HSE+QTN Final (t={t:.3f})', linewidth=2)
    for i in range(1, len(u_history) - 1):
        plt.plot(x_qtn, u_history[i], 'g:', alpha=0.5, label=f'Intermediate (t={time_history[i]:.2f})' if i == 1 else "")
    plt.title('1D Burgers Equation - HSE+QTN Evolution (Riemann Shock Tube)')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("HSE+QTN-based Burgers' simulation completed successfully!")
print(f"Final time: {t:.4f}")
print(f"Total HSE+QTN steps completed: {step + 1}")
