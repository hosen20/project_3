import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT
from scipy.interpolate import make_interp_spline

# ----------------------- Control Flags ----------------------- #
PLOT_INITIAL = True
PLOT_TIMESTEPS = True
PLOT_SPLINE = True

# ----------------- Physical and Numerical Setup ----------------- #
nu = 0.01
L = 1.0
N = 16
n_qubits = int(np.log2(N))
dx = L / N
x_vals = np.linspace(0, L, N)  # Includes full domain
CFL = 0.2
u_max = 1.0
dt_conv = CFL * dx / u_max
dt_diff = 0.5 * dx**2 / nu
dt = min(dt_conv, dt_diff)
t_final = 5.0  # Extended time to see evolution
nt = int(t_final / dt)
hbar = 1.0
m = 1.0

# ------------------ Initial Riemann State ------------------ #
def apply_smoothing_layers(qc, n_qubits):
    for layer in range(3):
        weight = 1 / (layer + 1)
        for i in range(n_qubits - 1):
            qc.h(i)
            qc.cx(i, i + 1)
            qc.rz(0.02 * weight, i + 1)
            qc.cx(i, i + 1)
            qc.h(i)
        for d, alpha in zip([2, 3], [0.01, 0.005]):
            for i in range(n_qubits - d):
                qc.crz(alpha * weight, i, i + d)
        qc.barrier()

def riemann_step_state(n_qubits):
    qc = QuantumCircuit(n_qubits)
    statevector = np.zeros(2**n_qubits, dtype=complex)
    for i in range(2**n_qubits):
        x = i * dx
        statevector[i] = 1.0 if x <= 0.5 else 0.0
    statevector /= np.linalg.norm(statevector)

    if PLOT_INITIAL:
        init_sv = Statevector(statevector)
        plt.figure()
        plt.plot(x_vals, np.abs(init_sv.data)**2)
        plt.title("Initial Riemann Step (Before Smoothing)")
        plt.xlabel("$x$")
        plt.ylabel(r"$|\psi(x, 0)|^2$")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    qc.initialize(statevector, range(n_qubits))
    qc.barrier()
    apply_smoothing_layers(qc, n_qubits)
    return qc

# ------------------ Quantum Operators ------------------ #
def quantum_prediction_operator(n_qubits, dt, nu, L):
    qc = QuantumCircuit(n_qubits)
    qc.append(QFT(n_qubits, do_swaps=True), range(n_qubits))

    k_vals = np.fft.fftfreq(2**n_qubits, d=dx) * 2 * np.pi
    for i in range(n_qubits):
        k = k_vals[i]
        theta = -(hbar**2 / (2 * m)) * (k**2) * dt / (2 * nu)
        qc.rz(theta, i)

    qc.append(QFT(n_qubits, do_swaps=True).inverse(), range(n_qubits))
    return qc

def quantum_nonlinear_operator(n_qubits, dt):
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.rz(-0.08 * dt, i)
    return qc

def quantum_smoothing_operator(n_qubits, dt):
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(0.07 * dt, i + 1)
        qc.cx(i, i + 1)
    for i in range(n_qubits - 2):
        qc.crz(0.035 * dt, i, i + 2)
    for i in range(n_qubits - 3):
        qc.crz(0.02 * dt, i, i + 3)
    return qc

def boundary_conditions_operator(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.ry(np.pi / 4, 0)
    qc.ry(-np.pi / 4, n_qubits - 1)
    qc.rz(np.pi / 6, 0)
    qc.rz(-np.pi / 6, n_qubits - 1)
    qc.cx(0, n_qubits - 1)
    return qc

# ------------------ Trotter Step Circuit ------------------ #
def trotter_step(n_qubits, dt, nu, L):
    qc = QuantumCircuit(n_qubits)
    ops = [
        quantum_prediction_operator(n_qubits, dt, nu, L),
        quantum_nonlinear_operator(n_qubits, dt),
        quantum_smoothing_operator(n_qubits, dt),
        boundary_conditions_operator(n_qubits),
    ]
    for op in ops:
        qc.compose(op, inplace=True)
    for i in range(n_qubits):
        qc.rz(0.01 * dt, i)
    for op in reversed(ops):
        qc.compose(op.inverse(), inplace=True)
    return qc

# ------------------ Run Simulation ------------------ #
qc_total = riemann_step_state(n_qubits)
state = Statevector.from_instruction(qc_total)

results = [np.abs(state.data)**2]
times = [0.0]
print(f"Running {nt} quantum Trotter steps with dt={dt:.6f}, total t={t_final}...")

step_circuit = trotter_step(n_qubits, dt, nu, L)
for _ in range(nt):
    state = state.evolve(step_circuit)
    results.append(np.abs(state.data)**2)
    times.append(times[-1] + dt)

print("Simulation complete.")

# ------------------ Plotting ------------------ #
if PLOT_TIMESTEPS:
    plt.figure(figsize=(10, 6))
    plot_indices = np.linspace(0, len(times)-1, min(len(times), 10), dtype=int)
    for idx in plot_indices:
        plt.plot(x_vals, results[idx], label=f't={times[idx]:.2f}')
    plt.xlabel('$x$')
    plt.ylabel(r'$|\psi(x,t)|^2$')
    plt.title('Quantum Burgers Equation Evolution (Riemann Step)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, results[-1], label='Final state', color='red')
    plt.xlabel('$x$')
    plt.ylabel(r'$|\psi(x,t)|^2$')
    plt.title('Final Quantum State Density')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if PLOT_SPLINE:
    plt.figure(figsize=(10, 6))
    for idx in plot_indices:
        y_vals = results[idx]
        spline = make_interp_spline(x_vals, y_vals, k=3)
        x_smooth = np.linspace(x_vals[0], x_vals[-1], 500)
        y_smooth = spline(x_smooth)
        plt.plot(x_smooth, y_smooth, label=f't={times[idx]:.2f}')
    plt.xlabel('$x$')
    plt.ylabel(r'$|\psi(x,t)|^2$')
    plt.title('Quantum Burgers Equation Evolution (Smoothed)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    spline_final = make_interp_spline(x_vals, results[-1], k=3)
    x_smooth_final = np.linspace(x_vals[0], x_vals[-1], 500)
    y_smooth_final = spline_final(x_smooth_final)
    plt.plot(x_smooth_final, y_smooth_final, label='Final state', color='red')
    plt.xlabel('$x$')
    plt.ylabel(r'$|\psi(x,t)|^2$')
    plt.title('Final Quantum State Density (Smoothed)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
