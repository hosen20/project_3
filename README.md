
 ## WISER 2025 Quantum Project
 ## Project 3 - Quantum Algorithm as a PDE Solver for Computational Fluid Dynamics (CFD)



### Team Information:
Team Member 1:
 - Full Name:Tania Jamshaid  
 - Womanium Program Enrollment ID - `gst-0CimQlsJ50lzdNx`


Team Member 2:
 - Full Name: Abdullah Kazi
 - Womanium Program Enrollment ID -  `gst-ivzL1FQDWcpVQ8F`


Team Member 3:
 - Full Name: Hussein Shiri
 - Womanium Program Enrollment ID - `gst-qQdmv4jGp82rZVv`


### Project Solution:
**Wiser Quantum Project 2025**
# :space_invader: Team Feyman prodigies 

  - [Project Description](#Projectdescription)
- [ Setup Instructions](#SetupInstructions)
-  [Team Introduction](#team-introduction)
- [Acknowledgement](#Acknowledgement)





## Project Description
This project aims to design and prototype a resource-efficient quantum-enhanced solver for partial differential equations (PDEs) in the context of Computational Fluid Dynamics (CFD). The focus is on leveraging emerging quantum algorithms to address some of the most challenging aspects of simulating nonlinear fluid flows—specifically, the 1D viscous Burgers’ equation in a shock tube configuration.

CFD is a cornerstone of modern science and engineering, underpinning advances in aerodynamics, energy systems, and climate modeling. However, classical high-fidelity solvers often face significant limitations when resolution demands increase or when dealing with stiff, nonlinear PDEs. The Burgers’ equation offers a tractable yet nontrivial test case for exploring new computational paradigms. It encapsulates both convective steepening and viscous smoothing—the same nonlinear and dissipative dynamics found in the Navier–Stokes equations—while avoiding the complexity of pressure projection. As a result, methods validated on Burgers can be extended to full-scale fluid dynamics problems with relative ease.

The core challenge of this project is to integrate quantum computing techniques into the PDE-solving pipeline without incurring prohibitive resource requirements. Two promising frameworks are available as starting points. The first is the Quantum Tensor-Network (QTN) approach, which compresses the velocity field into a Matrix-Product State representation. This format significantly reduces the state space while retaining essential physical information, allowing the flow field to be evolved efficiently using divergence-free projectors. The second is the Hydrodynamic Schrödinger Equation (HSE) formulation, which maps incompressible fluid motion onto the evolution of a quantum wavefunction. This reframing enables the use of universal quantum processors for direct time evolution via quantum gates, with the nonlinear and viscous terms incorporated through Trotterization or other operator-splitting schemes.



A typical hybrid solver workflow begins with classical preprocessing to set up the spatial grid, initial conditions, and boundary values. The velocity field is then encoded into a quantum-compatible representation—either as a compressed tensor network or as a normalized amplitude vector for qubits. Quantum circuits implement the time evolution step, alternating between spectral updates (via quantum Fourier transforms or momentum-space phase shifts) and position-space nonlinear updates (using rotation and controlled-phase gates). Measurements return statistical samples of the evolved state, which are post-processed classically to reconstruct the velocity profile. Boundary conditions are enforced between time steps, and the process repeats until the final simulation time is reached.

By conducting the Burgers’ equation study, the project aims to produce concrete benchmarks for quantum resource usage, numerical accuracy, and runtime performance. These benchmarks will inform future scaling to more complex CFD problems, such as full Navier–Stokes simulations, providing a pathway toward practical quantum-enhanced fluid dynamics in the coming years.


### Methods and objectives
Burgers’ Equation — Classical, Quantum-Inspired & Quantum Solvers
This repository implements multiple approaches for solving the 1D viscous Burgers’ equation, enabling benchmarking between analytic, classical numerical, quantum-inspired, and quantum algorithms.

Implemented Methods
1. Analytic Benchmark — Cole–Hopf Transform
Exact closed-form travelling viscous shock solution via Cole–Hopf transformation.

Serves as the ground truth for validation.

2. Classical Numerical Solver — Finite-Volume Godunov
Conservative finite-volume method with Godunov fluxes for shock capturing.

Explicit time-stepping, central differencing for diffusion.

CFL-stable, easy to implement, accurate shock handling.

3. Quantum Tensor Network (QTN) Approaches
Quantum-Inspired: Classical simulation of QTN evolution using MPS (via quimb).

Quantum Trotterization: MPS circuit encoding + Suzuki/Lie Trotter decomposition.

Scales with number of qubits → grid size = 2ⁿ.

4. HSE + QTN
Hydrodynamic Schrödinger Equation reformulation of Burgers’ equation.

Compressed MPS representation of wavefunctions & operators for scalable simulation.

5. Hybrid Quantum–Classical Solver
Amplitude encoding of velocity field.

Split-operator quantum evolution (QFT-based kinetic + nonlinear potential terms).

Classical boundary condition enforcement after each quantum step.

Error Mitigation & Scaling
Benchmarks on noiseless simulators, noisy backends (FakeManillaV2), and real QPUs.

Error mitigation via Zero Noise Extrapolation (ZNE) and Dynamic Decoupling (DD).

Scaling analysis for qubit count and higher-dimensional generalizations.

## Results

<img width="467" height="408" alt="Hybrid Quantum-Classical Burgers' Equation Solver" src="https://github.com/user-attachments/assets/540d26b0-b113-4d9c-92eb-1b353a2dbcfe" />
<img width="720" height="427" alt="Quantum_inspired_solution" src="https://github.com/user-attachments/assets/9699460c-339b-440a-9381-472d677d266f" />
<img width="607" height="390" alt="HSEHybrid" src="https://github.com/user-attachments/assets/f1571b82-5abb-4c2c-b545-a4d238e2bfa5" />









## Setup Instructions

#### 1. **Clone the Repository**

   Open your terminal or command prompt and run the following command to clone the repository:

   ```bash
   git clone 
```
Navigate into the project directory:
```bash
cd 
```
#### 2. **Create a Python Virtual Environment**
Create a virtual environment to manage dependencies:

```bash
python -m venv venv
```
#### 3. **Activate the Virtual Environment**
On Windows:
```bash
venv\Scripts\activate
```
On macOS and Linux:
```bash
source venv/bin/activate
```
#### 4. **Install Dependencies**
Install the required dependencies listed in the ['requirements.txt'] file:


- ## Usage of the Project

WISER_CFD_2025/
```bash
│
├── README.md                                  # Project overview and documentation
├── LICENSE                                    # License for use and distribution
│
├── Benchmarking/                              # Scripts and notebooks for performance and accuracy benchmarking
│   └── Update_Readme.md                       # Notes and documentation for benchmark results
│
├── Classical_solution/                        # Classical solver implementations (Cole–Hopf, Godunov)
│   └── Update_readme.md                       # Documentation for classical solvers
│
├── HSE+QTN/                                   # Hybrid Hydrodynamic Schrödinger Equation + Quantum Tensor Network method
│   └── Update_Readme.md                       # Documentation and usage notes
│
├── HSE/                                       # Pure Hydrodynamic Schrödinger Equation solver implementation
│   └── Add_files_via_upload                   # Latest code and data files
│
├── Hardware-QPU/                              # Runs on real quantum processing units
│   └── Add_files_via_upload                   # QPU execution scripts and outputs
│
├── Plot-Images/                               # Generated plots for results visualization
│   └── Update_readme.md                       # Plot descriptions and references
│
├── QTN/                                       # Quantum Tensor Network solver implementation
│   └── Added_real_QPU_solution                # QTN runs adapted for quantum hardware
│
├── Slide-Deck/                                # Presentation slides for project summary
│   └── Update_readme.md                       # Slide deck documentation
│
├── Technical-Report/                          # Detailed writeup of methods, results, and analysis
│   └── Update_readme.md                       # Report notes and links
│
├── requirements.txt                           # Python dependencies for running the project

```

## Team Introduction
**Team Name:** Feyn man Prodigies 

|   **Member Names**| **Abdullah Kazi**                      | **Tania Jamshaid** | **Hussein Shiri** | 
|----------------|-----------------------------------|----------------------------|----------------------------|
| **GitHub ID**  | AbdullahKazi500                   | Tania Jamshaid      | Hosen20 |



----------------------

## Acknowledgement
We would like to extend our heartfelt gratitude to WOMANIUM
Thank you for the opportunity to participate in this event, which has not only enriched our knowledge and skills but also connected us with a network of like-minded professionals and enthusiasts.


### Project Presentation Deck:

Slide deck and project descriptions linked [here](https://github.com/hosen20/CFD-Womanium-Project/tree/main/Slide-Deck)




