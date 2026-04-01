# Aircraft Takeoff & Propulsion Simulator

This project provides a high-fidelity numerical simulation for aircraft takeoff performance, integrating a physics-based BLDC motor-propeller model with a dynamic trajectory solver.

## 1. Propulsion Theory (BLDC Motor + Propeller)

The propulsion system's performance is determined by finding the equilibrium between the aerodynamic load of the propeller and the electrical characteristics of the BLDC motor.

### 1.1 Propeller Aerodynamics
The propeller's thrust ($T$) and absorbed power ($P_{prop}$) are governed by non-dimensional coefficients $C_t$ and $C_p$, which are functions of the Advance Ratio ($J$):

$$J = \frac{V}{n \cdot D} \tag{1}$$

Where:
- $V$ is the forward speed (m/s).
- $n$ is the rotational speed (rev/s).
- $D$ is the propeller diameter (m).

The dimensional thrust and power are then:
$$T = C_t \cdot \rho \cdot n^2 \cdot D^4 \tag{2}$$
$$P_{prop} = C_p \cdot \rho \cdot n^3 \cdot D^5 \tag{3}$$

In this project, $C_t$ and $C_p$ are predicted using a **PRS (Polynomial Response Surface)** surrogate model trained on experimental/simulation data.

### 1.2 Motor Electrical Model
A BLDC motor is modeled using three primary constants: $KV$ (velocity constant), $I_0$ (no-load current), and $R_m$ (internal resistance).

The voltage required to maintain a certain RPM and current is given by:
$$V_{motor} = \frac{RPM}{KV} + I \cdot R_m \tag{4}$$

The current draw ($I$) is proportional to the required mechanical torque (plus no-load losses):
$$I = I_0 + \frac{P_{prop} \cdot KV}{RPM} \tag{5}$$

### 1.3 Equilibrium Solver
The `BLDCMSolver` uses a numerical root-finder (`scipy.optimize.brentq`) to find the equilibrium rotational speed ($RPM_{eq}$) where the motor's electrical power matches the propeller's mechanical requirement:
$$f(RPM) = P_{in} - P_{prop} = 0 \tag{6}$$
subject to battery voltage ($V_{limit}$) and electronic speed controller ($P_{limit}$) constraints.

---

## 2. Takeoff Simulation

The takeoff trajectory is solved using a **4th-order Runge-Kutta (RK4)** numerical integration scheme.

### 2.1 Physics Equations
The simulation tracks the state vector $[x, y, v_x, v_y]$. The equations of motion are expressed in a Cartesian frame:

**Ground Roll Phase ($V < V_{rotate}$):**
The aircraft is constrained to the ground ($y = 0$).
$$L = \frac{1}{2} \rho V^2 S \cdot C_{L,ground} \tag{7}$$
$$D = \frac{1}{2} \rho V^2 S \cdot C_{D,ground} \tag{8}$$
$$N = W - L \tag{9}$$
$$a_x = \frac{T - D - \mu N}{m} \tag{10}$$

**Climb Phase ($V \ge V_{rotate}$):**
The aircraft rotates to $C_{L,max}$ and begins to climb. The forces are sumed in the velocity frame ($\text{tangent } t$ and $\text{normal } n$):
$$a_{t} = \frac{T - D - W \sin(\gamma)}{m} \tag{11}$$
$$a_{n} = \frac{L - W \cos(\gamma)}{m} \tag{12}$$

Where $\gamma$ is the flight path angle and $W = m \cdot g$ is the aircraft weight.

### 2.2 MTOW Optimization
The simulator uses a **Bisection** root-finding method to determine the maximum takeoff weight (MTOW) that allows the aircraft to clear a specific obstacle height (e.g., $h_{obs} = 0.9m$) within a given runway length (e.g., $x_{max} = 55m$):
$$f(m) = x(h=h_{obs}) - x_{max} = 0 \tag{13}$$

---

## 3. Usage

- **Performance Sweep**: Run `sweep_propulsion.py` to compare different motor/propeller combinations and find the highest EE.
- **Trajectory Analysis**: Run `takeoff.py` to simulate a specific mass and configuration.
- **Motor Characterization**: Run `motor_operation.py` to generate performance plots (Efficiency, Thrust, RPM) vs forward speed.
