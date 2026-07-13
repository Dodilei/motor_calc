# Project: Motor Calc & Takeoff Simulator

A physics-based simulation and optimization suite for Predicting BLDC motor-propeller performance and calculating aircraft takeoff trajectory.

## Core Features
- **BLDCM Performance Solver**: Calculates equilibrium RPM, thrust, current, and efficiency given a power or voltage limit and forward speed.
- **Aerodynamic Surrogate**: Uses a PRS-based surrogate model (from `prs_propeller_model.prs`) to predict propeller coefficients ($C_t$ and $C_p$).
- **Takeoff Simulation**: Numerical RK4 integration of the aircraft trajectory, including ground roll and airborne climb phases.
- **TOW Optimization**: Automated root-finding to determine the maximum takeoff weight for a specified runway length.
- **Propulsion Sweep**: Automated sweeps across motor and propeller databases to find the highest Net MTOW (MTOW - propulsion weight).

## Key Files
- **Shared Modules:** [bldcm/bldcm.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/bldcm/bldcm.py), [surrogate/prs.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/surrogate/prs.py), [takeoff.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/takeoff.py), [motor_db.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/motor_db.py), [aircraft_params.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/aircraft_params.py), [propeller_surrogate.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/propeller_surrogate.py)
- **Objective 1 — Static Analysis:** [motor_static.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/motor_static.py)
- **Objective 2 — Dynamic Analysis:** [motor_operation.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/motor_operation.py)
- **Objective 3 — Optimization:** [simulate_takeoff.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/simulate_takeoff.py) (single config), [sweep_propulsion.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/sweep_propulsion.py) (combination sweep)

## Current Optimal Configuration
- **Runway**: 55m
- **Power Limit**: 595W (canonical defaults live in [aircraft_params.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/aircraft_params.py))
- **Motor**: M310 (KV 310)
- **Propeller**: 19x8
- **Predicted MTOW**: ~12.96 kg
