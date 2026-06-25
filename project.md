# Project: Motor Calc & Takeoff Simulator

A physics-based simulation and optimization suite for Predicting BLDC motor-propeller performance and calculating aircraft takeoff trajectory.

## Core Features
- **BLDCM Performance Solver**: Calculates equilibrium RPM, thrust, current, and efficiency given a power or voltage limit and forward speed.
- **Aerodynamic Surrogate**: Uses a PRS-based surrogate model (from `prs_propeller_model.prs`) to predict propeller coefficients ($C_t$ and $C_p$).
- **Takeoff Simulation**: Numerical RK4 integration of the aircraft trajectory, including ground roll and airborne climb phases.
- **TOW Optimization**: Automated root-finding to determine the maximum takeoff weight for a specified runway length.
- **Propulsion Sweep**: Automated sweeps across motor and propeller databases to find the highest Net MTOW (MTOW - propulsion weight).

## Key Files
- `bldcm/bldcm.py`: Contains the `BLDCMSolver` class.
- `takeoff.py`: The main simulation and root-finding application.
- `motor_operation.py`: Script for sweep forward speed and plotting motor efficiency/performance.
- `sweep_propulsion.py`: Optimization script for motor/propeller combinations.
- `surrogate/prs.py`: Definition of the PRS surrogate model architecture.
- `propeller_surrogate.py`: Training and loading logic for the surrogate.

## Current Optimal Configuration
- **Runway**: 55m
- **Power Limit**: 600W
- **Motor**: M310 (KV 310)
- **Propeller**: 19x8
- **Predicted MTOW**: ~12.96 kg
