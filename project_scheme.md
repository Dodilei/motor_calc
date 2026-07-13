# Architecture: Three-Objective Motor Analysis Suite

## Core Modules (shared, never run directly)
- [bldcm/bldcm.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/bldcm/bldcm.py) — BLDC motor equilibrium solver
- [surrogate/prs.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/surrogate/prs.py) — PRS surrogate model for propeller Ct/Cp
- [surrogate/evaluation.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/surrogate/evaluation.py) — Surrogate cross-validation evaluator
- [takeoff.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/takeoff.py) — TakeoffSolver class and MTOW root-finding
- [motor_db.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/motor_db.py) — Motor database loading, parameter corrections, surrogate loading
- [aircraft_params.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/aircraft_params.py) — Canonical aircraft/system parameters (single source of truth)
- [propeller_surrogate.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/propeller_surrogate.py) — Surrogate training script + MODEL_PATH constant

## Entry Points (independently runnable)
- [motor_static.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/motor_static.py) — Objective 1: Static throttle sweep analysis
- [motor_operation.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/motor_operation.py) — Objective 2: Dynamic forward-speed sweep + thrust curve
- [simulate_takeoff.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/simulate_takeoff.py) — Objective 3a: Single-config takeoff simulation + MTOW optimization
- [sweep_propulsion.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/sweep_propulsion.py) — Objective 3b: Motor/propeller combination sweep

## Conventions
- Dot-prefixed directories for data/outputs: `.data/`, `.models/`, `.results/`, `.plots/`, `.output/`
- `.standalone/` for one-off scripts that are not part of the main workflow
- Generated images go to `.plots/`
- All shared aircraft/system parameters are defined in [aircraft_params.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/aircraft_params.py)
- Motor data access goes through [motor_db.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/motor_db.py)
- Each entry point can override defaults via CLI arguments
