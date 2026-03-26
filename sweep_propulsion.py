import numpy as np
import pandas as pd
from scipy.optimize import bisect
from bldcm.bldcm import BLDCMSolver
from surrogate.prs import PRSSurrogate
from propeller_surrogate import MODEL_PATH
from takeoff import AircraftParameters, TakeoffSolver


def find_tow_for_combination(solver, params, target_dist=55.0):
    sim = TakeoffSolver(solver, params, use_fast_thrust=False)

    def f(P):
        # Increased dt for speed during sweep
        dist = sim.simulate(P, dt=0.08)
        return dist - target_dist

    try:
        # Check signs at bounds
        if f(5.0) > 0:
            return 5.0
        if f(25.0) < 0:
            return 25.0

        opt_mass = bisect(f, 5.0, 25.0, xtol=0.2)
        return opt_mass
    except Exception:
        return None


def main():
    # 1. Load Surrogate
    surrogate_model = PRSSurrogate.load(MODEL_PATH)
    params = AircraftParameters()

    # 2. Define Databases
    motors = [
        {"name": "M310", "kv": 310, "i0": 1.66, "rm": 0.065, "weight": 0.70},
        {"name": "M400", "kv": 400, "i0": 2.20, "rm": 0.045, "weight": 0.60},
        {"name": "M200", "kv": 200, "i0": 1.10, "rm": 0.095, "weight": 0.85},
    ]

    props = [
        {"name": "18x8", "diam": 18, "pitch": 8, "weight": 0.10},
        {"name": "18x10", "diam": 18, "pitch": 10, "weight": 0.11},
        {"name": "19x8", "diam": 19, "pitch": 8, "weight": 0.12},
        {"name": "19x10", "diam": 19, "pitch": 10, "weight": 0.13},
    ]

    results = []

    print(
        f"{'Motor':<6} | {'Prop':<6} | {'PropWt':<6} | {'MotWt':<6} | {'MTOW':<8} | {'Net MTOW':<8}"
    )
    print("-" * 60)

    for motor in motors:
        for prop in props:
            # Initialize BLDCM Solver
            solver = BLDCMSolver(
                surrogate_model=surrogate_model,
                kv=motor["kv"],
                i0=motor["i0"],
                rm=motor["rm"],
                diameter=prop["diam"] * 0.0254,
                pitch=prop["pitch"],
            )

            mtow = find_tow_for_combination(solver, params)

            if mtow is not None:
                propulsion_wt = motor["weight"] + prop["weight"]
                net_mtow = mtow - propulsion_wt
                results.append(
                    {
                        "Motor": motor["name"],
                        "Prop": prop["name"],
                        "MTOW": mtow,
                        "Propulsion_Wt": propulsion_wt,
                        "Net_MTOW": net_mtow,
                    }
                )
                print(
                    f"{motor['name']:<6} | {prop['name']:<6} | {prop['weight']:<6.2f} | {motor['weight']:<6.2f} | {mtow:<8.3f} | {net_mtow:<8.3f}"
                )
            else:
                print(f"{motor['name']:<6} | {prop['name']:<6} | FAILED")

    df = pd.DataFrame(results)
    if not df.empty:
        best = df.loc[df["Net_MTOW"].idxmax()]
        print("\n" + "=" * 40)
        print("BEST COMBINATION FOUND:")
        print(f"Motor: {best['Motor']}")
        print(f"Propeller: {best['Prop']}")
        print(f"MTOW: {best['MTOW']:.3f} kg")
        print(f"Propulsion Weight: {best['Propulsion_Wt']:.3f} kg")
        print(f"Net MTOW: {best['Net_MTOW']:.3f} kg")
        print("=" * 40)


if __name__ == "__main__":
    main()
