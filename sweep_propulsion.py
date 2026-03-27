import numpy as np
import pandas as pd
from scipy.optimize import bisect
from bldcm.bldcm import BLDCMSolver
from surrogate.prs import PRSSurrogate
from propeller_surrogate import MODEL_PATH
from takeoff import AircraftParameters, TakeoffSolver
import itertools


BATTERY_VOLTAGE = 22.2


def find_tow_for_combination(solver, params, target_dist=55.0):
    sim = TakeoffSolver(solver, params, use_fast_thrust=True)
    t_static = sim._get_thrust(0.0)

    def f(P):
        # Increased dt for speed during sweep
        dist = sim.simulate(P, dt=0.08)
        return dist - target_dist

    try:
        # Check signs at bounds
        if f(5.0) > 0:
            return 5.0, t_static
        if f(20.0) < 0:
            return 20.0, t_static

        opt_mass = bisect(f, 5.0, 20.0, xtol=0.01)
        return opt_mass, t_static
    except Exception:
        return None, None


def main():
    # 1. Load Surrogate
    surrogate_model = PRSSurrogate.load(MODEL_PATH)
    params = AircraftParameters()

    # 2. Load and Clean Motor Database
    motor_df = pd.read_csv("./.data/tmotor_data.csv")
    # Clean: remove entries if missing rm, io, kv or io_vref
    motor_df = motor_df.dropna(subset=["rm", "io", "kv", "io_vref"])

    # Convert units: rm (mOhm -> Ohm), weight (g -> kg)
    motor_df["rm"] = motor_df["rm"] / 1000.0
    motor_df["weight"] = motor_df["weight"] / 1000.0

    # 3. Create Propeller Sweep
    diameters = range(16, 24)  # 16 to 23
    pitches = range(6, 13)  # 6 to 12

    props = []
    for d, p in itertools.product(diameters, pitches):
        # weight formula (kg): (12*diam + 4*pitch - 176) / 1000
        w = (12 * d + 4 * p - 176) / 1000.0
        props.append({"name": f"APC {d}x{p}E", "diam": d, "pitch": p, "weight": w})

    results = []

    print(
        f"{'Motor':<15} | {'Prop':<6} | {'PropWt':<6} | {'MotWt':<6} | {'TStatic':<8} | {'MTOW':<8} | {'Net MTOW':<8}"
    )
    print("-" * 85)

    for _, motor in motor_df.iterrows():
        for prop in props:
            # Corrections to motor parameters
            corr_kv = motor["kv"] * 1.05
            corr_io = motor["io"] * (1 - 0.01 * motor["io_vref"])
            corr_rm = (motor["rm"] * 0.95) * (1.035**3)

            # Initialize BLDCM Solver
            solver = BLDCMSolver(
                surrogate_model=surrogate_model,
                kv=corr_kv,
                i0=corr_io,
                rm=corr_rm,
                diameter=prop["diam"] * 0.0254,
                pitch=prop["pitch"],
            )

            mtow, t_static = find_tow_for_combination(solver, params)

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
                        "T_static": t_static,
                    }
                )
                print(
                    f"{motor['name']:<15} | {prop['name']:<6} | {prop['weight']:<6.3f} | {motor['weight']:<6.3f} | {t_static:<8.2f} | {mtow:<8.3f} | {net_mtow:<8.3f}"
                )

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
