import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import bisect

from bldcm.bldcm import BLDCMSolver
from surrogate.prs import PRSSurrogate
from propeller_surrogate import MODEL_PATH
from takeoff import AircraftParameters, TakeoffSolver


def find_tow_for_combination(solver, params, target_dist=55.0):
    sim = TakeoffSolver(solver, params, use_fast_thrust=True)
    t_static = sim._get_thrust(0.0)

    def f(P):
        # Increased dt for speed during sweep
        dist = sim.simulate(P, dt=0.05, max_steps=500)
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


def simulate_combination(motor_dict, prop_dict, params, surrogate_model):
    """
    Simulates a specific motor and propeller combination and
    returns the simulation results if successful.
    """
    # Corrections to motor parameters
    corr_kv = motor_dict["kv"] * 1.05
    corr_io = motor_dict["io"] * (1 - 0.01 * motor_dict["io_vref"])
    corr_rm = (motor_dict["rm"] * 0.95) * (1.035**3)

    # Initialize BLDCM Solver
    solver = BLDCMSolver(
        surrogate_model=surrogate_model,
        kv=corr_kv,
        i0=corr_io,
        rm=corr_rm,
        diameter=prop_dict["diam"] * 0.0254,
        pitch=prop_dict["pitch"],
    )

    mtow, t_static = find_tow_for_combination(solver, params)

    propulsion_wt = motor_dict["weight"] + prop_dict["weight"]
    total_pv = params.PV + propulsion_wt
    net_mtow = mtow - total_pv

    structural_eff = (net_mtow) / total_pv

    if mtow is not None:
        return {
            "Motor": motor_dict["name"],
            "Prop": prop_dict["name"],
            "Diameter": prop_dict["diam"],
            "Pitch": prop_dict["pitch"],
            "MTOW": mtow,
            "Propulsion_Wt": propulsion_wt,
            "Net_MTOW": net_mtow,
            "EE": structural_eff,
            "T_static": t_static,
            "kv": motor_dict["kv"],
            "io": motor_dict["io"],
            "rm": motor_dict["rm"],
            "io_vref": motor_dict["io_vref"],
        }
    return None


def get_motors(databases=["./.data/tmotor_data.csv", "./.data/mad_motor_data.csv"]):
    # Load and Clean Motor Database
    all_dfs = []
    for db in databases:
        # Load data with robustness to inconsistent line lengths (some files have extra metadata columns)
        df = pd.read_csv(db, on_bad_lines="skip")
        df = df.dropna(subset=["rm", "io", "kv", "io_vref"])

        # Select consistent columns (Units are now standardized to Ohm and kg across all datasets)
        cols = ["name", "kv", "io", "rm", "io_vref", "weight"]
        all_dfs.append(df[cols])

    motor_df = pd.concat(all_dfs, ignore_index=True)
    return motor_df



def get_props():
    # Create Propeller Sweep
    prop_perf_df = pd.read_csv(
        "./.data/prop_perfmap.csv", usecols=["Propeller", "Diameter", "Pitch"]
    )
    prop_perf_df = prop_perf_df.drop_duplicates()

    # Filter available props between 16;22 diam, 6;12 pitch
    prop_perf_df = prop_perf_df[
        (prop_perf_df["Diameter"] >= 16)
        & (prop_perf_df["Diameter"] <= 22)
        & (prop_perf_df["Pitch"] >= 6)
        & (prop_perf_df["Pitch"] <= 12)
    ]

    props = []
    for _, row in prop_perf_df.iterrows():
        d, p, name = row["Diameter"], row["Pitch"], row["Propeller"]
        # weight formula (kg): (12*diam + 4*pitch - 176) / 1000
        w = (12 * d + 4 * p - 176) / 1000.0
        props.append({"name": name, "diam": d, "pitch": p, "weight": w})
    return props


def update_ui(count, total_combinations, results, last_combo_str, results_dir):
    progress_val = count / total_combinations
    progress_pct = progress_val * 100
    bar_length = 30
    filled_length = int(bar_length * progress_val)
    bar = "=" * filled_length + "-" * (bar_length - filled_length)

    output = []
    # Fast clear screen using ANSI escape sequence
    output.append("\033[H\033[J")

    output.append(
        f"Propulsion Sweep (7 Threads): |{bar}| {progress_pct:6.2f}% ({count}/{total_combinations})"
    )

    if results:
        last = results[-1]
        output.append(
            f"Last Finished: {last['Motor']} + {last['Prop']} -> EE: {last['EE']:.3f}"
        )
    else:
        output.append("Last Finished: ---")

    output.append(f"Just simulated: {last_combo_str} ...")

    if results:
        df_temp = pd.DataFrame(results)

        # Periodic Save (every 100 iterations)
        if count % 100 == 0:
            partial_path = os.path.join(results_dir, "sweep_results_v2_partial.csv")
            try:
                df_temp.sort_values("EE", ascending=False).to_csv(
                    partial_path, index=False
                )
            except PermissionError:
                pass  # safely ignore if file is open

        top_5 = df_temp.nlargest(5, "EE")
        output.append("\n--- TOP 5 COMBINATIONS SO FAR ---")
        output.append(
            f"{' # ':<3} | {'Motor':<35} | {'Kv':<8} | {'Prop':<12} | {'EE':<10}"
        )
        output.append("-" * 100)
        for i, (_, row) in enumerate(top_5.iterrows()):
            output.append(
                f"{i + 1:^3} | {row['Motor'][:30]:<35} | {row['kv']:<8} | {row['Prop']:<12} | {row['EE']:<10.3f}"
            )
        output.append("-" * 100)

    print("\n".join(output), end="")


def main():
    # 1. Load Surrogate and Parameter
    surrogate_model = PRSSurrogate.load(MODEL_PATH)
    params = AircraftParameters()

    # 2. Extract Data
    motor_df = get_motors()
    props = get_props()

    # 3. Setup Results Directory
    results_dir = "./.results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results = []

    # Pre-build combinations
    combinations = []
    for _, motor in motor_df.iterrows():
        motor_dict = motor.to_dict()
        for prop in props:
            combinations.append((motor_dict, prop))

    total_combinations = len(combinations)
    count = 0

    print("Starting parallel sweep (7 threads)...")

    # 4. Multi-Thread Processing
    with ThreadPoolExecutor(max_workers=7) as executor:
        # Submit all tasks
        future_to_combo = {
            executor.submit(
                simulate_combination, m_dict, p_dict, params, surrogate_model
            ): (m_dict, p_dict)
            for m_dict, p_dict in combinations
        }

        # Handle completion in any order
        for future in as_completed(future_to_combo):
            count += 1
            m_dict, p_dict = future_to_combo[future]
            combo_str = f"{m_dict['name']} + {p_dict['name']}"

            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as exc:
                print(f"Combination {combo_str} generated an exception: {exc}")

            # 5. Iterative UI
            update_ui(count, total_combinations, results, combo_str, results_dir)

    # 6. Saving Final Results
    df = pd.DataFrame(results)
    if not df.empty:
        output_path = os.path.join(results_dir, "sweep_results_v2.csv")
        df.sort_values("EE", ascending=False).to_csv(output_path, index=False)

        best = df.loc[df["EE"].idxmax()]
        print("\n" + "=" * 40)
        print("BEST COMBINATION FOUND:")
        print(f"Motor: {best['Motor']}")
        print(f"Propeller: {best['Prop']}")
        print(f"MTOW: {best['MTOW']:.3f} kg")
        print(f"Propulsion Weight: {best['Propulsion_Wt']:.3f} kg")
        print(f"EE: {best['EE']:.3f}")
        print(f"Results saved to: {output_path}")
        print("=" * 40)


if __name__ == "__main__":
    main()
