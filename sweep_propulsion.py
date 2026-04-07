import os
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from bldcm.bldcm import BLDCMSolver
from takeoff import TakeoffSolver, apply_corrections, find_tow, load_surrogate


class AircraftParameters:
    def __init__(self, **overrides):
        self.g = 9.81
        self.rho = 1.225
        self.S_wing = 0.8
        self.CL_max = 1.969
        self.CL_ground = 0.997
        self.CD_ground = 0.057
        self.CD_max = 0.199
        self.mu = 0.04
        self.P_limit = 590.0
        self.V_limit = 23.0
        self.PV = 2.2
        for k, v in overrides.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)


def simulate_combination(motor_dict, prop_dict, params, surrogate_model):
    corr_kv, corr_io, corr_rm = apply_corrections(
        motor_dict["kv"], motor_dict["io"], motor_dict["rm"], motor_dict["io_vref"]
    )

    if corr_io < 0:
        return None

    solver = BLDCMSolver(
        surrogate_model=surrogate_model,
        kv=corr_kv,
        i0=corr_io,
        rm=corr_rm,
        diameter=prop_dict["diam"] * 0.0254,
        pitch=prop_dict["pitch"],
    )

    mtow, t_static, status = find_tow(
        solver,
        params,
        target_dist=55.0,
        bounds=(4.0, 28.0),
        use_fast_thrust=True,
        dt=0.05,
        max_steps=500,
    )

    if mtow is None:
        return None

    motor_wt = motor_dict["weight"]
    prop_wt = prop_dict["weight"]
    propulsion_wt = motor_wt + prop_wt
    total_pv = params.PV + propulsion_wt
    net_mtow = mtow - total_pv
    structural_eff = net_mtow / total_pv

    return {
        "Motor": motor_dict["name"],
        "Prop": prop_dict["name"],
        "Diameter": prop_dict["diam"],
        "Pitch": prop_dict["pitch"],
        "MTOW": mtow,
        "Motor_Wt": motor_wt,
        "Prop_Wt": prop_wt,
        "Propulsion_Wt": propulsion_wt,
        "Net_MTOW": net_mtow,
        "EE": structural_eff,
        "T_static": t_static,
        "Status": status,
        "kv": motor_dict["kv"],
        "io": motor_dict["io"],
        "rm": motor_dict["rm"],
        "io_vref": motor_dict["io_vref"],
    }


def get_motors(databases=None):
    if databases is None:
        databases = ["./.data/tmotor_data.csv", "./.data/mad_motor_data.csv"]

    all_dfs = []
    for db in databases:
        df = pd.read_csv(db, on_bad_lines="skip")
        df = df.dropna(subset=["rm", "io", "kv", "io_vref"])
        cols = ["name", "kv", "io", "rm", "io_vref", "weight"]
        all_dfs.append(df[cols])

    return pd.concat(all_dfs, ignore_index=True)


def get_props():
    prop_perf_df = pd.read_csv(
        "./.data/prop_perfmap.csv", usecols=["Propeller", "Diameter", "Pitch"]
    )
    prop_perf_df = prop_perf_df.drop_duplicates()
    prop_perf_df = prop_perf_df[
        (prop_perf_df["Diameter"] >= 16)
        & (prop_perf_df["Diameter"] <= 22)
        & (prop_perf_df["Pitch"] >= 6)
        & (prop_perf_df["Pitch"] <= 12)
    ]

    props = []
    for _, row in prop_perf_df.iterrows():
        d, p, name = row["Diameter"], row["Pitch"], row["Propeller"]
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
    output.append("\033[H\033[J")
    output.append(
        f"Propulsion Sweep (7 Threads): |{bar}| {progress_pct:6.2f}% ({count}/{total_combinations})"
    )

    if results:
        last = results[-1]
        output.append(
            f"Last Finished: {last['Motor']} + {last['Prop']} -> EE: {last['EE']:.3f} [{last['Status']}]"
        )
    else:
        output.append("Last Finished: ---")

    output.append(f"Just simulated: {last_combo_str} ...")

    if results:
        df_temp = pd.DataFrame(results)

        if count % 100 == 0:
            partial_path = os.path.join(results_dir, "sweep_results_partial.csv")
            try:
                df_temp.sort_values("EE", ascending=False).to_csv(
                    partial_path, index=False
                )
            except PermissionError:
                pass

        top_5 = df_temp.nlargest(5, "EE")
        output.append("\n--- TOP 5 COMBINATIONS SO FAR ---")
        output.append(
            f"{' # ':<3} | {'Motor':<35} | {'Kv':<8} | {'Prop':<12} | {'EE':<10} | {'Status':<15}"
        )
        output.append("-" * 110)
        for i, (_, row) in enumerate(top_5.iterrows()):
            output.append(
                f"{i + 1:^3} | {row['Motor'][:30]:<35} | {row['kv']:<8} | {row['Prop']:<12} | {row['EE']:<10.3f} | {row['Status']:<15}"
            )
        output.append("-" * 110)

    print("\n".join(output), end="")


def build_parser():
    p = argparse.ArgumentParser(
        description="Sweep motor+propeller combinations for optimal takeoff."
    )
    p.add_argument("--S_wing", type=float, help="Wing area (m^2)")
    p.add_argument("--CL_max", type=float, help="Max lift coefficient")
    p.add_argument("--CL_ground", type=float, help="Ground roll CL")
    p.add_argument("--CD_ground", type=float, help="Ground roll CD")
    p.add_argument("--CD_max", type=float, help="CD at CL_max")
    p.add_argument("--mu", type=float, help="Ground friction coefficient")
    p.add_argument("--P_limit", type=float, help="Power limit (W)")
    p.add_argument("--V_limit", type=float, help="Voltage limit (V)")
    p.add_argument("--PV", type=float, help="Empty weight without propulsion (kg)")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Build airplane parameters: hardcoded defaults + CLI overrides
    plane_keys = [
        "S_wing",
        "CL_max",
        "CL_ground",
        "CD_ground",
        "CD_max",
        "mu",
        "P_limit",
        "V_limit",
        "PV",
    ]
    overrides = {
        k: getattr(args, k) for k in plane_keys if getattr(args, k) is not None
    }
    params = AircraftParameters(**overrides)

    surrogate_model = load_surrogate()
    motor_df = get_motors()
    props = get_props()

    results_dir = "./.results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results = []

    combinations = []
    for _, motor in motor_df.iterrows():
        motor_dict = motor.to_dict()
        for prop in props:
            combinations.append((motor_dict, prop))

    total_combinations = len(combinations)
    count = 0

    print("Starting parallel sweep (7 threads)...")

    with ThreadPoolExecutor(max_workers=7) as executor:
        future_to_combo = {
            executor.submit(
                simulate_combination, m_dict, p_dict, params, surrogate_model
            ): (m_dict, p_dict)
            for m_dict, p_dict in combinations
        }

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

            update_ui(count, total_combinations, results, combo_str, results_dir)

    df = pd.DataFrame(results)
    if not df.empty:
        output_path = os.path.join(results_dir, "sweep_results.csv")
        df.sort_values("EE", ascending=False).to_csv(output_path, index=False)

        best = df.loc[df["EE"].idxmax()]
        print("\n" + "=" * 40)
        print("BEST COMBINATION FOUND:")
        print(f"Motor: {best['Motor']}")
        print(f"Propeller: {best['Prop']}")
        print(f"MTOW: {best['MTOW']:.3f} kg ({best['Status']})")
        print(f"Motor Weight: {best['Motor_Wt']:.3f} kg")
        print(f"Prop Weight: {best['Prop_Wt']:.3f} kg")
        print(f"Propulsion Weight: {best['Propulsion_Wt']:.3f} kg")
        print(f"EE: {best['EE']:.3f}")
        print(f"Results saved to: {output_path}")
        print("=" * 40)


if __name__ == "__main__":
    main()
