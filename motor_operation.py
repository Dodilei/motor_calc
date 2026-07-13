import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from bldcm.bldcm import BLDCMSolver
from motor_db import load_surrogate, apply_corrections
from aircraft_params import AircraftParameters


def sweep_forward_speed(
    solver, p_in_target: float, max_throttle: float, v_inf_range: np.ndarray
) -> pd.DataFrame:
    results = []

    for v_inf in v_inf_range:
        try:
            state = solver.solve_thrust(
                v_inf, p_in_target, max_throttle, return_state=True
            )
            state["V_inf"] = v_inf
            results.append(state)
        except RuntimeError as e:
            print(f"Solver failed at V_inf = {v_inf:.2f} m/s: {e}")

    return pd.DataFrame(results)


def plot_bldc_performance(df: pd.DataFrame, p_in_target: float, V_batt: float):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        f"BLDC-Propeller Equilibrium Performance ($P_{{in, target}}$ = {p_in_target} W)",
        fontsize=16,
    )

    # 1. Thrust vs Forward Speed
    sns.lineplot(
        data=df, x="V_inf", y="Thrust_N", ax=axes[0, 0], color="b", linewidth=2
    )
    axes[0, 0].set_ylabel("Thrust (N)")
    axes[0, 0].set_title("Thrust Degradation")

    # 2. RPM vs Forward Speed
    sns.lineplot(data=df, x="V_inf", y="RPM", ax=axes[0, 1], color="r", linewidth=2)
    axes[0, 1].set_ylabel("RPM")
    axes[0, 1].set_title("Equilibrium RPM (Unloading Effect)")

    # 3. Efficiency & Power vs Forward Speed
    ax_eff = axes[1, 0]
    sns.lineplot(
        data=df,
        x="V_inf",
        y="Efficiency",
        ax=ax_eff,
        color="g",
        linewidth=2,
        label="Efficiency",
    )
    ax_eff.set_ylabel("Efficiency")
    ax_eff.set_title("Efficiency & Power Distribution")

    ax_p_el = ax_eff.twinx()
    sns.lineplot(
        data=df,
        x="V_inf",
        y="P_el",
        ax=ax_p_el,
        color="teal",
        alpha=0.7,
        linewidth=2,
        label="P_el (W)",
    )
    ax_p_el.set_ylabel("Electrical Power (W)")
    if p_in_target is not None:
        ax_p_el.set_ylim(0, max(p_in_target, df["P_el"].max()) * 1.1)
    ax_p_el.grid(False)  # Avoid cluttered grid lines

    # Merge legends for Plot 3
    lines_e, labels_e = ax_eff.get_legend_handles_labels()
    lines_p, labels_p = ax_p_el.get_legend_handles_labels()
    ax_eff.legend(lines_e + lines_p, labels_e + labels_p, loc="lower left")
    ax_p_el.get_legend().remove()

    # 4. Advance Ratio vs Forward Speed
    sns.lineplot(
        data=df,
        x="V_inf",
        y="Advance_Ratio_J",
        ax=axes[1, 1],
        color="purple",
        linewidth=2,
    )
    axes[1, 1].set_ylabel("Advance Ratio (J)")
    axes[1, 1].set_title("Operating Advance Ratio")

    # 5. Electrical Telemetry (Voltage & Current)
    ax_volt = axes[2, 0]
    sns.lineplot(
        data=df,
        x="V_inf",
        y="Voltage_V",
        ax=ax_volt,
        color="darkorange",
        label="Voltage (V)",
    )
    ax_volt.axhline(
        V_batt, color="red", linestyle="--", alpha=0.7, label="6S Limit"
    )
    ax_volt.set_ylabel("Voltage (V)")
    ax_volt.set_title("Electrical Telemetry")
    ax_volt.set_ylim(0, df["Voltage_V"].max() * 1.1)

    ax_curr = ax_volt.twinx()
    sns.lineplot(
        data=df,
        x="V_inf",
        y="Batt_Current_A",
        ax=ax_curr,
        color="crimson",
        label="Current (A)",
    )
    ax_curr.set_ylabel("Current (A)")
    ax_curr.grid(False)  # Avoid cluttered grid lines

    # Merge legends for Plot 5
    lines_v, labels_v = ax_volt.get_legend_handles_labels()
    lines_c, labels_c = ax_curr.get_legend_handles_labels()
    ax_curr.legend(lines_v + lines_c, labels_v + labels_c, loc="lower right")
    ax_volt.get_legend().remove()

    # 6. Aero Characterization (Cp vs J)
    ax_cp = axes[2, 1]
    sns.lineplot(
        data=df, x="Advance_Ratio_J", y="cp", ax=ax_cp, color="magenta", linewidth=2
    )
    ax_cp.set_xlabel("Advance Ratio (J)")
    ax_cp.set_ylabel("Power Coefficient (Cp)")
    ax_cp.set_title("Aerodynamic Characterization ($C_p$ vs $J$)")

    plt.tight_layout()
    plt.savefig(".plots/motor_performance_enhanced.png")
    print("Plot saved to .plots/motor_performance_enhanced.png")
    plt.show()


def main():
    import argparse
    params = AircraftParameters()

    chosen_sys = AircraftParameters.load_chosen_system() or {}

    parser = argparse.ArgumentParser(description="BLDC Motor Operation Analysis")
    parser.add_argument("--thrust-curve", action="store_true", help="Enable thrust curve mode")
    parser.add_argument("--kv", type=float, default=chosen_sys.get("kv", 336.0), help="Motor KV constant")
    parser.add_argument("--i0", type=float, default=chosen_sys.get("io", 0.833), help="Motor no-load current (A)")
    parser.add_argument("--rm", type=float, default=chosen_sys.get("rm", 0.0421), help="Motor resistance (ohm)")
    parser.add_argument("--diam", type=float, default=chosen_sys.get("diam", 22.0), help="Propeller diameter (inches)")
    parser.add_argument("--pitch", type=float, default=chosen_sys.get("pitch", 10.0), help="Propeller pitch (inches)")
    parser.add_argument("--io-vref", type=float, default=chosen_sys.get("io_vref", 0.0), help="Reference voltage for I0 (0 means no correction)")
    parser.add_argument("--no-correction", action="store_true", help="Do not apply parameter corrections")
    parser.add_argument("--power", type=float, default=params.P_limit, help="Target power limit (W)")
    parser.add_argument("--v-max", type=float, default=20.0, help="Maximum velocity (m/s)")
    parser.add_argument("--points", type=int, default=21, help="Number of sweep points")

    args = parser.parse_args()

    kv, i0, rm = args.kv, args.i0, args.rm
    if not args.no_correction and args.io_vref > 0:
        kv, i0, rm = apply_corrections(kv, i0, rm, args.io_vref)

    surrogate_model = load_surrogate()
    solver = BLDCMSolver(
        surrogate_model=surrogate_model,
        kv=kv,
        i0=i0,
        rm=rm,
        diameter=args.diam * 0.0254,
        pitch=args.pitch,
        rest_voltage=chosen_sys.get("V_batt", params.V_batt),
    )

    if args.thrust_curve:
        v_inf_range = np.linspace(0, args.v_max, args.points)
        thrust_values = []

        print(f"Generating thrust curve data (P_limit={args.power}W)...")
        for v in v_inf_range:
            try:
                t = solver.solve_thrust(v, max_power=args.power)
                thrust_values.append(t)
                print(f"v={v:5.1f} m/s | T={t:7.3f} N")
            except Exception as e:
                print(f"v={v:5.1f} m/s | Error: {e}")
                thrust_values.append(0.0)

        v = v_inf_range
        t = np.array(thrust_values)

        # Simple polynomial fit (3rd degree)
        z = np.polyfit(v, t, 3)
        p = np.poly1d(z)

        print("\nPolynomial coefficients (3rd degree) for 0-20 m/s:")
        print(z)

        # Plotting code
        plt.figure(figsize=(10, 6))
        plt.scatter(v, t, label='Original (BLDCMSolver)', color='red')
        v_fine = np.linspace(0, args.v_max, 200)
        plt.plot(v_fine, p(v_fine), label='3-degree Polynomial Fit', linestyle='--')
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Thrust (N)')
        plt.title('Thrust vs Velocity Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('.plots/thrust_curve.png')
        print("\nCurve plot saved to '.plots/thrust_curve.png'")

        # Show error
        fit_errors = t - p(v)
        max_error = np.max(np.abs(fit_errors))
        print(f"Max fitting error: {max_error:.4f} N")
    else:
        speeds = np.linspace(0, args.v_max, args.points)
        df_results = sweep_forward_speed(solver, args.power, params.max_throttle, speeds)
        print(df_results.head())
        plot_bldc_performance(df_results, args.power, params.V_batt)


if __name__ == "__main__":
    main()
