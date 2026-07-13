import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from bldcm.bldcm import BLDCMSolver
from motor_db import load_surrogate, apply_corrections
from aircraft_params import AircraftParameters


def sweep_throttle(
    solver, max_power: float | None, throttle_range: np.ndarray
) -> pd.DataFrame:
    results = []
    v_inf = 0  # Static test

    for throttle in throttle_range:
        if throttle <= 0:
            continue
        try:
            # Set max_power to None if we purely want to sweep throttle without power cap
            state = solver.solve_thrust(v_inf, max_power, throttle, return_state=True)
            results.append(state)
        except RuntimeError as e:
            print(f"Solver failed at Throttle = {throttle:.2f}: {e}")

    return pd.DataFrame(results)


def plot_bldc_static_performance(df: pd.DataFrame, V_batt: float):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        "BLDC-Propeller Static Test Performance (varying throttle)",
        fontsize=16,
    )

    x_axis = "Throttle_t"

    # 1. Thrust vs Throttle
    sns.lineplot(data=df, x=x_axis, y="Thrust_N", ax=axes[0, 0], color="b", linewidth=2)
    axes[0, 0].set_ylabel("Thrust (N)")
    axes[0, 0].set_xlabel("Target Throttle")
    axes[0, 0].set_title("Thrust Generation")

    # 2. RPM vs Throttle
    sns.lineplot(data=df, x=x_axis, y="RPM", ax=axes[0, 1], color="r", linewidth=2)
    axes[0, 1].set_ylabel("RPM")
    axes[0, 1].set_xlabel("Target Throttle")
    axes[0, 1].set_title("Equilibrium RPM")

    # 3. Efficiency & Power vs Throttle
    ax_eff = axes[1, 0]
    sns.lineplot(
        data=df,
        x=x_axis,
        y="Efficiency",
        ax=ax_eff,
        color="g",
        linewidth=2,
        label="Efficiency",
    )
    ax_eff.set_ylabel("Efficiency")
    ax_eff.set_xlabel("Target Throttle")
    ax_eff.set_title("Efficiency & Power Distribution")

    ax_p_el = ax_eff.twinx()
    sns.lineplot(
        data=df,
        x=x_axis,
        y="P_el",
        ax=ax_p_el,
        color="teal",
        alpha=0.7,
        linewidth=2,
        label="P_el (W)",
    )
    ax_p_el.set_ylabel("Electrical Power (W)")
    ax_p_el.set_ylim(0, df["P_el"].max() * 1.1)
    ax_p_el.grid(False)  # Avoid cluttered grid lines

    # Merge legends for Plot 3
    lines_e, labels_e = ax_eff.get_legend_handles_labels()
    lines_p, labels_p = ax_p_el.get_legend_handles_labels()
    ax_eff.legend(lines_e + lines_p, labels_e + labels_p, loc="upper left")
    ax_p_el.get_legend().remove()

    # 4. Power & Efficiency vs RPM
    ax_rpm_eff = axes[1, 1]
    sns.lineplot(
        data=df,
        x="RPM",
        y="Efficiency",
        ax=ax_rpm_eff,
        color="g",
        linewidth=2,
        label="Efficiency",
    )
    ax_rpm_eff.set_ylabel("Efficiency")
    ax_rpm_eff.set_xlabel("RPM")
    ax_rpm_eff.set_title("Power & Efficiency vs RPM")

    ax_rpm_p = ax_rpm_eff.twinx()
    sns.lineplot(
        data=df,
        x="RPM",
        y="P_el",
        ax=ax_rpm_p,
        color="teal",
        alpha=0.7,
        linewidth=2,
        label="P_el (W)",
    )
    ax_rpm_p.set_ylabel("Electrical Power (W)")
    ax_rpm_p.grid(False)

    # Merge legends for Plot 4
    lines_e, labels_e = ax_rpm_eff.get_legend_handles_labels()
    lines_p, labels_p = ax_rpm_p.get_legend_handles_labels()
    ax_rpm_eff.legend(lines_e + lines_p, labels_e + labels_p, loc="upper left")
    ax_rpm_p.get_legend().remove()

    # 5. Electrical Telemetry (Actual Voltage & Current)
    ax_volt = axes[2, 0]
    sns.lineplot(
        data=df,
        x=x_axis,
        y="Voltage_V",
        ax=ax_volt,
        color="darkorange",
        label="Actual Voltage (V)",
    )
    ax_volt.axhline(
        V_batt, color="red", linestyle="--", alpha=0.7, label="6S Limit"
    )
    ax_volt.set_ylabel("Voltage (V)")
    ax_volt.set_xlabel("Target Throttle")
    ax_volt.set_title("Electrical Telemetry")
    ax_volt.set_ylim(0, max(V_batt, df["Voltage_V"].max()) * 1.1)

    ax_curr = ax_volt.twinx()
    sns.lineplot(
        data=df,
        x=x_axis,
        y="Batt_Current_A",
        ax=ax_curr,
        color="crimson",
        label="Battery Current (A)",
    )
    ax_curr.set_ylabel("Current (A)")
    ax_curr.grid(False)  # Avoid cluttered grid lines

    # Merge legends for Plot 5
    lines_v, labels_v = ax_volt.get_legend_handles_labels()
    lines_c, labels_c = ax_curr.get_legend_handles_labels()
    ax_curr.legend(lines_v + lines_c, labels_v + labels_c, loc="upper left")
    ax_volt.get_legend().remove()

    # 6. Thrust vs RPM
    ax_thrust_rpm = axes[2, 1]
    sns.lineplot(
        data=df, x="RPM", y="Thrust_N", ax=ax_thrust_rpm, color="b", linewidth=2
    )
    ax_thrust_rpm.set_xlabel("RPM")
    ax_thrust_rpm.set_ylabel("Thrust (N)")
    ax_thrust_rpm.set_title("Thrust vs RPM")

    plt.tight_layout()
    plt.savefig(".plots/motor_static_performance.png")
    print("Plot saved to .plots/motor_static_performance.png")
    plt.show()


def main():
    import argparse
    params = AircraftParameters()
    chosen_sys = AircraftParameters.load_chosen_system() or {}

    parser = argparse.ArgumentParser(description="BLDC Motor Static Analysis")
    parser.add_argument("--kv", type=float, default=chosen_sys.get("kv", 336.0), help="Motor KV constant")
    parser.add_argument("--i0", type=float, default=chosen_sys.get("io", 1.142), help="Motor no-load current (A)")
    parser.add_argument("--rm", type=float, default=chosen_sys.get("rm", 0.039), help="Motor resistance (ohm)")
    parser.add_argument("--diam", type=float, default=chosen_sys.get("diam", 21.0), help="Propeller diameter (inches)")
    parser.add_argument("--pitch", type=float, default=chosen_sys.get("pitch", 6.3), help="Propeller pitch (inches)")
    parser.add_argument("--io-vref", type=float, default=chosen_sys.get("io_vref", 0.0), help="Reference voltage for I0 (0 means no correction)")
    parser.add_argument("--no-correction", action="store_true", help="Do not apply parameter corrections")

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

    max_power = None  # No power cap for static sweep unless desired
    throttles = np.linspace(0.1, 1.0, 24)  # 10% to 100% throttle

    df_results = sweep_throttle(solver, max_power, throttles)
    print(df_results.head())
    plot_bldc_static_performance(df_results, params.V_batt)


if __name__ == "__main__":
    main()
