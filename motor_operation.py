import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from bldcm.bldcm import BLDCMSolver
from propeller_surrogate import MODEL_PATH
from surrogate.prs import PRSSurrogate

BATT_VOLTAGE = 22.2


def sweep_forward_speed(
    solver, p_in_target: float, max_voltage: float, v_inf_range: np.ndarray
) -> pd.DataFrame:
    results = []

    for v_inf in v_inf_range:
        try:
            state = solver.solve_thrust(
                v_inf, p_in_target, max_voltage, return_state=True
            )
            state["V_inf"] = v_inf
            results.append(state)
        except RuntimeError as e:
            print(f"Solver failed at V_inf = {v_inf:.2f} m/s: {e}")

    return pd.DataFrame(results)


def plot_bldc_performance(df: pd.DataFrame, p_in_target: float):
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
        BATT_VOLTAGE, color="red", linestyle="--", alpha=0.7, label="6S Limit"
    )
    ax_volt.set_ylabel("Voltage (V)")
    ax_volt.set_title("Electrical Telemetry")
    ax_volt.set_ylim(0, df["Voltage_V"].max() * 1.1)

    ax_curr = ax_volt.twinx()
    sns.lineplot(
        data=df,
        x="V_inf",
        y="Current_A",
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
    plt.savefig("motor_performance_enhanced.png")
    print("Plot saved to motor_performance_enhanced.png")
    plt.show()


def main():
    # Example Initialization (Requires the trained PRS surrogate and BLDCEquilibriumSolver)
    surrogate_model = PRSSurrogate.load(MODEL_PATH)

    # Dummy parameters for demonstration
    solver = BLDCMSolver(
        surrogate_model=surrogate_model,
        kv=320.0 * 1.05,
        i0=1.0 * (1 + 0.01 * (BATT_VOLTAGE - 18.0)),
        rm=(0.048 * 0.95) * (1.035**3),
        diameter=22 * 0.0254,
        pitch=10,
    )

    target_power = 600.0  # Watts
    speeds = np.linspace(0, 20, 20)  # 0 to 30 m/s

    df_results = sweep_forward_speed(solver, target_power, BATT_VOLTAGE, speeds)
    print(df_results.head())
    plot_bldc_performance(df_results, target_power)


if __name__ == "__main__":
    main()
