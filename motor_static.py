import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from bldcm.bldcm import BLDCMSolver
from propeller_surrogate import MODEL_PATH
from surrogate.prs import PRSSurrogate

BATT_VOLTAGE = 23.5


def sweep_voltage(
    solver, max_power: float | None, voltage_range: np.ndarray
) -> pd.DataFrame:
    results = []
    v_inf = 0.0  # Static test

    for voltage in voltage_range:
        if voltage <= 0:
            continue
        try:
            # Set max_power to None if we purely want to sweep voltage without power cap
            state = solver.solve_thrust(v_inf, max_power, voltage, return_state=True)
            state["Target_Voltage"] = voltage
            results.append(state)
        except RuntimeError as e:
            print(f"Solver failed at Voltage = {voltage:.2f} V: {e}")

    return pd.DataFrame(results)


def plot_bldc_static_performance(df: pd.DataFrame):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        "BLDC-Propeller Static Test Performance (varying voltage)",
        fontsize=16,
    )

    x_axis = "Target_Voltage"

    # 1. Thrust vs Voltage
    sns.lineplot(data=df, x=x_axis, y="Thrust_N", ax=axes[0, 0], color="b", linewidth=2)
    axes[0, 0].set_ylabel("Thrust (N)")
    axes[0, 0].set_xlabel("Target Voltage (V)")
    axes[0, 0].set_title("Thrust Generation")

    # 2. RPM vs Voltage
    sns.lineplot(data=df, x=x_axis, y="RPM", ax=axes[0, 1], color="r", linewidth=2)
    axes[0, 1].set_ylabel("RPM")
    axes[0, 1].set_xlabel("Target Voltage (V)")
    axes[0, 1].set_title("Equilibrium RPM")

    # 3. Efficiency & Power vs Voltage
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
    ax_eff.set_xlabel("Target Voltage (V)")
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
    ax_rpm_eff.legend(lines_e + labels_p, labels_e + labels_p, loc="upper left")
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
        BATT_VOLTAGE, color="red", linestyle="--", alpha=0.7, label="6S Limit"
    )
    ax_volt.set_ylabel("Voltage (V)")
    ax_volt.set_xlabel("Target Voltage (V)")
    ax_volt.set_title("Electrical Telemetry")
    ax_volt.set_ylim(0, max(BATT_VOLTAGE, df["Voltage_V"].max()) * 1.1)

    ax_curr = ax_volt.twinx()
    sns.lineplot(
        data=df,
        x=x_axis,
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
    plt.savefig("motor_static_performance.png")
    print("Plot saved to motor_static_performance.png")
    plt.show()


def main():
    # Example Initialization (Requires the trained PRS surrogate and BLDCEquilibriumSolver)
    surrogate_model = PRSSurrogate.load(MODEL_PATH)

    # Dummy parameters for demonstration
    solver = BLDCMSolver(
        surrogate_model=surrogate_model,
        kv=336,
        i0=0.833,
        rm=0.0421,
        diameter=22 * 0.0254,
        pitch=6.6,
    )

    max_power = None  # No power cap for static sweep unless desired
    voltages = np.linspace(1, BATT_VOLTAGE, 24)  # 1V to 24V

    df_results = sweep_voltage(solver, max_power, voltages)
    print(df_results.head())
    plot_bldc_static_performance(df_results)


if __name__ == "__main__":
    main()
