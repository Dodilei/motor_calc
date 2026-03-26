import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from bldcm.bldcm import BLDCMSolver
from propeller_surrogate import MODEL_PATH
from surrogate.prs import PRSSurrogate


def sweep_forward_speed(
    solver, p_in_target: float, v_inf_range: np.ndarray
) -> pd.DataFrame:
    results = []

    for v_inf in v_inf_range:
        try:
            state = solver.solve_for_target_power(target=p_in_target, v_inf=v_inf)
            state["V_inf"] = v_inf
            results.append(state)
        except RuntimeError as e:
            print(f"Solver failed at V_inf = {v_inf:.2f} m/s: {e}")

    return pd.DataFrame(results)


def plot_bldc_performance(df: pd.DataFrame, p_in_target: float):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"BLDC-Propeller Equilibrium Performance ($P_{{in}}$ = {p_in_target} W)",
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

    # 3. Efficiency vs Forward Speed
    sns.lineplot(
        data=df, x="V_inf", y="Efficiency", ax=axes[1, 0], color="g", linewidth=2
    )
    axes[1, 0].set_xlabel("Forward Speed (m/s)")
    axes[1, 0].set_ylabel("Overall System Efficiency")
    axes[1, 0].set_title("Efficiency Curve")

    # 4. Advance Ratio vs Forward Speed
    sns.lineplot(
        data=df,
        x="V_inf",
        y="Advance_Ratio_J",
        ax=axes[1, 1],
        color="purple",
        linewidth=2,
    )
    axes[1, 1].set_xlabel("Forward Speed (m/s)")
    axes[1, 1].set_ylabel("Advance Ratio (J)")
    axes[1, 1].set_title("Operating Advance Ratio")

    plt.tight_layout()
    plt.show()


def main():
    # Example Initialization (Requires the trained PRS surrogate and BLDCEquilibriumSolver)
    surrogate_model = PRSSurrogate.load(MODEL_PATH)

    # Dummy parameters for demonstration
    solver = BLDCMSolver(
        surrogate_model=surrogate_model,
        kv=310.0,
        i0=1.66,
        rm=0.065,
        diameter=18 * 0.0254,
        pitch=8,
    )

    target_power = 600.0  # Watts
    speeds = np.linspace(0, 20, 20)  # 0 to 30 m/s

    df_results = sweep_forward_speed(solver, target_power, speeds)
    print(df_results.head())
    plot_bldc_performance(df_results, target_power)


if __name__ == "__main__":
    main()
