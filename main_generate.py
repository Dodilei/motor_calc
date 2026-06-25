import numpy as np

from bldcm.bldcm import BLDCMSolver
from propeller_surrogate import MODEL_PATH
from surrogate.prs import PRSSurrogate


def sweep_forward_speed(
    solver, p_in_target: float, v_inf_range: np.ndarray
) -> np.ndarray:
    results = []

    for v_inf in v_inf_range:
        try:
            thrust = solver.solve_(target=p_in_target, v_inf=v_inf)
            results.append(thrust)
        except RuntimeError as e:
            print(f"Solver failed at V_inf = {v_inf:.2f} m/s: {e}")

    return np.array(results)


target_power = 600.0  # Watts
speeds = np.array([0.0, 10.0, 16.0])  # 0 to 30 m/s


def solve_motor(kv, i0, rm, diam, p):
    # Example Initialization (Requires the trained PRS surrogate and BLDCEquilibriumSolver)
    surrogate_model = PRSSurrogate.load(MODEL_PATH)

    # Dummy parameters for demonstration
    solver = BLDCMSolver(
        surrogate_model=surrogate_model,
        kv=kv,
        i0=i0,
        rm=rm,
        diameter=diam * 0.0254,
        pitch=p,
    )

    thrust_array = sweep_forward_speed(solver, target_power, speeds)


if __name__ == "__main__":
    main()
