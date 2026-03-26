import numpy as np
import matplotlib.pyplot as plt
from bldcm.bldcm import BLDCMSolver
from surrogate.prs import PRSSurrogate
from propeller_surrogate import MODEL_PATH

def generate_thrust_curve(p_limit=600.0, v_max=30.0, points=31):
    # Load model and initialize solver
    surrogate_model = PRSSurrogate.load(MODEL_PATH)
    bldcm_solver = BLDCMSolver(
        surrogate_model=surrogate_model,
        kv=310,
        i0=1.66,
        rm=0.065,
        diameter=18 * 0.0254,
        pitch=8,
    )

    v_inf_range = np.linspace(0, v_max, points)
    thrust_values = []

    print(f"Generating thrust curve data (P_limit={p_limit}W)...")
    for v in v_inf_range:
        try:
            t = bldcm_solver.solve_thrust(target=p_limit, v_inf=v)
            thrust_values.append(t)
            print(f"v={v:5.1f} m/s | T={t:7.3f} N")
        except Exception as e:
            print(f"v={v:5.1f} m/s | Error: {e}")
            thrust_values.append(0.0)

    return v_inf_range, np.array(thrust_values)

if __name__ == "__main__":
    v, t = generate_thrust_curve(v_max=20.0, points=21)
    
    # Simple polynomial fit (3rd degree)
    z = np.polyfit(v, t, 3)
    p = np.poly1d(z)
    
    print("\nPolynomial coefficients (3rd degree) for 0-20 m/s:")
    print(z)
    
    # Plotting code
    plt.figure(figsize=(10, 6))
    plt.scatter(v, t, label='Original (BLDCMSolver)', color='red')
    v_fine = np.linspace(0, 20, 200)
    plt.plot(v_fine, p(v_fine), label='3-degree Polynomial Fit', linestyle='--')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Thrust (N)')
    plt.title('Thrust vs Velocity Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('thrust_curve.png')
    print("\nCurve plot saved to 'thrust_curve.png'")
    
    # Show error
    fit_errors = t - p(v)
    max_error = np.max(np.abs(fit_errors))
    print(f"Max fitting error: {max_error:.4f} N")
