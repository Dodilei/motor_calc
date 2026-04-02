import cProfile
import pstats
import io
import numpy as np
from scipy.optimize import bisect
from bldcm.bldcm import BLDCMSolver
from surrogate.prs import PRSSurrogate
from propeller_surrogate import MODEL_PATH


class AircraftParameters:
    def __init__(self):
        self.g = 9.81
        self.rho = 1.225
        self.S_wing = 0.774  # m^2 (Typical for 2m span)
        self.CL_max = 1.969  # Max CL for air phase
        self.CL_ground = 0.997  # CL during ground roll
        self.CD_ground = 0.057  # Parasite drag + ground effect induction
        self.CD_max = 0.199  # CD at CL_max
        self.mu = 0.04  # Typical friction for grass/runway
        self.P_limit = 590.0  # Power limit in Watts
        self.V_limit = 23.0  # Voltage limit in Volts
        self.PV = 2.2  # Empty weight (kg) [without GMP]


class TakeoffSolver:
    def __init__(
        self,
        solver: BLDCMSolver,
        params: AircraftParameters,
        use_fast_thrust: bool = True,
    ):
        self.solver = solver
        self.p = params
        self.use_fast_thrust = use_fast_thrust
        self._thrust_poly = None

        if self.use_fast_thrust:
            self._fit_thrust_polynomial()

    def _fit_thrust_polynomial(self, v_range=(0, 25), pts=12):
        """Samples the solver to create a fast 3rd-degree polynomial fit."""
        v_samples = np.linspace(v_range[0], v_range[1], pts)
        t_samples = []
        for v in v_samples:
            try:
                t = self.solver.solve_thrust(
                    v, max_power=self.p.P_limit, max_voltage=self.p.V_limit
                )
                t_samples.append(t)
            except Exception:
                t_samples.append(0.0)

        # 3rd degree polynomial fit: T(v) = a*v^3 + b*v^2 + c*v + d
        coeffs = np.polyfit(v_samples, t_samples, 3)
        self._thrust_poly = np.poly1d(coeffs)

    def _get_thrust(self, v_inf):
        """Wrapper to get thrust. Uses fast polynomial fit if enabled."""
        if self.use_fast_thrust and self._thrust_poly is not None:
            v = np.clip(v_inf, 0.0, 30.0)
            return max(0.0, self._thrust_poly(v))

        try:
            return self.solver.solve_thrust(
                v_inf=v_inf, max_power=self.p.P_limit, max_voltage=self.p.V_limit
            )
        except Exception:
            return 0.0

    def _get_stall_speed(self, mass, cl):
        """Calculates stall speed for given mass and lift coefficient."""
        return np.sqrt((2 * mass * self.p.g) / (self.p.rho * self.p.S_wing * cl))

    def simulate(self, mass, h_obs=0.9, dt=0.01, max_steps=10000):
        """
        Simulates takeoff using RK4 integration from v=0 to h=h_obs.
        State: [x, y, vx, vy]
        """
        v_stall = self._get_stall_speed(mass, self.p.CL_max)
        v_rotate = 1.1 * v_stall  # Rotate at 10% above stall

        # Initial state: [x, y, vx, vy]
        state = np.array(
            [0.0, 0.0, 0.001, 0.0]
        )  # Start with tiny velocity to avoid div by zero if any

        def derivatives(s, m):
            x, y, vx, vy = s
            v_sq = vx**2 + vy**2
            v_mag = np.sqrt(v_sq)
            q = 0.5 * self.p.rho * v_sq * self.p.S_wing

            T = self._get_thrust(v_mag)
            W = m * self.p.g

            # Phase determination
            if v_mag < v_rotate and y <= 0.001:
                # GROUND ROLL
                L = q * self.p.CL_ground
                D = q * self.p.CD_ground
                N = max(0, W - L)
                D_total = D + self.p.mu * N

                ax = (T - D_total) / m

                ay = 0.0
                # If we are on ground, vy must be 0 or positive if we just reached v_rotate
                # But here we force 0 until rotate
                return np.array([vx, 0.0, ax, 0.0])
            else:
                # AIR BORN / CLIMB
                gamma = np.arctan2(vy, vx) if v_mag > 0.01 else 0.0

                L = q * self.p.CL_max
                D = q * self.p.CD_max

                # Forces in Velocity Frame
                F_tangent = T - D - W * np.sin(gamma)
                F_normal = L - W * np.cos(gamma)

                a_tangent = F_tangent / m
                a_normal = F_normal / m

                # Rotate to Cartesian
                ax = a_tangent * np.cos(gamma) - a_normal * np.sin(gamma)
                ay = a_tangent * np.sin(gamma) + a_normal * np.cos(gamma)

                return np.array([vx, vy, ax, ay])

        step = 0
        while state[1] < h_obs:
            if step >= max_steps:
                return 999 if state[1] < h_obs else state[0]
            if state[0] > 150:  # Fail safe for runway length
                return 150.1

            k1 = derivatives(state, mass)
            k2 = derivatives(state + 0.5 * dt * k1, mass)
            k3 = derivatives(state + 0.5 * dt * k2, mass)
            k4 = derivatives(state + dt * k3, mass)

            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Ground constraint
            if state[1] < 0:
                state[1] = 0
                state[3] = max(0, state[3])

            # Prevent going backwards
            if state[2] < 0:
                state[2] = 0

            step += 1

        return state[0]


def find_tow_for_distance(target_dist=55.0, use_fast_thrust=True):
    # Load model and initialize solver
    surrogate_model = PRSSurrogate.load(MODEL_PATH)
    bldcm_solver = BLDCMSolver(
        surrogate_model=surrogate_model,
        kv=330,
        i0=1.66,
        rm=0.065,
        diameter=18 * 0.0254,
        pitch=8,
    )

    params = AircraftParameters()
    sim = TakeoffSolver(bldcm_solver, params, use_fast_thrust=use_fast_thrust)

    def f(P):
        dist = sim.simulate(P)
        res = dist - target_dist
        print(
            f"Testing TOW: {P:5.2f} kg | Distance: {dist:7.2f} m | Residual: {res:7.2f}"
        )
        return res

    # Root finding using bisect
    # We need a range where f(P_low) < 0 and f(P_high) > 0
    # Higher mass = longer distance. So f(P_low) should be negative and f(P_high) positive.
    try:
        # Check signs at bounds first
        f_low = f(5.0)
        f_high = f(25.0)  # Increased upper bound to be safe
        print(f"Bounds check: f(5.0)={f_low:.2f}, f(25.0)={f_high:.2f}")

        if f_low > 0:
            print("Target distance is already exceeded at minimum mass (5kg).")
            return 5.0
        if f_high < 0:
            print("Target distance not reached even at maximum mass (25kg).")
            return 25.0

        opt_mass = bisect(f, 5.0, 25.0, xtol=0.01)
        return opt_mass
    except ValueError as e:
        print(f"Bisection failed: {e}")
        return None


if __name__ == "__main__":
    import sys

    use_fast = "--slow" not in sys.argv
    if "--profile" in sys.argv:
        print(f"Starting TOW optimization with profiling (Fast Thrust: {use_fast})...")
        pr = cProfile.Profile()
        pr.enable()
        optimal_tow = find_tow_for_distance(55.0, use_fast_thrust=use_fast)
        pr.disable()

        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(20)  # Show top 20 functions
        print(s.getvalue())
    else:
        print(f"Starting TOW optimization (Fast Thrust: {use_fast})...")
        print(
            "Tip: Run with --profile to see performance analysis, or --slow to use iterative solver."
        )
        optimal_tow = find_tow_for_distance(55.0, use_fast_thrust=use_fast)

    if optimal_tow:
        print(f"\nOptimal Takeoff Weight (TOW) for 55m runway: {optimal_tow:.3f} kg")
    else:
        print("\nCould not find a valid TOW for the given constraints.")
