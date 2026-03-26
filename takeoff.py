import numpy as np
from scipy.optimize import bisect
from bldcm.bldcm import BLDCMSolver
from surrogate.prs import PRSSurrogate
from propeller_surrogate import MODEL_PATH


class AircraftParameters:
    def __init__(self):
        self.g = 9.81
        self.rho = 1.225
        self.S_wing = 0.53  # m^2 (Typical for 1.8m span)
        self.CL_max = 1.4  # Max CL for air phase
        self.CL_ground = 0.4  # CL during ground roll
        self.CD_ground = 0.04  # Parasite drag + ground effect induction
        self.CD_max = 0.20  # CD at CL_max
        self.mu = 0.03  # Typical friction for grass/runway
        self.P_limit = 600.0  # Power limit in Watts


class TakeoffSolver:
    def __init__(self, solver: BLDCMSolver, params: AircraftParameters):
        self.solver = solver
        self.p = params

    def _get_thrust(self, v_inf):
        """Wrapper to get thrust from BLDCM solver at current speed."""
        try:
            return self.solver.solve_thrust(target=self.p.P_limit, v_inf=v_inf)
        except Exception:
            return 0.0

    def _get_stall_speed(self, mass, cl):
        """Calculates stall speed for given mass and lift coefficient."""
        return np.sqrt((2 * mass * self.p.g) / (self.p.rho * self.p.S_wing * cl))

    def simulate(self, mass, h_obs=0.9, dt=0.01):
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
        max_steps = 10000
        while state[1] < h_obs and step < max_steps:
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


def find_tow_for_distance(target_dist=55.0):
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

    params = AircraftParameters()
    sim = TakeoffSolver(bldcm_solver, params)

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
    print("Starting TOW optimization...")
    optimal_tow = find_tow_for_distance(55.0)
    if optimal_tow:
        print(f"\nOptimal Takeoff Weight (TOW) for 55m runway: {optimal_tow:.3f} kg")
    else:
        print("\nCould not find a valid TOW for the given constraints.")
