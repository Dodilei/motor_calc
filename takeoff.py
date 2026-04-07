import numpy as np
from scipy.optimize import bisect

from bldcm.bldcm import BLDCMSolver
from surrogate.prs import PRSSurrogate
from propeller_surrogate import MODEL_PATH


class TakeoffSolver:
    def __init__(
        self,
        solver: BLDCMSolver,
        params,
        use_fast_thrust: bool = True,
    ):
        self.solver = solver
        self.p = params
        self.use_fast_thrust = use_fast_thrust
        self._thrust_poly = None

        if self.use_fast_thrust:
            self._fit_thrust_polynomial()

    def _fit_thrust_polynomial(self, v_range=(0, 25), pts=12):
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

        coeffs = np.polyfit(v_samples, t_samples, 3)
        self._thrust_poly = np.poly1d(coeffs)

    def _get_thrust(self, v_inf):
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
        return np.sqrt((2 * mass * self.p.g) / (self.p.rho * self.p.S_wing * cl))

    def simulate(self, mass, h_obs=0.9, dt=0.01, max_steps=10000):
        v_stall = self._get_stall_speed(mass, self.p.CL_max)
        v_rotate = 1.1 * v_stall

        state = np.array([0.0, 0.0, 0.001, 0.0])

        def derivatives(s, m):
            x, y, vx, vy = s
            v_sq = vx**2 + vy**2
            v_mag = np.sqrt(v_sq)
            q = 0.5 * self.p.rho * v_sq * self.p.S_wing

            T = self._get_thrust(v_mag)
            W = m * self.p.g

            if v_mag < v_rotate and y <= 0.001:
                L = q * self.p.CL_ground
                D = q * self.p.CD_ground
                N = max(0, W - L)
                D_total = D + self.p.mu * N
                ax = (T - D_total) / m
                return np.array([vx, 0.0, ax, 0.0])
            else:
                gamma = np.arctan2(vy, vx) if v_mag > 0.01 else 0.0
                L = q * self.p.CL_max
                D = q * self.p.CD_max
                F_tangent = T - D - W * np.sin(gamma)
                F_normal = L - W * np.cos(gamma)
                a_tangent = F_tangent / m
                a_normal = F_normal / m
                ax = a_tangent * np.cos(gamma) - a_normal * np.sin(gamma)
                ay = a_tangent * np.sin(gamma) + a_normal * np.cos(gamma)
                return np.array([vx, vy, ax, ay])

        step = 0
        while state[1] < h_obs:
            if step >= max_steps:
                return 999 if state[1] < h_obs else state[0]
            if state[0] > 150:
                return 150.1

            k1 = derivatives(state, mass)
            k2 = derivatives(state + 0.5 * dt * k1, mass)
            k3 = derivatives(state + 0.5 * dt * k2, mass)
            k4 = derivatives(state + dt * k3, mass)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            if state[1] < 0:
                state[1] = 0
                state[3] = max(0, state[3])
            if state[2] < 0:
                state[2] = 0

            step += 1

        return state[0]


def find_tow(
    solver,
    params,
    target_dist=55.0,
    bounds=(4.0, 28.0),
    use_fast_thrust=True,
    dt=0.01,
    max_steps=10000,
):
    """Find max takeoff weight for a given distance constraint using bisection.

    Returns (mtow, t_static, status) where status is 'converged', 'saturated_low', or 'saturated_high'.
    """
    sim = TakeoffSolver(solver, params, use_fast_thrust=use_fast_thrust)
    t_static = sim._get_thrust(0.0)

    def f(P):
        dist = sim.simulate(P, dt=dt, max_steps=max_steps)
        return dist - target_dist

    try:
        f_low = f(bounds[0])
        if f_low > 0:
            return bounds[0], t_static, "saturated_low"

        f_high = f(bounds[1])
        if f_high < 0:
            return bounds[1], t_static, "saturated_high"

        opt_mass = bisect(f, bounds[0], bounds[1], xtol=0.01)
        return opt_mass, t_static, "converged"
    except Exception:
        return None, None, "error"


def apply_corrections(kv, io, rm, io_vref):
    """Apply grounding corrections to motor parameters.

    These corrections normalize the motor parameters to a common reference.
    The runtime voltage-dependent correction is applied in bldcm.py.
    """
    corr_kv = kv * 1.05
    corr_io = io / (1 + 0.01 * io_vref)
    corr_rm = (rm * 0.95) * (1.035**3)
    return corr_kv, corr_io, corr_rm


def load_surrogate():
    """Load the propeller surrogate model."""
    return PRSSurrogate.load(MODEL_PATH)


def lookup_motor(motor_name, kv, databases=None):
    """Look up motor parameters from CSV databases by name.

    Returns a dict with keys: name, kv, io, rm, io_vref, weight.
    Raises ValueError if motor not found.
    """
    import pandas as pd

    if databases is None:
        databases = ["./.data/tmotor_data.csv", "./.data/mad_motor_data.csv"]

    for db in databases:
        df = pd.read_csv(db, on_bad_lines="skip")
        match = df[(df["name"] == motor_name) & (df["kv"] == kv)]
        if not match.empty:
            row = match.iloc[0]
            return {
                "name": row["name"],
                "kv": row["kv"],
                "io": row["io"],
                "rm": row["rm"],
                "io_vref": row["io_vref"],
                "weight": row["weight"],
            }

    available_names = []
    for db in databases:
        df = pd.read_csv(db, on_bad_lines="skip")
        available_names.extend(df["name"].tolist())

    close = [n for n in available_names if motor_name.lower() in n.lower()]
    hint = f" Similar: {close[:5]}" if close else ""
    raise ValueError(f"Motor '{motor_name}' not found in databases.{hint}")
