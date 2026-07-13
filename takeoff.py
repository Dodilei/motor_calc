import numpy as np
from scipy.optimize import bisect

from bldcm.bldcm import BLDCMSolver

# Constants for motor temperature dynamics
K_COOL = 0.01  # Heat dissipation factor (Watts / gram*Celsius)


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
        self._current_poly = None

        if self.use_fast_thrust:
            self._fit_propulsion_polynomials()

    def _fit_propulsion_polynomials(self, v_range=(0, 25), pts=12):
        v_samples = np.linspace(v_range[0], v_range[1], pts)
        t_samples = []
        i_samples = []
        for v in v_samples:
            try:
                res = self.solver.solve_thrust(
                    v,
                    max_power=self.p.P_limit,
                    max_throttle=self.p.max_throttle,
                    return_state=True,
                )
                t_samples.append(res["Thrust_N"])
                i_samples.append(res["Motor_Current_A"])
            except Exception:
                t_samples.append(0.0)
                i_samples.append(0.0)

        t_coeffs = np.polyfit(v_samples, t_samples, 3)
        self._thrust_poly = np.poly1d(t_coeffs)

        i_coeffs = np.polyfit(v_samples, i_samples, 3)
        self._current_poly = np.poly1d(i_coeffs)

    def _get_thrust(self, v_inf):
        if self.use_fast_thrust and self._thrust_poly is not None:
            v = np.clip(v_inf, 0.0, 30.0)
            return max(0.0, self._thrust_poly(v))

        try:
            return self.solver.solve_thrust(
                v_inf=v_inf, max_power=self.p.P_limit, max_throttle=self.p.max_throttle
            )
        except Exception:
            return 0.0

    def _get_current(self, v_inf):
        if self.use_fast_thrust and self._current_poly is not None:
            v = np.clip(v_inf, 0.0, 30.0)
            return max(0.0, self._current_poly(v))

        try:
            res = self.solver.solve_thrust(
                v_inf=v_inf,
                max_power=self.p.P_limit,
                max_throttle=self.p.max_throttle,
                return_state=True,
            )
            return res["Motor_Current_A"]
        except Exception:
            return 0.0

    def _get_stall_speed(self, mass, cl):
        return np.sqrt((2 * mass * self.p.g) / (self.p.rho * self.p.S_wing * cl))

    def simulate(self, mass, h_obs=0.9, dt=0.01, max_steps=10000, track_history=False):
        v_stall = self._get_stall_speed(mass, self.p.CL_max)
        v_rotate = 1.1 * v_stall

        state = np.array([0.0, 0.0, 0.001, 0.0, 0.0])

        history = [] if track_history else None

        def derivatives(s, m):
            x, y, vx, vy, Qnorm = s
            v_sq = vx**2 + vy**2
            v_mag = np.sqrt(v_sq)
            q = 0.5 * self.p.rho * v_sq * self.p.S_wing

            T = self._get_thrust(v_mag)
            I_motor = self._get_current(v_mag)

            dQnorm = (I_motor**2 * self.solver.rm) - K_COOL * (Qnorm)

            W = m * self.p.g

            if v_mag < v_rotate and y <= 0.001:
                L = q * self.p.CL_ground
                D = q * self.p.CD_ground
                N = max(0, W - L)
                D_total = D + self.p.mu * N
                ax = (T - D_total) / m
                return np.array([vx, 0.0, ax, 0.0, dQnorm])
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
                return np.array([vx, vy, ax, ay, dQnorm])

        step = 0
        while state[1] < h_obs:
            if track_history:
                v_mag = np.sqrt(state[2] ** 2 + state[3] ** 2)
                try:
                    metrics = self.solver.solve_thrust(
                        v_mag,
                        max_power=self.p.P_limit,
                        max_throttle=self.p.max_throttle,
                        return_state=True,
                    )
                except Exception:
                    metrics = {}

                history.append(
                    {
                        "x": state[0],
                        "y": state[1],
                        "vx": state[2],
                        "vy": state[3],
                        "v_mag": v_mag,
                        "Qnorm": state[4],
                        **metrics,
                    }
                )

            if step >= max_steps:
                if track_history:
                    return 999 if state[1] < h_obs else state[0], history
                else:
                    return 999 if state[1] < h_obs else state[0]
            if state[0] > 150:
                if track_history:
                    return 150.1, history
                else:
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

        if track_history:
            return state[0], history
        else:
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

