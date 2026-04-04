from scipy.optimize import brentq
import numpy as np


class BLDCMSolver:
    def __init__(
        self,
        surrogate_model,
        kv: float,
        i0: float,
        rm: float,
        diameter: float,
        pitch: float,
        rho: float = 1.225,
    ):
        self.surrogate = surrogate_model
        self.kv = kv
        self.i0 = i0
        self.rm = rm
        self.diameter = diameter
        self.pitch = pitch
        self.rho = rho

    def _get_aero_coefficients(self, n_rpm: float, v_inf: float) -> tuple:
        # Construct input for the PRS surrogate
        v_inf_clp = max(0.0, v_inf)
        X_in = np.array([[self.diameter * 39.37, self.pitch, n_rpm, v_inf_clp]])

        # Predict Cp and Ct using the surrogate model
        predictions = self.surrogate.predict(X_in)

        # Extract values
        ct = predictions[0][0]
        cp = predictions[0][1]

        return cp, ct

    def _residual(
        self,
        n_rpm: float,
        v_inf: float,
        target_power: float | None = None,
        target_voltage: float | None = None,
    ) -> float:
        cp, _ = self._get_aero_coefficients(n_rpm, v_inf)

        # Calculate Propeller Power (Aerodynamic Load)
        n_rps = n_rpm / 60.0
        p_prop = cp * self.rho * (n_rps**3) * (self.diameter**5)

        # Calculate Motor Current
        v_est = n_rpm / self.kv + self.rm * (self.i0 + p_prop * self.kv) / n_rpm
        i_motor = (self.i0 * (1 + 0.01 * v_est)) + (p_prop * self.kv) / n_rpm

        # Calculate Motor Voltage
        v_motor = (n_rpm / self.kv) + (i_motor * self.rm)

        p_in_calc = v_motor * i_motor

        if target_power is None:
            return v_motor - target_voltage
        elif target_voltage is None:
            return p_in_calc - target_power
        else:
            return max(v_motor - target_voltage, p_in_calc - target_power)

    def solve_thrust(
        self,
        v_inf: float,
        max_power: float | None = 600,
        max_voltage: float | None = 24.2,
        rpm_bounds: tuple = (100, 15000),
        return_state: bool = False,
    ):
        residualf_args = (v_inf, max_power, max_voltage)

        brentq_kwargs = {
            "f": self._residual,
            "a": rpm_bounds[0],
            "b": rpm_bounds[1],
            "xtol": 1e-3,
        }

        try:
            n_eq: float = brentq(
                **brentq_kwargs,
                args=residualf_args,
            )  # pyright: ignore[reportAssignmentType]

        except ValueError:
            raise RuntimeError(
                f"Could not find equilibrium for ({max_power},{max_voltage}) at {v_inf} within RPM bounds."
            )

        # Retrieve final state at equilibrium
        cp, ct = self._get_aero_coefficients(n_eq, v_inf)
        n_rps = n_eq / 60.0

        # Final Aerodynamic metrics
        thrust = ct * self.rho * (n_rps**2) * (self.diameter**4)

        if not return_state:
            return thrust
        else:
            p_prop = cp * self.rho * (n_rps**3) * (self.diameter**5)
            j_adv = v_inf / (n_rps * self.diameter) if v_inf > 0 else 0.0

            # Final Electrical metrics
            v_est = n_eq / self.kv + self.rm * (self.i0 + p_prop * self.kv) / n_eq
            i_eq = (self.i0 + (1 + 0.01 * v_est)) + (p_prop * self.kv) / n_eq
            v_eq = (n_eq / self.kv) + (i_eq * self.rm)
            efficiency = p_prop / (v_eq * i_eq)

            return {
                "RPM": n_eq,
                "Voltage_V": v_eq,
                "Current_A": i_eq,
                "Thrust_N": thrust,
                "Efficiency": efficiency,
                "Advance_Ratio_J": j_adv,
                "P_el": v_eq * i_eq,
                "P_prop_W": p_prop,
                "cp": cp,
            }
