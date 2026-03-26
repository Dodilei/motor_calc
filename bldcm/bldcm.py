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
        v_inf_clp = max(0., v_inf)
        X_in = np.array([[self.diameter * 39.37, self.pitch, n_rpm, v_inf_clp]])

        # Predict Cp and Ct using the surrogate model
        predictions = self.surrogate.predict(X_in)

        # Extract values
        ct = predictions[0][0]
        cp = predictions[0][1]

        return cp, ct

    def _residual(
        self, n_rpm: float, v_inf: float, target: float, voltage=False
    ) -> float:
        cp, _ = self._get_aero_coefficients(n_rpm, v_inf)

        # Calculate Propeller Power (Aerodynamic Load)
        n_rps = n_rpm / 60.0
        p_prop = cp * self.rho * (n_rps**3) * (self.diameter**5)

        # Calculate Motor Current
        i_motor = self.i0 + (p_prop * self.kv) / n_rpm

        # Calculate Motor Voltage
        v_motor = (n_rpm / self.kv) + (i_motor * self.rm)

        # Calculate Electrical Input Power
        p_in_calc = v_motor * i_motor

        if voltage:
            return v_motor - target
        else:
            return p_in_calc - target

    def solve_for_target_power(
        self, target: float, v_inf: float, rpm_bounds: tuple = (100, 15000)
    ) -> dict:
        try:
            # Isolate the equilibrium RPM
            n_eq: float = brentq(
                f=self._residual,
                a=rpm_bounds[0],
                b=rpm_bounds[1],
                args=(v_inf, target),
                xtol=1e-3,
            )
        except ValueError:
            raise RuntimeError(
                f"Could not find equilibrium for Pin={target}W within RPM bounds {rpm_bounds}."
            )

        if self._residual(n_eq, v_inf, 24.2, True) > 0:
            n_eq: float = brentq(
                f=self._residual,
                a=rpm_bounds[0],
                b=rpm_bounds[1],
                args=(v_inf, 24.2, True),
                xtol=1e-3,
            )

        # Retrieve final state at equilibrium
        cp, ct = self._get_aero_coefficients(n_eq, v_inf)
        n_rps = n_eq / 60.0

        # Final Aerodynamic metrics
        p_prop = cp * self.rho * (n_rps**3) * (self.diameter**5)
        thrust = ct * self.rho * (n_rps**2) * (self.diameter**4)
        j_adv = v_inf / (n_rps * self.diameter) if v_inf > 0 else 0.0

        # Final Electrical metrics
        i_eq = self.i0 + (p_prop * self.kv) / n_eq
        v_eq = (n_eq / self.kv) + (i_eq * self.rm)
        efficiency = p_prop / (v_eq * i_eq)

        return {
            "RPM": n_eq,
            "Voltage_V": v_eq,
            "Current_A": i_eq,
            "Thrust_N": thrust,
            "Efficiency": efficiency,
            "Advance_Ratio_J": j_adv,
            "P_prop_W": p_prop,
            "cp": cp,
        }

    def solve_thrust(
        self, target: float, v_inf: float, rpm_bounds: tuple = (100, 15000)
    ):
        try:
            # Isolate the equilibrium RPM
            n_eq: float = brentq(
                f=self._residual,
                a=rpm_bounds[0],
                b=rpm_bounds[1],
                args=(v_inf, target),
                xtol=1e-3,
            )
        except ValueError:
            raise RuntimeError(
                f"Could not find equilibrium for Pin={target}W within RPM bounds {rpm_bounds}."
            )

        if self._residual(n_eq, v_inf, 24.2, True) > 0:
            n_eq: float = brentq(
                f=self._residual,
                a=rpm_bounds[0],
                b=rpm_bounds[1],
                args=(v_inf, 24.2, True),
                xtol=1e-3,
            )

        # Retrieve final state at equilibrium
        _, ct = self._get_aero_coefficients(n_eq, v_inf)
        n_rps = n_eq / 60.0

        # Final Aerodynamic metrics
        thrust = ct * self.rho * (n_rps**2) * (self.diameter**4)

        return thrust
