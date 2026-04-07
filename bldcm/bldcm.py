from scipy.optimize import brentq
import numpy as np

KV_EFF = 0.85
THRUST_EFF = 0.93
MPOWER_EFF = 1

ESC_RESIST = 0.003
BATT_RESIST = 0.0075

K_SWITCH = 0.01


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
        rest_voltage: float = 24,
    ):
        self.surrogate = surrogate_model
        self.kv = kv
        self.i0 = i0
        self.rm = rm
        self.diameter = diameter
        self.pitch = pitch
        self.rho = rho
        self.rest_voltage = rest_voltage

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
        target_dutycycle: float | None = None,
    ) -> float:
        cp, _ = self._get_aero_coefficients(n_rpm, v_inf)

        # Calculate Propeller Power (Aerodynamic Load)
        n_rps = n_rpm / 60.0
        p_prop = MPOWER_EFF * cp * self.rho * (n_rps**3) * (self.diameter**5)

        # Calculate Motor Current
        v_kv = n_rpm / (KV_EFF * self.kv)
        v_est = v_kv + self.rm * (self.i0 + p_prop / v_kv)
        i_motor = (self.i0 * (1 + 0.01 * v_est)) + (p_prop / v_kv)

        # Calculate Motor Voltage
        v_motor = v_kv + (i_motor * self.rm)

        # ESC conductive losses
        v_eff = v_motor + (i_motor * ESC_RESIST)

        discriminant = self.rest_voltage**2 - 4 * (i_motor * BATT_RESIST) * v_eff
        # Prevent NaN crashes from impossible optimizer guesses
        discriminant = max(0.0, discriminant)
        duty_cycle = (self.rest_voltage - np.sqrt(discriminant)) / (
            2 * i_motor * BATT_RESIST
        )

        v_in_calc = self.rest_voltage - (duty_cycle * i_motor * BATT_RESIST)

        switching_loss = K_SWITCH * v_in_calc * i_motor * duty_cycle * (1 - duty_cycle)

        p_in_calc = v_in_calc * (duty_cycle * i_motor) + switching_loss

        if target_power is None:
            return duty_cycle - target_dutycycle
        elif target_dutycycle is None:
            return p_in_calc - target_power
        else:
            return max(duty_cycle - target_dutycycle, p_in_calc - target_power)

    def solve_thrust(
        self,
        v_inf: float,
        max_power: float | None = 600,
        max_throttle: float | None = 1.0,
        rpm_bounds: tuple = (100, 15000),
        return_state: bool = False,
    ):
        residualf_args = (v_inf, max_power, max_throttle)
        print(max_throttle)
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
                f"Could not find equilibrium for ({max_power},{max_throttle}) at {v_inf} within RPM bounds."
            )

        # Retrieve final state at equilibrium
        cp, ct = self._get_aero_coefficients(n_eq, v_inf)
        n_rps = n_eq / 60.0

        # Final Aerodynamic metrics
        thrust = THRUST_EFF * ct * self.rho * (n_rps**2) * (self.diameter**4)

        if not return_state:
            return thrust
        else:
            p_prop = MPOWER_EFF * cp * self.rho * (n_rps**3) * (self.diameter**5)
            j_adv = v_inf / (n_rps * self.diameter) if v_inf > 0 else 0.0

            # Calculate Motor Current
            v_kv = n_eq / (KV_EFF * self.kv)
            v_est = v_kv + self.rm * (self.i0 + p_prop / v_kv)
            i_motor = (self.i0 * (1 + 0.01 * v_est)) + (p_prop / v_kv)

            # Calculate Motor Voltage
            v_motor = v_kv + (i_motor * self.rm)

            # ESC conductive losses
            v_eff = v_motor + (i_motor * ESC_RESIST)

            discriminant = self.rest_voltage**2 - 4 * (i_motor * BATT_RESIST) * v_eff
            # Prevent NaN crashes from impossible optimizer guesses
            discriminant = max(0.0, discriminant)
            duty_cycle = (self.rest_voltage - np.sqrt(discriminant)) / (
                2 * i_motor * BATT_RESIST
            )

            v_in_calc = self.rest_voltage - (duty_cycle * i_motor * BATT_RESIST)

            switching_loss = (
                K_SWITCH * v_in_calc * i_motor * duty_cycle * (1 - duty_cycle)
            )

            p_in_calc = v_in_calc * (duty_cycle * i_motor) + switching_loss

            efficiency = p_prop / p_in_calc
            print(v_eff, v_in_calc, switching_loss)
            print(v_eff * i_motor, p_in_calc)
            return {
                "RPM": n_eq,
                "Voltage_V": v_in_calc,
                "Duty_Cycle_D": duty_cycle,
                "Motor_Current_A": i_motor,
                "Batt_Current_A": (duty_cycle * i_motor),
                "Throttle_t": duty_cycle,
                "Thrust_N": thrust,
                "Efficiency": efficiency,
                "Advance_Ratio_J": j_adv,
                "P_el": p_in_calc,
                "P_prop_W": p_prop,
                "cp": cp,
            }
