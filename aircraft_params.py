class AircraftParameters:
    def __init__(self, **overrides):
        self.g = 9.81
        self.rho = 1.225
        self.S_wing = 0.77
        self.CL_max = 1.969
        self.CL_ground = 0.997
        self.CD_ground = 0.057
        self.CD_max = 0.199
        self.mu = 0.04
        self.P_limit = 595.0
        self.V_batt = 23.0
        self.max_throttle = 0.8
        self.PV = 2.0
        for k, v in overrides.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)

    @classmethod
    def load_chosen_system(cls):
        import json
        import os

        path = os.path.join(".data", "chosen_system.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return None
