import pandas as pd
from surrogate.prs import PRSSurrogate
from propeller_surrogate import MODEL_PATH

DEFAULT_DATABASES = ["./.data/tmotor_data.csv", "./.data/mad_motor_data.csv"]


def load_surrogate():
    """Load the propeller surrogate model."""
    return PRSSurrogate.load(MODEL_PATH)


def apply_corrections(kv, io, rm, io_vref):
    """Apply grounding corrections to motor parameters.

    These corrections normalize the motor parameters to a common reference.
    The runtime voltage-dependent correction is applied in bldcm.py.
    """
    corr_kv = kv * 1.05
    corr_io = io / (1 + 0.01 * io_vref)
    corr_rm = (rm * 0.95) * (1.035**3)
    return corr_kv, corr_io, corr_rm


def estimate_prop_weight(diam, pitch):
    """Estimate propeller weight using a linear regression formula.

    Parameters:
        diam (float): Propeller diameter in inches.
        pitch (float): Propeller pitch in inches.

    Returns:
        float: Estimated propeller weight in kg.
    """
    return (12 * diam + 4 * pitch - 176) / 1000.0


def lookup_motor(motor_name, kv, databases=None):
    """Look up motor parameters from CSV databases by name.

    Returns a dict with keys: name, kv, io, rm, io_vref, weight.
    Raises ValueError if motor not found.
    """
    if databases is None:
        databases = DEFAULT_DATABASES

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


def get_all_motors(databases=None):
    """Get all motors from the CSV databases."""
    if databases is None:
        databases = DEFAULT_DATABASES

    all_dfs = []
    for db in databases:
        df = pd.read_csv(db, on_bad_lines="skip")
        df = df.dropna(subset=["rm", "io", "kv", "io_vref"])
        cols = ["name", "kv", "io", "rm", "io_vref", "weight"]
        all_dfs.append(df[cols])

    return pd.concat(all_dfs, ignore_index=True)
