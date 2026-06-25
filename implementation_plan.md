# Refactor: Decouple `sweep_propulsion.py` and `takeoff.py`

## Resolved Decisions

| Item | Decision |
|---|---|
| Bisect bounds | `[4.0, 28.0]` (from `[5.0, 20.0]`) |
| Saturated results | Flagged in output, not excluded |
| Propeller lookup | Manual `--diam`/`--pitch` only |
| Code sharing | `TakeoffSolver` shared via module; `AircraftParameters` local to each script |
| Sweep CLI | Hardcoded defaults, CLI args supersede |
| Output | Add full propulsion weight breakdown |

---

## I₀ Correction — Suggestion

Current formula and its problem:
```python
corr_io = io * (1 - 0.01 * io_vref)   # Breaks when io_vref > 100
```

### Proposed: Voltage-ratio linear scaling

```python
corr_io = io * (V_operating / io_vref)
```

**Physics rationale**: I₀ is dominated by mechanical friction and iron losses, both of which scale with RPM. Since no-load RPM ∝ KV × V, I₀ measured at `io_vref` can be scaled to the operating voltage linearly:

- At `io_vref = 10V`, operating at `23V` → `corr_io = io × 2.3` (motor spins faster → more friction)
- At `io_vref = 20V`, operating at `23V` → `corr_io = io × 1.15` (small correction)
- At `io_vref = 358V`, operating at `23V` → `corr_io = io × 0.064` (huge motor tested at high V)

**Properties**:
- Never negative
- Correction ≈ 1.0 when `io_vref ≈ V_operating` (no distortion)
- Physically bounded: scales down for high-voltage-tested motors, scales up for low-voltage-tested motors

> [!IMPORTANT]
> **Compared to your original correction**: For a motor tested at `io_vref = 10V` with `V_operating = 23V`:
> - Your formula: `io × (1 - 0.1) = io × 0.90` (reduces I₀)
> - Proposed: `io × (23/10) = io × 2.3` (increases I₀)
> 
> These go in **opposite directions**. Your original correction *reduces* I₀ for low-vref motors. The voltage-ratio scaling *increases* it (because higher operating voltage → higher RPM → higher friction). 
> 
> Which behavior matched your single-motor validation data? If the motor had lower I₀ at operating conditions than at test conditions, a different mechanism may be at play (e.g., bearing break-in, thermal effects). Please confirm before I implement.

**Alternative (compromise)**: If you want a small downward correction that doesn't break:
```python
corr_io = io * max(0.1, 1 - 0.01 * io_vref)  # Clamped, never below 10%
```
This preserves your original formula's direction but prevents negatives. It's not physics-based, but it's safe.

---

## Proposed Changes

### File Structure

```
takeoff.py              → Shared module (TakeoffSolver class, utility functions)
simulate_takeoff.py     → [NEW] CLI script for single plane+propulsion simulation
sweep_propulsion.py     → Sweep script (imports TakeoffSolver from takeoff.py)
```

---

### [MODIFY] [takeoff.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/takeoff.py) → Shared module

- Keep `TakeoffSolver` class (unchanged logic)
- Move `AircraftParameters` out — it no longer lives here
- Remove `find_tow_for_distance()` and `if __name__ == "__main__"` block
- Keep as a clean importable module

Contents after refactor:
```python
# takeoff.py — Shared takeoff simulation module
from bldcm.bldcm import BLDCMSolver
from surrogate.prs import PRSSurrogate
from propeller_surrogate import MODEL_PATH

class TakeoffSolver:
    # ... (identical to current, but receives params as a generic object/dict)
    
def find_tow_for_combination(solver, params, target_dist, bounds=(4.0, 28.0)):
    # Shared bisection logic, used by both scripts
```

---

### [NEW] [simulate_takeoff.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/simulate_takeoff.py)

Replaces current `takeoff.py`'s `__main__` block. Features:

- **Local `AircraftParameters`** with hardcoded defaults
- **Propulsion params** via CLI: `--kv`, `--i0`, `--rm`, `--diam`, `--pitch`
- **Motor database lookup**: `--motor <name>` (searches `.data/*.csv`)
- **Parameter corrections** applied to motor params (same formulas as sweep)
- **Airplane param overrides** via CLI: `--S_wing`, `--P_limit`, `--V_batt`, `--PV`, etc.
- Retains `--profile` and `--slow` flags

```
python simulate_takeoff.py
python simulate_takeoff.py --kv 330 --i0 1.66 --rm 0.065 --diam 18 --pitch 8
python simulate_takeoff.py --motor "TMOTOR-AT4130" --diam 18 --pitch 8
python simulate_takeoff.py --P_limit 600 --target_dist 60
```

---

### [MODIFY] [sweep_propulsion.py](file:///c:/Users/User/Documents/Coisas/Study/UFGD/Aracs/2026/Projects/motor_calc/sweep_propulsion.py)

- **Import** `TakeoffSolver` from `takeoff.py` (shared code)
- **Define `AircraftParameters` locally** (copy of class, local defaults)
- **Bisect bounds** changed to `[4.0, 28.0]`
- **Quality flag** column: `"converged"`, `"saturated_low"`, `"saturated_high"`
- **Skip invalid motors**: negative corrected I₀ → skip with warning
- **Output columns**: add `Motor_Wt`, `Prop_Wt` (full propulsion weight breakdown)
- **CLI args** for airplane params (hardcoded defaults, CLI supersedes)
- **I₀ correction**: pending your decision above

---

## Verification Plan

### Automated
- `python simulate_takeoff.py` → verify matches current `takeoff.py` output
- `python simulate_takeoff.py --kv 330 --i0 1.66 --rm 0.065 --diam 18 --pitch 8` → same result
- `python simulate_takeoff.py --motor "TMOTOR-AT4130" --diam 18 --pitch 8` → valid lookup
- `python sweep_propulsion.py` → verify results contain quality flags and weight breakdown
- Verify `sweep_propulsion.py` has no dependency on `simulate_takeoff.py`
