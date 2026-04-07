import argparse
import cProfile
import pstats
import io

from bldcm.bldcm import BLDCMSolver
from takeoff import apply_corrections, find_tow, load_surrogate, lookup_motor


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
        self.P_limit = 590.0
        self.V_limit = 23.0
        self.PV = 2.2
        for k, v in overrides.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)


def build_parser():
    p = argparse.ArgumentParser(
        description="Takeoff simulation for a specific plane + propulsion setup."
    )

    # Propulsion: manual
    p.add_argument("--kv", type=float, help="Motor KV constant")
    p.add_argument("--i0", type=float, help="Motor no-load current (A)")
    p.add_argument("--rm", type=float, help="Motor internal resistance (Ohm)")
    p.add_argument("--diam", type=float, help="Propeller diameter (inches)")
    p.add_argument("--pitch", type=float, help="Propeller pitch (inches)")
    p.add_argument(
        "--io_vref",
        type=float,
        default=0.0,
        help="I0 reference voltage for correction (V). Set to 0 to skip correction.",
    )

    # Propulsion: from database
    p.add_argument(
        "--motor",
        type=str,
        help="Motor name to look up from database (overrides --kv/--i0/--rm)",
    )

    # Airplane parameters
    p.add_argument("--S_wing", type=float, help="Wing area (m^2)")
    p.add_argument("--CL_max", type=float, help="Max lift coefficient")
    p.add_argument("--CL_ground", type=float, help="Ground roll lift coefficient")
    p.add_argument("--CD_ground", type=float, help="Ground roll drag coefficient")
    p.add_argument("--CD_max", type=float, help="CD at CL_max")
    p.add_argument("--mu", type=float, help="Ground friction coefficient")
    p.add_argument("--P_limit", type=float, help="Power limit (W)")
    p.add_argument("--V_limit", type=float, help="Voltage limit (V)")
    p.add_argument("--PV", type=float, help="Empty weight without propulsion (kg)")

    # Simulation
    p.add_argument(
        "--target_dist", type=float, default=55.0, help="Target runway distance (m)"
    )
    p.add_argument(
        "--slow",
        action="store_true",
        help="Use iterative solver instead of polynomial fit",
    )
    p.add_argument("--profile", action="store_true", help="Enable cProfile profiling")
    p.add_argument(
        "--no_correction", action="store_true", help="Skip motor parameter corrections"
    )

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Build airplane parameters from defaults + CLI overrides
    plane_keys = [
        "S_wing",
        "CL_max",
        "CL_ground",
        "CD_ground",
        "CD_max",
        "mu",
        "P_limit",
        "V_limit",
        "PV",
    ]
    overrides = {
        k: getattr(args, k) for k in plane_keys if getattr(args, k) is not None
    }
    params = AircraftParameters(**overrides)

    # Resolve propulsion parameters
    if args.motor and args.kv is not None:
        motor = lookup_motor(args.motor, args.kv)
        kv = motor["kv"]
        i0 = motor["io"]
        rm = motor["rm"]
        io_vref = motor["io_vref"]
        print(
            f"Motor '{args.motor}': KV={kv} I0={i0} Rm={rm} io_vref={io_vref} Weight={motor['weight']}kg"
        )
    elif args.kv is not None and args.i0 is not None and args.rm is not None:
        kv = args.kv
        i0 = args.i0
        rm = args.rm
        io_vref = args.io_vref
    else:
        # Defaults (original takeoff.py values)
        kv, i0, rm, io_vref = 330, 1.66, 0.065, 0.0

    diam = args.diam if args.diam is not None else 18
    pitch = args.pitch if args.pitch is not None else 8

    # Apply corrections
    if not args.no_correction and io_vref > 0:
        kv, i0, rm = apply_corrections(kv, i0, rm, io_vref)
        print(f"Corrected: KV={kv:.1f} I0={i0:.3f} Rm={rm:.4f}")

    # Initialize
    surrogate_model = load_surrogate()
    solver = BLDCMSolver(
        surrogate_model=surrogate_model,
        kv=kv,
        i0=i0,
        rm=rm,
        diameter=diam * 0.0254,
        pitch=pitch,
    )

    use_fast = not args.slow

    def run_sim():
        print(f"Starting TOW optimization (Fast Thrust: {use_fast})...")
        mtow, t_static, status = find_tow(
            solver,
            params,
            target_dist=args.target_dist,
            use_fast_thrust=use_fast,
        )
        if mtow:
            print(
                f"\nOptimal TOW for {args.target_dist}m runway: {mtow:.3f} kg (status: {status})"
            )
            print(f"Structural efficiency: {(mtow - params.PV - )}")
            print(f"Static thrust: {t_static:.2f} N")
        else:
            print("\nCould not find a valid TOW for the given constraints.")

    if args.profile:
        print(f"Starting TOW optimization with profiling (Fast Thrust: {use_fast})...")
        pr = cProfile.Profile()
        pr.enable()
        run_sim()
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        print(s.getvalue())
    else:
        print(
            "Tip: Run with --profile to see performance analysis or --slow to use iterative solver."
        )
        run_sim()


if __name__ == "__main__":
    main()
