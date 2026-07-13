import argparse
import cProfile
import pstats
import io
import numpy as np
import matplotlib.pyplot as plt

from bldcm.bldcm import BLDCMSolver
from takeoff import find_tow
from motor_db import apply_corrections, load_surrogate, lookup_motor, estimate_prop_weight
from aircraft_params import AircraftParameters

# Constants for motor temperature dynamics
T_AMBIENT = 30.0  # Celsius
C_P = 0.55  # Specific heat of motor (Joules / gram*Kelvin)
# ALPHA_CU = 0.00393  # Temperature coefficient of copper


def build_parser():
    chosen_sys = AircraftParameters.load_chosen_system() or {}
    
    p = argparse.ArgumentParser(
        description="Takeoff simulation for a specific plane + propulsion setup."
    )

    # Propulsion: manual
    p.add_argument("--kv", type=float, default=chosen_sys.get("kv"), help="Motor KV constant")
    p.add_argument("--i0", type=float, default=chosen_sys.get("io"), help="Motor no-load current (A)")
    p.add_argument("--rm", type=float, default=chosen_sys.get("rm"), help="Motor internal resistance (Ohm)")
    p.add_argument("--diam", type=float, default=chosen_sys.get("diam"), help="Propeller diameter (inches)")
    p.add_argument("--pitch", type=float, default=chosen_sys.get("pitch"), help="Propeller pitch (inches)")
    p.add_argument(
        "--io_vref",
        type=float,
        default=chosen_sys.get("io_vref", 0.0),
        help="I0 reference voltage for correction (V). Set to 0 to skip correction.",
    )

    # Propulsion: from database
    p.add_argument(
        "--motor",
        type=str,
        help="Motor name to look up from database (overrides --kv/--i0/--rm)",
    )
    p.add_argument("--motor_wt", type=float, help="Motor weight (kg)")
    p.add_argument("--prop_wt", type=float, help="Propeller weight (kg)")

    # Airplane parameters
    p.add_argument("--S_wing", type=float, help="Wing area (m^2)")
    p.add_argument("--CL_max", type=float, help="Max lift coefficient")
    p.add_argument("--CL_ground", type=float, help="Ground roll lift coefficient")
    p.add_argument("--CD_ground", type=float, help="Ground roll drag coefficient")
    p.add_argument("--CD_max", type=float, help="CD at CL_max")
    p.add_argument("--mu", type=float, help="Ground friction coefficient")
    p.add_argument("--P_limit", type=float, help="Power limit (W)")
    p.add_argument("--V_batt", type=float, default=chosen_sys.get("V_batt"), help="Battery Voltage (V)")
    p.add_argument("--throttle", type=float, help="Throttle limit (V)")
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
        "V_batt",
        "throttle",
        "PV",
    ]
    overrides = {
        k: getattr(args, k) for k in plane_keys if getattr(args, k) is not None
    }
    params = AircraftParameters(**overrides)

    # Resolve propulsion parameters
    motor_wt = args.motor_wt
    if args.motor and args.kv is not None:
        motor = lookup_motor(args.motor, args.kv)
        kv = motor["kv"]
        i0 = motor["io"]
        rm = motor["rm"]
        io_vref = motor["io_vref"]
        if motor_wt is None:
            motor_wt = motor["weight"]
        print(
            f"Motor '{args.motor}': KV={kv} I0={i0} Rm={rm} io_vref={io_vref} Weight={motor_wt}kg"
        )
    elif args.kv is not None and args.i0 is not None and args.rm is not None:
        kv = args.kv
        i0 = args.i0
        rm = args.rm
        io_vref = args.io_vref
    else:
        # Defaults (original takeoff.py values)
        kv, i0, rm, io_vref = 330, 1.66, 0.065, 0.0

    if motor_wt is None:
        motor_wt = 0.150  # Default fallback if not in DB and not provided

    diam = args.diam if args.diam is not None else 18
    pitch = args.pitch if args.pitch is not None else 8

    prop_wt = args.prop_wt
    if prop_wt is None:
        prop_wt = estimate_prop_weight(diam, pitch)
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
        rest_voltage=params.V_batt,
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
            propulsion_wt = motor_wt + prop_wt
            total_pv = params.PV + propulsion_wt
            ee = (mtow - total_pv) / total_pv

            print(
                f"\nOptimal TOW for {args.target_dist}m runway: {mtow:.3f} kg (status: {status})"
            )
            print(
                f"Propulsion Weight: {propulsion_wt:.3f} kg (Motor: {motor_wt:.3f}, Prop: {prop_wt:.3f})"
            )
            print(f"Structural Efficiency (EE): {ee:.3f}")
            print(f"Static thrust: {t_static:.2f} N")

            # Plot simulation history
            plot_takeoff_history(solver, params, mtow, motor_wt)

        else:
            print("\nCould not find a valid TOW for the given constraints.")

    def plot_takeoff_history(solver, params, mass, motor_wt, dt=0.01):
        from takeoff import TakeoffSolver

        sim = TakeoffSolver(
            solver, params, use_fast_thrust=True
        )  # Use slow for better history accuracy

        print("Simulating final takeoff run with history tracking...")
        dist, history = sim.simulate(mass, track_history=True, dt=dt)

        # Plotting
        x = [h["x"] for h in history]

        fig = plt.figure(figsize=(12, 12), dpi=80)
        
        gs = fig.add_gridspec(3, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[2, 0])
        ax6 = fig.add_subplot(gs[2, 1])

        color1 = "tab:blue"
        ax1.set_xlabel("Runway Position (m)")
        ax1.set_ylabel("Speed (m/s)", color=color1)
        ax1.plot(x, [h["v_mag"] for h in history], color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1_twin = ax1.twinx()
        color2 = "tab:orange"
        ax1_twin.set_ylabel("Advance Ratio (J)", color=color2)
        ax1_twin.plot(x, [h.get("Advance_Ratio_J", 0) for h in history], color=color2)
        ax1_twin.tick_params(axis="y", labelcolor=color2)
        ax1.set_title("Speed and Advance Ratio")

        color1 = "tab:green"
        ax2.set_xlabel("Runway Position (m)")
        ax2.set_ylabel("Thrust (N)", color=color1)
        ax2.plot(x, [h.get("Thrust_N", 0) for h in history], color=color1)
        ax2.tick_params(axis="y", labelcolor=color1)
        ax2_twin = ax2.twinx()
        color2 = "tab:red"
        ax2_twin.set_ylabel("RPM", color=color2)
        ax2_twin.plot(x, [h.get("RPM", 0) for h in history], color=color2)
        ax2_twin.tick_params(axis="y", labelcolor=color2)
        ax2.set_title("Thrust and RPM")

        color1 = "tab:purple"
        ax3.set_xlabel("Runway Position (m)")
        ax3.set_ylabel("Motor Current (A)", color=color1)
        i_mot = [h.get("Motor_Current_A", 0) for h in history]
        ax3.plot(x, i_mot, color=color1)
        ax3.tick_params(axis="y", labelcolor=color1)
        
        ax3_twin = ax3.twinx()
        color2 = "tab:gray"
        ax3_twin.set_ylabel("Throttle", color=color2)
        throttles = [h.get("Throttle_t", 0) for h in history]
        ax3_twin.plot(x, throttles, color=color2)
        ax3_twin.tick_params(axis="y", labelcolor=color2)
        
        ax3.set_title("Motor Current and Throttle")
        i_min_mot = min(i_mot or [0])
        i_max_mot = max(i_mot or [0])
        ax3.set_ylim(0.7 * i_min_mot, 1.2 * i_max_mot)
        t_min = min(throttles or [0]) * 0.9
        ax3_twin.set_ylim(t_min, 1.0)

        color1 = "tab:brown"
        ax4.set_xlabel("Runway Position (m)")
        ax4.set_ylabel("Battery Current (A)", color=color1)
        i_batt = [h.get("Batt_Current_A", 0) for h in history]
        ax4.plot(x, i_batt, color=color1)
        ax4.tick_params(axis="y", labelcolor=color1)
        
        ax4_twin = ax4.twinx()
        color2 = "tab:pink"
        ax4_twin.set_ylabel("Voltage (V)", color=color2)
        ax4_twin.plot(x, [h.get("Voltage_V", 0) for h in history], color=color2)
        ax4_twin.tick_params(axis="y", labelcolor=color2)
        
        ax4.set_title("Battery Current and Voltage")
        i_min_batt = min(i_batt or [0])
        i_max_batt = max(i_batt or [0])
        ax4.set_ylim(0.7 * i_min_batt, 1.2 * i_max_batt)
        ax4_twin.set_ylim(0.5 * params.V_batt, 1.1 * params.V_batt)

        ax5.set_xlabel("Runway Position (m)")
        ax5.set_ylabel("Motor Temperature (°C)")
        T_motor = [
            T_AMBIENT + h.get("Qnorm", 0) / (1000 * motor_wt) / C_P for h in history
        ]
        ax5.plot(x, T_motor, color="tab:olive")
        ax5.set_title("Motor Temperature")

        ax6.set_xlabel("Runway Position (m)")
        ax6.set_ylabel("Height (m)")
        ax6.plot(x, [h["y"] for h in history], color="tab:cyan")
        ax6.set_title("Height")

        plt.tight_layout()
        plt.show()

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
