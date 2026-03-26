import cProfile
import pstats
import io
from takeoff import TakeoffSolver, AircraftParameters, BLDCMSolver, PRSSurrogate, MODEL_PATH

def run_single_sim():
    surrogate_model = PRSSurrogate.load(MODEL_PATH)
    bldcm_solver = BLDCMSolver(
        surrogate_model=surrogate_model,
        kv=310,
        i0=1.66,
        rm=0.065,
        diameter=18 * 0.0254,
        pitch=8,
    )
    params = AircraftParameters()
    sim = TakeoffSolver(bldcm_solver, params)
    
    print("Running a short simulation for profiling (max 500 steps)...")
    dist = sim.simulate(10.0, dt=0.01, max_steps=500)
    print(f"Simulation done. Distance: {dist:.2f} m")

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    run_single_sim()
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())
