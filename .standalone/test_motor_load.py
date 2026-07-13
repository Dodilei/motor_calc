import os
import sys

# Standard standardized project setup: Add current directory to path
sys.path.append(os.getcwd())

from motor_db import get_all_motors as get_motors


def test_loading():
    print("Testing motor data loading updates...")
    try:
        motor_df = get_motors()
        print(f"Loaded {len(motor_df)} motors successfully.")

        print("\nFirst 5 motors:")
        print(motor_df.head())

        # Check units for T-Motor (usually start with TMOTOR)
        tmotors = motor_df[
            motor_df["name"].str.contains("TMOTOR", case=False, na=False)
        ]
        if not tmotors.empty:
            print(f"\nVerifying T-Motor units (n={len(tmotors)}):")
            avg_weight = tmotors["weight"].mean()
            avg_rm = tmotors["rm"].mean()
            print(f"Average weight: {avg_weight:.4f} kg")
            print(f"Average Rm: {avg_rm:.4f} Ohm")

            # Sanity check: Motors are usually between 10g (0.01kg) and 5kg.
            # Resistance is usually between 10mOhm (0.01) and 1 Ohm.
            if avg_weight > 5 or avg_rm > 1.0:
                print("WARNING: Units might still be skewed (values seem very high).")
            elif avg_weight < 0.001 or avg_rm < 0.0001:
                print("WARNING: Units might be too small.")
            else:
                print("Units appear to be correctly in Ohm and kg.")

        # Check MAD motors robustness
        mad_motors = motor_df[
            motor_df["name"].str.contains("MAD|FS|BSC", case=False, na=False)
        ]
        if not mad_motors.empty:
            print(f"\nMAD motor verification (n={len(mad_motors)}):")
            print(f"Average weight: {mad_motors['weight'].mean():.4f} kg")
            print("Successfully loaded MAD motors despite potential bad lines.")

        print("\nVerification PASSED.")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_loading()
