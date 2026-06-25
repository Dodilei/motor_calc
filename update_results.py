import pandas as pd
import os


def update_sweep_results():
    # Paths
    results_path = os.path.join(".results", "sweep_results.csv")
    prop_perf_path = os.path.join(".data", "prop_perfmap.csv")
    output_path = os.path.join(".results", "sweep_results_2.csv")

    # Load data
    print(f"Loading {results_path}...")
    df_results = pd.read_csv(results_path)
    print(f"Loading {prop_perf_path}...")
    df_prop = pd.read_csv(prop_perf_path)

    # Create propeller mapping
    # Propeller, Diameter, Pitch
    prop_mapping = df_prop[["Propeller", "Diameter", "Pitch"]].drop_duplicates().copy()

    # Correction: 20x15E prop is actually 15 pitch, not 1.5
    prop_mapping.loc[prop_mapping["Propeller"] == "20x15E", "Pitch"] = 15.0

    # Merge
    print("Merging data...")
    df_merged = df_results.merge(
        prop_mapping, left_on="Prop", right_on="Propeller", how="left"
    )

    # Clean up merged columns (remove 'Propeller' as it's redundant with 'Prop')
    if "Propeller" in df_merged.columns:
        df_merged = df_merged.drop(columns=["Propeller"])

    # Reorder columns to put Diameter and Pitch next to Prop if desired
    # Current columns: Motor, Prop, MTOW, etc.
    # Let's put Diameter and Pitch after Prop
    cols = list(df_merged.columns)
    # Move Diameter, Pitch right after Prop
    prop_idx = cols.index("Prop")
    # Use pop and insert for better control
    pitch = cols.pop(cols.index("Pitch"))
    diameter = cols.pop(cols.index("Diameter"))
    cols.insert(prop_idx + 1, diameter)
    cols.insert(prop_idx + 2, pitch)

    df_merged = df_merged[cols]

    df_merged["io0"] = df_merged.io * (1 - 0.01 * df_merged["io_vref"])

    # Save to new file
    print(f"Saving updated results to {output_path}...")
    df_merged.to_csv(output_path, index=False)
    print("Done!")


if __name__ == "__main__":
    update_sweep_results()
