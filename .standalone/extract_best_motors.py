import pandas as pd
import os


def main():
    sweep_results_path = "./.results/sweep_results_final.csv"
    mad_data_path = "./.data/mad_motor_data.csv"
    tmotor_data_path = "./.data/tmotor_data.csv"
    output_path = "./.data/best_motors.csv"

    if not os.path.exists(sweep_results_path):
        print(f"Error: {sweep_results_path} not found.")
        return

    # 1. Load sweep results
    print(f"Loading {sweep_results_path}...")
    sweep_df = pd.read_csv(sweep_results_path)

    # 2. Find best unique motors (best EE for each motor name)
    # We sort by EE descending and drop duplicates by 'Motor' name
    best_unique_rows = sweep_df.sort_values("EE", ascending=False).drop_duplicates(
        "Motor"
    )

    # 3. Take top 50 unique motors
    top_50 = best_unique_rows.head(50).copy()
    print(f"Identifying top {len(top_50)} motors based on efficiency (EE)...")

    # 4. Load source databases to get original weights
    source_dfs = []
    if os.path.exists(mad_data_path):
        source_dfs.append(pd.read_csv(mad_data_path))
    if os.path.exists(tmotor_data_path):
        source_dfs.append(pd.read_csv(tmotor_data_path))

    if not source_dfs:
        print("Error: No source motor databases found.")
        return

    source_motors_list = pd.concat(source_dfs, ignore_index=True)

    # Drop rows with missing crucial data for matching or output
    source_motors_list = source_motors_list.dropna(subset=["name", "weight"])

    # Normalize numeric columns for reliable matching
    params = ["kv", "io", "rm", "io_vref"]
    for df in [top_50, source_motors_list]:
        for col in params:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Prepare final list
    final_motors = []

    # Matching logic
    for _, row in top_50.iterrows():
        motor_name = row["Motor"]
        kv, io, rm, io_vref = row["kv"], row["io"], row["rm"], row["io_vref"]

        # 1. Try exact name match
        match = source_motors_list[source_motors_list["name"] == motor_name]

        # 2. Try partial name match (case insensitive) if exact fails
        if match.empty:
            match = source_motors_list[
                source_motors_list.apply(
                    lambda x: (
                        x["name"].lower() in motor_name.lower()
                        or motor_name.lower() in x["name"].lower()
                    ),
                    axis=1,
                )
            ]
            if not match.empty:
                # Filter by KV to avoid mismatched variants in the same series
                match = match[abs(match["kv"] - kv) < 5.0]

        # 3. Try purely parametric matching if name logic fails
        if match.empty:
            # We use slightly larger tolerances for parametric matching
            match = source_motors_list[
                (abs(source_motors_list["kv"] - kv) < 2.0)
                & (abs(source_motors_list["rm"] - rm) < 0.005)
                & (abs(source_motors_list["io"] - io) < 0.2)
            ]

        if not match.empty:
            # Pick the result with closest KV if multiple matches
            match = match.copy()
            match["kv_diff"] = abs(match["kv"] - kv)
            selected_source = match.sort_values("kv_diff").iloc[0]

            final_motors.append(
                {
                    "name": motor_name,
                    "kv": kv,
                    "io": io,
                    "rm": rm,
                    "io_vref": io_vref,
                    "weight": selected_source["weight"],
                }
            )
        else:
            print(
                f"Warning: Could not find weight for '{motor_name}' (KV={kv}). Setting weight=0.0."
            )
            final_motors.append(
                {
                    "name": motor_name,
                    "kv": kv,
                    "io": io,
                    "rm": rm,
                    "io_vref": io_vref,
                    "weight": 0.0,
                }
            )

    # 5. Save results in the exact same format as mad_motor_data.csv
    result_df = pd.DataFrame(final_motors)
    result_df = result_df[["name", "kv", "io", "rm", "io_vref", "weight"]]
    result_df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(result_df)} entries.")


if __name__ == "__main__":
    main()
