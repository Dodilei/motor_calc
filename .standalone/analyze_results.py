import pandas as pd
import numpy as np
import os
import re

RESULTS_FILE = "./.results/sweep_results.csv"
MOTOR_DATA_FILE = "./.data/tmotor_data.csv"

def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found. Please run sweep_propulsion.py first.")
        return

    print(f"Loading results from {RESULTS_FILE}...")
    df = pd.read_csv(RESULTS_FILE)

    # 1. Ensure motor parameters are present
    if 'kv' not in df.columns:
        print("Motor parameters (kv, io, rm) missing in results. Joining with database...")
        if not os.path.exists(MOTOR_DATA_FILE):
            print(f"Error: {MOTOR_DATA_FILE} not found. Cannot join motor parameters.")
            return
        
        motors = pd.read_csv(MOTOR_DATA_FILE)
        # Align motor name columns for merging
        df = df.merge(motors[['name', 'kv', 'io', 'rm']], left_on='Motor', right_on='name', how='left')
        df.drop(columns=['name'], inplace=True)

    # 2. Extract diameter and drop diameter 23
    # Standard format: "APC 23x8E" or "23x8"
    def extract_diameter(prop_name):
        match = re.search(r'(\d+)x', str(prop_name))
        return int(match.group(1)) if match else None

    df['Diameter'] = df['Prop'].apply(extract_diameter)
    
    initial_count = len(df)
    df = df[df['Diameter'] != 23]
    print(f"Dropped {initial_count - len(df)} combinations with diameter 23.")

    # 3. Identify and drop outliers on Net_MTOW using IQR method
    Q1 = df['Net_MTOW'].quantile(0.25)
    Q3 = df['Net_MTOW'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_mask = (df['Net_MTOW'] < lower_bound) | (df['Net_MTOW'] > upper_bound)
    outliers = df[outlier_mask]
    
    if not outliers.empty:
        print(f"\n--- IDENTIFIED OUTLIERS ({len(outliers)}) ---")
        # Show a few if there are too many
        print(outliers[['Motor', 'Prop', 'Net_MTOW']].head(20).to_string(index=False))
        if len(outliers) > 20:
            print("...")
        df = df.drop(outliers.index)
        print(f"\nDropped {len(outliers)} outliers.")
    else:
        print("\nNo outliers identified.")

    # 4. Print 10 best combinations
    print("\n" + "="*80)
    print(f"{'TOP 10 BEST MOTOR-PROP COMBINATIONS':^80}")
    print("="*80)
    
    best_10 = df.nlargest(10, 'Net_MTOW')
    
    # Select and format columns for display
    display_cols = ['Motor', 'Prop', 'Net_MTOW', 'kv', 'io', 'rm']
    # Use better formatting for the table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(best_10[display_cols].to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()
