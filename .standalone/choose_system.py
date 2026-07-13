import os
import json
import sys

# Standard standardized project setup: Add current directory to path
sys.path.append(os.getcwd())

from motor_db import get_all_motors

def main():
    print("--- Motor and Propeller Selection ---")
    
    # 1. Search for motor
    motor_df = get_all_motors()
    search_term = input("Enter motor name (or part of it) to search: ").strip()
    
    matches = motor_df[motor_df['name'].str.contains(search_term, case=False, na=False)]
    
    if matches.empty:
        print(f"No motors found matching '{search_term}'.")
        return
        
    print(f"\nFound {len(matches)} matches:")
    for idx, (row_idx, row) in enumerate(matches.iterrows()):
        print(f"[{idx}] {row['name']} | KV: {row['kv']} | I0: {row['io']} | Rm: {row['rm']} | RefV: {row['io_vref']}")
        
    try:
        selection = int(input("\nSelect a motor by index: ").strip())
        selected_motor = matches.iloc[selection]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return
        
    print(f"Selected motor: {selected_motor['name']}")
    
    # 2. Ask for propeller
    try:
        diam = float(input("\nEnter propeller diameter (inches) [e.g. 22]: ").strip())
        pitch = float(input("Enter propeller pitch (inches) [e.g. 10]: ").strip())
    except ValueError:
        print("Invalid number.")
        return
        
    # 3. Ask for battery voltage
    try:
        v_batt = float(input("\nEnter battery voltage (V) [e.g. 23]: ").strip())
    except ValueError:
        print("Invalid number.")
        return
        
    # 4. Save to json
    chosen_system = {
        "motor_name": str(selected_motor['name']),
        "kv": float(selected_motor['kv']),
        "io": float(selected_motor['io']),
        "rm": float(selected_motor['rm']),
        "io_vref": float(selected_motor['io_vref']),
        "diam": diam,
        "pitch": pitch,
        "V_batt": v_batt
    }
    
    out_path = os.path.join(".data", "chosen_system.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(chosen_system, f, indent=4)
        
    print(f"\nSuccessfully saved chosen system to {out_path}!")
    print(json.dumps(chosen_system, indent=2))
    print("\nThis configuration will now be used as the default for the characterization and simulation scripts.")

if __name__ == "__main__":
    main()
