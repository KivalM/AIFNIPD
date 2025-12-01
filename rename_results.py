import os
from pathlib import Path

results_dir = Path("results/hyperparameters/qlearning")

if not results_dir.exists():
    print(f"Directory {results_dir} does not exist.")
    exit()

print(f"Scanning {results_dir}...")

for agent_dir in results_dir.iterdir():
    if not agent_dir.is_dir():
        continue
    
    name = agent_dir.name
    # Check if it matches the pattern QLearner_{lr}_{gamma}_{epsilon}_{mem}
    # And DOES NOT already end with _0.999 (or similar float)
    parts = name.split('_')
    
    # Expected parts: QLearner, lr, gamma, epsilon, mem -> 5 parts
    if len(parts) == 5 and parts[0] == "QLearner":
        new_name = f"{name}_0.999"
        new_path = agent_dir.parent / new_name
        
        print(f"Renaming {name} -> {new_name}")
        try:
            agent_dir.rename(new_path)
        except OSError as e:
            print(f"Error renaming {name}: {e}")
    else:
        print(f"Skipping {name} (does not match expected pattern or already renamed)")

print("Done.")
