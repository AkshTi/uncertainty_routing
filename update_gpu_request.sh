#!/bin/bash
# Script to update all run_exp scripts to request better GPUs

for script in run_exp{1..7}.sh; do
    # Replace the generic gpu:1 with specific GPU types
    # Uncomment ONE of the lines below based on what's available on your cluster
    
    # Option 1: Request V100 (good, widely available)
    sed -i '' 's/#SBATCH --gres=gpu:1/#SBATCH --gres=gpu:V100:1/' "$script"
    
    # Option 2: Request A100 (best, if available - 2-3x faster than V100)
    # sed -i '' 's/#SBATCH --gres=gpu:1/#SBATCH --gres=gpu:A100:1/' "$script"
    
    # Option 3: Request specific partition with better GPUs
    # sed -i '' 's/#SBATCH --partition=mit_normal_gpu/#SBATCH --partition=sched_mit_psfc_gpu_r8/' "$script"
    
    echo "Updated $script"
done

echo "Done! Review the changes and uncomment your preferred GPU option."
