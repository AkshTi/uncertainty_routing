#!/bin/bash
# Update all experiment scripts to use A100 GPUs

for i in {1..7}; do
    script="run_exp${i}.sh"
    
    # Update partition
    sed -i '' 's/#SBATCH --partition=.*/#SBATCH --partition=sched_mit_psfc_gpu_r8/' "$script"
    
    # Update GPU type
    sed -i '' 's/#SBATCH --gres=gpu:.*/#SBATCH --gres=gpu:A100:1/' "$script"
    
    # Add constraint if not present
    if ! grep -q "constraint=A100" "$script"; then
        sed -i '' '/#SBATCH --gres=gpu:A100:1/a\
#SBATCH --constraint=A100
' "$script"
    fi
    
    # Reduce time since A100s are faster
    case $i in
        1|3) sed -i '' 's/#SBATCH --time=04:00:00/#SBATCH --time=02:00:00/' "$script" ;;
        2) sed -i '' 's/#SBATCH --time=03:00:00/#SBATCH --time=01:30:00/' "$script" ;;
        4|7) sed -i '' 's/#SBATCH --time=05:00:00/#SBATCH --time=02:30:00/' "$script" ;;
        5) sed -i '' 's/#SBATCH --time=06:00:00/#SBATCH --time=03:00:00/' "$script" ;;
        6) sed -i '' 's/#SBATCH --time=04:00:00/#SBATCH --time=02:00:00/' "$script" ;;
    esac
    
    echo "✓ Updated $script for A100"
done

echo ""
echo "All scripts updated to use A100 GPUs on sched_mit_psfc_gpu_r8"
echo "Time limits reduced by ~50%"
