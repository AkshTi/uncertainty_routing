#!/bin/bash
# Check available GPU types on MIT Supercloud

echo "=== Available GPU Types ==="
sinfo -o "%20P %10G %10C %10m" | grep gpu

echo ""
echo "=== GPU Nodes and Availability ==="
sinfo -N -o "%12N %10P %10G %6t %10C %10m" | grep -E "NODELIST|gpu"

echo ""
echo "=== Current GPU Usage ==="
squeue -o "%.10i %.9P %.20j %.8u %.2t %.10M %.6D %.20R %.10b" | grep gpu | head -20

echo ""
echo "=== Recommended GPU Request ==="
echo "For fastest performance (< 30 min), add to your SLURM script:"
echo ""
echo "  #SBATCH --gres=gpu:A100:1"
echo "  #SBATCH --constraint=A100"
echo ""
echo "Or for V100:"
echo "  #SBATCH --gres=gpu:V100:1"
