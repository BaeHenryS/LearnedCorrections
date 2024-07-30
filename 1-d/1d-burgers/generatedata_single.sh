#!/bin/bash
#SBATCH --job-name=burgers_sim
#SBATCH --output=./jobs/burgers_sim.out
#SBATCH --error=./jobs/burgers_sim.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=24:00:00

# Load necessary modules (if any)
module load python

conda activate tf2.16.1_cuda12.1

# Set parameters
RESOLUTION=512
VISCOSITY=0.01
DT=0.001
OUTPUT="./output"
NUM_ITERS=8000

# Ensure the output directory exists
mkdir -p $OUTPUT

# Run the Python script with the specified output directory
/n/home13/henrybae/.conda/envs/tf2.16.1_cuda12.1/bin/python constructdata.py --resolution $RESOLUTION --viscosity $VISCOSITY --dt $DT --output $OUTPUT --num_iters $NUM_ITERS --start_sim 0 --end_sim 20