#!/bin/bash
#SBATCH --job-name=burgers_sim
#SBATCH --output=./jobs/burgers_sim_%A_%a.out
#SBATCH --error=./jobs/burgers_sim_%A_%a.err
#SBATCH --array=0-99
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00

# Load necessary modules (if any)
module load python

conda activate tf2.16.1_cuda12.1

# Define the start and end indices for each job
START_SIM=$((SLURM_ARRAY_TASK_ID * 10))
END_SIM=$((START_SIM + 9))

# Set other parameters
RESOLUTION=512
VISCOSITY=0.01
DT=0.001
OUTPUT='./output'
NUM_ITERS=1000

# Run the Python script
/n/home13/henrybae/.conda/envs/tf2.16.1_cuda12.1/bin/python constructdata.py --start_sim $START_SIM --end_sim $END_SIM --resolution $RESOLUTION --viscosity $VISCOSITY --dt $DT --output $OUTPUT --num_iters $NUM_ITERS
