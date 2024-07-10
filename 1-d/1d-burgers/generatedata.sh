#!/bin/bash
#SBATCH --job-name=burgers_sim
#SBATCH --output=./jobs/burgers_sim_%A_%a.out  # %A is replaced by job ID and %a by the array index
#SBATCH --error=./jobs/burgers_sim_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=24:00:00
#SBATCH --array=0-39

# Load necessary modules (if any)
module load python

conda activate tf2.16.1_cuda12.1

# Set parameters
RESOLUTION=512
VISCOSITY=0.01
DT=0.001
OUTPUT="./output"
NUM_ITERS=4000

# Calculate start and end sim based on SLURM_ARRAY_TASK_ID
# Each array job handles 10 simulations
START_SIM=$(($SLURM_ARRAY_TASK_ID * 10))
END_SIM=$(($START_SIM + 10))

# Ensure the output directory exists
mkdir -p $OUTPUT

# Run the Python script with the specified output directory and calculated start/end sim
/n/home13/henrybae/.conda/envs/tf2.16.1_cuda12.1/bin/python constructdata.py --resolution $RESOLUTION --viscosity $VISCOSITY --dt $DT --output $OUTPUT --num_iters $NUM_ITERS --start_sim $START_SIM --end_sim $END_SIM