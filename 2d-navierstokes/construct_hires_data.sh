#!/bin/bash
#SBATCH --job-name=navierstokes_hires_sim
#SBATCH --output=./jobs/navierstokes_hires_%A_%a.out
#SBATCH --error=./jobs/navierstokes_hires_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-10

# Debug information
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Calculate delay based on the job array index to stagger starts
DELAY=$(($SLURM_ARRAY_TASK_ID * 60))
echo "Delaying start by ${DELAY} seconds"

# Wait for the calculated delay
sleep ${DELAY}

# Run the Python script
/n/home13/henrybae/.conda/envs/tf2.16.1_cuda12.1/bin/python construct_hires_data.py

# Check exit status
status=$?
echo "Exit status: $status"
exit $status
