#!/bin/bash
#SBATCH --job-name=burgers_training
#SBATCH --output=./jobs/burgers_training_%j.out
#SBATCH --error=./jobs/burgers_training_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

module load python


# Execute the Python script
/n/home13/henrybae/.conda/envs/tf2.16.1_cuda12.1/bin/python train_updated_batch.py