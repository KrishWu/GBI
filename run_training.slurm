#!/bin/bash
#SBATCH --job-name=train_ranode
#SBATCH --output=logs/simulation_%A_%a.out
#SBATCH --error=logs/simulation_%A_%a.err
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=atlas-gpu
#SBATCH --gres=gpu:1

set -euo pipefail

# Load required modules
# module load python/3.9

# # Set up environment
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Activate virtual environment
conda activate /nfspool-0/home/krishjw/GBI/ranode/gbi_ranode_env

cd ranode

source setup.sh

law index

# Run the simulation with the current combination
law run FittingScanResults   --version simplified_test_2 --mx 100 --my 500 --s-ratio-index 11 --FittingScanResults-device cpu --BkgTemplateTraining-device cpu --BkgTemplateChecking-device cpu --PerfectBkgTemplateTraining-device cpu --RNodeTemplate-device cpu --PredictBkgProb-device cpu --ScanRANODE-device cpu --SampleModelBinSR-device cpu --PredictBkgProbGen-device cpu

# Deactivate virtual environment
deactivate

echo "Job completed with return code $? at: $(date)"
