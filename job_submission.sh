#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --account=def-agodbout
#SBATCH --mem=32000M		# memory per node
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10	# CPU cores/threads
#SBATCH --job-name=testing_pcam
#SBATCH --output=output/testing_pcam_jobid_%j.txt
#SBATCH --SBATCH --mail-user=example@gmail.com
#SBATCH --mail-type=ALL

SCRATCH="~/scratch"

echo "--- PCAM Job ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"

nvidia-smi

CACHE_BASE_DIR="${SCRATCH}/.cache" # Or /scratch/jezul/.cache

export TORCH_HOME="${CACHE_BASE_DIR}/torch"
mkdir -p $TORCH_HOME/hub/checkpoints # Ensure it exists

export HF_DATASETS_CACHE="${CACHE_BASE_DIR}/huggingface/datasets"
mkdir -p $HF_DATASETS_CACHE


ENV=~/workspace/TestProjects/PatchChamelyonProject/.venv/bin/activate

# load any required modules
module load StdEnv/2023 arrow cuda cudnn

# activate venv
source ${ENV}

echo "Python executable after venv activation: $(which python)"
echo "PyTorch version in venv: $(python -c 'import torch; print(torch.__version__)')"
echo "Is CUDA available to PyTorch in venv: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Variables for readability
OUTPUT_BASE_DIR="${SCRATCH}/pcam_projectt_runs"
RUN_SPECIFIC_DIR="${OUTPUT_BASE_DIR}/run_$(date +%Y%m%d_%H%M%S)_jobid_${SLURM_JOB_ID}"
mkdir -p "${RUN_SPECIFIC_DIR}"

echo "Output directory for this run: ${RUN_SPECIFIC_DIR}"

echo "Starting Python script execution..."

# run file
srun python PCamelyon.py --hpc_mode 1 --output_dir ${RUN_SPECIFIC_DIR}

echo "Python script finished with exit code $?"
echo "Job completed at $(date)"

