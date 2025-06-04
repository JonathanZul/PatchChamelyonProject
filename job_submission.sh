#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --account=def-agodbout
#SBATCH --mem=12000M		# memory per node
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10	# CPU cores/threads
#SBATCH --job-name=testing_pcam
#SBATCH --output=$SCRATCH/slurm_out/testing_pcam_jobid_%j.txt
#SBATCH --mail-user=jezulluna@upei.ca
#SBATCH --mail-type=ALL

SCRATCH="$SCRATCH"

echo "--- PCAM Job ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"

nvidia-smi

CACHE_BASE_DIR="${SCRATCH}/.cache" 

export TORCH_HOME="${CACHE_BASE_DIR}/torch"
mkdir -p $TORCH_HOME/hub/checkpoints

export HF_DATASETS_CACHE="${CACHE_BASE_DIR}/huggingface/datasets"
mkdir -p $HF_DATASETS_CACHE


ENV=.venv/bin/activate

# load any required modules
module load StdEnv/2023 arrow cuda cudnn

# activate venv
source ${ENV}

echo "Python executable after venv activation: $(which python)"
echo "PyTorch version in venv: $(python -c 'import torch; print(torch.__version__)')"
echo "Is CUDA available to PyTorch in venv: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Variables for readability
OUTPUT_BASE_DIR="${SCRATCH}/pcam_project_runs"
RUN_SPECIFIC_DIR="${OUTPUT_BASE_DIR}/run_$(date +%Y%m%d_%H%M%S)_jobid_${SLURM_JOB_ID}"
CONFIG_FILE_PATH="configs/hpc_config.yaml"
mkdir -p "${RUN_SPECIFIC_DIR}"

echo "Output directory for this run: ${RUN_SPECIFIC_DIR}"

echo "Starting Python script execution..."

# run file
python PCamelyon.py \
    --config "${CONFIG_FILE_PATH}" \
    --hpc_mode 1 \
    --output_dir "${RUN_SPECIFIC_DIR}"

echo "Python script finished with exit code $?"
echo "Job completed at $(date)"

