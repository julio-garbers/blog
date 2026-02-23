#!/bin/bash -l
#SBATCH --job-name=embed_lux
#SBATCH --output=/project/home/p200812/blog/bert_topic_websites_lux/script/slurm/embed_%a.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --account=p200812
#SBATCH --qos=default
#SBATCH --array=0-11

# =============================================================================
# Job Information
# =============================================================================

# Calculate year from array index (0 -> 2013, 1 -> 2014, ..., 11 -> 2024)
export YEAR=$((2013 + SLURM_ARRAY_TASK_ID))

echo "================================================================================"
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Array index: ${SLURM_ARRAY_TASK_ID}"
echo "Year: ${YEAR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "================================================================================"

# =============================================================================
# Environment Setup
# =============================================================================

module --force purge
module load env/release/2024
module load Python/3.11.10-GCCcore-13.3.0

source /project/home/p200812/blog/.venv/bin/activate

# Cache models in project directory (shared across array jobs)
export SENTENCE_TRANSFORMERS_HOME=/project/home/p200812/blog/bert_topic_websites_lux/models
export HF_HOME=/project/home/p200812/blog/bert_topic_websites_lux/models

# =============================================================================
# Run Embedding
# =============================================================================

echo ""
echo "[RUN] Starting embedding for year ${YEAR}..."
echo ""

uv run /project/home/p200812/blog/bert_topic_websites_lux/script/01_embed.py

EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "Job completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "================================================================================"

exit ${EXIT_CODE}
