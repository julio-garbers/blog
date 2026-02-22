#!/bin/bash -l
#SBATCH --job-name=bert_topic_lux
#SBATCH --output=/project/home/p200812/blog/bert_topic_websites_lux/script/slurm/bert_topic_%a.out
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
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
echo "================================================================================"

# =============================================================================
# Environment Setup
# =============================================================================

module --force purge
module load env/release/2024
module load Python/3.11.10-GCCcore-13.3.0

source /project/home/p200812/blog/.venv/bin/activate

# Cache embedding model in project directory
export SENTENCE_TRANSFORMERS_HOME=/project/home/p200812/blog/bert_topic_websites_lux/models

# =============================================================================
# Run BERTopic Analysis
# =============================================================================

echo ""
echo "[RUN] Starting BERTopic analysis for year ${YEAR}..."
echo ""

uv run /project/home/p200812/blog/bert_topic_websites_lux/script/01_bert_topic.py

EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "Job completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "================================================================================"

exit ${EXIT_CODE}
