#!/bin/bash -l
#SBATCH --job-name=bertopic_global_lux
#SBATCH --output=/project/home/p200812/blog/bert_topic_websites_lux/script/slurm/02_bert_topic.out
#SBATCH --partition=largemem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --account=p200812
#SBATCH --qos=default

# =============================================================================
# Job Information
# =============================================================================

echo "================================================================================"
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================================"

# =============================================================================
# Environment Setup
# =============================================================================

module --force purge
module load env/release/2024
module load Python/3.11.10-GCCcore-13.3.0

source /project/home/p200812/blog/.venv/bin/activate

# =============================================================================
# Run Global BERTopic
# =============================================================================

echo ""
echo "[RUN] Starting global BERTopic analysis..."
echo ""

uv run /project/home/p200812/blog/bert_topic_websites_lux/script/02_bert_topic.py

EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "Job completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "================================================================================"

exit ${EXIT_CODE}
