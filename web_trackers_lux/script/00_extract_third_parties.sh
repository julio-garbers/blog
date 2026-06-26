#!/bin/bash -l
#SBATCH --job-name=web_trackers_extract
#SBATCH --output=/project/home/p200812/blog/web_trackers_lux/script/slurm/00_extract_%a.out
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --account=p200812
#SBATCH --qos=default
#SBATCH --array=0-11

# =============================================================================
# Job Information
# =============================================================================

# Calculate year from array index (0 -> 2013, 1 -> 2014, ..., 11 -> 2024)
export WEB_TRACKERS_YEAR=$((2013 + SLURM_ARRAY_TASK_ID))

echo "================================================================================"
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Array index: ${SLURM_ARRAY_TASK_ID}"
echo "Year: ${WEB_TRACKERS_YEAR}"
echo "================================================================================"

# =============================================================================
# Environment Setup
# =============================================================================

module --force purge
module load env/release/2024
module load Python/3.11.10-GCCcore-13.3.0

source /project/home/p200812/blog/.venv/bin/activate

# =============================================================================
# Run Third-Party Extraction
# =============================================================================

echo ""
echo "[RUN] Extracting third-party requests for year ${WEB_TRACKERS_YEAR}..."
echo ""

uv run /project/home/p200812/blog/web_trackers_lux/script/00_extract_third_parties.py

EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "Job completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "================================================================================"

exit ${EXIT_CODE}
