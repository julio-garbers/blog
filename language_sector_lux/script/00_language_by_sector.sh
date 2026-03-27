#!/bin/bash -l
#SBATCH --job-name=lang_sector_lux
#SBATCH --output=/project/home/p200812/blog/language_sector_lux/script/slurm/00_language_by_sector.out
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --account=p200812
#SBATCH --qos=default

# =============================================================================
# Job Information
# =============================================================================

echo "================================================================================"
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "================================================================================"

# =============================================================================
# Environment Setup
# =============================================================================

module --force purge
module load env/release/2024
module load Python/3.11.10-GCCcore-13.3.0

source /project/home/p200812/blog/.venv/bin/activate

# =============================================================================
# Run Language x Sector Analysis
# =============================================================================

echo ""
echo "[RUN] Language x Sector analysis..."
echo ""

uv run /project/home/p200812/blog/language_sector_lux/script/00_language_by_sector.py

EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "Job completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "================================================================================"

exit ${EXIT_CODE}
