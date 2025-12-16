#!/bin/bash
#SBATCH --job-name=mk_hspc_export
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/viper/ptmp2/edrod/MKsegmentation/export_%j.log
#SBATCH --error=/viper/ptmp2/edrod/MKsegmentation/export_%j.err

module load python-waterboa/2024.06

cd /viper/ptmp2/edrod/MKsegmentation

# Export 2% data with OLD filter values (4000-75000 px² = 119-2232 µm²)
# This matches the original 2% segmentation parameters
python export_separate_mk_hspc.py \
    --base-dir /ptmp/edrod/unified_2pct \
    --czi-base /ptmp/edrod/2025_11_18 \
    --output-dir /viper/ptmp2/edrod/seg_tohtml \
    --samples-per-page 300 \
    --mk-min-area-um 119 \
    --mk-max-area-um 2232

echo "Export completed successfully"
