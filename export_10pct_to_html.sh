#!/bin/bash
#SBATCH --job-name=export_10pct
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/viper/ptmp2/edrod/MKsegmentation/export_10pct_%j.log
#SBATCH --error=/viper/ptmp2/edrod/MKsegmentation/export_10pct_%j.err

module load python-waterboa/2024.06

cd /viper/ptmp2/edrod/MKsegmentation

python export_separate_mk_hspc.py \
    --base-dir /ptmp/edrod/unified_10pct \
    --czi-base /ptmp/edrod/2025_11_18 \
    --output-dir /viper/ptmp2/edrod/seg_tohtml_10pct \
    --samples-per-page 300 \
    --mk-min-area-um 100 \
    --mk-max-area-um 2100

echo "Export completed successfully"
