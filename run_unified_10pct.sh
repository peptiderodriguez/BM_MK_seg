#!/bin/bash
#SBATCH --job-name=unified_10pct
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/viper/ptmp2/edrod/MKsegmentation/unified_10pct_%j.log
#SBATCH --error=/viper/ptmp2/edrod/MKsegmentation/unified_10pct_%j.err

module load python-waterboa/2024.06
module load rocm/6.3

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

cd /viper/ptmp2/edrod/MKsegmentation

# Get all CZI files
CZI_FILES=(
    /ptmp/edrod/2025_11_18/2025_11_18_FGC1.czi
    /ptmp/edrod/2025_11_18/2025_11_18_FGC2.czi
    /ptmp/edrod/2025_11_18/2025_11_18_FGC3.czi
    /ptmp/edrod/2025_11_18/2025_11_18_FGC4.czi
    /ptmp/edrod/2025_11_18/2025_11_18_FHU1.czi
    /ptmp/edrod/2025_11_18/2025_11_18_FHU2.czi
    /ptmp/edrod/2025_11_18/2025_11_18_FHU3.czi
    /ptmp/edrod/2025_11_18/2025_11_18_FHU4.czi
    /ptmp/edrod/2025_11_18/2025_11_18_MGC1.czi
    /ptmp/edrod/2025_11_18/2025_11_18_MGC2.czi
    /ptmp/edrod/2025_11_18/2025_11_18_MGC3.czi
    /ptmp/edrod/2025_11_18/2025_11_18_MGC4.czi
    /ptmp/edrod/2025_11_18/2025_11_18_MHU1.czi
    /ptmp/edrod/2025_11_18/2025_11_18_MHU2.czi
    /ptmp/edrod/2025_11_18/2025_11_18_MHU3.czi
    /ptmp/edrod/2025_11_18/2025_11_18_MHU4.czi
)

OUTPUT_BASE=/ptmp/edrod/unified_10pct

# Process each slide
for CZI in "${CZI_FILES[@]}"; do
    SLIDE_NAME=$(basename "$CZI" .czi)
    echo "Processing $SLIDE_NAME with 10% sampling..."

    python run_unified_FAST.py \
        --czi-path "$CZI" \
        --output-dir "${OUTPUT_BASE}/${SLIDE_NAME}" \
        --tile-size 4096 \
        --num-workers 10 \
        --mk-min-area-um 100 \
        --mk-max-area-um 2100 \
        --sample-fraction 0.10 \
        --calibration-samples 100

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to process $SLIDE_NAME"
    else
        echo "SUCCESS: Completed $SLIDE_NAME"
    fi
done

echo "All slides processed with 10% sampling"

# Automatically run HTML export
echo "Starting HTML export..."
python export_separate_mk_hspc.py \
    --base-dir /ptmp/edrod/unified_10pct \
    --czi-base /ptmp/edrod/2025_11_18 \
    --output-dir /viper/ptmp2/edrod/seg_tohtml_10pct \
    --samples-per-page 300 \
    --mk-min-area-um 100 \
    --mk-max-area-um 2100

if [ $? -ne 0 ]; then
    echo "ERROR: HTML export failed"
    exit 1
fi

echo "HTML export completed"

# Automatically push to GitHub
echo "Pushing to GitHub Pages..."
cd /viper/ptmp2/edrod/seg_tohtml_10pct

# Configure git user
git config --global user.email "edrod@mpcdf.mpg.de"
git config --global user.name "Claude Code Bot"

# Initialize git if needed
if [ ! -d .git ]; then
    git init
    git branch -M main
    git remote add origin https://github.com/peptiderodriguez/mk_hspc_review.git
fi

# Add all files
git add .

# Commit with timestamp
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
git commit -m "Update annotations - 10% sampling - $TIMESTAMP

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Push to GitHub
git push -u origin main --force

if [ $? -ne 0 ]; then
    echo "ERROR: GitHub push failed"
    exit 1
fi

echo "Successfully pushed to GitHub Pages!"
echo "View at: https://peptiderodriguez.github.io/mk_hspc_review/"
