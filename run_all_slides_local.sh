#!/bin/bash
# Process all 16 slides sequentially with 2 workers each
# Using RAM disk for fast I/O

CZI_BASE="/mnt/x/01_Users/EdRo_axioscan/bonemarrow/2025_11_18"
OUTPUT_BASE="/mnt/ramdisk/output"
FINAL_OUTPUT="$HOME/code/BM_MK_seg/output"

# Ensure output directories exist
mkdir -p "$OUTPUT_BASE"
mkdir -p "$FINAL_OUTPUT"

# Array of all slides
SLIDES=(
    "2025_11_18_FGC1"
    "2025_11_18_FGC2"
    "2025_11_18_FGC3"
    "2025_11_18_FGC4"
    "2025_11_18_FHU1"
    "2025_11_18_FHU2"
    "2025_11_18_FHU3"
    "2025_11_18_FHU4"
    "2025_11_18_MGC1"
    "2025_11_18_MGC2"
    "2025_11_18_MGC3"
    "2025_11_18_MGC4"
    "2025_11_18_MHU1"
    "2025_11_18_MHU2"
    "2025_11_18_MHU3"
    "2025_11_18_MHU4"
)

# Process each slide
for SLIDE in "${SLIDES[@]}"; do
    echo "=========================================="
    echo "Processing: $SLIDE"
    echo "Start time: $(date)"
    echo "=========================================="

    # Run segmentation on RAM disk
    python run_unified_FAST.py \
        --czi-path "$CZI_BASE/${SLIDE}.czi" \
        --output-dir "$OUTPUT_BASE/$SLIDE" \
        --tile-size 3000 \
        --num-workers 2 \
        --mk-min-area-um 100 \
        --mk-max-area-um 2100 \
        --sample-fraction 0.10 \
        --calibration-samples 100

    # Check if successful
    if [ $? -eq 0 ]; then
        echo "SUCCESS: $SLIDE completed"

        # Copy results from RAM disk to permanent storage
        echo "Copying results to permanent storage..."
        cp -r "$OUTPUT_BASE/$SLIDE" "$FINAL_OUTPUT/"

        # Clean up RAM disk to free space
        rm -rf "$OUTPUT_BASE/$SLIDE"

        echo "End time: $(date)"
    else
        echo "ERROR: $SLIDE failed"
        echo "Skipping to next slide..."
    fi

    echo ""
done

echo "=========================================="
echo "All slides processed!"
echo "Results in: $FINAL_OUTPUT"
echo "=========================================="
