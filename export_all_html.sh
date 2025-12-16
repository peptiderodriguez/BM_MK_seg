#!/bin/bash
# Export HTML review interface for all slides in unified segmentation output

UNIFIED_BASE="$1"
CZI_DIR="$2"
OUTPUT_BASE="$3"

if [ -z "$UNIFIED_BASE" ] || [ -z "$CZI_DIR" ] || [ -z "$OUTPUT_BASE" ]; then
    echo "Usage: $0 <unified_base_dir> <czi_dir> <output_base_dir>"
    echo "Example: $0 /ptmp/edrod/unified_2pct /ptmp/edrod/2025_11_18 /ptmp/edrod/review_2pct"
    exit 1
fi

echo "Exporting HTML for all slides..."
echo "Unified dir: $UNIFIED_BASE"
echo "CZI dir: $CZI_DIR"
echo "Output dir: $OUTPUT_BASE"
echo ""

module load python-waterboa/2024.06

for slide_dir in "$UNIFIED_BASE"/2025_11_18_*/; do
    slide_name=$(basename "$slide_dir")
    czi_path="$CZI_DIR/${slide_name}.czi"
    output_dir="$OUTPUT_BASE/$slide_name"

    if [ ! -f "$czi_path" ]; then
        echo "WARNING: CZI not found: $czi_path"
        continue
    fi

    echo "Processing $slide_name..."
    python /viper/ptmp2/edrod/MKsegmentation/export_unified_html.py \
        --unified-dir "$slide_dir" \
        --czi-path "$czi_path" \
        --output-dir "$output_dir"

    if [ $? -eq 0 ]; then
        echo "  ✓ Exported to $output_dir"
    else
        echo "  ✗ FAILED"
    fi
done

echo ""
echo "HTML export complete!"
echo "Open $OUTPUT_BASE/[SLIDE_NAME]/review.html to start annotating"
