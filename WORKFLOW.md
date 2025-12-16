# MK + HSPC Segmentation & Classification Workflow

Complete pipeline for segmenting, annotating, training classifiers, and applying them to full dataset.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. SAMPLE & ANNOTATE (10% of tiles)                            │
│    ├─ Segment 10% → Generate HTML → Push to GitHub             │
│    ├─ User annotates via GitHub Pages                          │
│    └─ Download annotations.json                                │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. TRAIN CLASSIFIERS                                            │
│    ├─ Convert annotations to training format                   │
│    ├─ Train separate MK and HSPC Random Forest classifiers     │
│    └─ Get classifier .pkl files                                │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. VALIDATE CLASSIFIERS                                         │
│    ├─ Check accuracy, precision, recall                        │
│    ├─ Must pass thresholds: 75% accuracy, 70% recall/precision │
│    └─ IF PASS → proceed to step 4                              │
│       IF FAIL → collect more annotations, retrain              │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. APPLY TO FULL DATASET (100% of tiles)                       │
│    ├─ Run segmentation with --mk-classifier --hspc-classifier  │
│    ├─ Classifiers filter detections in real-time               │
│    └─ Get final cell counts and features                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Sample & Annotate (10%)

### 1a. Run Segmentation on 10% of Tiles

**Current status:** Job 4896272 RUNNING (ETA ~4-6 hours)

```bash
# Submit batch job
sbatch /viper/ptmp2/edrod/MKsegmentation/run_unified_10pct.sh

# Monitor progress
squeue -j 4896272
tail -f /viper/ptmp2/edrod/MKsegmentation/unified_10pct_4896272.log
```

**What it does:**
- Processes 16 slides with 10% tile sampling
- MK size filter: **100-2100 µm²** (3360-70573 px²)
- Generates HTML for annotation
- Pushes to GitHub automatically

**Output:**
- Segmentation: `/ptmp/edrod/unified_10pct/`
- HTML: `/viper/ptmp2/edrod/seg_tohtml_10pct/`
- Live site: `https://peptiderodriguez.github.io/mk_hspc_review/`

**Expected cells:** ~4,500 MK, ~110,000 HSPC

---

### 1b. Annotate via GitHub Pages

1. Go to: `https://peptiderodriguez.github.io/mk_hspc_review/`
2. Review each cell image
3. Label as "Good" (positive) or "Bad" (negative)
4. Download annotations: `all_labels_combined.json`

---

### 1c. Transfer Annotations to Cluster

```bash
# From your local machine
scp all_labels_combined.json edrod@raven.mpcdf.mpg.de:/viper/ptmp2/edrod/MKsegmentation/annotations/all_labels_10pct.json
```

---

## Step 2: Train Classifiers

### 2a. Convert Annotations to Training Format

```bash
cd /viper/ptmp2/edrod/MKsegmentation

python convert_annotations_to_training.py \
    --annotations annotations/all_labels_10pct.json \
    --base-dir /ptmp/edrod/unified_10pct \
    --mk-min-area-um 100 \
    --mk-max-area-um 2100 \
    --output annotations/training_data_10pct.json
```

**What it does:**
- Recreates cell list in same order as HTML export
- Applies 100-2100 µm² MK filter (matches segmentation)
- Maps annotation IDs to actual cells
- Extracts 2,326 features per cell (22 morph + 256 SAM2 + 2048 ResNet)

**Output:** `annotations/training_data_10pct.json`

---

### 2b. Train Separate MK and HSPC Classifiers

```bash
python train_separate_classifiers.py \
    --training-data annotations/training_data_10pct.json \
    --output-mk mk_classifier_10pct.pkl \
    --output-hspc hspc_classifier_10pct.pkl \
    --morph-only
```

**Flags:**
- `--morph-only`: Use only 22 morphological features (faster, more interpretable)
- Without flag: Use all 2,326 features (SAM2 + ResNet embeddings)

**What it does:**
- Trains separate Random Forest for MK and HSPC
- Uses balanced class weights
- 5-fold cross-validation
- Saves confusion matrix and metrics

**Output:**
- `mk_classifier_10pct.pkl` (MK classifier)
- `hspc_classifier_10pct.pkl` (HSPC classifier)

---

## Step 3: Validate Classifiers

### 3a. Check Quality Thresholds

```bash
python validate_classifier.py \
    --mk-classifier mk_classifier_10pct.pkl \
    --hspc-classifier hspc_classifier_10pct.pkl \
    --min-accuracy 0.75 \
    --min-recall 0.70 \
    --min-precision 0.70
```

**Quality gates:**
- **Accuracy ≥ 75%**: Overall correctness
- **Recall ≥ 70%**: Don't miss true positives (sensitivity)
- **Precision ≥ 70%**: Avoid false positives (specificity)

**Output example:**
```
==================================================
MK: ✓ PASS
  Accuracy: 0.82, Precision: 0.78, Recall: 0.76
  Training samples: 850 (400 pos, 450 neg)

HSPC: ✓ PASS
  Accuracy: 0.80, Precision: 0.75, Recall: 0.72
  Training samples: 2100 (1050 pos, 1050 neg)

==================================================
✓ ALL CLASSIFIERS PASS VALIDATION
Ready to run on full dataset!
==================================================
```

---

### 3b. Decision Point

**IF ALL PASS:**
- ✅ Proceed to Step 4 (run on 100% dataset)

**IF ANY FAIL:**
- ❌ Collect more annotations
- Retrain with more data
- Re-validate

---

## Step 4: Apply to Full Dataset (100%)

### 4a. Create Batch Job for Full Segmentation

Create script `/viper/ptmp2/edrod/MKsegmentation/run_unified_100pct.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=unified_100pct
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/viper/ptmp2/edrod/MKsegmentation/unified_100pct_%j.log
#SBATCH --error=/viper/ptmp2/edrod/MKsegmentation/unified_100pct_%j.err

# Load modules
module load rocm/6.3
module load python-waterboa/2024.06

# Activate environment
source /viper/u2/edrod/.bashrc
conda activate seg_SAM2

# Set paths
SCRIPT_DIR="/viper/ptmp2/edrod/MKsegmentation"
OUTPUT_BASE="/ptmp/edrod/unified_100pct"
MK_CLASSIFIER="${SCRIPT_DIR}/mk_classifier_10pct.pkl"
HSPC_CLASSIFIER="${SCRIPT_DIR}/hspc_classifier_10pct.pkl"

# List of all 16 slides
CZI_FILES=(
    "/viper/u/edrod/HSPC/2025_11_18_FGC1/2025_11_18_FGC1.czi"
    "/viper/u/edrod/HSPC/2025_11_18_FGC2/2025_11_18_FGC2.czi"
    "/viper/u/edrod/HSPC/2025_11_18_FGC3/2025_11_18_FGC3.czi"
    "/viper/u/edrod/HSPC/2025_11_18_FGC4/2025_11_18_FGC4.czi"
    "/viper/u/edrod/HSPC/2025_11_18_FHU1/2025_11_18_FHU1.czi"
    "/viper/u/edrod/HSPC/2025_11_18_FHU2/2025_11_18_FHU2.czi"
    "/viper/u/edrod/HSPC/2025_11_18_FHU3/2025_11_18_FHU3.czi"
    "/viper/u/edrod/HSPC/2025_11_18_FHU4/2025_11_18_FHU4.czi"
    "/viper/u/edrod/HSPC/2025_11_18_MGC1/2025_11_18_MGC1.czi"
    "/viper/u/edrod/HSPC/2025_11_18_MGC2/2025_11_18_MGC2.czi"
    "/viper/u/edrod/HSPC/2025_11_18_MGC3/2025_11_18_MGC3.czi"
    "/viper/u/edrod/HSPC/2025_11_18_MGC4/2025_11_18_MGC4.czi"
    "/viper/u/edrod/HSPC/2025_11_18_MHU1/2025_11_18_MHU1.czi"
    "/viper/u/edrod/HSPC/2025_11_18_MHU2/2025_11_18_MHU2.czi"
    "/viper/u/edrod/HSPC/2025_11_18_MHU3/2025_11_18_MHU3.czi"
    "/viper/u/edrod/HSPC/2025_11_18_MHU4/2025_11_18_MHU4.czi"
)

# Process each slide
for CZI in "${CZI_FILES[@]}"; do
    SLIDE_NAME=$(basename $(dirname "$CZI"))

    echo "=================================================="
    echo "Processing ${SLIDE_NAME} with 100% sampling and classifiers..."
    echo "=================================================="

    python "${SCRIPT_DIR}/run_unified_FAST.py" \
        --czi-path "$CZI" \
        --output-dir "${OUTPUT_BASE}/${SLIDE_NAME}" \
        --tile-size 4096 \
        --num-workers 16 \
        --mk-classifier "$MK_CLASSIFIER" \
        --hspc-classifier "$HSPC_CLASSIFIER" \
        --mk-min-area-um 100 \
        --mk-max-area-um 2100 \
        --sample-fraction 1.0 \
        --calibration-samples 100
done

echo "All slides processed!"
echo "Output: ${OUTPUT_BASE}"
```

---

### 4b. Submit Job

```bash
sbatch /viper/ptmp2/edrod/MKsegmentation/run_unified_100pct.sh
```

**What it does:**
- Processes ALL tiles (100% sampling) from 16 slides
- MK size filter: 100-2100 µm²
- **Applies trained classifiers** to filter detections
- Each detection is classified before being saved
- Only "positive" cells (classifier predicts class=1) are kept

**ETA:** ~40-60 hours for all 16 slides @ 100% sampling

---

### 4c. Monitor Progress

```bash
# Check job status
squeue -u edrod

# Watch progress
tail -f /viper/ptmp2/edrod/MKsegmentation/unified_100pct_<JOB_ID>.log

# Check for errors
tail -f /viper/ptmp2/edrod/MKsegmentation/unified_100pct_<JOB_ID>.err
```

---

## How Classifiers Work

### Pipeline Behavior

**WITHOUT classifiers** (`--mk-classifier` omitted):
- All detections passing size filter are kept
- `apply_classifier()` returns `(True, 1.0)` for all cells
- Classifier confidence = 1.0

**WITH classifiers** (`--mk-classifier path/to/classifier.pkl`):
- Classifier evaluates each detection
- Returns `(is_positive, confidence)`
- If `is_positive=False`: cell is **removed** from results
- If `is_positive=True`: cell is kept with `classifier_confidence` score

### Where Filtering Happens

**run_unified_FAST.py:632-637 (MK cells):**
```python
# Apply classifier if available
is_positive, confidence = self.apply_classifier(morph, 'mk')

if not is_positive:
    # Remove from mask if classifier rejects
    mk_masks[mk_masks == mk_id] = 0
    continue
```

**run_unified_FAST.py:738-742 (HSPC cells):**
```python
# Apply classifier if available
is_positive, confidence = self.apply_classifier(morph, 'hspc')

if not is_positive:
    hspc_masks[hspc_masks == hspc_id] = 0
    continue
```

---

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--mk-min-area-um` | 100 | Minimum MK area in µm² |
| `--mk-max-area-um` | 2100 | Maximum MK area in µm² |
| `--sample-fraction` | 0.10 (10%) or 1.0 (100%) | Fraction of tiles to process |
| `--num-workers` | 16 | Optimal for 2 GPUs (8 per GPU) |
| `--mk-classifier` | path/to/classifier.pkl | Optional: MK classifier |
| `--hspc-classifier` | path/to/classifier.pkl | Optional: HSPC classifier |
| `--tile-size` | 4096 | Tile size for processing |
| `--calibration-samples` | 100 | Samples for SAM2 calibration |

---

## Expected Results

### 10% Sampling (Current)
- **MK cells:** ~4,500
- **HSPC cells:** ~110,000
- **Purpose:** Annotation and classifier training

### 100% Sampling (After classifier training)
- **MK cells:** ~45,000 (10x more)
- **HSPC cells:** ~1,100,000 (10x more)
- **Filtered by classifier:** Only "positive" cells kept
- **Purpose:** Final analysis dataset

---

## File Locations

### Scripts
- `/viper/ptmp2/edrod/MKsegmentation/run_unified_FAST.py` - Main segmentation
- `/viper/ptmp2/edrod/MKsegmentation/run_unified_10pct.sh` - 10% batch job
- `/viper/ptmp2/edrod/MKsegmentation/convert_annotations_to_training.py` - Annotation conversion
- `/viper/ptmp2/edrod/MKsegmentation/train_separate_classifiers.py` - Classifier training
- `/viper/ptmp2/edrod/MKsegmentation/validate_classifier.py` - Classifier validation

### Data
- `/ptmp/edrod/unified_10pct/` - 10% segmentation output
- `/ptmp/edrod/unified_100pct/` - 100% segmentation output (after training)
- `/viper/ptmp2/edrod/MKsegmentation/annotations/` - Annotation files
- `/viper/ptmp2/edrod/seg_tohtml_10pct/` - HTML for annotation

### Classifiers
- `mk_classifier_10pct.pkl` - Trained MK classifier
- `hspc_classifier_10pct.pkl` - Trained HSPC classifier

---

## Troubleshooting

### Validation fails (accuracy < 75%)
**Solution:** Collect more annotations, especially:
- False positives (labeled "Good" but should be "Bad")
- False negatives (labeled "Bad" but should be "Good")

### Low recall (< 70%)
**Problem:** Missing too many true positives

**Solution:**
- Review false negatives in confusion matrix
- Adjust class weights in `train_separate_classifiers.py`
- Collect more positive examples

### Low precision (< 70%)
**Problem:** Too many false positives

**Solution:**
- Review false positives in confusion matrix
- Use stricter features (consider morphological-only)
- Collect more negative examples

### GPU out of memory
**Solution:**
- Reduce `--num-workers` from 16 to 12 or 8
- Each worker uses ~4GB GPU memory
- Max safe: 10 workers per GPU (40GB A100)

---

## Notes

- **Pixel size:** 0.1725 µm/px
- **Conversion factor:** 0.02975625 (= 0.1725²)
- **100-2100 µm² = 3360-70573 px²**
- **GitHub repo:** `peptiderodriguez/mk_hspc_review`
- **Live site:** `https://peptiderodriguez.github.io/mk_hspc_review/`
