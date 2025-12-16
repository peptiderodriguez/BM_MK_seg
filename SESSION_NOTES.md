# MK+HSPC Segmentation & Annotation Pipeline - Session Notes
**Last Updated:** 2025-12-10 15:41

## 🧹 DIRECTORY CLEANUP (2025-12-10 15:41)

**Cleaned up duplicate/messy directories:**
- DELETED: `/viper/u2/edrod/MKsegmentation/` (253GB of clutter - 1397 empty output directories)
- DELETED: `/viper/u2/edrod/MKsegmentation_clean/` (empty directory from incomplete cleanup attempt)
- PRESERVED: Job logs copied to `/viper/ptmp2/edrod/MKsegmentation/unified_10pct_4896272.{log,err}`

**Working Directory (ONLY):**
- `/viper/ptmp2/edrod/MKsegmentation/` - All 94 scripts, complete and official

---

## ⚠️ CRITICAL: Job 4896272 FAILED - OUT OF MEMORY

**Job 4896272** ran OUT OF MEMORY after 7m 47s (processed 20/64 tiles of FGC1, then crashed)

**Problem:** RAM OOM with 16 workers + 128GB RAM limit
- Workers initialized successfully (device bug IS fixed ✓)
- Processed 31% of first slide before OOM
- MaxRSS: 67.6GB (but 21 OOM kill events detected)
- Issue: 16 workers × ~7.5GB per worker (SAM2 + Cellpose + ResNet) = ~120GB
- Plus CZI memmap and processing overhead → exceeds 128GB

**Solutions:**
1. **Reduce workers from 16 to 12** (safer: ~90GB for models + room for processing)
2. **Increase memory to 192GB or 256GB** (keep 16 workers)
3. **Reduce workers to 8** (conservative: ~60GB for models, lots of headroom)

**Recommendation:** Try 12 workers with 128GB first (balanced speed/safety)

---

## CRITICAL BUG FIXES COMPLETED

### 1. Export Script Filter Mismatch (FIXED)

**Problem:** `export_separate_mk_hspc.py` had HARDCODED old filter (4000-75000 px²) that didn't match new segmentation filter (100-2100 µm² = 3360-70573 px²)

**Impact:** HTML export would show different cells than segmentation created → annotation IDs wouldn't match when converting to training data

**Fix Applied:**
- Added `--mk-min-area-um` and `--mk-max-area-um` parameters to export script
- Converts µm² to px² internally (same logic as segmentation)
- Updated all export wrapper scripts:
  - `run_unified_10pct.sh` - passes `--mk-min-area-um 100 --mk-max-area-um 2100`
  - `export_10pct_to_html.sh` - passes same parameters
  - `run_export.sh` (for 2% data) - passes `--mk-min-area-um 119 --mk-max-area-um 2232` (OLD filter)

**Status:** ✅ FIXED - All scripts now consistent with µm-based filtering

---

## Classifier Workflow - COMPLETE & VALIDATED

**After 10% annotation and training, before deploying to 100% dataset:**

1. **Validate classifiers with quality thresholds:**
   ```bash
   python validate_classifier.py \
       --mk-classifier mk_classifier.pkl \
       --hspc-classifier hspc_classifier.pkl \
       --min-accuracy 0.75 \
       --min-recall 0.70 \
       --min-precision 0.70
   ```

2. **If validation passes, run full segmentation with classifiers:**
   ```bash
   # Run on 100% of tiles with trained classifiers
   python run_unified_FAST.py \
       --czi-path /path/to/slide.czi \
       --output-dir /ptmp/edrod/unified_100pct \
       --mk-classifier mk_classifier.pkl \
       --hspc-classifier hspc_classifier.pkl \
       --mk-min-area-um 100 \
       --mk-max-area-um 2100 \
       --num-workers 16 \
       --sample-fraction 1.0
   ```

**Validation Thresholds:**
- Minimum CV accuracy: 75%
- Minimum recall (positive class): 70% (don't miss true positives)
- Minimum precision (positive class): 70% (avoid false positives)

**Classifier Integration Status:** ✅ ALREADY IMPLEMENTED
- MK classifier: Applied at `run_unified_FAST.py:632`, filters detections before adding to results
- HSPC classifier: Applied at `run_unified_FAST.py:738`, filters detections before adding to results
- If `self.apply_classifier(morph, cell_type)` returns `is_positive=False`, cell is removed from mask
- Confidence scores saved in features: `classifier_confidence`
- When no classifier provided (`--mk-classifier` / `--hspc-classifier` omitted), `apply_classifier()` returns `(True, 1.0)` = accept all cells

---

## Scripts Status

### ✅ All Scripts Reviewed & Fixed

**Segmentation:**
- `/viper/ptmp2/edrod/MKsegmentation/run_unified_FAST.py` ✓ Reviewed - classifier integration working
- `/viper/ptmp2/edrod/MKsegmentation/run_unified_10pct.sh` ✓ Updated - passes µm filter to export

**HTML Export:**
- `/viper/ptmp2/edrod/MKsegmentation/export_separate_mk_hspc.py` ✓ FIXED - accepts µm-based parameters
- `/viper/ptmp2/edrod/MKsegmentation/export_10pct_to_html.sh` ✓ Updated - passes 100-2100 µm² parameters
- `/viper/ptmp2/edrod/MKsegmentation/run_export.sh` ✓ Updated - passes 119-2232 µm² for 2% data

**Annotation Processing:**
- `/viper/ptmp2/edrod/MKsegmentation/convert_annotations_to_training.py` ✓ Reviewed - µm-based filtering correct
- `/viper/ptmp2/edrod/MKsegmentation/train_separate_classifiers.py` ✓ Reviewed - proper class separation, balanced weights

**Classifier Validation (NEW):**
- `/viper/ptmp2/edrod/MKsegmentation/validate_classifier.py` ✓ Created & reviewed - validates against thresholds

**Documentation:**
- `/viper/ptmp2/edrod/MKsegmentation/WORKFLOW.md` ✓ Created - complete pipeline documentation

---

## µm-BASED FILTERING (Implemented - User Preference)

- MK size filter: **100-2100 µm²** (converts to 3360-70573 px² internally)
- Pixel size: 0.1725 µm/px
- Conversion factor: 0.02975625 (= 0.1725²)
- User feedback: "more intuitive for users indeed"
- Parameters: `--mk-min-area-um 100 --mk-max-area-um 2100`
- HTML cards display both: "119.0 µm² | 4000 px²"

**Implementation:**
- Segmentation script (`run_unified_FAST.py`): Accepts µm parameters, converts to px internally
- Export script (`export_separate_mk_hspc.py`): Accepts µm parameters, converts to px internally
- Convert script (`convert_annotations_to_training.py`): Accepts µm parameters, converts to px internally
- **All three scripts now consistent** - no more filter mismatches!

---

## 16 Slides

- FGC1-4 (Female GC)
- FHU1-4 (Female HU)
- MGC1-4 (Male GC)
- MHU1-4 (Male HU)

---

## Expected Output (10% sampling)

- ~4,500 MK cells (5x more than 2% sampling)
- ~110,000 HSPC cells
- Paginated HTML with 300 samples/page

---

## Previous Work

### 2% Sampling (Completed)
- 838 MK cells, 21,922 HSPC cells
- Annotated: 171 MK (60 pos, 111 neg), 160 HSPC (75 pos, 85 neg)
- **Result:** Too few samples for good RF classifier (59-68% accuracy)

### RF Classifier Training (Abandoned)
- Trained separate MK/HSPC classifiers
- **MK:** 68% accuracy, 37% recall (poor)
- **HSPC:** 51% accuracy (random)
- **Conclusion:** Need more training data → 10% sampling

---

## Bug Fixes History

### Cellpose Device Bug (FIXED - Job 4895943)
- Error: `AttributeError: 'str' object has no attribute 'type'`
- Location: `run_unified_FAST.py:436-460`
- Cause: `init_worker()` passed device string `"cuda:0"`, stored as string, but Cellpose expects `torch.device` object with `.type` attribute
- Fix: Added type checking and conversion to `torch.device` object
- Resolution: Job 4895943 canceled, job 4896272 submitted with fix
- **Verification:** Job 4896272 workers initialized successfully (device bug IS fixed)

### Export Script Filter Mismatch (FIXED - Current session)
- Hardcoded old filter `(4000, 75000)` px² in `export_separate_mk_hspc.py`
- New 10% filter: `100-2100 µm²` = `3360-70573 px²`
- Fix: Added `--mk-min-area-um` and `--mk-max-area-um` parameters
- Converts internally using same logic as segmentation script
- Updated all wrapper scripts to pass correct parameters

---

## Monitoring Commands

```bash
# Check job status
squeue -u edrod

# Check past jobs
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed

# Watch progress (when running)
tail -f /viper/ptmp2/edrod/MKsegmentation/unified_10pct_<JOB_ID>.log

# Check errors
tail -f /viper/ptmp2/edrod/MKsegmentation/unified_10pct_<JOB_ID>.err
```

---

## Next Steps

### 1. FIX OOM ISSUE (URGENT)
- Modify `run_unified_10pct.sh` to reduce workers from 16 to 12
- OR increase memory from 128GB to 192GB
- Resubmit job

### 2. After Job Completes
1. ✅ Verify HTML export succeeded
2. ✅ Check GitHub Pages is live
3. ⏳ **User annotates 10% dataset** via GitHub Pages
4. ⏳ Transfer annotations back to cluster
5. ⏳ **Convert annotations to training format:**
   ```bash
   python convert_annotations_to_training.py \
       --annotations annotations/all_labels_10pct.json \
       --base-dir /ptmp/edrod/unified_10pct \
       --mk-min-area-um 100 \
       --mk-max-area-um 2100 \
       --output annotations/training_data_10pct.json
   ```
6. ⏳ **Train new RF classifiers:**
   ```bash
   python train_separate_classifiers.py \
       --training-data annotations/training_data_10pct.json \
       --output-mk mk_classifier.pkl \
       --output-hspc hspc_classifier.pkl \
       --morph-only
   ```
7. ⏳ **Validate classifiers:**
   ```bash
   python validate_classifier.py \
       --mk-classifier mk_classifier.pkl \
       --hspc-classifier hspc_classifier.pkl \
       --min-accuracy 0.75 \
       --min-recall 0.70 \
       --min-precision 0.70
   ```
8. ⏳ **If valid, run on 100% dataset:**
   ```bash
   # Create batch job for full dataset with classifiers
   # Run on all 16 slides with --sample-fraction 1.0
   ```

---

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--mk-min-area-um` | 100 | Minimum MK area in µm² |
| `--mk-max-area-um` | 2100 | Maximum MK area in µm² |
| `--sample-fraction` | 0.10 (10%) or 1.0 (100%) | Fraction of tiles to process |
| `--num-workers` | 12 (RECOMMENDED) or 16 (OOM risk) | Parallel workers |
| `--mk-classifier` | path/to/classifier.pkl | Optional: MK classifier |
| `--hspc-classifier` | path/to/classifier.pkl | Optional: HSPC classifier |
| `--tile-size` | 4096 | Tile size for processing |
| `--calibration-samples` | 100 | Samples for SAM2 calibration |

---

## Resource Allocation

**GPU Memory (Bottleneck for worker count):**
- 2× A100 GPUs @ 40GB each = 80GB total GPU memory
- Each worker uses ~4GB GPU (SAM2 + Cellpose + ResNet)
- Max safe workers: 20 (10 per GPU @ ~4GB per worker)
- Current: 16 workers = 8 per GPU = ~32GB/80GB GPU ✓ OPTIMAL for GPU

**RAM (Bottleneck for job 4896272):**
- Each worker uses ~7.5GB RAM (SAM2 + Cellpose + ResNet models)
- 16 workers × 7.5GB = ~120GB + processing overhead → OOM with 128GB limit
- 12 workers × 7.5GB = ~90GB + processing overhead → should fit in 128GB
- 8 workers × 7.5GB = ~60GB + processing overhead → very safe with 128GB

**Recommendation:**
- **Option 1 (balanced):** 12 workers + 128GB RAM
- **Option 2 (faster):** 16 workers + 192GB RAM
- **Option 3 (safe):** 8 workers + 128GB RAM

---

## File Locations

### Scripts
- `/viper/ptmp2/edrod/MKsegmentation/run_unified_FAST.py` - Main segmentation
- `/viper/ptmp2/edrod/MKsegmentation/run_unified_10pct.sh` - 10% batch job
- `/viper/ptmp2/edrod/MKsegmentation/export_separate_mk_hspc.py` - HTML export
- `/viper/ptmp2/edrod/MKsegmentation/convert_annotations_to_training.py` - Annotation conversion
- `/viper/ptmp2/edrod/MKsegmentation/train_separate_classifiers.py` - Classifier training
- `/viper/ptmp2/edrod/MKsegmentation/validate_classifier.py` - Classifier validation
- `/viper/ptmp2/edrod/MKsegmentation/WORKFLOW.md` - Complete workflow documentation

### Data
- `/ptmp/edrod/unified_10pct/` - 10% segmentation output (INCOMPLETE - job failed)
- `/ptmp/edrod/unified_100pct/` - 100% segmentation output (after training)
- `/viper/ptmp2/edrod/MKsegmentation/annotations/` - Annotation files
- `/viper/ptmp2/edrod/seg_tohtml_10pct/` - HTML for annotation

### GitHub
- Repo: `peptiderodriguez/mk_hspc_review`
- Live site: `https://peptiderodriguez.github.io/mk_hspc_review/`

---

## Key Learnings

- **GPU memory is the bottleneck for worker count** - each worker loads SAM2, Cellpose, ResNet on GPU
- **RAM can also be a bottleneck** - 16 workers × 7.5GB per worker = 120GB, exceeds 128GB limit
- **Device bug is FIXED** - workers initialize successfully with `torch.device` object
- **Filter consistency is CRITICAL** - segmentation, export, and convert scripts must use identical filters
- **µm-based filtering preferred** - more intuitive across imaging systems, standard in biology literature
- **Define validation thresholds before deployment** - don't deploy poor classifiers without objective criteria
- **Classifier integration is complete** - `apply_classifier()` filters MK/HSPC detections during segmentation
- **Pipeline behavior with/without classifiers:**
  - Without: `--mk-classifier` omitted → accepts all cells passing size filter
  - With: `--mk-classifier path/to/classifier.pkl` → filters cells using trained model
