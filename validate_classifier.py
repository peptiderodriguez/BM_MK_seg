"""
Validate trained classifiers before deployment.

This script checks if trained classifiers meet minimum quality thresholds
before using them for full-scale segmentation.

Usage:
    python validate_classifier.py \
        --mk-classifier /viper/ptmp2/edrod/MKsegmentation/mk_classifier.pkl \
        --hspc-classifier /viper/ptmp2/edrod/MKsegmentation/hspc_classifier.pkl \
        --min-accuracy 0.75 \
        --min-recall 0.70 \
        --min-precision 0.70
"""

import argparse
import joblib
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score


def validate_classifier(clf_path, cell_type, min_accuracy=0.75, min_recall=0.70, min_precision=0.70):
    """
    Validate a trained classifier against quality thresholds.

    Returns:
        (passes, metrics_dict)
    """
    print(f"\n{'='*80}")
    print(f"VALIDATING {cell_type.upper()} CLASSIFIER")
    print(f"{'='*80}")

    # Load classifier
    print(f"\nLoading classifier: {clf_path}")
    clf_data = joblib.load(clf_path)

    clf = clf_data['classifier']
    X_train = clf_data['X_train']
    y_train = clf_data['y_train']
    cv_accuracy = clf_data.get('cv_accuracy', 0.0)

    print(f"Training samples: {len(X_train)}")
    print(f"  Positive: {sum(y_train)}")
    print(f"  Negative: {len(y_train) - sum(y_train)}")
    print(f"Stored CV accuracy: {cv_accuracy:.3f}")

    # Re-run cross-validation for detailed metrics
    n_splits = min(5, min(int(sum(y_train)), int(len(y_train) - sum(y_train))))

    if n_splits < 2:
        print("\nERROR: Not enough samples for cross-validation!")
        return False, {}

    print(f"\nRunning {n_splits}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_train, y_train, cv=cv)

    # Calculate metrics
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_train, y_pred, pos_label=1, zero_division=0)

    print(f"\n{'='*50}")
    print("CROSS-VALIDATION METRICS")
    print('='*50)
    print(f"  Accuracy:  {accuracy:.3f}  (min: {min_accuracy:.3f})")
    print(f"  Precision: {precision:.3f}  (min: {min_precision:.3f})")
    print(f"  Recall:    {recall:.3f}  (min: {min_recall:.3f})")
    print('='*50)

    print(f"\nDetailed classification report:")
    print(classification_report(y_train, y_pred, target_names=['Negative', 'Positive']))

    print(f"\nConfusion matrix:")
    cm = confusion_matrix(y_train, y_pred)
    print(f"              Predicted")
    print(f"              Neg    Pos")
    print(f"Actual Neg  [{cm[0,0]:4d}  {cm[0,1]:4d}]")
    print(f"       Pos  [{cm[1,0]:4d}  {cm[1,1]:4d}]")

    # Check thresholds
    passes_accuracy = accuracy >= min_accuracy
    passes_precision = precision >= min_precision
    passes_recall = recall >= min_recall

    print(f"\n{'='*50}")
    print("THRESHOLD CHECKS")
    print('='*50)
    print(f"  Accuracy:  {'✓ PASS' if passes_accuracy else '✗ FAIL'}")
    print(f"  Precision: {'✓ PASS' if passes_precision else '✗ FAIL'}")
    print(f"  Recall:    {'✓ PASS' if passes_recall else '✗ FAIL'}")
    print('='*50)

    passes_all = passes_accuracy and passes_precision and passes_recall

    if passes_all:
        print(f"\n✓ {cell_type.upper()} classifier PASSES validation!")
    else:
        print(f"\n✗ {cell_type.upper()} classifier FAILS validation!")
        print("\nRecommendations:")
        if not passes_accuracy:
            print("  - Collect more training samples")
            print("  - Review annotation quality")
        if not passes_recall:
            print("  - Model is missing too many true positives")
            print("  - Consider adjusting class weights")
            print("  - Review false negatives in training data")
        if not passes_precision:
            print("  - Model has too many false positives")
            print("  - Consider stricter classification threshold")
            print("  - Review false positives in training data")

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'n_samples': len(X_train),
        'n_positive': int(sum(y_train)),
        'n_negative': int(len(y_train) - sum(y_train))
    }

    return passes_all, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Validate trained classifiers before deployment'
    )
    parser.add_argument('--mk-classifier', type=str,
                       help='Path to trained MK classifier')
    parser.add_argument('--hspc-classifier', type=str,
                       help='Path to trained HSPC classifier')
    parser.add_argument('--min-accuracy', type=float, default=0.75,
                       help='Minimum cross-validation accuracy (default: 0.75)')
    parser.add_argument('--min-recall', type=float, default=0.70,
                       help='Minimum recall for positive class (default: 0.70)')
    parser.add_argument('--min-precision', type=float, default=0.70,
                       help='Minimum precision for positive class (default: 0.70)')

    args = parser.parse_args()

    if not args.mk_classifier and not args.hspc_classifier:
        parser.error("Must provide at least one of --mk-classifier or --hspc-classifier")

    results = {}
    all_pass = True

    if args.mk_classifier:
        passes, metrics = validate_classifier(
            args.mk_classifier, 'MK',
            args.min_accuracy, args.min_recall, args.min_precision
        )
        results['mk'] = {'passes': passes, 'metrics': metrics}
        all_pass = all_pass and passes

    if args.hspc_classifier:
        passes, metrics = validate_classifier(
            args.hspc_classifier, 'HSPC',
            args.min_accuracy, args.min_recall, args.min_precision
        )
        results['hspc'] = {'passes': passes, 'metrics': metrics}
        all_pass = all_pass and passes

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL VALIDATION SUMMARY")
    print('='*80)

    for cell_type, data in results.items():
        status = "✓ PASS" if data['passes'] else "✗ FAIL"
        m = data['metrics']
        print(f"\n{cell_type.upper()}: {status}")
        print(f"  Accuracy: {m['accuracy']:.3f}, Precision: {m['precision']:.3f}, Recall: {m['recall']:.3f}")
        print(f"  Training samples: {m['n_samples']} ({m['n_positive']} pos, {m['n_negative']} neg)")

    if all_pass:
        print(f"\n{'='*80}")
        print("✓ ALL CLASSIFIERS PASS VALIDATION")
        print("Ready to run on full dataset!")
        print('='*80)
        print("\nTo run segmentation with classifiers:")
        print("  python run_unified_FAST.py \\")
        print("      --czi-path /path/to/slide.czi \\")
        print("      --output-dir /path/to/output \\")
        if args.mk_classifier:
            print(f"      --mk-classifier {args.mk_classifier} \\")
        if args.hspc_classifier:
            print(f"      --hspc-classifier {args.hspc_classifier} \\")
        print("      --mk-min-area-um 100 \\")
        print("      --mk-max-area-um 2100 \\")
        print("      --num-workers 16")
        return 0
    else:
        print(f"\n{'='*80}")
        print("✗ VALIDATION FAILED")
        print("Do not use these classifiers on full dataset!")
        print("Collect more annotations and retrain.")
        print('='*80)
        return 1


if __name__ == "__main__":
    exit(main())
