"""
Train separate MK and HSPC classifiers from annotations.

Usage:
    python train_separate_classifiers.py \
        --training-data /viper/ptmp2/edrod/MKsegmentation/annotations/training_data.json \
        --output-mk /viper/ptmp2/edrod/MKsegmentation/mk_classifier.pkl \
        --output-hspc /viper/ptmp2/edrod/MKsegmentation/hspc_classifier.pkl
"""

import numpy as np
import json
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import argparse


def train_classifier(X, y, sample_ids, feature_names, output_path, cell_type_name):
    """Train a single classifier."""

    print(f"\n{'='*80}")
    print(f"TRAINING {cell_type_name.upper()} CLASSIFIER")
    print(f"{'='*80}")

    print(f"\nDataset:")
    print(f"  Total samples: {len(X)}")
    print(f"  Positive: {sum(y)}")
    print(f"  Negative: {len(y) - sum(y)}")
    print(f"  Features: {len(feature_names)}")

    if len(X) < 10:
        print(f"\nERROR: Need at least 10 samples, only have {len(X)}")
        return None

    # Convert to numpy array
    X_np = np.array(X)
    y_np = np.array(y)

    # Calculate class weights
    n_true = sum(y_np)
    n_false = len(y_np) - n_true

    if n_true > 0 and n_false > 0:
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_np),
            y=y_np
        )
        class_weight = {0: class_weights_array[0], 1: class_weights_array[1]}
        print(f"\n  Class weights: {class_weight}")
    else:
        class_weight = 'balanced'
        print("\n  WARNING: Only one class present!")

    # Train classifier
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation
    n_splits = min(5, min(int(sum(y_np)), int(len(y_np) - sum(y_np))))
    if n_splits >= 2:
        print(f"\nRunning {n_splits}-fold cross-validation...")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_np, y_np, cv=cv, scoring='accuracy')
        print(f"  CV Accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

        y_pred_cv = cross_val_predict(clf, X_np, y_np, cv=cv)
        print(f"\nCross-validation report:")
        print(classification_report(y_np, y_pred_cv, target_names=['Negative', 'Positive']))
        print(f"Confusion matrix:\n{confusion_matrix(y_np, y_pred_cv)}")
    else:
        print(f"\nSkipping CV (need at least 2 samples per class)")
        cv_scores = np.array([0.0])

    # Train on full dataset
    print("\nTraining on full dataset...")
    clf.fit(X_np, y_np)

    # Feature importance
    print("\nTop 15 most important features:")
    importance = sorted(
        zip(feature_names, clf.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    for i, (name, imp) in enumerate(importance[:15], 1):
        print(f"  {i:2d}. {name}: {imp:.4f}")

    # Save classifier
    classifier_data = {
        'classifier': clf,
        'feature_names': feature_names,
        'X_train': X_np,
        'y_train': y_np,
        'sample_ids': sample_ids,
        'training_samples': len(X_np),
        'cv_accuracy': float(cv_scores.mean()),
        'class_weight': class_weight,
        'cell_type': cell_type_name
    }

    joblib.dump(classifier_data, output_path)
    print(f"\nSaved classifier to: {output_path}")

    return clf


def main():
    parser = argparse.ArgumentParser(
        description='Train separate MK and HSPC classifiers'
    )
    parser.add_argument('--training-data', type=str, required=True,
                       help='Path to training_data.json')
    parser.add_argument('--output-mk', type=str, required=True,
                       help='Output path for MK classifier')
    parser.add_argument('--output-hspc', type=str, required=True,
                       help='Output path for HSPC classifier')
    parser.add_argument('--morph-only', action='store_true',
                       help='Use only morphological features (exclude ResNet/SAM2)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("LOADING TRAINING DATA")
    print("="*80)

    # Load training data
    with open(args.training_data, 'r') as f:
        data = json.load(f)

    X_all = data['X']
    y_all = data['y']
    sample_ids_all = data['sample_ids']
    training_samples = data['training_samples']

    print(f"\nTotal samples loaded: {len(X_all)}")

    # Split by cell type
    mk_indices = [i for i, s in enumerate(training_samples) if s['cell_type'] == 'mk']
    hspc_indices = [i for i, s in enumerate(training_samples) if s['cell_type'] == 'hspc']

    print(f"  MK samples: {len(mk_indices)}")
    print(f"  HSPC samples: {len(hspc_indices)}")

    # Get feature names
    all_feature_names = sorted(X_all[0].keys())

    if args.morph_only:
        feature_names = [f for f in all_feature_names
                        if not f.startswith('sam2_') and not f.startswith('resnet_')]
        print(f"\nUsing {len(feature_names)} morphological features (excluding embeddings)")
    else:
        feature_names = all_feature_names
        print(f"\nUsing all {len(feature_names)} features")

    # Convert feature dicts to arrays
    def extract_features(indices):
        X = []
        y = []
        ids = []
        for i in indices:
            feat_dict = X_all[i]
            X.append([feat_dict.get(name, 0.0) for name in feature_names])
            y.append(y_all[i])
            ids.append(sample_ids_all[i])
        return X, y, ids

    # Train MK classifier
    X_mk, y_mk, ids_mk = extract_features(mk_indices)
    train_classifier(X_mk, y_mk, ids_mk, feature_names, args.output_mk, "MK")

    # Train HSPC classifier
    X_hspc, y_hspc, ids_hspc = extract_features(hspc_indices)
    train_classifier(X_hspc, y_hspc, ids_hspc, feature_names, args.output_hspc, "HSPC")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
