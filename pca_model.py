"""PCA model training and management."""

import numpy as np
import joblib
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import config
from feature_extractor import extract_features_batch
from dataset_loader import load_dataset


def create_classifier(classifier_type=None):
    """
    Create a classifier instance.

    Args:
        classifier_type: 'logistic', 'svm', or 'random_forest'
                        Defaults to config.CLASSIFIER_TYPE

    Returns:
        Untrained classifier instance
    """
    if classifier_type is None:
        classifier_type = config.CLASSIFIER_TYPE

    if classifier_type == 'logistic':
        return LogisticRegression(
            max_iter=config.LR_MAX_ITER,
            C=config.LR_C,
            random_state=config.RANDOM_STATE,
            verbose=1
        )
    elif classifier_type == 'svm':
        return SVC(
            kernel=config.SVM_KERNEL,
            C=config.SVM_C,
            gamma=config.SVM_GAMMA,
            random_state=config.RANDOM_STATE,
            probability=True,  # Enable probability estimates
            verbose=True
        )
    elif classifier_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            random_state=config.RANDOM_STATE,
            verbose=1
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


def train_pca_detector(train_data_path, n_components=None, classifier_type=None):
    """
    Train PCA + classifier on training dataset.

    Args:
        train_data_path: Path to training data directory
        n_components: Number of PCA components (None = auto based on variance)
        classifier_type: Type of classifier to use

    Returns:
        Tuple of (pca_model, scaler, classifier, metadata)
    """
    print("\n" + "=" * 70)
    print("TRAINING PCA-BASED AI IMAGE DETECTOR")
    print("=" * 70)

    # Load dataset
    print("\nStep 1: Loading dataset...")
    image_paths, labels = load_dataset(train_data_path)
    print(f"Loaded {len(image_paths)} images")
    print(f"  Real images: {sum(1 for l in labels if l == 0)}")
    print(f"  AI images:   {sum(1 for l in labels if l == 1)}")

    # Extract features
    print("\nStep 2: Extracting features...")
    X = extract_features_batch(image_paths, verbose=True)
    y = np.array(labels)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Features per image: {X.shape[1]}")

    # Check for NaN or infinite values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("Warning: Found NaN or infinite values in features. Cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Split data
    print("\nStep 3: Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")

    # Normalize features
    print("\nStep 4: Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    print("Features normalized (zero mean, unit variance)")

    # Apply PCA
    print("\nStep 5: Applying PCA...")
    if n_components is None:
        # Auto-determine based on variance threshold
        pca = PCA(n_components=config.PCA_VARIANCE_THRESHOLD, random_state=config.RANDOM_STATE)
    else:
        pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    n_components_used = pca.n_components_
    variance_explained = np.sum(pca.explained_variance_ratio_)

    print(f"PCA components: {n_components_used}")
    print(f"Variance explained: {variance_explained:.2%}")
    print(f"Reduced dimension: {X.shape[1]} -> {n_components_used}")

    # Train classifier
    print(f"\nStep 6: Training {classifier_type or config.CLASSIFIER_TYPE} classifier...")
    classifier = create_classifier(classifier_type)
    classifier.fit(X_train_pca, y_train)

    # Evaluate on validation set
    print("\nStep 7: Evaluating on validation set...")
    train_score = classifier.score(X_train_pca, y_train)
    val_score = classifier.score(X_val_pca, y_val)

    print(f"Training accuracy:   {train_score:.2%}")
    print(f"Validation accuracy: {val_score:.2%}")

    # Create metadata
    metadata = {
        'n_components': int(n_components_used),
        'variance_explained': float(variance_explained),
        'n_features': int(X.shape[1]),
        'classifier_type': classifier_type or config.CLASSIFIER_TYPE,
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'train_accuracy': float(train_score),
        'val_accuracy': float(val_score),
    }

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return pca, scaler, classifier, metadata


def save_model(pca_model, scaler, classifier, metadata):
    """
    Save trained models and metadata to disk.

    Args:
        pca_model: Trained PCA model
        scaler: Fitted StandardScaler
        classifier: Trained classifier
        metadata: Dictionary with model information
    """
    print("\nSaving models...")

    # Save PCA model
    joblib.dump(pca_model, config.PCA_MODEL_FILE)
    print(f"  PCA model saved to: {config.PCA_MODEL_FILE}")

    # Save scaler
    joblib.dump(scaler, config.SCALER_FILE)
    print(f"  Scaler saved to: {config.SCALER_FILE}")

    # Save classifier
    joblib.dump(classifier, config.CLASSIFIER_MODEL_FILE)
    print(f"  Classifier saved to: {config.CLASSIFIER_MODEL_FILE}")

    # Save metadata
    with open(config.METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to: {config.METADATA_FILE}")

    print("\nAll models saved successfully!")


def load_model():
    """
    Load pre-trained models from disk.

    Returns:
        Tuple of (pca_model, scaler, classifier, metadata)
    """
    import os

    # Check if all files exist
    required_files = [
        config.PCA_MODEL_FILE,
        config.SCALER_FILE,
        config.CLASSIFIER_MODEL_FILE,
        config.METADATA_FILE
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

    # Load models
    pca_model = joblib.load(config.PCA_MODEL_FILE)
    scaler = joblib.load(config.SCALER_FILE)
    classifier = joblib.load(config.CLASSIFIER_MODEL_FILE)

    # Load metadata
    with open(config.METADATA_FILE, 'r') as f:
        metadata = json.load(f)

    return pca_model, scaler, classifier, metadata


def print_model_info():
    """Print information about the loaded model."""
    try:
        _, _, _, metadata = load_model()

        print("\n" + "=" * 70)
        print("MODEL INFORMATION")
        print("=" * 70)
        print(f"Classifier type:     {metadata.get('classifier_type', 'Unknown')}")
        print(f"PCA components:      {metadata.get('n_components', 'Unknown')}")
        print(f"Variance explained:  {metadata.get('variance_explained', 0):.2%}")
        print(f"Original features:   {metadata.get('n_features', 'Unknown')}")
        print(f"Training samples:    {metadata.get('train_samples', 'Unknown')}")
        print(f"Training accuracy:   {metadata.get('train_accuracy', 0):.2%}")
        print(f"Validation accuracy: {metadata.get('val_accuracy', 0):.2%}")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("No trained model found. Please train a model first.")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "info":
        print_model_info()
    else:
        print("Usage:")
        print("  python pca_model.py info  - Display model information")
