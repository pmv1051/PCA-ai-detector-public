"""Model evaluation utilities."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from dataset_loader import load_dataset
from detector import AIImageDetector


def evaluate_model(test_data_path, save_plots=True, plot_dir='evaluation_plots'):
    """
    Evaluate model on test dataset.

    Args:
        test_data_path: Path to test data directory
        save_plots: Whether to save evaluation plots
        plot_dir: Directory to save plots

    Returns:
        Dictionary with evaluation metrics
    """
    import os

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # Load test dataset
    print("\nLoading test dataset...")
    image_paths, y_true = load_dataset(test_data_path)
    print(f"Test images: {len(image_paths)}")
    print(f"  Real: {sum(1 for l in y_true if l == 0)}")
    print(f"  AI:   {sum(1 for l in y_true if l == 1)}")

    # Load detector
    print("\nLoading detector...")
    detector = AIImageDetector()

    # Make predictions
    print("\nMaking predictions...")
    results = detector.predict_batch(image_paths, verbose=True)

    # Extract predictions and probabilities
    y_pred = [r['label'] for r in results]
    y_proba = [r['probabilities']['ai_generated'] for r in results]

    # Calculate metrics
    print("\nCalculating metrics...")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-Score:  {f1:.2%}")
    print(f"ROC AUC:   {auc:.4f}")

    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("               Real    AI")
    print(f"Actual Real   {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       AI     {cm[1][0]:5d}  {cm[1][1]:5d}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'AI Generated']))

    # Create plots
    if save_plots:
        print(f"\nSaving plots to {plot_dir}/...")
        os.makedirs(plot_dir, exist_ok=True)

        # Confusion matrix heatmap
        plot_confusion_matrix(cm, save_path=os.path.join(plot_dir, 'confusion_matrix.png'))

        # ROC curve
        plot_roc_curve(y_true, y_proba, save_path=os.path.join(plot_dir, 'roc_curve.png'))

        # Confidence distribution
        plot_confidence_distribution(results, y_true, save_path=os.path.join(plot_dir, 'confidence_dist.png'))

        print("Plots saved successfully!")

    print("=" * 70)

    # Return metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(auc),
        'confusion_matrix': cm.tolist(),
        'n_samples': len(y_true),
        'n_correct': int(np.sum(np.array(y_true) == np.array(y_pred))),
    }

    return metrics


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Real', 'AI Generated'],
           yticklabels=['Real', 'AI Generated'],
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_proba, save_path='roc_curve.png'):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confidence_distribution(results, y_true, save_path='confidence_dist.png'):
    """Plot confidence distribution for correct and incorrect predictions."""
    confidences_correct = [r['confidence'] for r, t in zip(results, y_true) if r['label'] == t]
    confidences_incorrect = [r['confidence'] for r, t in zip(results, y_true) if r['label'] != t]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Correct predictions
    ax1.hist(confidences_correct, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Correct Predictions (n={len(confidences_correct)})')
    ax1.axvline(np.mean(confidences_correct), color='red', linestyle='--',
                label=f'Mean: {np.mean(confidences_correct):.2%}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Incorrect predictions
    if len(confidences_incorrect) > 0:
        ax2.hist(confidences_incorrect, bins=20, edgecolor='black', alpha=0.7, color='red')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Incorrect Predictions (n={len(confidences_incorrect)})')
        ax2.axvline(np.mean(confidences_incorrect), color='blue', linestyle='--',
                    label=f'Mean: {np.mean(confidences_incorrect):.2%}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No incorrect predictions!',
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Incorrect Predictions (n=0)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_errors(test_data_path, n_errors=10):
    """
    Analyze misclassified images.

    Args:
        test_data_path: Path to test data
        n_errors: Number of errors to display

    Returns:
        List of error cases
    """
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    # Load test dataset
    image_paths, y_true = load_dataset(test_data_path)

    # Load detector and predict
    detector = AIImageDetector()
    results = detector.predict_batch(image_paths, verbose=True)

    # Find errors
    errors = []
    for img_path, true_label, result in zip(image_paths, y_true, results):
        if result['label'] != true_label:
            errors.append({
                'image_path': img_path,
                'true_label': 'Real' if true_label == 0 else 'AI Generated',
                'predicted_label': result['prediction'],
                'confidence': result['confidence'],
            })

    print(f"\nTotal errors: {len(errors)}")

    if len(errors) > 0:
        print(f"\nShowing top {min(n_errors, len(errors))} errors by confidence:")
        print("-" * 70)

        # Sort by confidence (most confident mistakes first)
        errors_sorted = sorted(errors, key=lambda x: x['confidence'], reverse=True)

        for i, error in enumerate(errors_sorted[:n_errors], 1):
            print(f"\n{i}. {error['image_path']}")
            print(f"   True: {error['true_label']}")
            print(f"   Predicted: {error['predicted_label']} (confidence: {error['confidence']:.1%})")

    return errors


if __name__ == "__main__":
    import sys
    import config

    test_path = config.TEST_DIR if len(sys.argv) < 2 else sys.argv[1]

    # Run evaluation
    metrics = evaluate_model(test_path, save_plots=True)

    # Analyze errors
    errors = analyze_errors(test_path, n_errors=10)
