"""AI image detection using trained PCA model."""

import numpy as np
from feature_extractor import extract_features, extract_features_batch
from pca_model import load_model


class AIImageDetector:
    """
    AI image detector using PCA and classifier.

    Usage:
        detector = AIImageDetector()
        result = detector.predict('path/to/image.jpg')
        print(result['prediction'], result['confidence'])
    """

    def __init__(self, model_path=None):
        """
        Initialize detector with pre-trained model.

        Args:
            model_path: Optional path to model directory
                       If None, uses default paths from config
        """
        print("Loading AI image detector...")

        # Load pre-trained models
        self.pca, self.scaler, self.classifier, self.metadata = load_model()

        print(f"Model loaded successfully!")
        print(f"  Classifier: {self.metadata.get('classifier_type', 'Unknown')}")
        print(f"  Accuracy: {self.metadata.get('val_accuracy', 0):.2%}")

    def _preprocess_features(self, features):
        """
        Preprocess features: scale and apply PCA.

        Args:
            features: Raw feature vector or matrix

        Returns:
            PCA-transformed features
        """
        # Handle single sample vs batch
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Clean features
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Apply PCA
        features_pca = self.pca.transform(features_scaled)

        return features_pca

    def predict(self, image_path):
        """
        Predict whether an image is AI-generated or real.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with:
                - prediction: 'AI Generated' or 'Real'
                - label: 1 for AI, 0 for Real
                - confidence: probability (0-1)
                - probabilities: [prob_real, prob_ai]
        """
        # Extract features
        features = extract_features(image_path)

        # Preprocess
        features_pca = self._preprocess_features(features)

        # Predict
        prediction_label = self.classifier.predict(features_pca)[0]

        # Get probabilities
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(features_pca)[0]
            confidence = probabilities[prediction_label]
        else:
            # SVM decision function as confidence
            decision = self.classifier.decision_function(features_pca)[0]
            probabilities = [1 - decision, decision] if decision > 0 else [abs(decision), 0]
            confidence = max(probabilities)

        # Format result
        prediction_text = 'AI Generated' if prediction_label == 1 else 'Real'

        result = {
            'prediction': prediction_text,
            'label': int(prediction_label),
            'confidence': float(confidence),
            'probabilities': {
                'real': float(probabilities[0]),
                'ai_generated': float(probabilities[1])
            }
        }

        return result

    def predict_batch(self, image_paths, verbose=True):
        """
        Classify multiple images efficiently.

        Args:
            image_paths: List of image file paths
            verbose: Show progress bar

        Returns:
            List of prediction dictionaries
        """
        # Extract features for all images
        features_batch = extract_features_batch(image_paths, verbose=verbose)

        # Preprocess
        features_pca = self._preprocess_features(features_batch)

        # Predict
        predictions = self.classifier.predict(features_pca)

        # Get probabilities
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(features_pca)
        else:
            # Fallback for classifiers without predict_proba
            probabilities = np.zeros((len(predictions), 2))
            probabilities[range(len(predictions)), predictions] = 1.0

        # Format results
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            prediction_text = 'AI Generated' if pred == 1 else 'Real'
            confidence = probs[pred]

            result = {
                'image_path': image_paths[i],
                'prediction': prediction_text,
                'label': int(pred),
                'confidence': float(confidence),
                'probabilities': {
                    'real': float(probs[0]),
                    'ai_generated': float(probs[1])
                }
            }
            results.append(result)

        return results

    def get_model_info(self):
        """Get information about the loaded model."""
        return self.metadata


def detect_image(image_path, show_details=True):
    """
    Standalone function to detect if an image is AI-generated.

    Args:
        image_path: Path to image file
        show_details: Print detailed results

    Returns:
        Prediction dictionary
    """
    detector = AIImageDetector()
    result = detector.predict(image_path)

    if show_details:
        print("\n" + "=" * 70)
        print(f"Image: {image_path}")
        print("=" * 70)
        print(f"Prediction:  {result['prediction']}")
        print(f"Confidence:  {result['confidence']:.1%}")
        print(f"\nProbabilities:")
        print(f"  Real:         {result['probabilities']['real']:.1%}")
        print(f"  AI Generated: {result['probabilities']['ai_generated']:.1%}")
        print("=" * 70)

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detector.py <image_path>")
        print("Example: python detector.py test_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    detect_image(image_path, show_details=True)
