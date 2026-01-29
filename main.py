"""
AI Image Detection using PCA

Command line interface for training and using PCA based AI image detector.
"""

import argparse
import sys
import os
import config
from dataset_loader import (
    print_dataset_info, setup_data_directories, download_sample_dataset,
    download_cifake_dataset
)
from pca_model import train_pca_detector, save_model, print_model_info
from detector import AIImageDetector, detect_image
from evaluate import evaluate_model, analyze_errors


def cmd_setup(args):
    """Setup data directories."""
    setup_data_directories()
    if args.download_info:
        download_sample_dataset()


def cmd_download(args):
    """Download CIFAKE dataset from Kaggle."""
    download_cifake_dataset()


def cmd_train(args):
    """Train PCA model."""
    # Check if training data exists
    if not os.path.exists(args.data_path):
        print(f"Error: Training data directory not found: {args.data_path}")
        print("\nPlease run 'python main.py setup' to create directories.")
        return

    # Print dataset info
    if not print_dataset_info(args.data_path):
        print("\nNo training data found. Please add images to the data directory.")
        return

    # Train model
    try:
        pca, scaler, classifier, metadata = train_pca_detector(
            train_data_path=args.data_path,
            n_components=args.n_components,
            classifier_type=args.classifier
        )

        # Save model
        save_model(pca, scaler, classifier, metadata)

        print("\nModel training complete!")
        print(f"Validation accuracy: {metadata['val_accuracy']:.2%}")

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


def cmd_predict(args):
    """Predict on single image or batch."""
    try:
        # Check if model exists
        if not os.path.exists(config.PCA_MODEL_FILE):
            print("Error: No trained model found.")
            print("Please train a model first using: python main.py train")
            return

        # Single image
        if args.image:
            if not os.path.exists(args.image):
                print(f"Error: Image not found: {args.image}")
                return

            result = detect_image(args.image, show_details=True)

            # Additional info if verbose
            if args.verbose:
                detector = AIImageDetector()
                info = detector.get_model_info()
                print(f"\nModel info:")
                print(f"  Classifier: {info.get('classifier_type', 'Unknown')}")
                print(f"  PCA components: {info.get('n_components', 'Unknown')}")

        # Batch of images
        elif args.batch:
            if not os.path.exists(args.batch):
                print(f"Error: Directory not found: {args.batch}")
                return

            from dataset_loader import get_image_paths
            image_paths = get_image_paths(args.batch)

            if len(image_paths) == 0:
                print(f"No images found in {args.batch}")
                return

            print(f"\nProcessing {len(image_paths)} images...")

            detector = AIImageDetector()
            results = detector.predict_batch(image_paths, verbose=True)

            # Print summary
            print("\n" + "=" * 70)
            print("BATCH PREDICTION RESULTS")
            print("=" * 70)

            ai_count = sum(1 for r in results if r['label'] == 1)
            real_count = len(results) - ai_count

            print(f"Total images: {len(results)}")
            print(f"  AI Generated: {ai_count} ({ai_count/len(results):.1%})")
            print(f"  Real:         {real_count} ({real_count/len(results):.1%})")

            # Show individual results if not too many
            if args.verbose or len(results) <= 20:
                print("\nIndividual results:")
                for r in results:
                    filename = os.path.basename(r['image_path'])
                    print(f"  {filename:40s} -> {r['prediction']:15s} ({r['confidence']:.1%})")

        else:
            print("Error: Please specify --image or --batch")

    except Exception as e:
        print(f"\nError during prediction: {e}")
        import traceback
        traceback.print_exc()


def cmd_evaluate(args):
    """Evaluate model on test set."""
    try:
        # Check if model exists
        if not os.path.exists(config.PCA_MODEL_FILE):
            print("Error: No trained model found.")
            print("Please train a model first using: python main.py train")
            return

        # Check if test data exists
        if not os.path.exists(args.data_path):
            print(f"Error: Test data directory not found: {args.data_path}")
            return

        # Print test dataset info
        if not print_dataset_info(args.data_path):
            print("\nNo test data found.")
            return

        # Evaluate
        metrics = evaluate_model(
            test_data_path=args.data_path,
            save_plots=args.save_plots,
            plot_dir=args.plot_dir
        )

        # Analyze errors if requested
        if args.analyze_errors:
            analyze_errors(args.data_path, n_errors=args.n_errors)

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()


def cmd_info(args):
    """Display model information."""
    print_model_info()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='AI Image Detection using PCA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup data directories
  python main.py setup --download-info

  # Download CIFAKE dataset from Kaggle
  python main.py download

  # Train model
  python main.py train --data-path ./data/train

  # Predict single image
  python main.py predict --image photo.jpg

  # Predict batch of images
  python main.py predict --batch ./test_images/

  # Evaluate on test set
  python main.py evaluate --data-path ./data/test

  # Show model info
  python main.py info
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Setup command
    parser_setup = subparsers.add_parser('setup', help='Setup data directories')
    parser_setup.add_argument('--download-info', action='store_true',
                            help='Show dataset download instructions')
    parser_setup.set_defaults(func=cmd_setup)

    # Download command
    parser_download = subparsers.add_parser('download', help='Download CIFAKE dataset from Kaggle')
    parser_download.set_defaults(func=cmd_download)

    # Train command
    parser_train = subparsers.add_parser('train', help='Train PCA model')
    parser_train.add_argument('--data-path', default=config.TRAIN_DIR,
                            help=f'Path to training data (default: {config.TRAIN_DIR})')
    parser_train.add_argument('--n-components', type=int, default=None,
                            help='Number of PCA components (default: auto)')
    parser_train.add_argument('--classifier', choices=['logistic', 'svm', 'random_forest'],
                            default=config.CLASSIFIER_TYPE,
                            help=f'Classifier type (default: {config.CLASSIFIER_TYPE})')
    parser_train.set_defaults(func=cmd_train)

    # Predict command
    parser_predict = subparsers.add_parser('predict', help='Predict on images')
    group = parser_predict.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', help='Path to single image')
    group.add_argument('--batch', help='Path to directory of images')
    parser_predict.add_argument('--verbose', action='store_true',
                              help='Show detailed output')
    parser_predict.set_defaults(func=cmd_predict)

    # Evaluate command
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate model on test set')
    parser_eval.add_argument('--data-path', default=config.TEST_DIR,
                           help=f'Path to test data (default: {config.TEST_DIR})')
    parser_eval.add_argument('--save-plots', action='store_true', default=True,
                           help='Save evaluation plots')
    parser_eval.add_argument('--plot-dir', default='evaluation_plots',
                           help='Directory to save plots')
    parser_eval.add_argument('--analyze-errors', action='store_true',
                           help='Analyze misclassified images')
    parser_eval.add_argument('--n-errors', type=int, default=10,
                           help='Number of errors to display')
    parser_eval.set_defaults(func=cmd_evaluate)

    # Info command
    parser_info = subparsers.add_parser('info', help='Display model information')
    parser_info.set_defaults(func=cmd_info)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
