"""Dataset loading and management utilities."""

import os
import glob
from pathlib import Path
import config


def get_image_paths(data_dir, class_name=None):
    """
    Get paths to all images in a directory.

    Args:
        data_dir: Root directory containing images
        class_name: Optional class subdirectory ('real' or 'ai_generated')

    Returns:
        List of image file paths
    """
    if class_name:
        search_dir = os.path.join(data_dir, class_name)
    else:
        search_dir = data_dir

    # Support common image formats
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_paths = []

    for ext in extensions:
        pattern = os.path.join(search_dir, '**', ext)
        image_paths.extend(glob.glob(pattern, recursive=True))

    return sorted(image_paths)


def load_dataset(data_dir, class_labels=None):
    """
    Load dataset with labels.

    Expected directory structure:
        data_dir/
            real/
                image1.jpg
                image2.jpg
                ...
            ai_generated/
                image1.jpg
                image2.jpg
                ...

    Args:
        data_dir: Root directory containing class subdirectories
        class_labels: Dict mapping class names to labels
                     Default: {'real': 0, 'ai_generated': 1}

    Returns:
        Tuple of (image_paths, labels)
    """
    if class_labels is None:
        class_labels = {'real': 0, 'ai_generated': 1}

    image_paths = []
    labels = []

    for class_name, label in class_labels.items():
        class_dir = os.path.join(data_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue

        class_images = get_image_paths(data_dir, class_name)
        image_paths.extend(class_images)
        labels.extend([label] * len(class_images))

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}")

    return image_paths, labels


def verify_dataset(data_dir):
    """
    Verify dataset structure and print statistics.

    Args:
        data_dir: Root directory to verify

    Returns:
        Dict with dataset statistics
    """
    stats = {
        'total_images': 0,
        'real_images': 0,
        'ai_images': 0,
        'valid': False
    }

    real_dir = os.path.join(data_dir, 'real')
    ai_dir = os.path.join(data_dir, 'ai_generated')

    if os.path.exists(real_dir):
        real_images = get_image_paths(data_dir, 'real')
        stats['real_images'] = len(real_images)

    if os.path.exists(ai_dir):
        ai_images = get_image_paths(data_dir, 'ai_generated')
        stats['ai_images'] = len(ai_images)

    stats['total_images'] = stats['real_images'] + stats['ai_images']
    stats['valid'] = stats['total_images'] > 0

    return stats


def print_dataset_info(data_dir):
    """Print detailed dataset information."""
    print(f"\nDataset Directory: {data_dir}")
    print("=" * 60)

    stats = verify_dataset(data_dir)

    if not stats['valid']:
        print("No images found!")
        print("\nExpected structure:")
        print("  data_dir/")
        print("    real/")
        print("      *.jpg, *.png, etc.")
        print("    ai_generated/")
        print("      *.jpg, *.png, etc.")
        return False

    print(f"Real images:        {stats['real_images']:,}")
    print(f"AI-generated images: {stats['ai_images']:,}")
    print(f"Total images:       {stats['total_images']:,}")

    if stats['real_images'] > 0 and stats['ai_images'] > 0:
        balance_ratio = stats['real_images'] / stats['ai_images']
        print(f"Balance ratio:      {balance_ratio:.2f} (real/ai)")

        if 0.8 <= balance_ratio <= 1.2:
            print("Dataset is well balanced.")
        else:
            print("Warning: Dataset is imbalanced!")

    print("=" * 60)
    return True


def setup_data_directories():
    """
    Create data directory structure if it doesn't exist.

    Creates:
        data/
            train/
                real/
                ai_generated/
            test/
                real/
                ai_generated/
    """
    dirs_to_create = [
        config.TRAIN_DIR,
        os.path.join(config.TRAIN_DIR, 'real'),
        os.path.join(config.TRAIN_DIR, 'ai_generated'),
        config.TEST_DIR,
        os.path.join(config.TEST_DIR, 'real'),
        os.path.join(config.TEST_DIR, 'ai_generated'),
    ]

    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

    print("Data directories created:")
    print(f"  {config.TRAIN_DIR}")
    print(f"  {config.TEST_DIR}")
    print("\nPlease place your images in the appropriate subdirectories:")
    print("  data/train/real/ - Real photos for training")
    print("  data/train/ai_generated/ - AI images for training")
    print("  data/test/real/ - Real photos for testing")
    print("  data/test/ai_generated/ - AI images for testing")


def download_cifake_dataset():
    """
    Download CIFAKE dataset using kagglehub and organize it.

    Returns:
        str: Path to downloaded dataset, or None if failed
    """
    try:
        import kagglehub
        import shutil
        from tqdm import tqdm

        print("\n" + "=" * 70)
        print("DOWNLOADING CIFAKE DATASET")
        print("=" * 70)
        print("\nThis will download ~2GB of data from Kaggle.")
        print("Make sure you have:")
        print("  1. A Kaggle account")
        print("  2. Kaggle API credentials configured (~/.kaggle/kaggle.json)")
        print("\nFor setup instructions, visit:")
        print("  https://www.kaggle.com/docs/api")
        print("=" * 70)

        # Download latest version
        print("\nDownloading CIFAKE dataset from Kaggle...")
        path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")

        print(f"\nDataset downloaded to: {path}")

        # Organize the dataset
        print("\nOrganizing dataset into train/test directories...")
        organize_cifake_dataset(path)

        print("\n" + "=" * 70)
        print("DATASET DOWNLOAD COMPLETE!")
        print("=" * 70)
        print(f"\nDataset organized in:")
        print(f"  Training: {config.TRAIN_DIR}")
        print(f"  Testing:  {config.TEST_DIR}")

        return path

    except ImportError:
        print("\nError: kagglehub not installed.")
        print("Install it with: pip install kagglehub")
        return None
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure you have a Kaggle account")
        print("  2. Set up Kaggle API credentials:")
        print("     - Go to https://www.kaggle.com/settings")
        print("     - Click 'Create New API Token'")
        print("     - Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
        print("  3. Alternatively, manually download from:")
        print("     https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
        return None


def organize_cifake_dataset(source_path, train_split=0.8):
    """
    Organize downloaded CIFAKE dataset into train/test directories.

    Args:
        source_path: Path to downloaded CIFAKE dataset
        train_split: Fraction of data to use for training (default: 0.8)
    """
    import shutil
    from tqdm import tqdm
    import random

    # Setup directories
    setup_data_directories()

    # CIFAKE structure: source_path/train/ and source_path/test/
    # Each contains REAL/ and FAKE/ subdirectories

    # Map CIFAKE directories to our structure
    cifake_train = os.path.join(source_path, 'train')
    cifake_test = os.path.join(source_path, 'test')

    if not os.path.exists(cifake_train):
        print(f"Warning: Expected directory not found: {cifake_train}")
        print("Dataset structure may have changed. Please organize manually.")
        return

    # Process training data
    print("\nOrganizing training data...")

    # REAL images (CIFAKE uses 'REAL' folder)
    cifake_real = os.path.join(cifake_train, 'REAL')
    if os.path.exists(cifake_real):
        real_images = get_image_paths(cifake_real)
        random.shuffle(real_images)

        # Split into train/test
        n_train = int(len(real_images) * train_split)
        train_real = real_images[:n_train]
        test_real = real_images[n_train:]

        # Copy to train directory
        dest_dir = os.path.join(config.TRAIN_DIR, 'real')
        for img_path in tqdm(train_real, desc="Copying REAL images (train)"):
            filename = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(dest_dir, filename))

        # Copy to test directory
        dest_dir = os.path.join(config.TEST_DIR, 'real')
        for img_path in tqdm(test_real, desc="Copying REAL images (test)"):
            filename = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(dest_dir, filename))

    # FAKE images (AI-generated, CIFAKE uses 'FAKE' folder)
    cifake_fake = os.path.join(cifake_train, 'FAKE')
    if os.path.exists(cifake_fake):
        fake_images = get_image_paths(cifake_fake)
        random.shuffle(fake_images)

        # Split into train/test
        n_train = int(len(fake_images) * train_split)
        train_fake = fake_images[:n_train]
        test_fake = fake_images[n_train:]

        # Copy to train directory
        dest_dir = os.path.join(config.TRAIN_DIR, 'ai_generated')
        for img_path in tqdm(train_fake, desc="Copying FAKE images (train)"):
            filename = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(dest_dir, filename))

        # Copy to test directory
        dest_dir = os.path.join(config.TEST_DIR, 'ai_generated')
        for img_path in tqdm(test_fake, desc="Copying FAKE images (test)"):
            filename = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(dest_dir, filename))

    print("\nDataset organization complete!")


def download_sample_dataset():
    """
    Attempt to download CIFAKE dataset, or show manual instructions.
    """
    # Try automatic download
    path = download_cifake_dataset()

    if path is None:
        # Show manual instructions
        print("\n" + "=" * 70)
        print("MANUAL DATASET DOWNLOAD INSTRUCTIONS")
        print("=" * 70)

        print("\n1. CIFAKE Dataset (Recommended):")
        print("   - Visit: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
        print("   - Download the dataset")
        print("   - Extract to a temporary directory")
        print("   - Run: python -c \"from dataset_loader import organize_cifake_dataset; organize_cifake_dataset('path/to/extracted/dataset')\"")
        print("   - Contains 60,000 real + 60,000 AI-generated images")

        print("\n2. Manual Collection:")
        print("   Real images:")
        print("     - COCO dataset: https://cocodataset.org/")
        print("     - ImageNet: https://www.image-net.org/")
        print("     - Flickr images")
        print("   ")
        print("   AI-generated images:")
        print("     - Generate using Stable Diffusion, DALL-E, Midjourney")
        print("     - DiffusionDB: https://github.com/poloclub/diffusiondb")

        print("\n3. After collecting:")
        print("   - Place images in data/train/real/ and data/train/ai_generated/")
        print("   - Reserve 10-20% of images for data/test/ directories")
        print("   - Ensure balanced classes (equal number of real and AI images)")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    # Test dataset utilities
    setup_data_directories()
    download_sample_dataset()

    if os.path.exists(config.TRAIN_DIR):
        print("\nChecking training data...")
        print_dataset_info(config.TRAIN_DIR)

    if os.path.exists(config.TEST_DIR):
        print("\nChecking test data...")
        print_dataset_info(config.TEST_DIR)
