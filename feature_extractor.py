"""Feature extraction module for AI image detection."""

import numpy as np
import cv2
from PIL import Image
from scipy import fft
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern
import config


def load_image(image_path):
    """Load and preprocess image."""
    img = Image.open(image_path)

    # Convert to RGB if needed
    if img.mode != config.COLOR_MODE:
        if config.COLOR_MODE == 'RGB':
            img = img.convert('RGB')
        elif config.COLOR_MODE == 'GRAY':
            img = img.convert('L')

    # Resize to standard size
    img = img.resize(config.IMAGE_SIZE, Image.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32) / 255.0

    return img_array


def extract_frequency_features(image):
    """
    Extract frequency domain features using DCT and FFT.

    AI-generated images often have different frequency distributions
    compared to real photos, especially in high-frequency components.
    """
    features = []

    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    else:
        gray = image

    if config.EXTRACT_DCT:
        # Discrete Cosine Transform (JPEG-like compression analysis)
        dct = cv2.dct(gray.astype(np.float32))

        # Get top-left coefficients (low frequencies)
        dct_flat = dct.flatten()
        dct_sorted = np.sort(np.abs(dct_flat))[::-1]
        dct_features = dct_sorted[:config.DCT_COMPONENTS]
        features.append(dct_features)

    if config.EXTRACT_FFT:
        # Fast Fourier Transform
        fft_result = fft.fft2(gray)
        fft_shifted = fft.fftshift(fft_result)
        magnitude_spectrum = np.abs(fft_shifted)

        # Extract radial frequency distribution
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2

        # Sample points at different radii
        fft_features = []
        for r in np.linspace(1, min(h, w) // 2, config.FFT_COMPONENTS):
            r = int(r)
            y, x = center_y - r, center_x
            if 0 <= y < h and 0 <= x < w:
                fft_features.append(magnitude_spectrum[y, x])

        features.append(np.array(fft_features))

    if len(features) > 0:
        return np.concatenate(features)
    return np.array([])


def extract_noise_features(image):
    """
    Extract noise pattern features.

    Real cameras have sensor-specific noise patterns.
    AI-generated images have more uniform, synthetic noise.
    """
    features = []

    if not config.EXTRACT_NOISE:
        return np.array([])

    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    else:
        gray = image

    # Apply denoising to estimate noise
    denoised = cv2.fastNlMeansDenoising((gray * 255).astype(np.uint8), h=10)
    denoised = denoised.astype(np.float32) / 255.0

    noise = gray - denoised

    # Noise statistics
    noise_mean = np.mean(noise)
    noise_std = np.std(noise)
    noise_skew = skew(noise.flatten())
    noise_kurt = kurtosis(noise.flatten())

    # Local noise variance
    kernel_size = 8
    local_vars = []
    h, w = noise.shape
    for i in range(0, h - kernel_size, kernel_size):
        for j in range(0, w - kernel_size, kernel_size):
            block = noise[i:i+kernel_size, j:j+kernel_size]
            local_vars.append(np.var(block))

    local_var_mean = np.mean(local_vars)
    local_var_std = np.std(local_vars)

    features = [noise_mean, noise_std, noise_skew, noise_kurt, local_var_mean, local_var_std]

    return np.array(features)


def extract_statistical_features(image):
    """
    Extract statistical features from color channels and edges.

    Includes histogram moments, edge statistics, and color distribution.
    """
    features = []

    if not config.EXTRACT_STATS:
        return np.array([])

    # Handle RGB or grayscale
    if len(image.shape) == 3:
        channels = [image[:, :, i] for i in range(3)]
    else:
        channels = [image]

    # Per-channel statistics
    for channel in channels:
        # Histogram moments
        channel_flat = channel.flatten()
        features.extend([
            np.mean(channel_flat),
            np.std(channel_flat),
            skew(channel_flat),
            kurtosis(channel_flat),
        ])

    # Edge statistics (using Sobel)
    if len(image.shape) == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)

    edge_flat = edge_magnitude.flatten()
    features.extend([
        np.mean(edge_flat),
        np.std(edge_flat),
        np.percentile(edge_flat, 90),  # Strong edges
    ])

    return np.array(features)


def extract_texture_features(image):
    """
    Extract texture features using Local Binary Patterns (LBP).

    LBP captures local texture patterns that may differ between
    AI-generated and real images.
    """
    features = []

    if not config.EXTRACT_TEXTURE:
        return np.array([])

    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)

    # Compute LBP
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

    # LBP histogram
    n_bins = n_points + 2  # uniform patterns + non-uniform
    hist, _ = np.histogram(lbp.flatten(), bins=n_bins, range=(0, n_bins), density=True)

    features = hist

    return np.array(features)


def extract_features(image_path):
    """
    Extract all features from an image.

    Args:
        image_path: Path to the image file

    Returns:
        Feature vector as numpy array
    """
    # Load image
    image = load_image(image_path)

    # Extract all feature types
    feature_list = []

    freq_features = extract_frequency_features(image)
    if len(freq_features) > 0:
        feature_list.append(freq_features)

    noise_features = extract_noise_features(image)
    if len(noise_features) > 0:
        feature_list.append(noise_features)

    stat_features = extract_statistical_features(image)
    if len(stat_features) > 0:
        feature_list.append(stat_features)

    texture_features = extract_texture_features(image)
    if len(texture_features) > 0:
        feature_list.append(texture_features)

    # Concatenate all features
    if len(feature_list) > 0:
        all_features = np.concatenate(feature_list)
    else:
        all_features = np.array([])

    return all_features


def _extract_features_worker(img_path):
    """Worker function for parallel feature extraction."""
    try:
        return extract_features(img_path)
    except Exception:
        return None


def extract_features_batch(image_paths, verbose=True, n_workers=None):
    """
    Extract features from multiple images using parallel processing.

    Args:
        image_paths: List of image file paths
        verbose: Show progress bar
        n_workers: Number of parallel workers (default: CPU count)

    Returns:
        Feature matrix (n_images, n_features)
    """
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count

    if n_workers is None:
        n_workers = cpu_count()

    features_list = []

    with Pool(processes=n_workers) as pool:
        if verbose:
            results = list(tqdm(
                pool.imap(_extract_features_worker, image_paths),
                total=len(image_paths),
                desc=f"Extracting features ({n_workers} workers)"
            ))
        else:
            results = pool.map(_extract_features_worker, image_paths)

    # Filter out failed extractions
    features_list = [f for f in results if f is not None]

    if len(features_list) == 0:
        return np.array([])

    return np.array(features_list)
