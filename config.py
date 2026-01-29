"""Configuration parameters for AI image detection system."""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Image preprocessing
IMAGE_SIZE = (256, 256)  # Resize all images to this size
COLOR_MODE = 'RGB'  # 'RGB' or 'GRAY'

# Feature extraction
EXTRACT_DCT = True  # Discrete Cosine Transform features
EXTRACT_FFT = True  # Fast Fourier Transform features
EXTRACT_NOISE = True  # Noise pattern features
EXTRACT_STATS = True  # Statistical features
EXTRACT_TEXTURE = True  # Texture features (LBP)

# DCT/FFT parameters
DCT_COMPONENTS = 100  # Number of DCT coefficients to keep
FFT_COMPONENTS = 100  # Number of FFT components to keep

# PCA parameters
PCA_N_COMPONENTS = None  # None = auto (95% variance), or specify number
PCA_VARIANCE_THRESHOLD = 0.95  # Explain 95% of variance

# Classifier parameters
CLASSIFIER_TYPE = 'logistic'  # 'logistic', 'svm', or 'random_forest'
RANDOM_STATE = 42

# Logistic Regression parameters
LR_MAX_ITER = 1000
LR_C = 1.0

# SVM parameters
SVM_KERNEL = 'rbf'
SVM_C = 1.0
SVM_GAMMA = 'scale'

# Random Forest parameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 20

# Training parameters
TEST_SPLIT = 0.2  # Fraction of data for validation
BATCH_SIZE = 32  # For batch processing

# Model files
PCA_MODEL_FILE = os.path.join(MODEL_DIR, 'pca_model.pkl')
CLASSIFIER_MODEL_FILE = os.path.join(MODEL_DIR, 'classifier_model.pkl')
SCALER_FILE = os.path.join(MODEL_DIR, 'scaler.pkl')
METADATA_FILE = os.path.join(MODEL_DIR, 'metadata.json')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
