"""
Configuration file for managing paths and versions across the water polo tracking project.
"""

from pathlib import Path
import yaml

# Get the project root directory (two levels up from this file)
ROOT_DIR = Path(__file__).parent.parent

# Load configuration from YAML file
CONFIG_PATH = ROOT_DIR / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Base paths
MODELS_DIR = ROOT_DIR / config['paths']['models']
DATA_DIR = ROOT_DIR / config['paths']['data']
PROCESSED_DIR = ROOT_DIR / config['paths']['processed']
RAW_DIR = ROOT_DIR / config['paths']['raw']
LABEL_STUDIO_DIR = ROOT_DIR / config['paths']['label_studio']
VIDEOS_DIR = Path(config['paths']['videos'])  # Use absolute path as is

# Model versions
MODEL_VERSIONS = config['model_versions']

# Dataset versions
DATASET_VERSIONS = config['dataset_versions']

# Video file
VIDEO_FILE = config['video']

# Detection parameters
DETECTION_CONFIG = config['detection']

# Model paths
def get_model_path(model_type: str) -> Path:
    """Get the path to a model file based on its type."""
    version = MODEL_VERSIONS[model_type]
    return MODELS_DIR / model_type / f"best_{model_type}_{version}.pt"

# Dataset paths
def get_dataset_path(model_type: str, split: str = None) -> Path:
    """Get the path to a dataset directory based on its type and split."""
    version = DATASET_VERSIONS[model_type]
    path = PROCESSED_DIR / model_type / version
    if split:
        path = path / split
    return path

# Label Studio paths
def get_label_studio_export_path(model_type: str, split: str) -> Path:
    """Get the path for Label Studio export JSON files."""
    return LABEL_STUDIO_DIR / "exports" / model_type / split / f"{model_type}.json"

def get_label_studio_import_path(model_type: str) -> Path:
    """Get the path for Label Studio import labels."""
    return LABEL_STUDIO_DIR / "imports" / model_type / "labels"

# Video paths
def get_video_path() -> Path:
    """Get the path to the video file."""
    return VIDEOS_DIR / VIDEO_FILE

# Detection parameters
def get_detection_params(model_type: str) -> dict:
    """Get detection parameters for a specific model type."""
    return {
        'confidence_threshold': DETECTION_CONFIG['confidence_threshold'],
        'frame_interval': DETECTION_CONFIG['frame_interval'],
        'min_detections': DETECTION_CONFIG['min_detections'][model_type],
        'max_detections': DETECTION_CONFIG['max_detections'][model_type]
    }