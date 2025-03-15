"""
Simplified configuration for minimal debugging.
Avoids the self-reference issue in the main config.py file.
"""
import torch
import os
import logging
from datetime import datetime

# ======================================================
# Global configuration and defaults for debugging
# ======================================================

# Output and logging
OUTPUT_DIR = "analysis"
LOG_LEVEL = logging.INFO  # Use INFO for more detailed logging during debugging
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"debug_{TIMESTAMP}.log")

# S3 storage configuration
USE_COMPANY_S3 = False  # Disable company S3 for debugging

# Analysis features
REPORT_VARIANCE = True
DO_BASELINE = True
NUM_RANDOM_BASELINES = 10  # Reduced for faster debugging

# Device configuration - simplified for debugging
def get_default_device():
    """Determine the best available device for computation."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

# Default to CPU for consistent debugging
DEFAULT_DEVICE = 'cpu'  # Override this manually if needed

# Regression configuration - reduced for debugging
RCOND_SWEEP_LIST = [1e-15, 1e-10, 1e-5] + list(torch.logspace(-8, -3, 10).tolist())  # Reduced list

# Example sweep and run for debugging
SWEEPS = {
    '20250304052839': 'transformer',
    '20250304060315': 'rnn',
}

# Markov analysis settings
MAX_MARKOV_ORDER = 2  # Reduced for debugging

# Checkpoint processing settings
PROCESS_ALL_CHECKPOINTS = False  # Only process specified checkpoints for debugging
MAX_CHECKPOINTS = 1  # Just process one checkpoint for debugging

# Layer configurations
TRANSFORMER_ACTIVATION_KEYS = (
    ['blocks.0.hook_resid_pre'] +
    [f'blocks.{i}.hook_resid_post' for i in range(4)] +  # Reduced number of layers
    ['ln_final.hook_normalized']
)

# Memory management
MEMORY_EFFICIENT = True
MAX_CACHE_SIZE_GB = 2.0  # Reduced for debugging

# Error handling
MAX_RETRIES = 1  # Reduced for debugging
RETRY_DELAY = 1  # Reduced for debugging

# Create directory for logs
os.makedirs(LOG_DIR, exist_ok=True) 