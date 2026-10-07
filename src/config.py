import os
import torch
import numpy as np

# Global Reproducibility Seeds
SEED = 42

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(SEED)

# Environment Paths Configuration
BASE_DIR = r"d:/Grammer scoring engine"
DATA_DIR = os.path.join(BASE_DIR, "shl-hiring-assessment-2026", "Dataset_Final")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

CONFIG = {
    'base_dir': BASE_DIR,
    'data_dir': DATA_DIR,
    'cache_dir': CACHE_DIR,
    'outputs_dir': OUTPUTS_DIR,
    'figures_dir': FIGURES_DIR,
    'sr': 16000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

for path_key in ['cache_dir', 'outputs_dir', 'figures_dir']:
    os.makedirs(CONFIG[path_key], exist_ok=True)
