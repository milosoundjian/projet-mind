from pathlib import Path
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parent

def load_config(name: str):
    """Load YAML config using OmegaConf."""
    path = CONFIG_DIR / f"{name}.yaml"
    cfg = OmegaConf.load(path)
    return cfg