from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
FIGURES_DIR = EXPERIMENTS_DIR / "figures"
MODELS_DIR = EXPERIMENTS_DIR / "models"