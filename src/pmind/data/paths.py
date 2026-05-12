from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

NOTEBOOKS_DIR = ROOT / "notebooks"
RESULTS_DIR = ROOT / "results"
SCRIPTS_DIR = ROOT / "scripts"

MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
RUNS_DIR = RESULTS_DIR / "runs"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"