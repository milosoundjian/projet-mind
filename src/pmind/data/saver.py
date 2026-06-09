from . import paths
import matplotlib.pyplot as plt

def save_figure(name, out_dir=paths.FIGURES_DIR, format="pdf"):
    file_path = out_dir / f"{name}.{format}"
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    return file_path

