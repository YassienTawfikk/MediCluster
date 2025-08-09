from pathlib import Path

from scipy.signal import spectrogram


def get_base_dir():
    """Return the project root by locating the top-level directory containing 'src'."""
    here = Path().resolve()
    for parent in [here] + list(here.parents):
        if (parent / "src").is_dir():
            return parent
    raise RuntimeError("Project root not found — 'src/' directory is missing.")


# Project Base Directory
base_dir = get_base_dir()

# Raw Data Directories
data_dir = base_dir / "data"
raw_data_dir = data_dir / "raw"
processed_data_dir = data_dir / "processed"
curated_data_dir = data_dir / "curated"

# Output Directories
output_dir = base_dir / "outputs"
model_dir = output_dir / "models"
figures_dir = output_dir / "figures"
docs_dir = output_dir / "docs"

# Directories List
data_dir_list = [raw_data_dir, processed_data_dir, curated_data_dir]
output_dir_list = [model_dir, figures_dir, docs_dir]
