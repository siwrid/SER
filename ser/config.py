"""Project-wide paths and label definitions."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "visualizations"

FEATURES_PKL = DATA_DIR / "IEMOCAP_features.pkl"
TEXT_CSV = DATA_DIR / "iemocap_text.csv"

# Integer labels used in IEMOCAP_features.pkl. Label 5 ("frustrated") is
# excluded from every experiment in this project.
EMOTIONS = {
    0: "happy",
    1: "sad",
    2: "neutral",
    3: "angry",
    4: "excited",
}
EXCLUDED_LABEL = 5
