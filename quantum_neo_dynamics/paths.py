from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / "data"

CIRCUITS_DIR = DATA_DIR / "circ"
AQC_CIRCUITS_DIR = CIRCUITS_DIR / "aqc"
VQE_CIRCUITS_DIR = CIRCUITS_DIR / "vqe"
PRODUCT_CIRCUITS_DIR = CIRCUITS_DIR / "product" 

HAM_DIR = DATA_DIR / "ham"

TARGETS_DIR = DATA_DIR / "targets"
BACKENDS_DIR = DATA_DIR / "backends"

LOG_DIR = ROOT_DIR / "logs"

RESULTS_DIR = ROOT_DIR / "results"

HARDWARE_EXPERIMENTS_JOBDATA = RESULTS_DIR / "hardware_experiments_jobdata.csv"