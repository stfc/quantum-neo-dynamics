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

FIGURES_DIR = ROOT_DIR / "figures"
ZNE_HARDWARE_FIGURES_DIR = FIGURES_DIR / "zne_hardware"

RESULTS_DIR = ROOT_DIR / "results"

STATEVECTOR_RESULTS_FILE = RESULTS_DIR / "statevector_data.json"

HARDWARE_EXPERIMENTS_JOBDATA = RESULTS_DIR / "hardware_experiments_jobdata.csv"
HARDWARE_EXPERIMENTS_RESULTSFILE = RESULTS_DIR / "ZNE_hardware.csv"

MPLSTYLE_FILE = FIGURES_DIR / "proton.mplstyle"