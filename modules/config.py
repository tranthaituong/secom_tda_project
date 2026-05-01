"""
Configuration file for SECOM TDA Project
=======================================

Chứa các tham số mặc định cho toàn bộ pipeline.
Có thể import và sử dụng trong các module khác.
"""

from pathlib import Path
from typing import List

# =============================================================================
# Paths
# =============================================================================

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# Data files
SECOM_DATA_FILE = DATA_DIR / "secom.data"
SECOM_LABELS_FILE = DATA_DIR / "secom_labels.data"

# Output files
PCA_DATA_FILE = OUTPUT_DIR / "data_pca.npy"
WINDOWS_DICT_FILE = OUTPUT_DIR / "windows_dict.npy"
LABELS_RAW_FILE = OUTPUT_DIR / "labels_raw.npy"
TOPO_FEATURES_FILE = OUTPUT_DIR / "topo_features.npy"
TOPO_SCORES_FILE = OUTPUT_DIR / "topo_scores.npy"
TOPO_DIAGRAMS_FILE = OUTPUT_DIR / "topo_diagrams.npy"
ML_PREDS_FILE = OUTPUT_DIR / "ml_preds.npy"
ML_PARAM_LOG_FILE = OUTPUT_DIR / "ml_param_log.csv"
ABLATION_RESULTS_FILE = OUTPUT_DIR / "ablation_results.csv"

# Visualization outputs
TOPO_HEATMAP_FILE = OUTPUT_DIR / "topo_ablation_heatmap.png"
METRIC_COMPARISON_FILE = OUTPUT_DIR / "metric_comparison.png"
DIAGRAM_PDF_FILE = OUTPUT_DIR / "diagram_sidebyside.pdf"
ANALYSIS_NOTES_FILE = OUTPUT_DIR / "analysis_notes.md"


# =============================================================================
# Data Processing Parameters (NV1)
# =============================================================================

# Số lượng thành phần PCA để thử nghiệm
DEFAULT_PCA_COMPONENTS: List[int] = [2, 3, 5]

# Kích thước sliding window (L)
DEFAULT_WINDOW_SIZES: List[int] = [20, 30, 50]

# Bước nhảy giữa các windows
DEFAULT_STRIDE: int = 1

# =============================================================================
# TDA Parameters (NV3)
# =============================================================================

# Chiều homology cần tính (0=components, 1=loops/cycles)
DEFAULT_HOMOLOGY_DIMENSIONS: List[int] = [1]  # Chỉ H1

# Phân vị để xác định ngưỡng anomaly
# Ngưỡng 95% nghĩa là top 5% persistence scores được coi là anomaly
DEFAULT_THRESHOLD_PERCENTILE: float = 95.0

# Ngưỡng tối đa cho Rips filtration
# None = auto-calculate based on data
DEFAULT_MAX_EDGE: float = None

# Giới hạn số lượng windows để tính toán (None = tất cả)
# Giảm giá trị này nếu bộ nhớ không đủ
DEFAULT_MAX_SAMPLES: int = None


# =============================================================================
# ML Baseline Parameters (NV4)
# =============================================================================

# Các giá trị contamination (tỷ lệ anomaly ước tính) để thử nghiệm
DEFAULT_CONTAMINATION_VALUES: List[float] = [0.01, 0.05, 0.07, 0.1]

# Số lượng trees trong Isolation Forest
DEFAULT_N_ESTIMATORS: int = 100

# Random seed cho reproducibility
DEFAULT_RANDOM_STATE: int = 42

# Kernel cho One-Class SVM
DEFAULT_SVM_KERNEL: str = 'rbf'

# Gamma cho One-Class SVM ('scale', 'auto', hoặc float)
DEFAULT_SVM_GAMMA: str = 'scale'


# =============================================================================
# Evaluation Parameters (NV5)
# =============================================================================

# Average method cho metrics ('binary', 'micro', 'macro', 'weighted')
DEFAULT_AVERAGE_METHOD: str = 'binary'


# =============================================================================
# Visualization Parameters (NV6)
# =============================================================================

# Figure sizes (width, height) in inches
HEATMAP_FIGSIZE: tuple = (8, 6)
BAR_CHART_FIGSIZE: tuple = (9, 6)
DIAGRAM_FIGSIZE: tuple = (12, 5)

# DPI cho output images
DEFAULT_DPI: int = 300

# Số frames hiển thị trong persistence diagram comparison
DEFAULT_N_FRAMES_DISPLAY: int = 5


# =============================================================================
# Pipeline Configuration
# =============================================================================

class PipelineConfig:
    """
    Class cấu hình cho toàn bộ pipeline.
    
    Sử dụng:
        config = PipelineConfig()
        config.pca_components = [2, 3]
        config.window_sizes = [20, 30]
    """
    
    def __init__(self):
        # Data Processing
        self.pca_components = DEFAULT_PCA_COMPONENTS.copy()
        self.window_sizes = DEFAULT_WINDOW_SIZES.copy()
        self.stride = DEFAULT_STRIDE
        
        # TDA
        self.homology_dimensions = DEFAULT_HOMOLOGY_DIMENSIONS.copy()
        self.threshold_percentile = DEFAULT_THRESHOLD_PERCENTILE
        self.max_edge = DEFAULT_MAX_EDGE
        self.max_samples = DEFAULT_MAX_SAMPLES
        
        # ML Baselines
        self.contamination_values = DEFAULT_CONTAMINATION_VALUES.copy()
        self.n_estimators = DEFAULT_N_ESTIMATORS
        self.random_state = DEFAULT_RANDOM_STATE
        self.svm_kernel = DEFAULT_SVM_KERNEL
        self.svm_gamma = DEFAULT_SVM_GAMMA
        
        # Paths
        self.data_dir = DATA_DIR
        self.output_dir = OUTPUT_DIR
        
    def to_dict(self) -> dict:
        """Chuyển cấu hình thành dictionary."""
        return {
            'pca_components': self.pca_components,
            'window_sizes': self.window_sizes,
            'stride': self.stride,
            'homology_dimensions': self.homology_dimensions,
            'threshold_percentile': self.threshold_percentile,
            'contamination_values': self.contamination_values,
            'random_state': self.random_state,
        }


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
