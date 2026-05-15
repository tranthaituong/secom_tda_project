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
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

# Data files
SECOM_DATA_FILE = DATA_DIR / "raw" / "secom.data"
SECOM_LABELS_FILE = DATA_DIR / "raw" / "secom_labels.data"

# Intermediate directories
# TDA results are now stored in OUTPUT_DIR


# =============================================================================
# NV1: Data Processing Parameters
# =============================================================================

# Số lượng thành phần PCA để thử nghiệm
DEFAULT_PCA_COMPONENTS: List[int] = [2, 3, 5]

# Kích thước sliding window (L)
DEFAULT_WINDOW_SIZES: List[int] = [20, 30, 50]

# Bước nhảy giữa các windows
DEFAULT_STRIDE: int = 1

# Ngưỡng NaN (loại bỏ cột có NaN > threshold)
NAN_THRESHOLD: float = 0.50

# Số láng giềng cho KNNImputer
KNN_NEIGHBORS: int = 5


# =============================================================================
# NV1: POD Parameters
# =============================================================================

# Số Mode cố định cho POD_5D
N_POD_FIXED: int = 5

# Ngưỡng năng lượng cho POD_95
ENERGY_TARGET: float = 0.95


# =============================================================================
# NV3: TDA Parameters
# =============================================================================

# Chiều homology cần tính (0=components, 1=loops/cycles)
DEFAULT_HOMOLOGY_DIMENSIONS: List[int] = [1]  # Chỉ H1

# Phân vị để tính epsilon cho Vietoris-Rips
DEFAULT_EPS_PERCENTILE: int = 60

# Phân vị để xác định ngưỡng anomaly
# Ngưỡng 95% nghĩa là top 5% persistence scores được coi là anomaly
DEFAULT_THRESHOLD_PERCENTILE: float = 95.0

# Ngưỡng tối đa cho Rips filtration
# None = auto-calculate based on data
DEFAULT_MAX_EDGE: float = None

# Giới hạn số lượng windows để tính toán (None = tất cả)
# Giảm giá trị này nếu bộ nhớ không đủ
DEFAULT_MAX_SAMPLES: int = None

# Grid search parameters
GRID_SEARCH_NOISE_LEVELS: int = 30
GRID_SEARCH_THRESHOLD_LEVELS: int = 30


# =============================================================================
# NV4: ML Baseline Parameters
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
# NV5: Evaluation Parameters
# =============================================================================

# Average method cho metrics ('binary', 'micro', 'macro', 'weighted')
DEFAULT_AVERAGE_METHOD: str = 'binary'


# =============================================================================
# NV6: Visualization Parameters
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
# ML Datasets Mapping
# =============================================================================

# Các file dữ liệu đã xử lý để chạy ML baselines
ML_DATASETS: dict = {
    'PCA_5D': 'secom_processed_pca5.csv',
    'POD_5D': 'secom_processed_pod5.csv',
    'POD_95': 'secom_processed_pod95.csv'
}


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
        self.nan_threshold = NAN_THRESHOLD
        self.knn_neighbors = KNN_NEIGHBORS
        
        # POD Parameters
        self.n_pod_fixed = N_POD_FIXED
        self.energy_target = ENERGY_TARGET
        
        # TDA
        self.homology_dimensions = DEFAULT_HOMOLOGY_DIMENSIONS.copy()
        self.eps_percentile = DEFAULT_EPS_PERCENTILE
        self.threshold_percentile = DEFAULT_THRESHOLD_PERCENTILE
        self.max_edge = DEFAULT_MAX_EDGE
        self.max_samples = DEFAULT_MAX_SAMPLES
        self.grid_noise_levels = GRID_SEARCH_NOISE_LEVELS
        self.grid_threshold_levels = GRID_SEARCH_THRESHOLD_LEVELS
        
        # ML Baselines
        self.contamination_values = DEFAULT_CONTAMINATION_VALUES.copy()
        self.n_estimators = DEFAULT_N_ESTIMATORS
        self.random_state = DEFAULT_RANDOM_STATE
        self.svm_kernel = DEFAULT_SVM_KERNEL
        self.svm_gamma = DEFAULT_SVM_GAMMA
        
        # Paths
        self.data_dir = DATA_DIR
        self.processed_dir = PROCESSED_DIR
        self.output_dir = OUTPUT_DIR
        
        # ML Datasets
        self.ml_datasets = ML_DATASETS.copy()
    
    def to_dict(self) -> dict:
        """Chuyển cấu hình thành dictionary."""
        return {
            'pca_components': self.pca_components,
            'window_sizes': self.window_sizes,
            'stride': self.stride,
            'nan_threshold': self.nan_threshold,
            'knn_neighbors': self.knn_neighbors,
            'n_pod_fixed': self.n_pod_fixed,
            'energy_target': self.energy_target,
            'eps_percentile': self.eps_percentile,
            'threshold_percentile': self.threshold_percentile,
            'contamination_values': self.contamination_values,
            'random_state': self.random_state,
        }


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
