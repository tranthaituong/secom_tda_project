"""
SECOM TDA Project - Modules
===========================

Các module xử lý cho dự án phân tích dữ liệu SECOM sử dụng
Topological Data Analysis (TDA) cho anomaly detection.

Modules:
    - data_processing: Xử lý dữ liệu SECOM, PCA, sliding windows
    - tda_features: Trích xuất đặc trưng topological (Persistent Homology)
    - ml_baselines: Isolation Forest, One-Class SVM baselines
    - evaluation: Đánh giá và so sánh các phương pháp
    - visualization: Tạo heatmaps, charts, PDF reports
"""

__version__ = "1.0.0"
__author__ = "GTMT Project"

from . import data_processing
from . import tda_features
from . import ml_baselines
from . import evaluation
from . import visualization

__all__ = [
    'data_processing',
    'tda_features',
    'ml_baselines',
    'evaluation',
    'visualization',
]
