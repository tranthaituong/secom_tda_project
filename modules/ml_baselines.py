"""
Module ML Baselines - Nhiệm vụ 4 (NV4)
=====================================

Module này cài đặt các baseline models cho anomaly detection:
1. Isolation Forest (IF): Phát hiện anomaly dựa trên việc cô lập điểm bất thường
2. One-Class SVM (OCSVM): Học boundary của dữ liệu normal

Điểm khác biệt so với version cũ:
- Chạy trên dữ liệu đã xử lý (CSV files từ data/processed/)
- Hỗ trợ cả PCA và POD datasets
- Sử dụng sliding windows trực tiếp trên CSV

Các tham số được thử nghiệm:
- contamination: Tỷ lệ anomaly ước tính (0.01, 0.05, 0.07, 0.1)
- kernel, gamma: Tham số cho SVM

Author: GTMT Project
"""

import csv
import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


# ============================================================================
# ĐƯỜNG DẪN MẶC ĐỊNH
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DEFAULT_PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ============================================================================
# DATASET MAPPING
# ============================================================================

# Các file dữ liệu đã xử lý để chạy ML baselines
ML_DATASETS = {
    'PCA_5D': 'secom_processed_pca5.csv',
    'POD_5D': 'secom_processed_pod5.csv',
    'POD_95': 'secom_processed_pod95.csv'
}

# Các tham số ML
L_LIST = [20, 30, 50]
ANOMALY_FRACTIONS = [0.01, 0.05, 0.07, 0.1]


# ============================================================================
# LABEL HANDLING
# ============================================================================

class MLLabels:
    """
    Quản lý việc căn chỉnh nhãn với sliding windows.
    """
    
    @staticmethod
    def align_labels_with_windows(
        y_orig: np.ndarray,
        window_size: int,
        stride: int = 1
    ) -> np.ndarray:
        """
        Cắt nhãn gốc để khớp với số lượng sliding window.
        
        Với mỗi window, nhãn được gán cho điểm cuối cùng của window:
        window[i, i+1, ..., i+L-1] -> label[i+L-1]
        
        Args:
            y_orig: Mảng nhãn gốc (-1=normal, 1=anomaly).
            window_size: Kích thước window L.
            stride: Bước nhảy giữa các windows.
            
        Returns:
            Mảng nhãn đã căn chỉnh với shape (n_windows,).
        """
        aligned_labels = []
        for i in range(0, len(y_orig) - window_size + 1, stride):
            window_label = y_orig[i + window_size - 1]
            aligned_labels.append(window_label)
        return np.array(aligned_labels)
    
    @staticmethod
    def convert_to_binary(
        y: np.ndarray,
        positive_label: int = 1,
        negative_label: int = -1
    ) -> np.ndarray:
        """
        Chuyển đổi nhãn về dạng binary (0 và 1).
        
        Args:
            y: Mảng nhãn gốc.
            positive_label: Giá trị nhãn positive (anomaly) trong dữ liệu gốc.
            negative_label: Giá trị nhãn negative (normal) trong dữ liệu gốc.
            
        Returns:
            Mảng binary với 1=anomaly, 0=normal.
        """
        return np.where(y == positive_label, 1, 0)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_flattened_windows(
    data_df: pd.DataFrame,
    L: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cắt sliding window từ DataFrame và flatten (trải phẳng) trực tiếp.
    
    Args:
        data_df: DataFrame với các cột feature và 'Label'
        L: Kích thước window
        stride: Bước nhảy
        
    Returns:
        X_2d: Ma trận 2D đã flatten (n_windows, L * n_features)
        y_aligned: Mảng nhãn đã căn chỉnh (n_windows,)
    """
    X = data_df.drop(columns=['Label']).values
    y_raw = data_df['Label'].values
    
    X_windows = []
    y_windows = []
    
    for i in range(0, len(X) - L + 1, stride):
        window = X[i:i + L]
        X_windows.append(window.flatten())
        y_windows.append(y_raw[i + L - 1])
    
    return np.array(X_windows), np.array(y_windows)


def run_ml_models(
    X_2d: np.ndarray,
    y_aligned: np.ndarray,
    contamination_values: List[float]
) -> Tuple[Dict, Dict, np.ndarray]:
    """
    Huấn luyện Isolation Forest và One-Class SVM.
    
    Args:
        X_2d: Ma trận features đã flatten
        y_aligned: Nhãn đã căn chỉnh
        contamination_values: Danh sách contamination fractions
        
    Returns:
        iso_preds: Dict predictions từ Isolation Forest
        svm_preds: Dict predictions từ One-Class SVM
        labels_true: Mảng nhãn đã căn chỉnh
    """
    iso_preds = {}
    svm_preds = {}
    
    for fraction in contamination_values:
        # Isolation Forest
        iso_model = IsolationForest(
            contamination=fraction,
            random_state=42,
            n_estimators=100
        )
        y_pred_iso = iso_model.fit_predict(X_2d)
        # Chuyển: 1 (inlier) -> -1, -1 (outlier) -> 1
        iso_preds[fraction] = np.where(y_pred_iso == 1, -1, 1)
        
        # One-Class SVM
        svm_model = OneClassSVM(
            nu=fraction,
            kernel='rbf',
            gamma='scale'
        )
        y_pred_svm = svm_model.fit_predict(X_2d)
        svm_preds[fraction] = np.where(y_pred_svm == 1, -1, 1)
    
    return iso_preds, svm_preds, y_aligned


# ============================================================================
# MLBaselineRunner
# ============================================================================

class MLBaselineRunner:
    """
    Chạy các baseline ML models với ablation study.
    """
    
    def __init__(
        self,
        processed_dir = None,
        output_dir = None
    ):
        """
        Khởi tạo ML Baseline Runner.
        
        Args:
            processed_dir: Thư mục chứa file CSV đã xử lý
            output_dir: Thư mục lưu kết quả ML
        """
        self.processed_dir = processed_dir or DEFAULT_PROCESSED_DIR
        self.output_dir = output_dir or DEFAULT_PROCESSED_DIR
        self.logger = logging.getLogger(__name__)
        
        self.results_ = {
            'iso': {},
            'svm': {},
            'labels_true': {}
        }
        self.param_logs_ = []
    
    def run_ablation_study(self) -> Dict:
        """
        Chạy ablation study cho tất cả các cấu hình.
        
        Returns:
            Dictionary chứa tất cả predictions và labels.
        """
        self.logger.info("=" * 60)
        self.logger.info("BẮT ĐẦU ABLATION STUDY - ML BASELINES")
        self.logger.info("=" * 60)
        
        for dataset_name, file_name in ML_DATASETS.items():
            file_path = os.path.join(self.processed_dir, file_name)
            
            if not os.path.exists(file_path):
                self.logger.warning(f"Không tìm thấy {file_path}. Bỏ qua.")
                continue
            
            self.logger.info(f"\n  Xử lý: {dataset_name}")
            
            df = pd.read_csv(file_path)
            
            self.results_['iso'][dataset_name] = {}
            self.results_['svm'][dataset_name] = {}
            self.results_['labels_true'][dataset_name] = {}
            
            for L in L_LIST:
                self.logger.info(f"    - Window L={L}")
                
                # Tạo flattened windows
                X_2d, y_aligned = create_flattened_windows(df, L)
                
                self.results_['labels_true'][dataset_name][L] = y_aligned
                self.results_['iso'][dataset_name][L] = {}
                self.results_['svm'][dataset_name][L] = {}
                
                for fraction in ANOMALY_FRACTIONS:
                    # Chạy ML models
                    iso_preds, svm_preds, _ = run_ml_models(
                        X_2d, y_aligned, [fraction]
                    )
                    
                    self.results_['iso'][dataset_name][L][fraction] = iso_preds[fraction]
                    self.results_['svm'][dataset_name][L][fraction] = svm_preds[fraction]
                    
                    # Log params
                    self.param_logs_.append({
                        'dataset': dataset_name,
                        'L': L,
                        'model': 'Isolation Forest',
                        'params': f"contamination={fraction}"
                    })
                    self.param_logs_.append({
                        'dataset': dataset_name,
                        'L': L,
                        'model': 'One-Class SVM',
                        'params': f"nu={fraction}, kernel='rbf', gamma='scale'"
                    })
                
                self.logger.info(f"      Shape: {X_2d.shape}, Anomalies: {np.sum(y_aligned == 1)}")
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("HOÀN TẤT ABLATION STUDY")
        self.logger.info("=" * 60)
        
        return self.results_
    
    def save_outputs(self) -> Dict:
        """
        Lưu kết quả ra file.
        
        Output files:
            - ml_preds.npy: Dictionary chứa tất cả predictions
            - ml_param_log.csv: CSV log các tham số đã thử
            
        Returns:
            Dictionary đường dẫn các file đã lưu.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info("Lưu kết quả ML Baselines...")
        
        files = {}
        
        # Lưu predictions
        preds_path = os.path.join(self.output_dir, 'ml_preds.npy')
        np.save(preds_path, self.results_, allow_pickle=True)
        files['ml_preds'] = preds_path
        self.logger.info(f"  - Đã lưu: {preds_path}")
        
        # Lưu param log
        log_path = os.path.join(self.output_dir, 'ml_param_log.csv')
        csv_columns = ['dataset', 'L', 'model', 'params']
        
        with open(log_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for log in self.param_logs_:
                writer.writerow(log)
        
        files['ml_param_log'] = log_path
        self.logger.info(f"  - Đã lưu: {log_path}")
        
        return files


# ============================================================================
# MLBaselineManager
# ============================================================================

class MLBaselineManager:
    """
    Quản lý việc chạy ML baselines một cách đơn giản.
    """
    
    def __init__(
        self,
        processed_dir = None,
        output_dir = None
    ):
        """
        Khởi tạo ML Baseline Manager.
        
        Args:
            processed_dir: Thư mục chứa file CSV đã xử lý
            output_dir: Thư mục lưu kết quả
        """
        self.processed_dir = processed_dir or DEFAULT_PROCESSED_DIR
        self.output_dir = output_dir or DEFAULT_PROCESSED_DIR
        self.logger = logging.getLogger(__name__)
    
    def run(self) -> Dict:
        """
        Chạy toàn bộ pipeline ML baselines.
        
        Returns:
            Dictionary chứa kết quả và đường dẫn file.
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("BẮT ĐẦU ML BASELINES")
        self.logger.info("=" * 60)
        
        # Khởi tạo runner
        runner = MLBaselineRunner(
            processed_dir=self.processed_dir,
            output_dir=self.output_dir
        )
        
        # Chạy ablation study
        results = runner.run_ablation_study()
        
        # Lưu outputs
        saved_files = runner.save_outputs()
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("HOÀN TẤT ML BASELINES")
        self.logger.info("=" * 60)
        
        return {
            'results': results,
            'param_logs': runner.param_logs_,
            'saved_files': saved_files
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Hàm main để chạy ML Baselines độc lập.
    
    Usage:
        python -m modules.ml_baselines
    """
    import argparse
    
    base_dir = Path(__file__).parent.parent
    default_processed = base_dir / "data" / "processed"
    
    parser = argparse.ArgumentParser(description='Chạy ML Baselines')
    parser.add_argument('--processed-dir', default=str(default_processed),
                       help='Thư mục chứa file CSV đã xử lý')
    parser.add_argument('--output-dir', default=str(default_processed),
                       help='Thư mục lưu kết quả')
    
    args = parser.parse_args()
    
    manager = MLBaselineManager(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir
    )
    
    results = manager.run()
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
