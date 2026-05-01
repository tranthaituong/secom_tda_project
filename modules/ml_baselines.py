"""
Module ML Baselines - Nhiệm vụ 4 (NV4)
=====================================

Module này cài đặt các baseline models cho anomaly detection:
1. Isolation Forest (IF): Phát hiện anomaly dựa trên việc cô lập điểm bất thường
2. One-Class SVM (OCSVM): Học boundary của dữ liệu normal

Các tham số được thử nghiệm:
- contamination: Tỷ lệ anomaly ước tính (0.01, 0.05, 0.07, 0.1)
- kernel, gamma: Tham số cho SVM

Phương pháp so sánh:
- Baseline methods vs Topology (H1) features
- Ablation study với các cấu hình PCA và window size khác nhau

Author: GTMT Project
"""

import csv
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class MLLabels:
    """
    Quản lý việc căn chỉnh nhãn với sliding windows.
    
    Do sử dụng sliding windows, số lượng samples giảm đi.
    Cần căn chỉnh nhãn để khớp với số lượng windows.
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


class IsolationForestModel:
    """
    Wrapper cho Isolation Forest với logging và tuning.
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int = 42,
        n_estimators: int = 100
    ):
        """
        Khởi tạo Isolation Forest.
        
        Args:
            contamination: Tỷ lệ anomaly ước tính trong dữ liệu.
            random_state: Seed cho reproducibility.
            n_estimators: Số lượng trees trong forest.
        """
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model = None
        
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Train và predict với Isolation Forest.
        
        Isolation Forest trả về:
        - 1 cho inliers (normal)
        - -1 cho outliers (anomaly)
        
        Args:
            X: Ma trận features với shape (n_samples, n_features).
            
        Returns:
            Mảng nhãn với 1=anomaly, -1=normal.
        """
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators
        )
        predictions = self.model.fit_predict(X)
        return predictions


class OneClassSVMModel:
    """
    Wrapper cho One-Class SVM với logging và tuning.
    """
    
    def __init__(
        self,
        nu: float = 0.1,
        kernel: str = 'rbf',
        gamma: str = 'scale'
    ):
        """
        Khởi tạo One-Class SVM.
        
        Args:
            nu: Tham số nu (0 < nu <= 1), tương tự như contamination.
            kernel: Loại kernel ('rbf', 'linear', 'poly').
            gamma: Tham số gamma cho kernel 'rbf' ('scale', 'auto', hoặc float).
        """
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = None
        
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Train và predict với One-Class SVM.
        
        Args:
            X: Ma trận features với shape (n_samples, n_features).
            
        Returns:
            Mảng nhãn với 1=inlier (normal), -1=outlier (anomaly).
        """
        self.model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma
        )
        predictions = self.model.fit_predict(X)
        return predictions


class MLBaselineRunner:
    """
    Chạy các baseline ML models với ablation study.
    
    Thuộc tính:
        windows_dict: Dictionary windows {n: {L: windows_array}}.
        labels_raw: Nhãn gốc từ SECOM dataset.
        n_components_list: Danh sách số PCA components.
        window_sizes_list: Danh sách kích thước window.
        contamination_values: Danh sách contamination fractions.
    """
    
    def __init__(
        self,
        windows_dict: Dict,
        labels_raw: np.ndarray,
        n_components_list: List[int] = None,
        window_sizes_list: List[int] = None,
        contamination_values: List[float] = None
    ):
        """
        Khởi tạo ML Baseline Runner.
        
        Args:
            windows_dict: Dictionary windows từ data_processing.
            labels_raw: Nhãn gốc (-1=normal, 1=anomaly).
            n_components_list: Danh sách PCA components (default: [2, 3, 5]).
            window_sizes_list: Danh sách window sizes (default: [20, 30, 50]).
            contamination_values: Danh sách contamination fractions.
        """
        self.windows_dict = windows_dict
        self.labels_raw = labels_raw
        self.n_components_list = n_components_list or [2, 3, 5]
        self.window_sizes_list = window_sizes_list or [20, 30, 50]
        self.contamination_values = contamination_values or [0.01, 0.05, 0.07, 0.1]
        
        self.logger = logging.getLogger(__name__)
        self.results_ = {}
        self.param_logs_ = []
        
    def _flatten_windows(self, X_3d: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi windows 3D thành 2D cho ML models.
        
        Shape: (n_windows, window_size, n_features) 
             -> (n_windows, window_size * n_features)
        
        Args:
            X_3d: Mảng 3D windows.
            
        Returns:
            Mảng 2D đã flatten.
        """
        return X_3d.reshape(X_3d.shape[0], -1)
    
    def run_ablation_study(self, verbose: bool = True) -> Dict:
        """
        Chạy ablation study cho tất cả các cấu hình.
        
        Với mỗi cấu hình (n, L, contamination):
        1. Flatten windows data
        2. Căn chỉnh labels
        3. Train và predict với Isolation Forest
        4. Train và predict với One-Class SVM
        
        Returns:
            Dictionary chứa tất cả predictions và labels.
        """
        self.logger.info("=" * 60)
        self.logger.info("BẮT ĐẦU ABLATION STUDY - ML BASELINES")
        self.logger.info("=" * 60)
        
        self.results_ = {
            'iso': {},
            'svm': {},
            'labels_true': {}
        }
        
        total_configs = (
            len(self.n_components_list) * 
            len(self.window_sizes_list) * 
            len(self.contamination_values)
        )
        current_config = 0
        
        for n in self.n_components_list:
            self.results_['iso'][n] = {}
            self.results_['svm'][n] = {}
            self.results_['labels_true'][n] = {}
            
            for L in self.window_sizes_list:
                self.results_['iso'][n][L] = {}
                self.results_['svm'][n][L] = {}
                
                # 1. Flatten dữ liệu windows
                X_3d = self.windows_dict[n][L]
                X_2d = self._flatten_windows(X_3d)
                
                if verbose:
                    self.logger.info(f"\n--- PCA(n={n}), L={L} ---")
                    self.logger.info(f"  Input shape: {X_3d.shape} -> {X_2d.shape}")
                
                # 2. Căn chỉnh nhãn
                y_aligned = MLLabels.align_labels_with_windows(
                    self.labels_raw, window_size=L
                )
                self.results_['labels_true'][n][L] = y_aligned
                
                if verbose:
                    n_anomalies = np.sum(y_aligned == 1)
                    self.logger.info(f"  Aligned labels: {len(y_aligned)} samples, "
                                   f"{n_anomalies} anomalies")
                
                # 3. Chạy với các giá trị contamination khác nhau
                for fraction in self.contamination_values:
                    current_config += 1
                    
                    # Isolation Forest
                    iso_model = IsolationForestModel(
                        contamination=fraction,
                        random_state=42
                    )
                    y_pred_iso = iso_model.fit_predict(X_2d)
                    # Chuyển -1/1 thành 1/-1 để phù hợp với convention
                    y_pred_iso = np.where(y_pred_iso == 1, -1, 1)
                    self.results_['iso'][n][L][fraction] = y_pred_iso
                    
                    self.param_logs_.append({
                        'n_components': n,
                        'L': L,
                        'model': 'Isolation Forest',
                        'params': f"contamination={fraction}"
                    })
                    
                    # One-Class SVM
                    svm_model = OneClassSVMModel(nu=fraction)
                    y_pred_svm = svm_model.fit_predict(X_2d)
                    y_pred_svm = np.where(y_pred_svm == 1, -1, 1)
                    self.results_['svm'][n][L][fraction] = y_pred_svm
                    
                    self.param_logs_.append({
                        'n_components': n,
                        'L': L,
                        'model': 'One-Class SVM',
                        'params': f"nu={fraction}, kernel='rbf', gamma='scale'"
                    })
                    
                    if verbose:
                        iso_acc = np.mean(y_pred_iso == y_aligned) * 100
                        self.logger.info(
                            f"  [{current_config}/{total_configs}] "
                            f"contamination={fraction}: "
                            f"ISO accuracy={iso_acc:.1f}%"
                        )
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("HOÀN TẤT ABLATION STUDY")
        self.logger.info("=" * 60)
        
        return self.results_
    
    def save_outputs(self, output_dir: str = "outputs") -> Dict:
        """
        Lưu kết quả ra file.
        
        Output files:
            - ml_preds.npy: Dictionary chứa tất cả predictions
            - ml_param_log.csv: CSV log các tham số đã thử
            
        Args:
            output_dir: Thư mục lưu file.
            
        Returns:
            Dictionary đường dẫn các file đã lưu.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Lưu kết quả ML Baselines...")
        
        files = {}
        
        # Lưu predictions
        preds_path = output_path / "ml_preds.npy"
        np.save(preds_path, self.results_, allow_pickle=True)
        files['ml_preds'] = str(preds_path)
        self.logger.info(f"  - Đã lưu: {preds_path}")
        
        # Lưu param log
        log_path = output_path / "ml_param_log.csv"
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=['n_components', 'L', 'model', 'params']
            )
            writer.writeheader()
            writer.writerows(self.param_logs_)
        files['ml_param_log'] = str(log_path)
        self.logger.info(f"  - Đã lưu: {log_path}")
        
        return files
    
    def get_predictions(
        self,
        model: str,
        n: int,
        L: int,
        contamination: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lấy predictions cho một cấu hình cụ thể.
        
        Args:
            model: 'iso' hoặc 'svm'.
            n: Số PCA components.
            L: Kích thước window.
            contamination: Giá trị contamination.
            
        Returns:
            Tuple (predictions, labels_true).
        """
        if model == 'iso':
            preds = self.results_['iso'][n][L][contamination]
        else:
            preds = self.results_['svm'][n][L][contamination]
        
        labels = self.results_['labels_true'][n][L]
        
        return preds, labels


class MLBaselineManager:
    """
    Quản lý việc chạy ML baselines một cách đơn giản.
    """
    
    def __init__(
        self,
        windows_dict_path: str = "outputs/windows_dict.npy",
        labels_path: str = "outputs/labels_raw.npy",
        output_dir: str = "outputs"
    ):
        """
        Khởi tạo ML Baseline Manager.
        
        Args:
            windows_dict_path: Đường dẫn file windows_dict.npy.
            labels_path: Đường dẫn file labels_raw.npy.
            output_dir: Thư mục lưu kết quả.
        """
        self.windows_dict_path = windows_dict_path
        self.labels_path = labels_path
        self.output_dir = output_dir
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
        
        # Load data
        self.logger.info(f"Đọc data từ: {self.windows_dict_path}")
        data = np.load(self.windows_dict_path, allow_pickle=True)
        windows_dict = data.item() if hasattr(data, 'item') else data
        
        self.logger.info(f"Đọc labels từ: {self.labels_path}")
        labels_raw = np.load(self.labels_path)
        
        # Khởi tạo runner
        runner = MLBaselineRunner(
            windows_dict=windows_dict,
            labels_raw=labels_raw
        )
        
        # Chạy ablation study
        results = runner.run_ablation_study()
        
        # Lưu outputs
        saved_files = runner.save_outputs(self.output_dir)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("HOÀN TẤT ML BASELINES")
        self.logger.info("=" * 60)
        
        return {
            'results': results,
            'param_logs': runner.param_logs_,
            'saved_files': saved_files
        }


def main():
    """
    Hàm main để chạy ML Baselines độc lập.
    
    Usage:
        python -m modules.ml_baselines
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Chạy ML Baselines')
    parser.add_argument('--windows', default='outputs/windows_dict.npy',
                       help='Đường dẫn windows_dict.npy')
    parser.add_argument('--labels', default='outputs/labels_raw.npy',
                       help='Đường dẫn labels_raw.npy')
    parser.add_argument('--output', default='outputs', help='Thư mục output')
    
    args = parser.parse_args()
    
    manager = MLBaselineManager(
        windows_dict_path=args.windows,
        labels_path=args.labels,
        output_dir=args.output
    )
    
    results = manager.run()
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
