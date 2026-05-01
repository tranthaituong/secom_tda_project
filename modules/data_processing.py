"""
Module xử lý dữ liệu SECOM - Nhiệm vụ 1 (NV1)
==============================================

Module này thực hiện các bước tiền xử lý dữ liệu:
1. Load dữ liệu từ file SECOM (.data)
2. Xử lý giá trị khuyết (NaN) bằng mean imputation
3. Loại bỏ các feature có variance = 0 (constant features)
4. Chuẩn hóa dữ liệu bằng StandardScaler
5. Giảm chiều bằng PCA (Principal Component Analysis)
6. Tạo sliding windows cho phân tích time-series

Dataset: SECOM Semiconductor Manufacturing Dataset
- 1567 samples x 591 features
- Labels: -1 (pass/normal), 1 (fail/anomaly)
- 104 failures trong tập dữ liệu

Author: GTMT Project
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def setup_logger(log_file: str = None) -> logging.Logger:
    """
    Thiết lập logger để ghi log ra file và console.
    
    Args:
        log_file: Đường dẫn file log. Nếu None, chỉ log ra console.
        
    Returns:
        Logger instance đã cấu hình.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    stride: int = 1
) -> np.ndarray:
    """
    Tạo các cửa sổ trượt (sliding windows) từ chuỗi thời gian.
    
    Args:
        data: Mảng 2D với shape (n_samples, n_features).
        window_size: Kích thước cửa sổ (L) - số lượng samples trong mỗi window.
        stride: Bước nhảy giữa các windows (mặc định=1).
        
    Returns:
        Mảng 3D với shape (n_windows, window_size, n_features).
        
    Example:
        >>> data = np.random.rand(100, 5)  # 100 samples, 5 features
        >>> windows = create_sliding_windows(data, window_size=20, stride=1)
        >>> windows.shape
        (81, 20, 5)
    """
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i : i + window_size])
    return np.array(windows)


class SECOMDataProcessor:
    """
    Class xử lý dữ liệu SECOM cho phân tích anomaly detection.
    
    Attributes:
        data_file: Đường dẫn file dữ liệu secom.data
        labels_file: Đường dẫn file nhãn secom_labels.data
        pca_components: Số lượng thành phần PCA
        window_sizes: Danh sách kích thước cửa sổ trượt
        output_dir: Thư mục lưu kết quả
    """
    
    def __init__(
        self,
        data_file: str = "data/secom.data",
        labels_file: str = "data/secom_labels.data",
        pca_components: list = None,
        window_sizes: list = None,
        output_dir: str = "outputs"
    ):
        """
        Khởi tạo DataProcessor.
        
        Args:
            data_file: Đường dẫn file dữ liệu SECOM (space-separated).
            labels_file: Đường dẫn file nhãn (-1=pass, 1=fail).
            pca_components: Danh sách số thành phần PCA cần fit (default: [2, 3, 5]).
            window_sizes: Danh sách kích thước window L (default: [20, 30, 50]).
            output_dir: Thư mục lưu các file output.
        """
        self.data_file = data_file
        self.labels_file = labels_file
        self.pca_components = pca_components or [2, 3, 5]
        self.window_sizes = window_sizes or [20, 30, 50]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Các thuộc tính sẽ được set trong quá trình xử lý
        self.data_raw = None
        self.labels_raw = None
        self.data_filled = None
        self.data_filtered = None
        self.data_scaled = None
        self.pca_results = {}
        self.windows_dict = {}
        self.scaler = None
        self.logger = None
    
    def load_data(self, logger: logging.Logger = None) -> tuple:
        """
        Đọc dữ liệu từ file SECOM.
        
        Args:
            logger: Logger instance (tùy chọn).
            
        Returns:
            Tuple (data, labels) dạng DataFrame.
            
        Raises:
            FileNotFoundError: Nếu file không tồn tại.
            ValueError: Nếu dữ liệu trống.
        """
        self.logger = logger or setup_logger()
        
        self.logger.info("=" * 60)
        self.logger.info("BƯỚC 1: Đọc dữ liệu SECOM")
        self.logger.info("=" * 60)
        
        try:
            self.data_raw = pd.read_csv(self.data_file, sep=' ', header=None)
            self.labels_raw = pd.read_csv(self.labels_file, sep=' ', header=None)
        except FileNotFoundError as e:
            self.logger.error(f"Không tìm thấy file: {e}")
            raise
        
        self.logger.info(f"  - Kích thước data: {self.data_raw.shape}")
        self.logger.info(f"  - Kích thước labels: {self.labels_raw.shape}")
        
        # Trích xuất cột nhãn (cột đầu tiên) và bỏ timestamp
        self.labels_raw = self.labels_raw.iloc[:, 0].values.astype(int)
        
        n_failures = np.sum(self.labels_raw == 1)
        self.logger.info(f"  - Số lượng mẫu bình thường (label=-1): {np.sum(self.labels_raw == -1)}")
        self.logger.info(f"  - Số lượng mẫu bất thường (label=1): {n_failures}")
        
        return self.data_raw, self.labels_raw
    
    def fill_missing_values(self) -> pd.DataFrame:
        """
        Xử lý giá trị khuyết (NaN) bằng mean imputation.
        
        Các bước:
        1. Thay NaN bằng mean của từng cột
        2. Nếu cột toàn NaN (mean cũng NaN), thay bằng 0
        
        Returns:
            DataFrame đã xử lý NaN.
        """
        self.logger.info("=" * 60)
        self.logger.info("BƯỚC 2: Xử lý giá trị khuyết (NaN)")
        self.logger.info("=" * 60)
        
        total_nans_before = self.data_raw.isna().sum().sum()
        self.logger.info(f"  - Tổng số NaN ban đầu: {total_nans_before}")
        
        self.data_filled = self.data_raw.fillna(self.data_raw.mean())
        self.data_filled = self.data_filled.fillna(0)
        
        total_nans_after = self.data_filled.isna().sum().sum()
        self.logger.info(f"  - Số NaN sau xử lý: {total_nans_after}")
        
        return self.data_filled
    
    def remove_constant_features(self) -> pd.DataFrame:
        """
        Loại bỏ các feature có variance = 0 (giá trị không đổi).
        
        Các feature constant không mang thông tin cho ML models.
        
        Returns:
            DataFrame đã loại bỏ constant features.
        """
        self.logger.info("=" * 60)
        self.logger.info("BƯỚC 3: Loại bỏ constant features (variance=0)")
        self.logger.info("=" * 60)
        
        variances = self.data_filled.var()
        zero_var_cols = variances[variances == 0].index.tolist()
        
        self.logger.info(f"  - Số cột bị xóa: {len(zero_var_cols)}")
        
        self.data_filtered = self.data_filled.drop(columns=zero_var_cols)
        self.logger.info(f"  - Kích thước sau khi loại bỏ: {self.data_filtered.shape}")
        
        return self.data_filtered
    
    def standardize(self) -> np.ndarray:
        """
        Chuẩn hóa dữ liệu bằng StandardScaler (z-score normalization).
        
        Mỗi feature được biến đổi thành: (x - mean) / std
        
        Returns:
            Mảng numpy đã chuẩn hóa.
        """
        self.logger.info("=" * 60)
        self.logger.info("BƯỚC 4: Chuẩn hóa dữ liệu (StandardScaler)")
        self.logger.info("=" * 60)
        
        self.scaler = StandardScaler()
        self.data_scaled = self.scaler.fit_transform(self.data_filtered)
        
        self.logger.info(f"  - Kích thước dữ liệu sau scaling: {self.data_scaled.shape}")
        self.logger.info(f"  - Mean của scaled data (should ≈ 0): {np.mean(self.data_scaled):.6f}")
        
        return self.data_scaled
    
    def apply_pca(self) -> dict:
        """
        Áp dụng PCA để giảm chiều dữ liệu.
        
        Chạy PCA với nhiều giá trị n_components để thử nghiệm:
        - PCA=2: Giữ lại thông tin chính, loại bỏ nhiễu
        - PCA=3: Thêm 1 chiều phụ
        - PCA=5: Thêm thông tin chi tiết hơn
        
        Returns:
            Dictionary với key là n_components, value là data đã transform.
        """
        self.logger.info("=" * 60)
        self.logger.info("BƯỚC 5: Áp dụng PCA (Principal Component Analysis)")
        self.logger.info("=" * 60)
        
        for n in self.pca_components:
            pca = PCA(n_components=n)
            transformed = pca.fit_transform(self.data_scaled)
            self.pca_results[n] = transformed
            
            explained_var = sum(pca.explained_variance_ratio_) * 100
            self.logger.info(f"  - PCA(n={n}): shape={transformed.shape}, "
                           f"explained_variance={explained_var:.2f}%")
        
        return self.pca_results
    
    def create_windows(self, stride: int = 1) -> dict:
        """
        Tạo sliding windows cho dữ liệu PCA.
        
        Args:
            stride: Bước nhảy giữa các windows (default=1).
            
        Returns:
            Dictionary lồng nhau: windows_dict[n_components][L] = windows_array
        """
        self.logger.info("=" * 60)
        self.logger.info("BƯỚC 6: Tạo Sliding Windows")
        self.logger.info("=" * 60)
        
        self.windows_dict = {}
        
        for n in self.pca_components:
            self.windows_dict[n] = {}
            pca_data = self.pca_results[n]
            
            for L in self.window_sizes:
                windows = create_sliding_windows(pca_data, window_size=L, stride=stride)
                self.windows_dict[n][L] = windows
                self.logger.info(f"  - PCA(n={n}), L={L}: shape={windows.shape}")
        
        return self.windows_dict
    
    def save_outputs(self) -> dict:
        """
        Lưu các kết quả đã xử lý ra file .npy.
        
        Output files:
            - data_pca.npy: Dictionary các ma trận PCA {n: data}
            - windows_dict.npy: Dictionary lồng nhau {n: {L: windows}}
            - labels_aligned.npy: Mảng nhãn đã căn chỉnh với số windows
            
        Returns:
            Dictionary chứa đường dẫn các file đã lưu.
        """
        self.logger.info("=" * 60)
        self.logger.info("LƯU KẾT QUẢ")
        self.logger.info("=" * 60)
        
        # Lưu PCA results
        pca_path = self.output_dir / "data_pca.npy"
        np.save(pca_path, self.pca_results, allow_pickle=True)
        self.logger.info(f"  - Đã lưu PCA data: {pca_path}")
        
        # Lưu windows dictionary
        windows_path = self.output_dir / "windows_dict.npy"
        np.save(windows_path, self.windows_dict, allow_pickle=True)
        self.logger.info(f"  - Đã lưu windows dict: {windows_path}")
        
        # Lưu nhãn gốc
        labels_path = self.output_dir / "labels_raw.npy"
        np.save(labels_path, self.labels_raw)
        self.logger.info(f"  - Đã lưu raw labels: {labels_path}")
        
        return {
            'pca_data': str(pca_path),
            'windows_dict': str(windows_path),
            'labels_raw': str(labels_path)
        }
    
    def process_all(self, log_file: str = None) -> dict:
        """
        Chạy toàn bộ pipeline xử lý dữ liệu.
        
        Pipeline bao gồm:
        1. Load data
        2. Fill NaN
        3. Remove constant features
        4. Standardize
        5. Apply PCA
        6. Create sliding windows
        7. Save outputs
        
        Args:
            log_file: Đường dẫn file log (tùy chọn).
            
        Returns:
            Dictionary chứa các kết quả và đường dẫn file.
        """
        self.logger = setup_logger(str(self.output_dir / log_file) if log_file else None)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("BẮT ĐẦU QUÁ TRÌNH XỬ LÝ DỮ LIỆU SECOM")
        self.logger.info("=" * 60)
        
        # Chạy pipeline
        self.load_data()
        self.fill_missing_values()
        self.remove_constant_features()
        self.standardize()
        self.apply_pca()
        self.create_windows()
        saved_files = self.save_outputs()
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("HOÀN TẤT QUÁ TRÌNH XỬ LÝ DỮ LIỆU")
        self.logger.info("=" * 60)
        
        return {
            'windows_dict': self.windows_dict,
            'pca_results': self.pca_results,
            'labels_raw': self.labels_raw,
            'scaler': self.scaler,
            'saved_files': saved_files
        }


def main():
    """
    Hàm main để chạy xử lý dữ liệu độc lập.
    
    Usage:
        python -m modules.data_processing
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Xử lý dữ liệu SECOM')
    parser.add_argument('--data', default='data/secom.data', help='Đường dẫn file data')
    parser.add_argument('--labels', default='data/secom_labels.data', help='Đường dẫn file labels')
    parser.add_argument('--output', default='outputs', help='Thư mục output')
    parser.add_argument('--pca', nargs='+', type=int, default=[2, 3, 5],
                       help='Danh sách số PCA components')
    parser.add_argument('--windows', nargs='+', type=int, default=[20, 30, 50],
                       help='Danh sách kích thước window')
    
    args = parser.parse_args()
    
    processor = SECOMDataProcessor(
        data_file=args.data,
        labels_file=args.labels,
        pca_components=args.pca,
        window_sizes=args.windows,
        output_dir=args.output
    )
    
    results = processor.process_all(log_file='data_processing.log')
    
    return results


if __name__ == "__main__":
    main()
