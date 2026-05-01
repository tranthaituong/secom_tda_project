"""
Module TDA (Topological Data Analysis) Features - Nhiệm vụ 3 (NV3)
===================================================================

Module này trích xuất đặc trưng topological từ dữ liệu SECOM sử dụng
Persistent Homology để phát hiện anomaly trong quá trình sản xuất bán dẫn.

Lý thuyết:
- Persistent Homology theo dõi các cấu trúc topological (connected components, 
  loops, voids) khi thay đổi ngưỡng similarity.
- H1 (homology dimension 1) bắt các vòng lặp/cycles - phù hợp để phát hiện
  chu kỳ rung động bất thường trong máy móc.
- Persistence = Death - Birth: độ bền của cấu trúc topological

Ứng dụng:
- Trong sản xuất bán dẫn, lỗi máy móc thường biểu hiện qua các chu kỳ rung
  động bất thường mà H1 có thể bắt được.

Author: GTMT Project
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

# TDA library - ripser cho persistence homology
try:
    from ripser import ripser
    from persim import plot_diagrams
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    logging.warning("Ripser not available. TDA features will use placeholder.")


class TDAFeatureExtractor:
    """
    Trích xuất đặc trưng topological sử dụng Persistent Homology.
    
    Attributes:
        homology_dimensions: Các chiều đồng đều cần tính (default: [1] cho H1 loops).
        max_edge: Ngưỡng tối đa cho Rips complex.
        n_samples: Giới hạn số lượng mẫu để tính toán (None = tất cả).
    """
    
    def __init__(
        self,
        homology_dimensions: List[int] = None,
        max_edge: float = None,
        n_samples: int = None,
        threshold_percentile: float = 95.0
    ):
        """
        Khởi tạo TDA Feature Extractor.
        
        Args:
            homology_dimensions: Danh sách chiều đồng đều (0=components, 1=loops).
            max_edge: Ngưỡng max cho Rips filtration (auto-calculate nếu None).
            n_samples: Giới hạn samples (None = tất cả).
            threshold_percentile: Phân vị để xác định ngưỡng anomaly (default: 95).
        """
        self.homology_dimensions = homology_dimensions or [1]
        self.max_edge = max_edge
        self.n_samples = n_samples
        self.threshold_percentile = threshold_percentile
        
        self.logger = logging.getLogger(__name__)
        self.diagrams_ = {}
        self.features_ = {}
        
    def _check_ripser(self):
        """Kiểm tra xem ripser có được cài đặt không."""
        if not RIPSER_AVAILABLE:
            raise ImportError(
                "Ripser is required for TDA features. "
                "Please install: pip install ripser persim"
            )
    
    def _compute_rips_persistence(
        self,
        point_cloud: np.ndarray,
        max_dim: int = 1
    ) -> Dict:
        """
        Tính Persistent Homology sử dụng Ripser.
        
        Args:
            point_cloud: Ma trận điểm với shape (n_points, n_dimensions).
            max_dim: Chiều homology tối đa cần tính.
            
        Returns:
            Dictionary chứa persistence diagrams.
        """
        self._check_ripser()
        
        params = {
            'maxdim': max_dim,
            'metric': 'euclidean'
        }
        
        if self.max_edge is not None:
            params['thresh'] = self.max_edge
        
        result = ripser(point_cloud, **params)
        return result['dgms']
    
    def _extract_h1_persistence(self, diagram: np.ndarray) -> np.ndarray:
        """
        Trích xuất persistence values từ H1 diagram.
        
        Persistence = Death - Birth. Giá trị lớn = cấu trúc bền vững.
        
        Args:
            diagram: H1 persistence diagram từ ripser.
            
        Returns:
            Mảng 1D chứa persistence values (bỏ qua infinite persistence).
        """
        if len(diagram) == 0:
            return np.array([0.0])
        
        # Bỏ qua điểm có death = inf (không bao giờ chết)
        finite_diagram = diagram[np.isfinite(diagram[:, 1])]
        
        if len(finite_diagram) == 0:
            return np.array([0.0])
        
        # Tính persistence = death - birth
        persistence = finite_diagram[:, 1] - finite_diagram[:, 0]
        return persistence
    
    def compute_topo_features(
        self,
        windows_dict: Dict,
        verbose: bool = True
    ) -> Dict:
        """
        Tính đặc trưng topological cho tất cả các cấu hình windows.
        
        Args:
            windows_dict: Dictionary lồng nhau {n_components: {L: windows_array}}.
            verbose: Có in progress không.
            
        Returns:
            Dictionary {n: {L: topo_features_array}}.
        """
        self._check_ripser()
        
        self.logger.info("=" * 60)
        self.logger.info("TÍNH ĐẶC TRƯNG TOPOLOGICAL (TDA)")
        self.logger.info("=" * 60)
        
        self.features_ = {}
        self.diagrams_ = {}
        
        for n in sorted(windows_dict.keys()):
            self.features_[n] = {}
            self.diagrams_[n] = {}
            
            for L in sorted(windows_dict[n].keys()):
                if verbose:
                    self.logger.info(f"  - Đang xử lý: PCA(n={n}), L={L}")
                
                windows = windows_dict[n][L]
                n_windows = len(windows)
                
                # Giới hạn số lượng windows nếu cần
                if self.n_samples and n_windows > self.n_samples:
                    windows = windows[:self.n_samples]
                
                topo_features = []
                diagrams_for_config = []
                
                # Tính TDA cho từng window
                for i, window in enumerate(windows):
                    # Tính persistence homology
                    diagrams = self._compute_rips_persistence(window, max_dim=1)
                    
                    # Trích xuất H1 persistence
                    if len(diagrams) > 1 and len(diagrams[1]) > 0:
                        h1_persistence = self._extract_h1_persistence(diagrams[1])
                        diagrams_for_config.append(diagrams[1])
                    else:
                        h1_persistence = np.array([0.0])
                    
                    topo_features.append(h1_persistence)
                    
                    if verbose and (i + 1) % 100 == 0:
                        self.logger.info(f"      Đã xử lý {i + 1}/{n_windows} windows")
                
                self.features_[n][L] = topo_features
                self.diagrams_[n][L] = diagrams_for_config
                
                self.logger.info(f"      Hoàn tất: {n_windows} windows")
        
        return self.features_
    
    def extract_max_persistence(
        self,
        features_dict: Dict = None
    ) -> Dict:
        """
        Trích xuất max persistence làm anomaly score cho mỗi window.
        
        Ngưỡng 95th percentile thường được dùng để xác định anomaly:
        - Windows có max persistence > ngưỡng = bất thường
        
        Args:
            features_dict: Dictionary features (dùng self.features_ nếu None).
            
        Returns:
            Dictionary {n: {L: (scores, threshold)}}.
        """
        features_dict = features_dict or self.features_
        
        self.logger.info("Trích xuất Max Persistence scores...")
        
        results = {}
        
        for n in sorted(features_dict.keys()):
            results[n] = {}
            
            for L in sorted(features_dict[n].keys()):
                features = features_dict[n][L]
                max_persistence = []
                
                for window_features in features:
                    if len(window_features) > 0:
                        max_persistence.append(np.max(window_features))
                    else:
                        max_persistence.append(0.0)
                
                max_persistence = np.array(max_persistence)
                threshold = np.percentile(max_persistence, self.threshold_percentile)
                
                results[n][L] = {
                    'scores': max_persistence,
                    'threshold': threshold
                }
                
                self.logger.info(f"  - PCA(n={n}), L={L}: "
                               f"threshold={threshold:.4f}")
        
        return results
    
    def predict_anomaly(
        self,
        scores_dict: Dict = None,
        labels_raw: np.ndarray = None
    ) -> Dict:
        """
        Chuyển scores thành nhãn anomaly dựa trên ngưỡng percentile.
        
        Args:
            scores_dict: Dictionary scores từ extract_max_persistence.
            labels_raw: Nhãn gốc để đồng bộ với windows (tùy chọn).
            
        Returns:
            Dictionary {n: {L: (predictions, labels_aligned)}}.
        """
        scores_dict = scores_dict or self.extract_max_persistence()
        
        predictions = {}
        
        for n in sorted(scores_dict.keys()):
            predictions[n] = {}
            
            for L in sorted(scores_dict[n].keys()):
                scores = scores_dict[n][L]['scores']
                threshold = scores_dict[n][L]['threshold']
                
                # Binary prediction: 1 = anomaly, 0 = normal
                preds = (scores >= threshold).astype(int)
                predictions[n][L] = preds
                
                n_anomalies = np.sum(preds)
                self.logger.info(f"  - PCA(n={n}), L={L}: "
                               f"{n_anomalies}/{len(preds)} windows detected as anomaly")
        
        return predictions
    
    def save_outputs(self, output_dir: str = "outputs") -> Dict:
        """
        Lưu các kết quả TDA ra file.
        
        Output files:
            - topo_features.npy: Dictionary đặc trưng topological
            - topo_diagrams.npy: Dictionary persistence diagrams
            - topo_scores.npy: Dictionary max persistence scores
            
        Args:
            output_dir: Thư mục lưu file.
            
        Returns:
            Dictionary đường dẫn các file đã lưu.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Lưu kết quả TDA...")
        
        files = {}
        
        # Lưu features
        features_path = output_path / "topo_features.npy"
        np.save(features_path, self.features_, allow_pickle=True)
        files['topo_features'] = str(features_path)
        self.logger.info(f"  - Đã lưu: {features_path}")
        
        # Lưu diagrams (chỉ lưu một phần để tiết kiệm space)
        diagrams_path = output_path / "topo_diagrams.npy"
        np.save(diagrams_path, self.diagrams_, allow_pickle=True)
        files['topo_diagrams'] = str(diagrams_path)
        self.logger.info(f"  - Đã lưu: {diagrams_path}")
        
        # Lưu scores
        scores_dict = self.extract_max_persistence()
        scores_path = output_path / "topo_scores.npy"
        np.save(scores_path, scores_dict, allow_pickle=True)
        files['topo_scores'] = str(scores_path)
        self.logger.info(f"  - Đã lưu: {scores_path}")
        
        return files


class TDAManager:
    """
    Quản lý việc trích xuất và sử dụng TDA features.
    
    Cung cấp interface đơn giản để chạy toàn bộ pipeline TDA.
    """
    
    def __init__(
        self,
        windows_dict_path: str = "outputs/windows_dict.npy",
        output_dir: str = "outputs"
    ):
        """
        Khởi tạo TDA Manager.
        
        Args:
            windows_dict_path: Đường dẫn file windows_dict.npy từ data_processing.
            output_dir: Thư mục lưu kết quả.
        """
        self.windows_dict_path = windows_dict_path
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
    def run_tda_pipeline(
        self,
        homology_dimensions: List[int] = None,
        threshold_percentile: float = 95.0
    ) -> Dict:
        """
        Chạy toàn bộ pipeline TDA.
        
        Args:
            homology_dimensions: Các chiều homology cần tính.
            threshold_percentile: Phân vị ngưỡng anomaly.
            
        Returns:
            Dictionary chứa kết quả và đường dẫn file.
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("BẮT ĐẦU PIPELINE TDA")
        self.logger.info("=" * 60)
        
        # Load windows data
        self.logger.info(f"Đọc windows data từ: {self.windows_dict_path}")
        data = np.load(self.windows_dict_path, allow_pickle=True)
        windows_dict = data.item() if hasattr(data, 'item') else data
        
        # Khởi tạo extractor
        extractor = TDAFeatureExtractor(
            homology_dimensions=homology_dimensions,
            threshold_percentile=threshold_percentile
        )
        
        # Tính features
        extractor.compute_topo_features(windows_dict)
        
        # Lưu outputs
        saved_files = extractor.save_outputs(self.output_dir)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("HOÀN TẤT PIPELINE TDA")
        self.logger.info("=" * 60)
        
        return {
            'features': extractor.features_,
            'diagrams': extractor.diagrams_,
            'scores': extractor.extract_max_persistence(),
            'saved_files': saved_files
        }


def main():
    """
    Hàm main để chạy TDA độc lập.
    
    Usage:
        python -m modules.tda_features
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Trích xuất đặc trưng TDA')
    parser.add_argument('--windows', default='outputs/windows_dict.npy',
                       help='Đường dẫn windows_dict.npy')
    parser.add_argument('--output', default='outputs', help='Thư mục output')
    parser.add_argument('--percentile', type=float, default=95.0,
                       help='Phan vi nguong anomaly')
    
    args = parser.parse_args()
    
    manager = TDAManager(
        windows_dict_path=args.windows,
        output_dir=args.output
    )
    
    results = manager.run_tda_pipeline(threshold_percentile=args.percentile)
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
