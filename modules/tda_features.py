"""
Module TDA (Topological Data Analysis) Features - Nhiệm vụ 3 (NV3)
===================================================================

Module này trích xuất đặc trưng topological từ dữ liệu SECOM sử dụng:
1. Vietoris-Rips Filtration (cài đặt thủ công)
2. Persistent Homology - H1 (loops/cycles)
3. Lọc nhiễu topo (noise filtering)
4. Grid search cho ngưỡng lọc nhiễu và ngưỡng phân loại

Lý thuyết:
- Persistent Homology theo dõi các cấu trúc topological khi thay đổi ngưỡng similarity.
- H1 (homology dimension 1) bắt các vòng lặp/cycles.
- Persistence = Death - Birth: độ bền của cấu trúc topological.

Author: GTMT Project
"""

import logging
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import combinations
from tqdm import tqdm


# ============================================================================
# ĐƯỜNG DẪN MẶC ĐỊNH
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DEFAULT_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DEFAULT_TDA_DIR = BASE_DIR / "outputs"


# ============================================================================
# VIETORIS-RIPS PERSISTENCE (CÀI ĐẶT THỦ CÔNG)
# ============================================================================

class VietorisRipsRaw:
    """
    Cài đặt Vietoris-Rips Filtration và Persistent Homology thủ công.
    
    Không cần Ripser - sử dụng thuật toán matrix reduction.
    """
    
    def __init__(self, max_dim: int = 1, eps_percentile: int = 60):
        """
        Khởi tạo Vietoris-Rips filter.
        
        Args:
            max_dim: Chiều homology tối đa (0=components, 1=loops, 2=triangles)
            eps_percentile: Phân vị để tính ngưỡng epsilon cho filtration
        """
        self.max_dim = max_dim
        self.eps_percentile = eps_percentile
    
    def compute_persistence(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Tính Persistent Homology từ point cloud.
        
        Args:
            X: Ma trận điểm với shape (n_points, n_dimensions)
            
        Returns:
            List of persistence diagrams [dgm0, dgm1, ...]
        """
        # Tính ma trận khoảng cách Euclidean
        D = np.sqrt(((X[:, None, :] - X[None, :, :])**2).sum(axis=2))
        n = D.shape[0]
        
        # Tính epsilon từ phân vị khoảng cách
        tri = D[np.triu_indices(n, k=1)]
        if len(tri) == 0:
            return [np.empty((0, 2)), np.empty((0, 2))]
        
        eps = np.percentile(tri, self.eps_percentile)
        
        # Xây dựng danh sách simplices (filtration)
        simplices = []
        
        # 0-simplices (điểm) - birth = 0
        for i in range(n):
            simplices.append((0.0, 0, (i,)))
        
        # 1-simplices (cạnh) - birth = khoảng cách
        for i, j in combinations(range(n), 2):
            if D[i, j] <= eps:
                simplices.append((D[i, j], 1, (i, j)))
        
        # 2-simplices (tam giác) - birth = max của 3 cạnh
        if self.max_dim >= 1:
            for i, j, k in combinations(range(n), 3):
                d = max(D[i, j], D[i, k], D[j, k])
                if d <= eps:
                    simplices.append((d, 2, (i, j, k)))
        
        # Sắp xếp theo birth time và dimension
        simplices.sort(key=lambda s: (s[0], s[1]))
        
        # Tạo index map
        s2i = {s[2]: idx for idx, s in enumerate(simplices)}
        
        # Matrix reduction (coboundary matrix)
        columns = [set() for _ in range(len(simplices))]
        for j, (_, dim, verts) in enumerate(simplices):
            if dim > 0:
                for skip in range(len(verts)):
                    face = tuple(verts[m] for m in range(len(verts)) if m != skip)
                    if face in s2i:
                        columns[j].add(s2i[face])
        
        # Reduce coboundary matrix
        pivot_to_col = {}
        dgms = [[] for _ in range(self.max_dim + 1)]
        
        for j in range(len(columns)):
            while columns[j]:
                pivot = max(columns[j])
                if pivot in pivot_to_col:
                    columns[j].symmetric_difference_update(columns[pivot_to_col[pivot]])
                else:
                    pivot_to_col[pivot] = j
                    dgms[simplices[pivot][1]].append([simplices[pivot][0], simplices[j][0]])
                    break
        
        return [np.array(d) if len(d) > 0 else np.empty((0, 2)) for d in dgms]


# ============================================================================
# TDA FEATURES VỚI LỌC NHIỄU VÀ GRID SEARCH
# ============================================================================

class TDAFeatureExtractor:
    """
    Trích xuất đặc trưng topological với lọc nhiễu.
    
    Thuộc tính mới:
    - eps_percentile: Ngưỡng cho Vietoris-Rips
    - Grid search cho noise_threshold và classification_threshold
    """
    
    def __init__(
        self,
        eps_percentile: int = 60,
        max_dim: int = 1,
        window_sizes: List[int] = None
    ):
        self.eps_percentile = eps_percentile
        self.max_dim = max_dim
        self.window_sizes = window_sizes or [20, 30, 50]
        self.rips = VietorisRipsRaw(max_dim=max_dim, eps_percentile=eps_percentile)
        self.logger = logging.getLogger(__name__)
        self.results = {}
    
    def filter_dgms(
        self,
        dgms_list: List,
        noise_threshold: float
    ) -> np.ndarray:
        """
        Lọc nhiễu topo từ list persistence diagrams.
        
        Args:
            dgms_list: List các diagrams
            noise_threshold: Ngưỡng lọc (persistence < threshold = nhiễu)
            
        Returns:
            Mảng max persistence đã lọc
        """
        filtered_max_pers = []
        for dgms in dgms_list:
            h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
            if h1.size > 0:
                pers = h1[:, 1] - h1[:, 0]
                valid_pers = pers[pers > noise_threshold]
                filtered_max_pers.append(np.max(valid_pers) if valid_pers.size > 0 else 0.0)
            else:
                filtered_max_pers.append(0.0)
        return np.array(filtered_max_pers)
    
    def load_processed_data(self, processed_dir = None) -> List[Tuple[str, pd.DataFrame]]:
        """
        Load tất cả file CSV đã xử lý từ thư mục processed.
        
        Returns:
            List of (filename, DataFrame) tuples
        """
        processed_path = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
        data_files = []
        
        csv_files = sorted(processed_path.glob('secom_processed_*.csv'))
        for f in csv_files:
            df = pd.read_csv(f)
            data_files.append((f.stem, df))
        
        return data_files
    
    def compute_tda_for_file(
        self,
        file_name: str,
        df: pd.DataFrame
    ) -> Dict:
        """
        Tính TDA cho một file đã xử lý.
        
        Args:
            file_name: Tên file (không có extension)
            df: DataFrame với các cột feature và 'Label'
            
        Returns:
            Dictionary chứa results cho mỗi window size
        """
        X = df.drop(columns=['Label']).values
        y = np.where(df['Label'].values == 1, 1, 0)
        
        results = {}
        
        for L in self.window_sizes:
            # Tạo sliding windows
            X_windows = [X[i:i+L] for i in range(len(X) - L + 1)]
            y_windows = y[L-1:]
            
            # Tính TDA cho từng window
            dgms_list = []
            for window in tqdm(X_windows, desc=f"TDA {file_name} L={L}", leave=False):
                dgms = self.rips.compute_persistence(window)
                dgms_list.append(dgms)
            
            results[L] = {
                'dgms': dgms_list,
                'labels': y_windows,
                'source': file_name
            }
        
        return results
    
    def run_full_tda(
        self,
        processed_dir = None,
        save_dir = None
    ) -> Dict:
        """
        Chạy TDA cho tất cả file đã xử lý.
        
        Args:
            processed_dir: Thư mục chứa file CSV đã xử lý
            save_dir: Thư mục lưu kết quả TDA
            
        Returns:
            Dictionary chứa tất cả kết quả
        """
        processed_dir = processed_dir or DEFAULT_PROCESSED_DIR
        save_dir = save_dir or DEFAULT_TDA_DIR
        os.makedirs(save_dir, exist_ok=True)
        
        # Load tất cả file
        data_files = self.load_processed_data(processed_dir)
        
        all_results = {}
        
        for file_name, df in tqdm(data_files, desc="Processing files"):
            self.logger.info(f"  Xử lý: {file_name}")
            file_results = self.compute_tda_for_file(file_name, df)
            all_results[file_name] = file_results
            
            # Lưu từng file
            for L, result in file_results.items():
                save_path = os.path.join(save_dir, f"{file_name}_L{L}.pkl")
                with open(save_path, 'wb') as f:
                    pickle.dump(result, f)
        
        self.results = all_results
        self.logger.info(f"\n[OK] Đã lưu TDA vào {save_dir}/")
        
        return all_results


class TDAEvaluatorWithGridSearch:
    """
    Đánh giá TDA với Grid Search cho:
    - noise_threshold: Ngưỡng lọc nhiễu topo
    - classification_threshold: Ngưỡng phân loại anomaly
    """
    
    def __init__(
        self,
        tda_dir = None,
        n_noise_levels: int = 30,
        n_threshold_levels: int = 30
    ):
        self.tda_dir = tda_dir or DEFAULT_TDA_DIR
        self.n_noise_levels = n_noise_levels
        self.n_threshold_levels = n_threshold_levels
        self.logger = logging.getLogger(__name__)
        self.rips = VietorisRipsRaw(max_dim=1, eps_percentile=60)
    
    def filter_dgms(self, dgms_list: List, noise_threshold: float) -> np.ndarray:
        """Lọc nhiễu và trả về max persistence."""
        filtered_max_pers = []
        for dgms in dgms_list:
            h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
            if h1.size > 0:
                pers = h1[:, 1] - h1[:, 0]
                valid_pers = pers[pers > noise_threshold]
                filtered_max_pers.append(np.max(valid_pers) if valid_pers.size > 0 else 0.0)
            else:
                filtered_max_pers.append(0.0)
        return np.array(filtered_max_pers)
    
    def grid_search(self) -> Tuple[List, Dict]:
        """
        Grid search 2D để tìm ngưỡng lọc nhiễu và ngưỡng phân loại tối ưu.
        
        Returns:
            results_table: List of results
            global_best: Dict với config tốt nhất
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        tda_files = sorted([f for f in os.listdir(self.tda_dir) if f.endswith('.pkl')])
        results_table = []
        global_best = {'f1': -1, 'dgms_sample': None, 'noise': 0, 'threshold': 0, 'title': ''}
        
        self.logger.info("🚀 ĐANG QUÉT 2D GRID SEARCH...")
        
        for f_name in tda_files:
            with open(os.path.join(self.tda_dir, f_name), 'rb') as f:
                data = pickle.load(f)
            
            y_true = np.array(data['labels'])
            
            # Tìm max persistence để đặt range
            all_pers = []
            for dgms in data['dgms']:
                h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
                if h1.size > 0:
                    all_pers.extend(h1[:, 1] - h1[:, 0])
            
            max_possible = np.max(all_pers) if all_pers else 0.5
            
            # Grid cho noise threshold
            noise_grid = np.linspace(0.0, max_possible * 0.8, self.n_noise_levels)
            
            best_f1, best_p, best_r = -1, 0, 0
            best_noise, best_t = 0, 0
            
            for noise_th in noise_grid:
                max_pers_array = self.filter_dgms(data['dgms'], noise_th)
                
                # Grid cho classification threshold
                thresholds = np.linspace(
                    max_pers_array.min() if max_pers_array.min() > 0 else 0,
                    max_pers_array.max() if max_pers_array.max() > 0 else 1,
                    self.n_threshold_levels
                )
                
                for t in thresholds:
                    preds = (max_pers_array > t).astype(int)
                    f1 = f1_score(y_true, preds, zero_division=0)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_p = precision_score(y_true, preds, zero_division=0)
                        best_r = recall_score(y_true, preds, zero_division=0)
                        best_noise = noise_th
                        best_t = t
            
            results_table.append([
                data['source'], data['L'],
                best_f1, best_p, best_r,
                best_noise, best_t
            ])
            
            # Cập nhật best toàn cục
            if best_f1 > global_best['f1']:
                error_indices = np.where(y_true == 1)[0]
                if len(error_indices) > 0:
                    global_best = {
                        'f1': best_f1,
                        'noise': best_noise,
                        'threshold': best_t,
                        'dgms_sample': data['dgms'][error_indices[0]],
                        'title': f"BEST: {data['source']} | L={data['L']} | Noise={best_noise:.3f} | Thresh={best_t:.3f}"
                    }
        
        return results_table, global_best
    
    def print_results(self, results_table: List):
        """In bảng kết quả."""
        print("\n" + "="*110)
        print(f"{'Source Data':<25} | {'L':<4} | {'F1':<8} | {'Precision':<10} | {'Recall':<8} | {'Lọc Nhiễu':<10} | {'Threshold':<10}")
        print("-" * 110)
        for res in results_table:
            print(f"{res[0]:<25} | {res[1]:<4} | {res[2]:.4f} | {res[3]:.4f}    | {res[4]:.4f} | {res[5]:.4f}     | {res[6]:.4f}")
        print("="*110)


# ============================================================================
# TDAManager (COMPATIBLE VỚI PIPELINE CŨ)
# ============================================================================

class TDAManager:
    """
    Quản lý việc trích xuất TDA features.
    
    Interface đơn giản để chạy toàn bộ pipeline TDA.
    """
    
    def __init__(
        self,
        processed_dir = None,
        tda_dir = None,
        eps_percentile: int = 60,
        window_sizes: List[int] = None
    ):
        self.processed_dir = processed_dir or DEFAULT_PROCESSED_DIR
        self.tda_dir = tda_dir or DEFAULT_TDA_DIR
        self.eps_percentile = eps_percentile
        self.window_sizes = window_sizes or [20, 30, 50]
        self.logger = logging.getLogger(__name__)
    
    def run_tda_pipeline(self) -> Dict:
        """
        Chạy toàn bộ pipeline TDA.
        
        Returns:
            Dictionary chứa kết quả TDA.
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("BẮT ĐẦU PIPELINE TDA")
        self.logger.info("=" * 60)
        
        # Khởi tạo extractor
        extractor = TDAFeatureExtractor(
            eps_percentile=self.eps_percentile,
            window_sizes=self.window_sizes
        )
        
        # Chạy TDA
        results = extractor.run_full_tda(
            processed_dir=self.processed_dir,
            save_dir=self.tda_dir
        )
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("HOÀN TẤT PIPELINE TDA")
        self.logger.info("=" * 60)
        
        return results
    
    def run_evaluation(self) -> Tuple[List, Dict]:
        """
        Chạy grid search evaluation.
        
        Returns:
            (results_table, global_best)
        """
        evaluator = TDAEvaluatorWithGridSearch(tda_dir=self.tda_dir)
        results_table, global_best = evaluator.grid_search()
        evaluator.print_results(results_table)
        
        return results_table, global_best


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Hàm main để chạy TDA độc lập.
    
    Usage:
        python -m modules.tda_features
    """
    import argparse
    
    base_dir = Path(__file__).parent.parent
    default_processed = base_dir / "data" / "processed"
    default_tda = base_dir / "data" / "tda_raw"
    
    parser = argparse.ArgumentParser(description='Trích xuất đặc trưng TDA')
    parser.add_argument('--processed-dir', default=str(default_processed),
                       help='Thư mục chứa file CSV đã xử lý')
    parser.add_argument('--tda-dir', default=str(default_tda),
                       help='Thư mục lưu kết quả TDA')
    parser.add_argument('--eps', type=int, default=60,
                       help='Phân vị cho epsilon (default: 60)')
    parser.add_argument('--windows', nargs='+', type=int, default=[20, 30, 50],
                       help='Danh sách kích thước window')
    parser.add_argument('--eval', action='store_true',
                       help='Chạy evaluation sau khi tính TDA')
    
    args = parser.parse_args()
    
    manager = TDAManager(
        processed_dir=args.processed_dir,
        tda_dir=args.tda_dir,
        eps_percentile=args.eps,
        window_sizes=args.windows
    )
    
    # Chạy TDA
    manager.run_tda_pipeline()
    
    # Chạy evaluation nếu được yêu cầu
    if args.eval:
        manager.run_evaluation()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
