"""
Module Evaluation - Nhiệm vụ 5 (NV5)
====================================

Module này đánh giá hiệu suất của các phương pháp:
1. Topology (H1) - TDA-based anomaly detection
2. Isolation Forest (ISO) - ML baseline
3. One-Class SVM (SVM) - ML baseline

Các metrics được tính toán:
- Precision: Tỷ lệ True Positives / (True Positives + False Positives)
- Recall: Tỷ lệ True Positives / (True Positives + False Negatives)
- F1-Score: Harmonic mean của Precision và Recall
- AUC: Area Under ROC Curve

Đặc biệt quan trọng với imbalanced data (SECOM có ~6.6% anomaly):
- Precision cao: Ít false alarms
- Recall cao: Phát hiện được nhiều anomaly thực sự
- F1 cân bằng giữa Precision và Recall

Author: GTMT Project
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Tính toán tất cả các metrics cho binary classification.
    
    Args:
        y_true: Nhãn thực tế (0=normal, 1=anomaly).
        y_pred: Nhãn dự đoán (0=normal, 1=anomaly).
        
    Returns:
        Tuple (precision, recall, f1, auc).
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        # Trường hợp chỉ có 1 lớp trong dữ liệu
        auc = 0.5
    
    return precision, recall, f1, auc


def calculate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, int]:
    """
    Tính confusion matrix và trả về dạng dictionary.
    
    Args:
        y_true: Nhãn thực tế.
        y_pred: Nhãn dự đoán.
        
    Returns:
        Dictionary với TN, FP, FN, TP.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp)
    }


class EvaluationMetrics:
    """
    Tính toán và lưu trữ các metrics cho tất cả models và configurations.
    """
    
    def __init__(self):
        """Khởi tạo EvaluationMetrics."""
        self.results_ = []
        self.best_by_model_ = {}
        
    def add_result(
        self,
        model: str,
        n: int,
        L: int,
        param: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        topo_score: np.ndarray = None
    ):
        """
        Thêm kết quả đánh giá cho một model.
        
        Args:
            model: Tên model ('Topology', 'ISO', 'SVM').
            n: Số PCA components.
            L: Kích thước window.
            param: Tham số model (contamination/threshold).
            y_true: Nhãn thực tế (0/1).
            y_pred: Nhãn dự đoán (0/1).
            topo_score: Scores cho Topology model (nếu có).
        """
        precision, recall, f1, auc = calculate_all_metrics(y_true, y_pred)
        cm = calculate_confusion_matrix(y_true, y_pred)
        
        result = {
            'Model': model,
            'n': n,
            'L': L,
            'Param': param,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc,
            **cm
        }
        
        self.results_.append(result)
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Trả về DataFrame chứa tất cả kết quả.
        
        Returns:
            DataFrame với các cột: Model, n, L, Param, Precision, Recall, F1, AUC.
        """
        return pd.DataFrame(self.results_)
    
    def get_best_by_model(self) -> pd.DataFrame:
        """
        Lấy kết quả tốt nhất (theo F1) cho mỗi model.
        
        Returns:
            DataFrame với best config cho mỗi model.
        """
        df = self.get_dataframe()
        self.best_by_model_ = df.loc[df.groupby('Model')['F1'].idxmax()]
        return self.best_by_model_
    
    def get_best_overall(self) -> Dict[str, Any]:
        """
        Lấy kết quả tốt nhất trong tất cả các models.
        
        Returns:
            Dictionary chứa best config và metrics.
        """
        df = self.get_dataframe()
        best_idx = df['F1'].idxmax()
        best_row = df.loc[best_idx]
        return best_row.to_dict()
    
    def save_results(self, output_path: str):
        """
        Lưu kết quả ra file CSV.
        
        Args:
            output_path: Đường dẫn file CSV.
        """
        df = self.get_dataframe()
        df.to_csv(output_path, index=False)


class AblationStudyEvaluator:
    """
    Chạy ablation study để đánh giá tất cả các cấu hình.
    """
    
    def __init__(
        self,
        topo_scores_path: str = "outputs/topo_scores.npy",
        ml_preds_path: str = "outputs/ml_preds.npy",
        output_dir: str = "outputs"
    ):
        """
        Khởi tạo Ablation Study Evaluator.
        
        Args:
            topo_scores_path: Đường dẫn file topo_scores.npy.
            ml_preds_path: Đường dẫn file ml_preds.npy.
            output_dir: Thư mục lưu kết quả.
        """
        self.topo_scores_path = topo_scores_path
        self.ml_preds_path = ml_preds_path
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        self.topo_data_ = None
        self.ml_data_ = None
        self.evaluator_ = EvaluationMetrics()
        
    def load_data(self):
        """Load dữ liệu từ các file đã lưu."""
        self.logger.info("Đọc dữ liệu đánh giá...")
        
        # Load TDA scores
        topo_scores_data = np.load(self.topo_scores_path, allow_pickle=True)
        self.topo_data_ = topo_scores_data.item() if hasattr(topo_scores_data, 'item') else topo_scores_data
        
        # Load ML predictions
        ml_data = np.load(self.ml_preds_path, allow_pickle=True)
        self.ml_data_ = ml_data.item() if hasattr(ml_data, 'item') else ml_data
        
        self.logger.info("  - Đã load TDA scores và ML predictions")
    
    def _align_labels_for_windows(
        self,
        y_orig: np.ndarray,
        window_size: int,
        stride: int = 1
    ) -> np.ndarray:
        """Căn chỉnh nhãn với sliding windows."""
        aligned = []
        for i in range(0, len(y_orig) - window_size + 1, stride):
            aligned.append(y_orig[i + window_size - 1])
        return np.array(aligned)
    
    def _convert_to_binary(
        self,
        y: np.ndarray,
        positive_val: int = 1
    ) -> np.ndarray:
        """Chuyển nhãn về binary (0/1)."""
        return np.where(y == positive_val, 1, 0)
    
    def evaluate_topo(self, threshold_percentile: float = 95.0) -> Dict:
        """
        Đánh giá mô hình Topology (H1).
        
        Args:
            threshold_percentile: Phân vị để xác định ngưỡng anomaly.
            
        Returns:
            Dictionary kết quả đánh giá.
        """
        self.logger.info("=" * 60)
        self.logger.info("ĐÁNH GIÁ TOPOLOGY (H1)")
        self.logger.info("=" * 60)
        
        # Load labels
        labels_path = self.output_dir / "labels_raw.npy"
        if labels_path.exists():
            y_orig = np.load(labels_path)
        else:
            self.logger.warning("Labels file not found, using default")
            return {}
        
        topo_results = {}
        
        for n in sorted(self.topo_data_.keys()):
            topo_results[n] = {}
            
            for L in sorted(self.topo_data_[n].keys()):
                # Lấy scores
                scores_data = self.topo_data_[n][L]
                if isinstance(scores_data, dict):
                    scores = scores_data['scores']
                    threshold = scores_data['threshold']
                else:
                    scores = scores_data
                    threshold = np.percentile(scores, threshold_percentile)
                
                # Chuyển scores thành binary predictions
                preds = (scores >= threshold).astype(int)
                
                # Căn chỉnh labels
                y_aligned = self._align_labels_for_windows(y_orig, L)
                y_binary = self._convert_to_binary(y_aligned)
                
                # Tính metrics
                precision, recall, f1, auc = calculate_all_metrics(y_binary, preds)
                
                self.evaluator_.add_result(
                    model='Topology',
                    n=n,
                    L=L,
                    param=f'Threshold={threshold:.3f}',
                    y_true=y_binary,
                    y_pred=preds
                )
                
                topo_results[n][L] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'threshold': threshold
                }
                
                self.logger.info(
                    f"  PCA(n={n}), L={L}: "
                    f"F1={f1:.4f}, AUC={auc:.4f}"
                )
        
        return topo_results
    
    def evaluate_ml_baselines(self) -> Dict:
        """
        Đánh giá các mô hình ML Baselines (ISO, SVM).
        
        Returns:
            Dictionary kết quả đánh giá.
        """
        self.logger.info("=" * 60)
        self.logger.info("ĐÁNH GIÁ ML BASELINES")
        self.logger.info("=" * 60)
        
        # Load labels
        labels_path = self.output_dir / "labels_raw.npy"
        y_orig = np.load(labels_path)
        
        ml_results = {'ISO': {}, 'SVM': {}}
        contamination_values = [0.01, 0.05, 0.07, 0.1]
        
        for model_name in ['ISO', 'SVM']:
            for n in sorted(self.ml_data_[model_name.lower()].keys()):
                ml_results[model_name][n] = {}
                
                for L in sorted(self.ml_data_[model_name.lower()][n].keys()):
                    ml_results[model_name][n][L] = {}
                    
                    for fraction in contamination_values:
                        # Lấy predictions
                        preds_raw = self.ml_data_[model_name.lower()][n][L][fraction]
                        preds = self._convert_to_binary(preds_raw)
                        
                        # Căn chỉnh labels
                        y_aligned = self._align_labels_for_windows(y_orig, L)
                        y_binary = self._convert_to_binary(y_aligned)
                        
                        # Tính metrics
                        precision, recall, f1, auc = calculate_all_metrics(
                            y_binary, preds
                        )
                        
                        self.evaluator_.add_result(
                            model=model_name,
                            n=n,
                            L=L,
                            param=str(fraction),
                            y_true=y_binary,
                            y_pred=preds
                        )
                        
                        ml_results[model_name][n][L][fraction] = {
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'auc': auc
                        }
                        
                        self.logger.info(
                            f"  {model_name}(n={n}), L={L}, "
                            f"contamination={fraction}: "
                            f"F1={f1:.4f}, AUC={auc:.4f}"
                        )
        
        return ml_results
    
    def run_full_evaluation(self) -> Dict:
        """
        Chạy đánh giá đầy đủ cho tất cả các models.
        
        Returns:
            Dictionary chứa tất cả kết quả đánh giá.
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("BẮT ĐẦU ABLATION STUDY ĐÁNH GIÁ")
        self.logger.info("=" * 60)
        
        # Load data
        self.load_data()
        
        # Đánh giá Topology
        topo_results = self.evaluate_topo()
        
        # Đánh giá ML Baselines
        ml_results = self.evaluate_ml_baselines()
        
        # Lưu kết quả
        results_path = self.output_dir / "ablation_results.csv"
        self.evaluator_.save_results(str(results_path))
        self.logger.info(f"\nĐã lưu kết quả: {results_path}")
        
        # Tổng kết
        best_overall = self.evaluator_.get_best_overall()
        best_df = self.evaluator_.get_best_by_model()
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("KẾT QUẢ TỔNG KẾT")
        self.logger.info("=" * 60)
        
        self.logger.info("\nBest F1 theo Model:")
        for _, row in best_df.iterrows():
            self.logger.info(
                f"  {row['Model']}: n={row['n']}, L={row['L']}, "
                f"F1={row['F1']:.4f}, AUC={row['AUC']:.4f}"
            )
        
        self.logger.info(f"\nBest Overall: {best_overall['Model']} "
                       f"(n={best_overall['n']}, L={best_overall['L']}) "
                       f"với F1={best_overall['F1']:.4f}")
        
        return {
            'topology': topo_results,
            'ml_baselines': ml_results,
            'all_results': self.evaluator_.get_dataframe(),
            'best_by_model': best_df,
            'best_overall': best_overall
        }


def main():
    """
    Hàm main để chạy evaluation độc lập.
    
    Usage:
        python -m modules.evaluation
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Đánh giá các mô hình')
    parser.add_argument('--topo', default='outputs/topo_scores.npy',
                       help='Đường dẫn topo_scores.npy')
    parser.add_argument('--ml', default='outputs/ml_preds.npy',
                       help='Đường dẫn ml_preds.npy')
    parser.add_argument('--output', default='outputs', help='Thư mục output')
    
    args = parser.parse_args()
    
    evaluator = AblationStudyEvaluator(
        topo_scores_path=args.topo,
        ml_preds_path=args.ml,
        output_dir=args.output
    )
    
    results = evaluator.run_full_evaluation()
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
