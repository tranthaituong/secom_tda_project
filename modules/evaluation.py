"""
Module Evaluation - Nhiệm vụ 5 (NV5)
=====================================

Module này đánh giá hiệu suất của các phương pháp:
1. Topology (H1) - TDA-based anomaly detection với Grid Search
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
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)


# ============================================================================
# ĐƯỜNG DẪN MẶC ĐỊNH
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DEFAULT_TDA_DIR = BASE_DIR / "data" / "tda_raw"
DEFAULT_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"


# ============================================================================
# METRICS
# ============================================================================

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
        auc = 0.5
    
    return precision, recall, f1, auc


def calculate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, int]:
    """Tính confusion matrix và trả về dạng dictionary."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp)
    }


# ============================================================================
# TDA EVALUATION WITH GRID SEARCH
# ============================================================================

class TDAEvaluator:
    """
    Đánh giá TDA với Grid Search cho noise_threshold và classification_threshold.
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
    
    def evaluate_all(self) -> Tuple[List, Dict]:
        """
        Grid search để tìm ngưỡng tối ưu.
        
        Returns:
            (results_table, global_best)
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        tda_files = sorted([f for f in os.listdir(self.tda_dir) if f.endswith('.pkl')])
        results_table = []
        global_best = {
            'f1': -1, 'precision': 0, 'recall': 0,
            'noise': 0, 'threshold': 0, 'source': '', 'L': 0
        }
        
        self.logger.info("🚀 GRID SEARCH cho TDA...")
        
        for f_name in tda_files:
            # Trích xuất L từ tên file (ví dụ: secom_processed_pca2_L20.pkl -> L=20)
            import re
            L_match = re.search(r'_L(\d+)', f_name)
            L = int(L_match.group(1)) if L_match else 0
            
            with open(os.path.join(self.tda_dir, f_name), 'rb') as f:
                data = pickle.load(f)
            
            y_true = np.array(data['labels'])
            
            # Tìm max persistence range
            all_pers = []
            for dgms in data['dgms']:
                h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
                if h1.size > 0:
                    all_pers.extend(h1[:, 1] - h1[:, 0])
            
            max_possible = np.max(all_pers) if all_pers else 0.5
            
            # Grid search
            noise_grid = np.linspace(0.0, max_possible * 0.8, self.n_noise_levels)
            
            best_f1, best_p, best_r = -1, 0, 0
            best_noise, best_t = 0, 0
            
            for noise_th in noise_grid:
                max_pers_array = self.filter_dgms(data['dgms'], noise_th)
                
                min_val = max_pers_array.min() if max_pers_array.min() > 0 else 0
                max_val = max_pers_array.max() if max_pers_array.max() > 0 else 1
                
                thresholds = np.linspace(min_val, max_val, self.n_threshold_levels)
                
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
                data['source'], L,
                best_f1, best_p, best_r,
                best_noise, best_t
            ])
            
            if best_f1 > global_best['f1']:
                global_best = {
                    'f1': best_f1,
                    'precision': best_p,
                    'recall': best_r,
                    'noise': best_noise,
                    'threshold': best_t,
                    'source': data['source'],
                    'L': L
                }
        
        return results_table, global_best


# ============================================================================
# ML EVALUATION
# ============================================================================

class MLEvaluator:
    """
    Đánh giá ML baselines từ file ml_preds.npy.
    """
    
    def __init__(
        self,
        ml_preds_path: str = 'data/processed/ml_preds.npy'
    ):
        self.ml_preds_path = ml_preds_path
        self.logger = logging.getLogger(__name__)
    
    def evaluate_all(self) -> List:
        """Đánh giá tất cả các cấu hình ML."""
        data = np.load(self.ml_preds_path, allow_pickle=True).item()
        
        results_table = []
        contamination_values = [0.01, 0.05, 0.07, 0.1]
        
        for model_key, model_name in [('iso', 'ISO'), ('svm', 'SVM')]:
            for dataset_name in data[model_key].keys():
                for L in data[model_key][dataset_name].keys():
                    y_true = data['labels_true'][dataset_name][L]
                    y_binary = np.where(y_true == 1, 1, 0)
                    
                    for fraction in contamination_values:
                        y_pred = data[model_key][dataset_name][L][fraction]
                        y_pred_binary = np.where(y_pred == 1, 1, 0)
                        
                        precision, recall, f1, auc = calculate_all_metrics(y_binary, y_pred_binary)
                        
                        results_table.append([
                            model_name, dataset_name, L, fraction,
                            f1, precision, recall, auc
                        ])
        
        return results_table


# ============================================================================
# COMBINED EVALUATION
# ============================================================================

class AblationStudyEvaluator:
    """
    Chạy ablation study để đánh giá tất cả các cấu hình.
    """
    
    def __init__(
        self,
        tda_dir = None,
        ml_output_dir = None,
        output_dir = None
    ):
        self.tda_dir = tda_dir or DEFAULT_TDA_DIR
        self.ml_output_dir = ml_output_dir or DEFAULT_PROCESSED_DIR
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.logger = logging.getLogger(__name__)
    
    def run_full_evaluation(self) -> Dict:
        """
        Chạy đánh giá đầy đủ.
        
        Returns:
            Dictionary chứa tất cả kết quả đánh giá.
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("BẮT ĐẦU ABLATION STUDY ĐÁNH GIÁ")
        self.logger.info("=" * 60)
        
        all_results = []
        
        # 1. Đánh giá TDA với Grid Search
        self.logger.info("\n--- ĐÁNH GIÁ TDA (Grid Search) ---")
        tda_evaluator = TDAEvaluator(tda_dir=self.tda_dir)
        tda_results, tda_best = tda_evaluator.evaluate_all()
        
        for res in tda_results:
            all_results.append({
                'Model': 'Topology',
                'Source': res[0],
                'L': res[1],
                'F1': res[2],
                'Precision': res[3],
                'Recall': res[4],
                'Noise_Threshold': res[5],
                'Class_Threshold': res[6]
            })
        
        # In bảng kết quả TDA
        print("\n" + "="*110)
        print(f"{'Source Data':<25} | {'L':<4} | {'F1':<8} | {'Precision':<10} | {'Recall':<8} | {'Noise':<10} | {'Threshold':<10}")
        print("-" * 110)
        for res in tda_results:
            print(f"{res[0]:<25} | {res[1]:<4} | {res[2]:.4f} | {res[3]:.4f}    | {res[4]:.4f} | {res[5]:.4f}     | {res[6]:.4f}")
        print("="*110)
        
        # 2. Đánh giá ML
        self.logger.info("\n--- ĐÁNH GIÁ ML BASELINES ---")
        ml_preds_path = os.path.join(self.ml_output_dir, 'ml_preds.npy')
        
        if os.path.exists(ml_preds_path):
            ml_evaluator = MLEvaluator(ml_preds_path=ml_preds_path)
            ml_results = ml_evaluator.evaluate_all()
            
            for res in ml_results:
                all_results.append({
                    'Model': res[0],
                    'Source': res[1],
                    'L': res[2],
                    'Param': res[3],
                    'F1': res[4],
                    'Precision': res[5],
                    'Recall': res[6],
                    'AUC': res[7]
                })
        
        # Tạo DataFrame và lưu
        df = pd.DataFrame(all_results)
        results_path = self.output_dir / "ablation_results.csv"
        df.to_csv(results_path, index=False)
        self.logger.info(f"\nĐã lưu kết quả: {results_path}")
        
        # Tổng kết
        self.logger.info("\n" + "=" * 60)
        self.logger.info("KẾT QUẢ TỔNG KẾT")
        self.logger.info("=" * 60)
        
        # Best TDA
        self.logger.info(f"\nBest TDA: {tda_best['source']}, L={tda_best['L']}")
        self.logger.info(f"  F1={tda_best['f1']:.4f}, Precision={tda_best['precision']:.4f}, Recall={tda_best['recall']:.4f}")
        
        # Best ML
        if len(df[df['Model'] != 'Topology']) > 0:
            best_ml = df[df['Model'] != 'Topology'].loc[df[df['Model'] != 'Topology']['F1'].idxmax()]
            self.logger.info(f"\nBest ML: {best_ml['Model']}, {best_ml['Source']}, L={best_ml['L']}")
            self.logger.info(f"  F1={best_ml['F1']:.4f}")
        
        return {
            'tda_results': tda_results,
            'tda_best': tda_best,
            'ml_results': ml_results if 'ml_results' in dir() else [],
            'all_results': df,
            'best_overall': df.loc[df['F1'].idxmax()].to_dict() if len(df) > 0 else {}
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Hàm main để chạy evaluation độc lập.
    
    Usage:
        python -m modules.evaluation
    """
    import argparse
    
    base_dir = Path(__file__).parent.parent
    default_tda = base_dir / "data" / "tda_raw"
    default_processed = base_dir / "data" / "processed"
    default_output = base_dir / "outputs"
    
    parser = argparse.ArgumentParser(description='Đánh giá các mô hình')
    parser.add_argument('--tda-dir', default=str(default_tda),
                       help='Thư mục chứa file TDA .pkl')
    parser.add_argument('--ml-dir', default=str(default_processed),
                       help='Thư mục chứa ml_preds.npy')
    parser.add_argument('--output', default=str(default_output), help='Thư mục output')
    
    args = parser.parse_args()
    
    evaluator = AblationStudyEvaluator(
        tda_dir=args.tda_dir,
        ml_output_dir=args.ml_dir,
        output_dir=args.output
    )
    
    results = evaluator.run_full_evaluation()
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
