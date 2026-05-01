#!/usr/bin/env python3
"""
SECOM TDA Project - Main Entry Point
=====================================

Dự án phân tích dữ liệu SECOM (sản xuất bán dẫn) sử dụng:
1. Xử lý dữ liệu và giảm chiều (PCA + Sliding Windows)
2. Trích xuất đặc trưng Topological (Persistent Homology - H1)
3. Baseline ML models (Isolation Forest, One-Class SVM)
4. Đánh giá và so sánh các phương pháp
5. Visualization và phân tích kết quả

Dataset: SECOM Semiconductor Manufacturing Dataset
- 1567 samples x 591 features
- ~6.6% anomaly rate (104 failures)
- Mục tiêu: Phát hiện anomaly trong quá trình sản xuất chip

Usage:
    python main.py --all              # Chạy toàn bộ pipeline
    python main.py --data             # Chỉ chạy data processing
    python main.py --tda             # Chỉ chạy TDA features
    python main.py --ml              # Chỉ chạy ML baselines
    python main.py --eval            # Chỉ chạy evaluation
    python main.py --viz             # Chỉ chạy visualization

Author: GTMT Project
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules import (
    data_processing,
    tda_features,
    ml_baselines,
    evaluation,
    visualization
)


def setup_logging(log_file: str = None, level: int = logging.INFO):
    """
    Thiết lập logging cho toàn bộ pipeline.
    
    Args:
        log_file: Đường dẫn file log (tùy chọn).
        level: Logging level.
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def run_data_processing(args) -> dict:
    """
    Chạy bước xử lý dữ liệu (NV1).
    
    Args:
        args: Command line arguments.
        
    Returns:
        Dictionary chứa kết quả xử lý.
    """
    logging.info("=" * 70)
    logging.info("BƯỚC 1: XỬ LÝ DỮ LIỆU SECOM")
    logging.info("=" * 70)
    
    processor = data_processing.SECOMDataProcessor(
        data_file=args.data_file,
        labels_file=args.labels_file,
        pca_components=args.pca_components,
        window_sizes=args.window_sizes,
        output_dir=args.output_dir
    )
    
    results = processor.process_all(log_file='data_processing.log')
    
    logging.info("Hoàn tất xử lý dữ liệu")
    
    return results


def run_tda_features(args) -> dict:
    """
    Chạy bước trích xuất TDA features (NV3).
    
    Args:
        args: Command line arguments.
        
    Returns:
        Dictionary chứa kết quả TDA.
    """
    logging.info("=" * 70)
    logging.info("BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG TDA (TOPOLOGY)")
    logging.info("=" * 70)
    
    manager = tda_features.TDAManager(
        windows_dict_path=f"{args.output_dir}/windows_dict.npy",
        output_dir=args.output_dir
    )
    
    results = manager.run_tda_pipeline(
        threshold_percentile=args.threshold_percentile
    )
    
    logging.info("Hoàn tất trích xuất TDA features")
    
    return results


def run_ml_baselines(args) -> dict:
    """
    Chạy bước ML Baselines (NV4).
    
    Args:
        args: Command line arguments.
        
    Returns:
        Dictionary chứa kết quả ML.
    """
    logging.info("=" * 70)
    logging.info("BƯỚC 3: CHẠY ML BASELINES")
    logging.info("=" * 70)
    
    manager = ml_baselines.MLBaselineManager(
        windows_dict_path=f"{args.output_dir}/windows_dict.npy",
        labels_path=f"{args.output_dir}/labels_raw.npy",
        output_dir=args.output_dir
    )
    
    results = manager.run()
    
    logging.info("Hoàn tất ML Baselines")
    
    return results


def run_evaluation(args) -> dict:
    """
    Chạy bước đánh giá (NV5).
    
    Args:
        args: Command line arguments.
        
    Returns:
        Dictionary chứa kết quả đánh giá.
    """
    logging.info("=" * 70)
    logging.info("BƯỚC 4: ĐÁNH GIÁ VÀ ABLATION STUDY")
    logging.info("=" * 70)
    
    evaluator = evaluation.AblationStudyEvaluator(
        topo_scores_path=f"{args.output_dir}/topo_scores.npy",
        ml_preds_path=f"{args.output_dir}/ml_preds.npy",
        output_dir=args.output_dir
    )
    
    results = evaluator.run_full_evaluation()
    
    logging.info("Hoàn tất đánh giá")
    
    return results


def run_visualization(args) -> dict:
    """
    Chạy bước visualization (NV6).
    
    Args:
        args: Command line arguments.
        
    Returns:
        Dictionary chứa đường dẫn các file đã tạo.
    """
    logging.info("=" * 70)
    logging.info("BƯỚC 5: TẠO VISUALIZATIONS")
    logging.info("=" * 70)
    
    manager = visualization.VisualizationManager(
        results_csv=f"{args.output_dir}/ablation_results.csv",
        topo_diagrams=f"{args.output_dir}/topo_diagrams.npy",
        output_dir=args.output_dir
    )
    
    output_files = manager.run_visualizations()
    
    logging.info("Hoàn tất visualizations")
    
    return output_files


def run_full_pipeline(args):
    """
    Chạy toàn bộ pipeline theo thứ tự.
    
    Args:
        args: Command line arguments.
    """
    logging.info("\n" + "=" * 70)
    logging.info("BẮT ĐẦU PIPELINE HOÀN CHỈNH")
    logging.info("=" * 70)
    
    results = {}
    
    # 1. Data Processing
    results['data'] = run_data_processing(args)
    
    # 2. TDA Features
    results['tda'] = run_tda_features(args)
    
    # 3. ML Baselines
    results['ml'] = run_ml_baselines(args)
    
    # 4. Evaluation
    results['eval'] = run_evaluation(args)
    
    # 5. Visualization
    results['viz'] = run_visualization(args)
    
    # Summary
    logging.info("\n" + "=" * 70)
    logging.info("TÓM TẮT KẾT QUẢ")
    logging.info("=" * 70)
    
    eval_results = results['eval']
    if 'best_overall' in eval_results:
        best = eval_results['best_overall']
        logging.info(f"Best Model: {best.get('Model', 'N/A')}")
        logging.info(f"Best Configuration: n={best.get('n', 'N/A')}, L={best.get('L', 'N/A')}")
        logging.info(f"Best F1-Score: {best.get('F1', 0):.4f}")
        logging.info(f"Best AUC: {best.get('AUC', 0):.4f}")
    
    logging.info("\nCác file đầu ra đã được lưu trong thư mục:")
    logging.info(f"  - {args.output_dir}/")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SECOM TDA Project - Anomaly Detection using Topological Data Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all                    # Chạy toàn bộ pipeline
  python main.py --data --pca 2 3        # Chỉ xử lý dữ liệu với PCA=2,3
  python main.py --tda --percentile 90    # Chạy TDA với ngưỡng 90th percentile
  python main.py --ml                     # Chỉ chạy ML baselines
  python main.py --eval                   # Chỉ đánh giá kết quả
  python main.py --viz                    # Chỉ tạo visualizations

Output Directory:
  Mặc định: outputs/
  Các file được tạo:
    - data_pca.npy, windows_dict.npy, labels_raw.npy
    - topo_features.npy, topo_scores.npy, topo_diagrams.npy
    - ml_preds.npy, ml_param_log.csv
    - ablation_results.csv
    - topo_ablation_heatmap.png, metric_comparison.png
    - diagram_sidebyside.pdf, analysis_notes.md
        """
    )
    
    # Pipeline flags
    parser.add_argument('--all', action='store_true',
                       help='Chạy toàn bộ pipeline')
    parser.add_argument('--data', action='store_true',
                       help='Chỉ chạy xử lý dữ liệu')
    parser.add_argument('--tda', action='store_true',
                       help='Chỉ chạy TDA features')
    parser.add_argument('--ml', action='store_true',
                       help='Chỉ chạy ML baselines')
    parser.add_argument('--eval', action='store_true',
                       help='Chỉ chạy đánh giá')
    parser.add_argument('--viz', action='store_true',
                       help='Chỉ chạy visualization')
    
    # Data processing arguments
    parser.add_argument('--data-file', default='data/secom.data',
                       help='Đường dẫn file dữ liệu SECOM')
    parser.add_argument('--labels-file', default='data/secom_labels.data',
                       help='Đường dẫn file nhãn')
    parser.add_argument('--pca', dest='pca_components', nargs='+', type=int,
                       default=[2, 3, 5],
                       help='Danh sách số PCA components (default: 2 3 5)')
    parser.add_argument('--windows', dest='window_sizes', nargs='+', type=int,
                       default=[20, 30, 50],
                       help='Danh sách kích thước window (default: 20 30 50)')
    
    # TDA arguments
    parser.add_argument('--percentile', dest='threshold_percentile', type=float,
                       default=95.0,
                       help='Phân vị ngưỡng anomaly (default: 95)')
    
    # Output arguments
    parser.add_argument('--output-dir', default='outputs',
                       help='Thư mục lưu kết quả (default: outputs)')
    parser.add_argument('--log-file',
                       help='Đường dẫn file log (tùy chọn)')
    parser.add_argument('--verbose', action='store_true',
                       help='Bật chế độ verbose')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_file=args.log_file, level=log_level)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine which step(s) to run
    run_all = args.all or not any([args.data, args.tda, args.ml, args.eval, args.viz])
    
    results = {}
    
    if run_all:
        results = run_full_pipeline(args)
    else:
        if args.data:
            results['data'] = run_data_processing(args)
        if args.tda:
            results['tda'] = run_tda_features(args)
        if args.ml:
            results['ml'] = run_ml_baselines(args)
        if args.eval:
            results['eval'] = run_evaluation(args)
        if args.viz:
            results['viz'] = run_visualization(args)
    
    logging.info("\n" + "=" * 70)
    logging.info("HOÀN TẤT")
    logging.info("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
