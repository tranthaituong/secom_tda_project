#!/usr/bin/env python3
"""
SECOM TDA Project - Main Entry Point
=====================================

Dự án phân tích dữ liệu SECOM (sản xuất bán dẫn) sử dụng:
1. Xử lý dữ liệu và giảm chiều (PCA + POD/SVD)
2. Trích xuất đặc trưng Topological (Vietoris-Rips + Persistent Homology)
3. Baseline ML models (Isolation Forest, One-Class SVM)
4. Đánh giá và so sánh các phương pháp
5. Visualization và phân tích kết quả

Dataset: SECOM Semiconductor Manufacturing Dataset
- 1567 samples x 591 features
- ~6.6% anomaly rate (104 failures)
- Mục tiêu: Phát hiện anomaly trong quá trình sản xuất chip

Usage:
    python main.py --all              # Chạy toàn bộ pipeline
    python main.py --data            # Chỉ chạy data processing (PCA + POD)
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

# Base directory
BASE_DIR = Path(__file__).parent

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
    
    Pipeline:
    1. Load Data
    2. Remove high NaN columns (>50%)
    3. Remove zero variance columns
    4. KNN Imputation
    5. StandardScaler
    6. PCA (2D, 3D, 5D)
    7. POD/SVD (POD_5D, POD_95)
    
    Args:
        args: Command line arguments.
        
    Returns:
        Dictionary chứa kết quả xử lý.
    """
    logging.info("=" * 70)
    logging.info("BƯỚC 1: XỬ LÝ DỮ LIỆU SECOM (PCA + POD)")
    logging.info("=" * 70)
    
    results = data_processing.run_full_pipeline()
    
    logging.info("Hoàn tất xử lý dữ liệu")
    
    return results


def run_tda_features(args) -> dict:
    """
    Chạy bước trích xuất TDA features (NV3).
    
    Pipeline:
    1. Load processed CSV files (PCA/POD)
    2. Vietoris-Rips Persistent Homology
    3. H1 (loops) feature extraction
    4. Save raw TDA coordinates to .pkl files
    
    Args:
        args: Command line arguments.
        
    Returns:
        Dictionary chứa kết quả TDA.
    """
    logging.info("=" * 70)
    logging.info("BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG TDA (Vietoris-Rips)")
    logging.info("=" * 70)
    
    manager = tda_features.TDAManager(
        processed_dir=args.processed_dir,
        tda_dir=args.tda_dir,
        eps_percentile=args.eps_percentile,
        window_sizes=args.window_sizes
    )
    
    results = manager.run_tda_pipeline()
    
    logging.info("Hoàn tất trích xuất TDA features")
    
    return results


def run_ml_baselines(args) -> dict:
    """
    Chạy bước ML Baselines (NV4).
    
    Chạy trên dữ liệu đã xử lý (CSV files):
    - PCA_5D, POD_5D, POD_95
    
    Args:
        args: Command line arguments.
        
    Returns:
        Dictionary chứa kết quả ML.
    """
    logging.info("=" * 70)
    logging.info("BƯỚC 3: CHẠY ML BASELINES (Isolation Forest + OCSVM)")
    logging.info("=" * 70)
    
    manager = ml_baselines.MLBaselineManager(
        processed_dir=args.processed_dir,
        output_dir=args.ml_output_dir
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
    logging.info("BƯỚC 4: ĐÁNH GIÁ VÀ SO SÁNH")
    logging.info("=" * 70)
    
    evaluator = evaluation.AblationStudyEvaluator(
        tda_dir=args.tda_dir,
        ml_output_dir=args.ml_output_dir,
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
        results_csv=os.path.join(args.output_dir, 'ablation_results.csv'),
        tda_dir=args.tda_dir,
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
    
    eval_results = results.get('eval', {})
    if 'best_overall' in eval_results:
        best = eval_results['best_overall']
        logging.info(f"Best Model: {best.get('Model', 'N/A')}")
        logging.info(f"Best Configuration: {best.get('Config', 'N/A')}")
        logging.info(f"Best F1-Score: {best.get('F1', 0):.4f}")
    
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
  python main.py --data                   # Chỉ xử lý dữ liệu (PCA + POD)
  python main.py --tda                    # Chỉ tính TDA features
  python main.py --ml                     # Chỉ chạy ML baselines
  python main.py --eval                   # Chỉ đánh giá kết quả
  python main.py --viz                    # Chỉ tạo visualizations

Output Directories:
  - data/raw/          : Dữ liệu gốc SECOM
  - data/processed/     : Dữ liệu đã xử lý (CSV files: PCA, POD)
  - data/tda_raw/      : Tọa độ TDA thô (.pkl files)
  - outputs/           : Kết quả đánh giá và visualization
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
    parser.add_argument('--data-file', default=str(BASE_DIR / 'data' / 'raw' / 'secom.data'),
                       help='Đường dẫn file dữ liệu SECOM')
    parser.add_argument('--labels-file', default=str(BASE_DIR / 'data' / 'raw' / 'secom_labels.data'),
                       help='Đường dẫn file nhãn')
    parser.add_argument('--processed-dir', default=str(BASE_DIR / 'data' / 'processed'),
                       help='Thư mục lưu dữ liệu đã xử lý (CSV)')
    
    # PCA arguments
    parser.add_argument('--pca', dest='pca_components', nargs='+', type=int,
                       default=[2, 3, 5],
                       help='Danh sách số PCA components (default: 2 3 5)')
    
    # Window arguments
    parser.add_argument('--windows', dest='window_sizes', nargs='+', type=int,
                       default=[20, 30, 50],
                       help='Danh sách kích thước window (default: 20 30 50)')
    
    # TDA arguments
    parser.add_argument('--eps', dest='eps_percentile', type=int,
                       default=60,
                       help='Phân vị cho epsilon trong Vietoris-Rips (default: 60)')
    parser.add_argument('--tda-dir', default=str(BASE_DIR / 'data' / 'tda_raw'),
                       help='Thư mục lưu kết quả TDA thô')
    
    # ML arguments
    parser.add_argument('--ml-output-dir', default=str(BASE_DIR / 'data' / 'processed'),
                       help='Thư mục lưu kết quả ML')
    
    # Output arguments
    parser.add_argument('--output-dir', default=str(BASE_DIR / 'outputs'),
                       help='Thư mục lưu kết quả (default: outputs)')
    parser.add_argument('--log-file',
                       help='Đường dẫn file log (tùy chọn)')
    parser.add_argument('--verbose', action='store_true',
                       help='Bật chế độ verbose')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_file=args.log_file, level=log_level)
    
    # Create output directories
    Path(BASE_DIR / 'data' / 'raw').mkdir(parents=True, exist_ok=True)
    Path(BASE_DIR / 'data' / 'processed').mkdir(parents=True, exist_ok=True)
    Path(BASE_DIR / 'data' / 'tda_raw').mkdir(parents=True, exist_ok=True)
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
