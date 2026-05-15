"""
Module Visualization - Nhiệm vụ 6 (NV6)
=======================================

Module này tạo các visualization để phân tích kết quả:
1. Heatmap so sánh F1-Score giữa các methods và configs
2. Bar chart so sánh các mô hình
3. Persistence diagrams comparison
4. Analysis notes

Visualization outputs:
- topo_ablation_heatmap.png: Heatmap F1-Score
- metric_comparison.png: So sánh F1-Score tối ưu giữa các models
- diagram_sidebyside.pdf: So sánh Persistence Diagrams
- analysis_notes.md: Ghi chú phân tích

Author: GTMT Project
"""

import logging
import os
import pickle
import base64
import io
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ============================================================================
# ĐƯỜNG DẪN MẶC ĐỊNH
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DEFAULT_TDA_DIR = BASE_DIR / "data" / "tda_raw"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"


# Visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available for visualization.")

try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except Exception:
    WEASYPRINT_AVAILABLE = False
    logging.warning("WeasyPrint not available - PDF export disabled.")


# ============================================================================
# HEATMAP & BAR CHART
# ============================================================================

class AblationHeatmapPlotter:
    """Tạo heatmap từ kết quả ablation study."""
    
    def __init__(self, style: str = 'whitegrid'):
        self.style = style
    
    def plot_tda_heatmap(
        self,
        results_table: List,
        output_path: str = "outputs/tda_heatmap.png",
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 300
    ) -> Optional[plt.Figure]:
        """Tạo heatmap cho kết quả TDA."""
        if not PLOTTING_AVAILABLE:
            return None
        
        # Chuyển thành DataFrame
        df = pd.DataFrame(results_table, columns=[
            'Source', 'L', 'F1', 'Precision', 'Recall', 'Noise', 'Threshold'
        ])
        
        # Pivot cho heatmap
        pivot = df.pivot(index='Source', columns='L', values='F1')
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.set_theme(style=self.style)
        sns.heatmap(
            pivot, annot=True, fmt=".4f",
            cmap="YlGnBu",
            cbar_kws={'label': 'F1-Score'},
            linewidths=0.5, ax=ax
        )
        
        ax.set_title('TDA F1-Score Heatmap (Grid Search)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Window Size (L)', fontsize=12)
        ax.set_ylabel('Data Source', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        logging.info(f"Đã lưu heatmap: {output_path}")
        return fig
    
    def plot_metric_comparison(
        self,
        df: pd.DataFrame,
        output_path: str = "outputs/metric_comparison.png",
        figsize: Tuple[int, int] = (9, 6),
        dpi: int = 300
    ) -> Optional[plt.Figure]:
        """Tạo bar chart so sánh F1-Score tối ưu."""
        if not PLOTTING_AVAILABLE:
            return None
        
        # Get best F1 for each model/source
        best_per_model = df.loc[df.groupby(['Model', 'Source'])['F1'].idxmax()]
        best_per_model = best_per_model.sort_values('F1', ascending=False)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Tạo labels
        best_per_model['Label'] = best_per_model['Model'] + ' - ' + best_per_model['Source'].astype(str)
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(best_per_model)))
        
        bars = ax.barh(
            best_per_model['Label'],
            best_per_model['F1'],
            color=colors
        )
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.annotate(
                f'{width:.4f}',
                xy=(width, bar.get_y() + bar.get_height()/2),
                ha='left', va='center',
                fontsize=10
            )
        
        ax.set_title('F1-Score Comparison (Best per Method)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('F1-Score', fontsize=12)
        ax.set_xlim(0, best_per_model['F1'].max() * 1.2)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        logging.info(f"Đã lưu bar chart: {output_path}")
        return fig


# ============================================================================
# PERSISTENCE DIAGRAM
# ============================================================================

class PersistenceDiagramPlotter:
    """Tạo visualization cho Persistence Diagrams."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Chuyển figure thành base64 string."""
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()
    
    def plot_best_diagram(
        self,
        tda_dir: str,
        best_source: str,
        best_L: int,
        noise_threshold: float,
        output_path: str = "outputs/best_persistence_diagram.png",
        n_frames: int = 5
    ) -> Optional[str]:
        """Vẽ persistence diagram cho cấu hình tốt nhất."""
        if not PLOTTING_AVAILABLE:
            return None
        
        # Load TDA data
        pkl_path = os.path.join(tda_dir, f"{best_source}_L{best_L}.pkl")
        if not os.path.exists(pkl_path):
            self.logger.warning(f"Không tìm thấy {pkl_path}")
            return None
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Tìm frame có anomaly và vẽ
        error_indices = [i for i, label in enumerate(data['labels']) if label == 1]
        
        if not error_indices:
            self.logger.info("Không có anomaly trong data")
            return None
        
        # Vẽ persistence diagram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        for idx in error_indices[:n_frames]:
            dgms = data['dgms'][idx]
            h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
            
            if h1.size > 0:
                # Lọc nhiễu
                pers = h1[:, 1] - h1[:, 0]
                valid_mask = pers > noise_threshold
                filtered_h1 = h1[valid_mask]
                
                if filtered_h1.size > 0:
                    ax1.scatter(filtered_h1[:, 0], filtered_h1[:, 1], alpha=0.6)
        
        # Đường chéo
        max_val = max(ax1.get_xlim()[1], ax1.get_ylim()[1]) * 1.1
        ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        ax1.set_xlabel('Birth')
        ax1.set_ylabel('Death')
        ax1.set_title(f'Best Config: {best_source}, L={best_L}')
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # Barcode
        ax2.set_title('H1 Barcode (Anomaly Frames)')
        
        plt.suptitle(f'Persistence Diagram - Noise Threshold: {noise_threshold:.4f}', fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Đã lưu diagram: {output_path}")
        return output_path


# ============================================================================
# ANALYSIS NOTES
# ============================================================================

class AnalysisNotesGenerator:
    """Tạo file analysis_notes.md tổng hợp kết quả."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_notes(
        self,
        tda_results: List,
        tda_best: Dict,
        ml_results: List,
        output_path: str = "outputs/analysis_notes.md"
    ) -> str:
        """Tạo file analysis notes."""
        
        notes = f"""# Analysis Notes: SECOM TDA Project

## 1. Tổng quan

Dự án sử dụng Topological Data Analysis (TDA) để phát hiện anomaly trong sản xuất bán dẫn.

## 2. Kết quả TDA (Grid Search)

| Source | L | F1 | Precision | Recall | Noise Threshold | Classification Threshold |
|--------|---|-----|-----------|--------|-----------------|-------------------------|
"""
        
        for res in tda_results:
            notes += f"| {res[0]} | {res[1]} | {res[2]:.4f} | {res[3]:.4f} | {res[4]:.4f} | {res[5]:.4f} | {res[6]:.4f} |\n"
        
        notes += f"""
### Cấu hình tốt nhất:
- **Source:** {tda_best.get('source', 'N/A')}
- **Window Size (L):** {tda_best.get('L', 'N/A')}
- **F1-Score:** {tda_best.get('f1', 0):.4f}
- **Precision:** {tda_best.get('precision', 0):.4f}
- **Recall:** {tda_best.get('recall', 0):.4f}
- **Noise Threshold:** {tda_best.get('noise', 0):.4f}

## 3. Kết quả ML Baselines

"""
        
        if ml_results:
            notes += "| Model | Dataset | L | Param | F1 | Precision | Recall | AUC |\n"
            notes += "|-------|---------|---|-------|-----|-----------|--------|-----|\n"
            for res in ml_results:
                notes += f"| {res[0]} | {res[1]} | {res[2]} | {res[3]} | {res[4]:.4f} | {res[5]:.4f} | {res[6]:.4f} | {res[7]:.4f} |\n"
        
        notes += """
## 4. Giải thích kết quả

### Hiện tượng "Loãng đặc trưng" (L lớn)
- Khi tăng chiều dài cửa sổ L, các chu kỳ lỗi bị trộn lẫn với dữ liệu bình thường
- Độ bền (Persistence) của H1 loops giảm xuống
- Điểm H1 chìm vào vùng nhiễu (gần đường chéo y=x)

### Ý nghĩa vật lý
- Lỗi máy móc biểu hiện qua chu kỳ rung động bất thường
- H1 bắt được cấu hình hình học của các chu kỳ này
- Ngưỡng lọc nhiễu giúp loại bỏ "vòng lặp giả"

## 5. Đề xuất cải tiến

1. Kết hợp H1 với đặc trưng thống kê truyền thống
2. Thử nghiệm với các giá trị POD modes khác nhau
3. Sử dụng Weighted TDA với trọng số thời gian
4. Kết hợp với LSTM/Transformer cho time-series

## 6. Tham số mặc định

| Module | Tham số | Giá trị |
|--------|---------|---------|
| Data Processing | PCA Components | [2, 3, 5] |
| Data Processing | POD Modes | [5, 95%] |
| Data Processing | KNN Neighbors | 5 |
| TDA | eps_percentile | 60 |
| TDA | Grid Search | 30x30 |
| ML | Contamination | [0.01, 0.05, 0.07, 0.1] |
| ML | n_estimators (IF) | 100 |
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(notes)
        
        self.logger.info(f"Đã lưu analysis notes: {output_path}")
        return output_path


# ============================================================================
# VISUALIZATION MANAGER
# ============================================================================

class VisualizationManager:
    """Quản lý việc tạo tất cả visualizations."""
    
    def __init__(
        self,
        results_csv: str = None,
        tda_dir = None,
        output_dir = None
    ):
        self.results_csv = results_csv
        self.tda_dir = tda_dir or DEFAULT_TDA_DIR
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.logger = logging.getLogger(__name__)
    
    def run_visualizations(
        self,
        tda_results: List = None,
        tda_best: Dict = None,
        ml_results: List = None
    ) -> Dict:
        """
        Chạy tất cả visualizations.
        
        Args:
            tda_results: Kết quả TDA từ evaluation
            tda_best: Cấu hình TDA tốt nhất
            ml_results: Kết quả ML
            
        Returns:
            Dictionary đường dẫn các file đã tạo.
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TẠO VISUALIZATIONS")
        self.logger.info("=" * 60)
        
        output_files = {}
        
        # 1. TDA Heatmap
        if tda_results:
            self.logger.info("Tạo TDA heatmap...")
            heatmap_plotter = AblationHeatmapPlotter()
            heatmap_path = str(self.output_dir / "tda_heatmap.png")
            heatmap_plotter.plot_tda_heatmap(tda_results, heatmap_path)
            output_files['heatmap'] = heatmap_path
        
        # 2. Metric comparison (from CSV if available)
        if self.results_csv and os.path.exists(self.results_csv):
            self.logger.info("Tạo metric comparison...")
            df = pd.read_csv(self.results_csv)
            heatmap_plotter = AblationHeatmapPlotter()
            comparison_path = str(self.output_dir / "metric_comparison.png")
            heatmap_plotter.plot_metric_comparison(df, comparison_path)
            output_files['comparison'] = comparison_path
        
        # 3. Persistence diagram
        if tda_best and self.tda_dir:
            self.logger.info("Tạo persistence diagram...")
            diagram_plotter = PersistenceDiagramPlotter()
            diagram_path = str(self.output_dir / "best_persistence_diagram.png")
            diagram_plotter.plot_best_diagram(
                self.tda_dir,
                tda_best.get('source', ''),
                tda_best.get('L', 30),
                tda_best.get('noise', 0),
                diagram_path
            )
            output_files['diagram'] = diagram_path
        
        # 4. Analysis notes
        if tda_results or ml_results:
            self.logger.info("Tạo analysis notes...")
            notes_generator = AnalysisNotesGenerator()
            notes_path = str(self.output_dir / "analysis_notes.md")
            notes_generator.generate_notes(
                tda_results or [],
                tda_best or {},
                ml_results or [],
                notes_path
            )
            output_files['notes'] = notes_path
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("HOÀN TẤT VISUALIZATIONS")
        self.logger.info("=" * 60)
        
        return output_files


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Hàm main để chạy visualizations độc lập."""
    import argparse
    
    base_dir = Path(__file__).parent.parent
    default_tda = base_dir / "data" / "tda_raw"
    default_output = base_dir / "outputs"
    
    parser = argparse.ArgumentParser(description='Tạo visualizations')
    parser.add_argument('--results', default=str(default_output / 'ablation_results.csv'),
                       help='Đường dẫn file kết quả')
    parser.add_argument('--tda-dir', default=str(default_tda),
                       help='Thư mục chứa file TDA .pkl')
    parser.add_argument('--output', default=str(default_output), help='Thư mục output')
    
    args = parser.parse_args()
    
    manager = VisualizationManager(
        results_csv=args.results,
        tda_dir=args.tda_dir,
        output_dir=args.output
    )
    
    output_files = manager.run_visualizations()
    
    print("\nĐã tạo các file visualization:")
    for name, path in output_files.items():
        print(f"  - {name}: {path}")
    
    return output_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
