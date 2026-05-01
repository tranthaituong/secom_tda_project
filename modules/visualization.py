"""
Module Visualization - Nhiệm vụ 6 (NV6)
=======================================

Module này tạo các visualization để phân tích kết quả:
1. Heatmap ablation study cho Topology (F1-Score theo PCA và L)
2. Bar chart so sánh các mô hình
3. Persistence diagrams comparison
4. Birth-Death tables

Visualization outputs:
- topo_ablation_heatmap.png: Heatmap F1-Score của Topology
- metric_comparison.png: So sánh F1-Score tối ưu giữa các models
- diagram_sidebyside.pdf: So sánh Persistence Diagrams
- analysis_notes.md: Ghi chú phân tích

Author: GTMT Project
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import base64
import io

# Visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
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


class AblationHeatmapPlotter:
    """
    Tạo heatmap từ kết quả ablation study.
    """
    
    def __init__(self, style: str = 'whitegrid'):
        """
        Khởi tạo AblationHeatmapPlotter.
        
        Args:
            style: Seaborn style (default: 'whitegrid').
        """
        self.style = style
        
    def plot_topo_ablation(
        self,
        df: pd.DataFrame,
        output_path: str = "outputs/topo_ablation_heatmap.png",
        figsize: Tuple[int, int] = (8, 6),
        dpi: int = 300
    ) -> plt.Figure:
        """
        Tạo heatmap cho kết quả Topology ablation study.
        
        Args:
            df: DataFrame chứa kết quả ablation.
            output_path: Đường dẫn lưu hình.
            figsize: Kích thước figure (width, height).
            dpi: Độ phân giải hình.
            
        Returns:
            Matplotlib Figure object.
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError("Matplotlib is required for plotting.")
        
        # Filter Topology results
        topo_df = df[df['Model'] == 'Topology']
        
        if len(topo_df) == 0:
            logging.warning("No Topology results found in DataFrame.")
            return None
        
        # Create pivot table
        pivot = topo_df.pivot(index='n', columns='L', values='F1')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.set_theme(style=self.style)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".4f",
            cmap="YlGnBu",
            cbar_kws={'label': 'F1-Score'},
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title(
            'Topology (H1) Ablation Study: F1-Score theo PCA và L',
            fontsize=14,
            fontweight='bold',
            pad=15
        )
        ax.set_xlabel('Độ dài cửa sổ trượt (L)', fontsize=12)
        ax.set_ylabel('Số chiều PCA (n_components)', fontsize=12)
        
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
    ) -> plt.Figure:
        """
        Tạo bar chart so sánh F1-Score tối ưu giữa các models.
        
        Args:
            df: DataFrame chứa kết quả.
            output_path: Đường dẫn lưu hình.
            figsize: Kích thước figure.
            dpi: Độ phân giải.
            
        Returns:
            Matplotlib Figure object.
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError("Matplotlib is required for plotting.")
        
        # Get best F1 for each model
        best_models = df.loc[df.groupby('Model')['F1'].idxmax()].sort_values(
            by='F1', ascending=False
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # Plot bar chart
        sns.barplot(
            x='Model',
            y='F1',
            data=best_models,
            hue='Model',
            palette=colors[:len(best_models)],
            legend=False,
            ax=ax
        )
        
        # Add value labels on bars
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f"{height:.4f}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center',
                va='center',
                fontsize=12,
                color='black',
                xytext=(0, 8),
                textcoords='offset points'
            )
        
        ax.set_title(
            'So sánh F1-Score tối ưu giữa Topology và Baselines',
            fontsize=14,
            fontweight='bold',
            pad=15
        )
        ax.set_ylabel('Khả năng phát hiện lỗi (F1-Score)', fontsize=12)
        ax.set_xlabel('Phương pháp', fontsize=12)
        ax.set_ylim(0, best_models['F1'].max() + 0.1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        logging.info(f"Đã lưu bar chart: {output_path}")
        
        return fig


class PersistenceDiagramPlotter:
    """
    Tạo visualization cho Persistence Diagrams.
    """
    
    def __init__(self):
        """Khởi tạo PersistenceDiagramPlotter."""
        self.logger = logging.getLogger(__name__)
        
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Chuyển figure thành base64 string."""
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()
    
    def plot_diagram_comparison(
        self,
        diagrams_data: Dict,
        best_config: Dict,
        worst_config: Dict,
        output_path: str = "outputs/diagram_sidebyside.pdf",
        n_frames: int = 5
    ) -> str:
        """
        Tạo PDF so sánh 2 persistence diagrams.
        
        Args:
            diagrams_data: Dictionary chứa diagrams.
            best_config: Dict với keys 'n' và 'L' cho config tốt nhất.
            worst_config: Dict với keys 'n' và 'L' cho config xấu nhất.
            output_path: Đường dẫn lưu PDF.
            n_frames: Số frames để hiển thị.
            
        Returns:
            Đường dẫn file đã lưu.
        """
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Cannot create diagram comparison without matplotlib.")
            return None
        
        if not WEASYPRINT_AVAILABLE:
            self.logger.warning("Cannot create PDF without weasyprint.")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot best config
        n_best, l_best = best_config['n'], best_config['L']
        pts_best = [
            pt for frame in diagrams_data.get(str(n_best), {}).get(str(l_best), [])[:n_frames]
            for pt in frame
        ]
        b_best = [p[0] for p in pts_best]
        d_best = [p[1] for p in pts_best]
        
        ax1.scatter(b_best, d_best, alpha=0.6, color='#2c3e50', label='H1 Points')
        max_val = max(max(b_best), max(d_best)) * 1.1 if b_best else 1
        ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        ax1.set_title(f'Tối ưu: PCA={n_best}, L={l_best}', fontsize=12)
        ax1.set_xlabel('Birth')
        ax1.set_ylabel('Death')
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # Plot worst config
        n_worst, l_worst = worst_config['n'], worst_config['L']
        pts_worst = [
            pt for frame in diagrams_data.get(str(n_worst), {}).get(str(l_worst), [])[:n_frames]
            for pt in frame
        ]
        b_worst = [p[0] for p in pts_worst]
        d_worst = [p[1] for p in pts_worst]
        
        ax2.scatter(b_worst, d_worst, alpha=0.6, color='#e74c3c', label='H1 Points')
        max_val2 = max(max(b_worst), max(d_worst)) * 1.1 if b_worst else 1
        ax2.plot([0, max_val2], [0, max_val2], 'r--', alpha=0.5)
        ax2.set_title(f'Suy giảm: PCA={n_worst}, L={l_worst}', fontsize=12)
        ax2.set_xlabel('Birth')
        ax2.set_ylabel('Death')
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        img_b64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        # Create HTML content
        html_content = f'''
        <html>
        <head>
        <style>
            @page {{ size: A4 landscape; margin: 15mm; background-color: #ffffff; }}
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333; }}
            .header {{ text-align: center; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; margin-bottom: 20px; }}
            .chart-container {{ text-align: center; margin-top: 30px; }}
            .chart-container img {{ width: 100%; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .caption {{ font-style: italic; color: #666; margin-top: 15px; font-size: 11pt; line-height: 1.5; }}
            .stats-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
            .stats-table th {{ background-color: #f8f9fa; color: #2c3e50; }}
        </style>
        </head>
        <body>
            <div class="header">
                <h1>So sánh Persistence Diagrams</h1>
                <p>Phân tích ảnh hưởng của tham số Window Length (L) đến H1 features</p>
            </div>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{img_b64}">
                <div class="caption">
                    <strong>Hình:</strong> So sánh persistence diagrams giữa cấu hình tối ưu và cấu hình có L quá lớn.
                    Độ bền (Persistence) của H1 loops giảm khi L tăng.
                </div>
            </div>
        </body>
        </html>
        '''
        
        HTML(string=html_content).write_pdf(output_path)
        self.logger.info(f"Đã lưu PDF: {output_path}")
        
        return output_path


class AnalysisNotesGenerator:
    """
    Tạo file analysis_notes.md tổng hợp kết quả.
    """
    
    def __init__(self):
        """Khởi tạo AnalysisNotesGenerator."""
        self.logger = logging.getLogger(__name__)
        
    def generate_notes(
        self,
        results_df: pd.DataFrame,
        best_config: Dict,
        output_path: str = "outputs/analysis_notes.md"
    ) -> str:
        """
        Tạo file analysis_notes.md từ kết quả đánh giá.
        
        Args:
            results_df: DataFrame chứa tất cả kết quả.
            best_config: Dictionary chứa config tốt nhất.
            output_path: Đường dẫn lưu file markdown.
            
        Returns:
            Đường dẫn file đã lưu.
        """
        best_by_model = results_df.loc[results_df.groupby('Model')['F1'].idxmax()]
        
        # Find best topology config
        topo_best = best_by_model[best_by_model['Model'] == 'Topology'].iloc[0]
        
        notes = f"""# Analysis Notes: SECOM TDA Project

## 1. Tổng quan kết quả

Dựa trên ablation study, các cấu hình tối ưu cho mỗi phương pháp:

| Model | PCA (n) | L | F1-Score | AUC |
|-------|---------|---|----------|-----|
"""
        
        for _, row in best_by_model.iterrows():
            notes += f"| {row['Model']} | {row['n']} | {row['L']} | {row['F1']:.4f} | {row['AUC']:.4f} |\n"
        
        notes += f"""
## 2. Cấu hình tối ưu Topology (H1)

- **Số chiều PCA:** {int(topo_best['n'])}
- **Kích thước Window:** {int(topo_best['L'])}
- **F1-Score:** {topo_best['F1']:.4f}
- **AUC:** {topo_best['AUC']:.4f}
- **Precision:** {topo_best['Precision']:.4f}
- **Recall:** {topo_best['Recall']:.4f}

## 3. Giải thích hiện tượng "Loãng đặc trưng" (L lớn)

Khi tăng chiều dài cửa sổ L:
- Các chu kỳ lỗi (cycles) bị trộn lẫn với quá nhiều dữ liệu bình thường
- Độ bền (Persistence) của các vòng lặp H1 bị giảm xuống
- Điểm H1 chìm vào vùng nhiễu (gần đường chéo Y=X)
- Recall giảm mạnh do mô hình không còn phân biệt được vòng lặp lỗi

## 4. Ý nghĩa vật lý của H1 trong sản xuất Chip

- Lỗi máy móc trong sản xuất bán dẫn biểu hiện qua các chu kỳ rung động bất thường
- Đặc trưng H1 của Persistent Homology bắt được cấu hình hình học của các chu kỳ này
- **Ngưỡng 95th Percentile:** Giúp loại bỏ "vòng lặp giả" do nhiễu hệ thống

## 5. Đề xuất cải tiến

1. Kết hợp H1 với các đặc trưng thống kê truyền thống để tăng Precision
2. Thử nghiệm với các giá trị PCA trung gian (ví dụ PCA=3, 4)
3. Sử dụng Weighted TDA với trọng số theo thời gian
4. Kết hợp với sequence models (LSTM, Transformer) cho time-series

## 6. Tham số mặc định của Pipeline

| Module | Tham số | Giá trị mặc định |
|--------|---------|------------------|
| Data Processing | PCA Components | [2, 3, 5] |
| Data Processing | Window Sizes | [20, 30, 50] |
| TDA | Homology Dimension | 1 (H1) |
| TDA | Threshold Percentile | 95 |
| ML Baselines | Contamination | [0.01, 0.05, 0.07, 0.1] |
| ML Baselines | n_estimators (IF) | 100 |
| ML Baselines | kernel (SVM) | rbf |
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(notes)
        
        self.logger.info(f"Đã lưu analysis notes: {output_path}")
        
        return output_path


class BirthDeathTableExporter:
    """
    Xuất Birth-Death table từ persistence diagrams.
    """
    
    def __init__(self):
        """Khởi tạo BirthDeathTableExporter."""
        self.logger = logging.getLogger(__name__)
    
    def export_to_csv(
        self,
        diagrams_data: Dict,
        output_path: str = "outputs/birth_death_table.csv",
        max_frames_per_config: int = 10
    ) -> str:
        """
        Xuất birth-death points ra CSV.
        
        Args:
            diagrams_data: Dictionary chứa persistence diagrams.
            output_path: Đường dẫn lưu CSV.
            max_frames_per_config: Số frames tối đa mỗi config.
            
        Returns:
            Đường dẫn file đã lưu.
        """
        table_rows = []
        
        for n_str, l_dict in diagrams_data.items():
            for l_str, frames in l_dict.items():
                n = int(n_str)
                L = int(l_str)
                for f_idx, frame in enumerate(frames[:max_frames_per_config]):
                    for pt in frame:
                        if len(pt) >= 2:
                            b, d = pt[0], pt[1]
                            persistence = d - b
                            table_rows.append({
                                'PCA_Components': n,
                                'Window_L': L,
                                'Frame_Index': f_idx,
                                'Birth': round(b, 10),
                                'Death': round(d, 10),
                                'Persistence': round(persistence, 10)
                            })
        
        df = pd.DataFrame(table_rows)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Đã lưu birth-death table: {output_path} ({len(df)} rows)")
        
        return output_path


class VisualizationManager:
    """
    Quản lý việc tạo tất cả visualizations.
    """
    
    def __init__(
        self,
        results_csv: str = "outputs/ablation_results.csv",
        topo_diagrams: str = "outputs/topo_diagrams.npy",
        output_dir: str = "outputs"
    ):
        """
        Khởi tạo VisualizationManager.
        
        Args:
            results_csv: Đường dẫn file kết quả ablation.
            topo_diagrams: Đường dẫn file persistence diagrams.
            output_dir: Thư mục lưu outputs.
        """
        self.results_csv = results_csv
        self.topo_diagrams = topo_diagrams
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
    def run_visualizations(self) -> Dict:
        """
        Chạy tất cả visualizations.
        
        Returns:
            Dictionary đường dẫn các file đã tạo.
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TẠO VISUALIZATIONS")
        self.logger.info("=" * 60)
        
        output_files = {}
        
        # Load results
        df = pd.read_csv(self.results_csv)
        
        # 1. Create heatmap
        self.logger.info("Tạo ablation heatmap...")
        heatmap_plotter = AblationHeatmapPlotter()
        heatmap_path = str(self.output_dir / "topo_ablation_heatmap.png")
        heatmap_plotter.plot_topo_ablation(df, heatmap_path)
        output_files['heatmap'] = heatmap_path
        
        # 2. Create metric comparison
        self.logger.info("Tạo metric comparison chart...")
        comparison_path = str(self.output_dir / "metric_comparison.png")
        heatmap_plotter.plot_metric_comparison(df, comparison_path)
        output_files['comparison'] = comparison_path
        
        # 3. Create analysis notes
        self.logger.info("Tạo analysis notes...")
        notes_generator = AnalysisNotesGenerator()
        best_config = df.loc[df.groupby('Model')['F1'].idxmax()].iloc[0].to_dict()
        notes_path = str(self.output_dir / "analysis_notes.md")
        notes_generator.generate_notes(df, best_config, notes_path)
        output_files['notes'] = notes_path
        
        # 4. Create diagram comparison (if diagrams available)
        diagrams_path = Path(self.topo_diagrams)
        if diagrams_path.exists():
            try:
                self.logger.info("Tạo diagram comparison...")
                diagrams_data = np.load(diagrams_path, allow_pickle=True).item()
                
                # Export birth-death table to CSV
                self.logger.info("Xuất birth-death table...")
                bd_exporter = BirthDeathTableExporter()
                bd_csv_path = str(self.output_dir / "birth_death_table.csv")
                bd_exporter.export_to_csv(diagrams_data, bd_csv_path)
                output_files['birth_death_csv'] = bd_csv_path
                
                # Best: PCA=2, L=30; Worst: PCA=2, L=50
                diagram_plotter = PersistenceDiagramPlotter()
                pdf_path = str(self.output_dir / "diagram_sidebyside.pdf")
                diagram_plotter.plot_diagram_comparison(
                    diagrams_data,
                    best_config={'n': 2, 'L': 30},
                    worst_config={'n': 2, 'L': 50},
                    output_path=pdf_path
                )
                output_files['diagram_pdf'] = pdf_path
            except Exception as e:
                self.logger.warning(f"Không thể tạo diagram comparison: {e}")
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("HOÀN TẤT VISUALIZATIONS")
        self.logger.info("=" * 60)
        
        return output_files


def main():
    """
    Hàm main để chạy visualizations độc lập.
    
    Usage:
        python -m modules.visualization
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Tạo visualizations')
    parser.add_argument('--results', default='outputs/ablation_results.csv',
                       help='Đường dẫn file kết quả ablation')
    parser.add_argument('--diagrams', default='outputs/topo_diagrams.npy',
                       help='Đường dẫn file diagrams')
    parser.add_argument('--output', default='outputs', help='Thư mục output')
    
    args = parser.parse_args()
    
    manager = VisualizationManager(
        results_csv=args.results,
        topo_diagrams=args.diagrams,
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
