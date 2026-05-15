"""
Module xử lý dữ liệu SECOM - Nhiệm vụ 1 (NV1)
==============================================

Module này thực hiện các bước tiền xử lý dữ liệu:
1. Load dữ liệu từ file SECOM (.data)
2. Xử lý giá trị khuyết (NaN) bằng KNNImputer
3. Loại bỏ các feature có variance = 0 (constant features)
4. Chuẩn hóa dữ liệu bằng StandardScaler
5. Giảm chiều bằng PCA và POD/SVD
6. Tạo sliding windows cho phân tích time-series

Dataset: SECOM Semiconductor Manufacturing Dataset
- 1567 samples x 591 features
- Labels: -1 (pass/normal), 1 (fail/anomaly)
- 104 failures trong tập dữ liệu

Author: GTMT Project
"""

import logging
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD


# ============================================================================
# CẤU HÌNH HỆ THỐNG
# ============================================================================

# Đường dẫn gốc của project
BASE_DIR = Path(__file__).parent.parent

# Các thư mục con
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TDA_RAW_DIR = BASE_DIR / "data" / "tda_raw"

# Tham số xử lý
NAN_THRESHOLD = 0.50   # Loại bỏ cột trống > 50%
KNN_NEIGHBORS = 5      # Số láng giềng điền khuyết
N_POD_FIXED = 5        # Số Mode cố định cho POD_5D
ENERGY_TARGET = 0.95   # Ngưỡng năng lượng cho POD_95
PCA_FIXED_DIMS = [2, 3, 5]  # Các chiều cố định muốn xuất


# ============================================================================
# TIỀN XỬ LÝ (PREPROCESSING)
# ============================================================================

def setup_workspace():
    """Tự động thiết lập cấu trúc thư mục và di chuyển dữ liệu gốc"""
    print("=== BƯỚC 0: THIẾT LẬP KHÔNG GIAN LÀM VIỆC ===")
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    files_to_move = ['secom.data', 'secom_labels.data', 'secom.names']
    for f in files_to_move:
        if os.path.exists(f):
            shutil.move(f, os.path.join(RAW_DIR, f))
            print(f" -> Đã di chuyển {f} vào {RAW_DIR}")
        elif not os.path.exists(os.path.join(RAW_DIR, f)):
            print(f" !! Cảnh báo: Không tìm thấy {f}")


def load_data() -> tuple:
    """
    Đọc dữ liệu từ file SECOM.

    Returns
    -------
    X : pd.DataFrame  - Ma trận đặc trưng (n_samples x n_features_raw)
    y : pd.Series     - Nhãn lớp {-1, +1}
    """
    data_path = os.path.join(RAW_DIR, 'secom.data')
    label_path = os.path.join(RAW_DIR, 'secom_labels.data')

    X = pd.read_csv(data_path, sep=r'\s+', header=None)
    y = pd.read_csv(label_path, sep=r'\s+', header=None).iloc[:, 0]

    print(f"    Kích thước X ban đầu : {X.shape}  (samples x features)")
    print(f"    Phân phối nhãn       : {dict(y.value_counts())}")

    return X, y


def remove_high_nan_cols(X: pd.DataFrame) -> pd.DataFrame:
    """
    Loại bỏ các cột có tỷ lệ NaN > NAN_THRESHOLD (mặc định 50%).
    """
    nan_ratio = X.isna().mean()
    keep_cols = nan_ratio[nan_ratio <= NAN_THRESHOLD].index
    n_removed = X.shape[1] - len(keep_cols)
    X = X[keep_cols]
    print(f"    Đã loại bỏ : {n_removed} cột có NaN > {NAN_THRESHOLD*100:.0f}%")
    print(f"    Còn lại    : {X.shape[1]} cột")
    return X


def remove_zero_variance_cols(X: pd.DataFrame) -> pd.DataFrame:
    """
    Loại bỏ các cột hằng số (phương sai = 0).
    """
    variances = X.var(skipna=True)
    non_const_cols = variances[variances > 0].index
    n_removed = X.shape[1] - len(non_const_cols)
    X = X[non_const_cols]
    print(f"    Đã loại bỏ : {n_removed} cột hằng số")
    print(f"    Còn lại    : {X.shape[1]} cột")
    return X


def impute_missing(X: pd.DataFrame) -> np.ndarray:
    """
    Điền khuyết bằng KNNImputer(n_neighbors=5, weights='distance').
    """
    from sklearn.impute import KNNImputer
    print(f"    Đang điền khuyết KNN (k={KNN_NEIGHBORS}, weights='distance')...")
    imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS, weights='distance')
    X_imputed = imputer.fit_transform(X)
    remaining_nan = np.isnan(X_imputed).sum()
    print(f"    NaN còn lại sau imputation : {remaining_nan}")
    return X_imputed


def standardize(X_imputed: np.ndarray) -> tuple:
    """
    Chuẩn hóa về Mean = 0, Std = 1 bằng StandardScaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    print(f"    Mean sau scaling : {X_scaled.mean():.6f}  (~ 0)")
    print(f"    Std  sau scaling : {X_scaled.std():.6f}   (~ 1)")
    return X_scaled, scaler


# ============================================================================
# GIẢM CHIỀU - PCA VÀ POD/SVD
# ============================================================================

def run_pca(X_scaled: np.ndarray, y: pd.Series) -> dict:
    """
    Áp dụng PCA cho các chiều cố định (2, 3, 5).

    Returns
    -------
    pca_results : dict  - {n: (X_pca, pca_model)}
    """
    pca_results = {}

    for n in PCA_FIXED_DIMS:
        pca = PCA(n_components=n)
        X_pca = pca.fit_transform(X_scaled)
        pca_results[n] = (X_pca, pca)

        explained_var = sum(pca.explained_variance_ratio_) * 100
        print(f"    PCA {n} chiều - Phương sai giữ được: {explained_var:.2f}%")

        # Lưu CSV
        df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n)])
        df_pca['Label'] = y.values
        output_path = os.path.join(PROCESSED_DIR, f'secom_processed_pca{n}.csv')
        df_pca.to_csv(output_path, index=False)

    return pca_results


def run_pod(X_scaled: np.ndarray, y: pd.Series) -> dict:
    """
    Áp dụng POD (Proper Orthogonal Decomposition) = SVD.

    Tạo 2 phiên bản:
    - POD_5D: 5 mode cố định (so sánh với PCA)
    - POD_95: Số mode động để giữ >= 95% năng lượng

    Returns
    -------
    pod_results : dict  - {'5D': (X_pod, n_modes), '95': (X_pod, n_modes)}
    """
    print("\n" + "=" * 70)
    print("POD / SVD  (Proper Orthogonal Decomposition)")
    print("=" * 70)

    # Tính phổ năng lượng đầy đủ bằng Full SVD
    print("\n[2.1] Tính phổ năng lượng đầy đủ...")
    _, singular_values, _ = np.linalg.svd(X_scaled, full_matrices=False)
    print(f"    Top-5 Singular Values: {singular_values[:5]}")

    # Tính tỷ lệ năng lượng
    total_energy = np.sum(singular_values ** 2)
    energy_ratios = (singular_values ** 2) / total_energy

    print(f"\n    {'Mode':>6} | {'sigma_k':>12} | {'Energy E_k':>12} | {'Cumul. Energy':>14}")
    print(f"    {'---':>6} | {'---':>12} | {'---':>12} | {'---':>14}")
    cumul = 0.0
    for i, (sv, er) in enumerate(zip(singular_values[:15], energy_ratios[:15])):
        cumul += er
        print(f"    {i+1:>6} | {sv:>12.4f} | {er*100:>11.4f}% | {cumul*100:>13.4f}%")

    # Tìm số mode cho 95% năng lượng
    cumulative_energy = np.cumsum(energy_ratios)
    r_95 = int(np.searchsorted(cumulative_energy, ENERGY_TARGET)) + 1
    r_95 = min(r_95, len(energy_ratios))
    achieved_energy = cumulative_energy[r_95 - 1]

    print(f"\n[2.2] Tìm số Mode cho {ENERGY_TARGET*100:.0f}% năng lượng...")
    print(f"    r = {r_95:>4} Mode  ->  CE = {achieved_energy*100:.4f}%  >= {ENERGY_TARGET*100:.0f}%  OK")

    pod_results = {}

    # POD_5D - 5 mode cố định
    print(f"\n[2.3] POD {N_POD_FIXED} Mode cố định (POD_{N_POD_FIXED}D)...")
    svd_5d = TruncatedSVD(n_components=N_POD_FIXED, algorithm='randomized', n_iter=10, random_state=42)
    X_pod_5d = svd_5d.fit_transform(X_scaled)
    pod_results['5D'] = (X_pod_5d, N_POD_FIXED)
    print(f"    Shape: {X_pod_5d.shape}, Năng lượng: {sum(energy_ratios[:N_POD_FIXED])*100:.4f}%")

    df_pod5 = pd.DataFrame(X_pod_5d, columns=[f'POD_Mode_{i+1}' for i in range(N_POD_FIXED)])
    df_pod5['Label'] = y.values
    output_path_5d = os.path.join(PROCESSED_DIR, 'secom_processed_pod5.csv')
    df_pod5.to_csv(output_path_5d, index=False)
    print(f"    Đã lưu: {output_path_5d}")

    # POD_95 - Mode động theo năng lượng
    print(f"\n[2.4] POD {r_95} Mode  >= {ENERGY_TARGET*100:.0f}% Năng lượng (POD_95)...")
    svd_95 = TruncatedSVD(n_components=r_95, algorithm='randomized', n_iter=10, random_state=42)
    X_pod_95 = svd_95.fit_transform(X_scaled)
    pod_results['95'] = (X_pod_95, r_95)
    print(f"    Shape: {X_pod_95.shape}, Năng lượng: {sum(energy_ratios[:r_95])*100:.4f}%")

    df_pod95 = pd.DataFrame(X_pod_95, columns=[f'POD_Mode_{i+1}' for i in range(r_95)])
    df_pod95['Label'] = y.values
    output_path_95 = os.path.join(PROCESSED_DIR, 'secom_processed_pod95.csv')
    df_pod95.to_csv(output_path_95, index=False)
    print(f"    Đã lưu: {output_path_95}")

    return pod_results


# ============================================================================
# SLIDING WINDOWS
# ============================================================================

def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    stride: int = 1
) -> np.ndarray:
    """
    Tạo các cửa sổ trượt (sliding windows) từ chuỗi thời gian.

    Args:
        data: Mảng 2D với shape (n_samples, n_features).
        window_size: Kích thước cửa sổ (L).
        stride: Bước nhảy giữa các windows.

    Returns:
        Mảng 3D với shape (n_windows, window_size, n_features).
    """
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i : i + window_size])
    return np.array(windows)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_full_pipeline():
    """
    Chạy toàn bộ pipeline xử lý dữ liệu SECOM.
    Bao gồm: Preprocessing -> PCA -> POD -> Sliding Windows
    """
    print("\n" + "=" * 70)
    print("SECOM TDA PIPELINE - PREPROCESSING")
    print("=" * 70)

    # Bước 0: Setup workspace
    setup_workspace()

    # Bước 1: Load data
    print("\n" + "-" * 70)
    print("BƯỚC 1: Đọc dữ liệu SECOM")
    print("-" * 70)
    X, y = load_data()
    print(f"    Kích thước X ban đầu: {X.shape}")

    # Bước 2: Lọc NaN > 50%
    print("\n" + "-" * 70)
    print("BƯỚC 2: Lọc cột có NaN > 50%")
    print("-" * 70)
    X = remove_high_nan_cols(X)

    # Bước 3: Lọc cột hằng số
    print("\n" + "-" * 70)
    print("BƯỚC 3: Loại bỏ cột hằng số (variance = 0)")
    print("-" * 70)
    X = remove_zero_variance_cols(X)

    # Bước 4: Điền khuyết KNN
    print("\n" + "-" * 70)
    print("BƯỚC 4: Điền khuyết bằng KNNImputer")
    print("-" * 70)
    X_imputed = impute_missing(X)

    # Bước 5: Chuẩn hóa
    print("\n" + "-" * 70)
    print("BƯỚC 5: Chuẩn hóa (StandardScaler -> Mean=0, Std=1)")
    print("-" * 70)
    X_scaled, scaler = standardize(X_imputed)
    print(f"    Kích thước X_scaled: {X_scaled.shape}")

    # Bước 6: PCA
    print("\n" + "-" * 70)
    print("BƯỚC 6: Giảm chiều bằng PCA")
    print("-" * 70)
    pca_results = run_pca(X_scaled, y)

    # Bước 7: POD/SVD
    print("\n" + "-" * 70)
    print("BƯỚC 7: Giảm chiều bằng POD (SVD)")
    print("-" * 70)
    pod_results = run_pod(X_scaled, y)

    # Tổng kết
    print("\n" + "=" * 70)
    print("TỔNG KẾT PREPROCESSING")
    print("=" * 70)
    print(f"  Dữ liệu gốc           : {X.shape[0]} mẫu × {X.shape[1]} đặc trưng")
    print(f"  PCA files đã lưu     : {', '.join([f'PCA {n}D' for n in PCA_FIXED_DIMS])}")
    print(f"  POD_5D (5 modes)     : {PROCESSED_DIR}/secom_processed_pod5.csv")
    print(f"  POD_95 ({pod_results['95'][1]} modes)  : {PROCESSED_DIR}/secom_processed_pod95.csv")
    print("\n  Pipeline hoàn tất. Sẵn sàng cho bước TDA tiếp theo.")
    print("=" * 70 + "\n")

    return {
        'X_scaled': X_scaled,
        'y': y,
        'pca_results': pca_results,
        'pod_results': pod_results,
        'scaler': scaler
    }


if __name__ == "__main__":
    run_full_pipeline()
