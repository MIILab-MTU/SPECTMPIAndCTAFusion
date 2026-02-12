# -*- coding: utf-8 -*-
"""
angle_statistics_with_clureg.py
在原角度评估脚本基础上，加入基于 clureg_model.mat 的 'CluReg' 方法：
- 自动在候选目录查找并读取模型
- 直接对 SPECT apex 做非刚性变换
- 与原有方法一并统计并出图

依赖：numpy / pandas / matplotlib / scipy（loadmat, cKDTree, cdist）
可选：mat73（若 .mat 为 v7.3 HDF5）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

# ================= 基础路径与输出 =================
base_dir = r"data\Apex_data"
out_dir = r'final_result'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
output_excel = os.path.join(out_dir, "angle_statistics_all_methods.xlsx")
output_plot  = os.path.join(out_dir, "angle_plot_all_methods.png")

# 原有方法 + CluReg
methods = ['ICP', 'SICP', 'CPD_Affine', 'CPD_Rigid', 'CluReg', 'FFD', 'BCPD++']

# —— CluReg 模型候选目录（按需增删顺序）——
clureg_dir_tpl_candidates = [
    r"preprocess\cloud_result212\{patient}",
]

# ================= I/O：点云与 apex =================
def load_point_cloud(file_path):
    try:
        points = np.loadtxt(file_path, delimiter=None)
        if points.ndim == 1 and points.shape[0] == 3:
            points = points[np.newaxis, :]
        if points.shape[1] != 3:
            raise ValueError(f"Expected 3D points, got {points.shape[1]} columns")
        return points
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_apex_points(file_path):
    """读取包含两行形如 'CTA Apex: [x y z]' 与 'SPECT Apex: [x y z]' 的文本"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Apex file {file_path} does not exist")
        cta_apex = None
        spect_apex = None
        with open(file_path, 'r') as f:
            for line in f:
                s = line.strip()
                if s.startswith("CTA Apex:"):
                    coords = s.replace("CTA Apex:", "").strip()
                    cta_apex = np.fromstring(coords.strip("[]"), sep=' ')
                elif s.startswith("SPECT Apex:"):
                    coords = s.replace("SPECT Apex:", "").strip()
                    spect_apex = np.fromstring(coords.strip("[]"), sep=' ')
        if cta_apex is None or spect_apex is None:
            raise ValueError("Missing CTA or SPECT apex in file")
        if cta_apex.shape != (3,) or spect_apex.shape != (3,):
            raise ValueError(f"Invalid apex shapes: CTA={cta_apex.shape}, SPECT={spect_apex.shape}")
        return cta_apex, spect_apex
    except Exception as e:
        print(f"Error loading apex points from {file_path}: {e}")
        return None, None

# ================= 传统方法：参数读取与应用 =================
def load_transform_params(method, save_dir):
    try:
        if method == 'ICP':
            T = np.loadtxt(os.path.join(save_dir, 'icp_tfm.txt'))
            R = T[:3, :3]; t = T[:3, 3]; s = 1.0
            v = y_down = None
            return s, R, t, v, y_down
        elif method == 'SICP':
            s = np.loadtxt(os.path.join(save_dir, 'result_sicp_s.txt'))
            R = np.loadtxt(os.path.join(save_dir, 'result_sicp_R.txt'))
            t = np.loadtxt(os.path.join(save_dir, 'result_sicp_T.txt'))
            v = y_down = None
            return s, R, t, v, y_down
        elif method == 'CPD_Affine':
            R = np.loadtxt(os.path.join(save_dir, 'result_affine_R.txt'))
            t = np.loadtxt(os.path.join(save_dir, 'result_affine_T.txt'))
            s = 1.0; v = y_down = None
            return s, R, t, v, y_down
        elif method == 'CPD_Rigid':
            s = np.loadtxt(os.path.join(save_dir, 'result_rigid_s.txt'))
            R = np.loadtxt(os.path.join(save_dir, 'result_rigid_R.txt'))
            t = np.loadtxt(os.path.join(save_dir, 'result_rigid_T.txt'))
            v = y_down = None
            return s, R, t, v, y_down
        elif method == 'BCPD++':
            s = np.loadtxt(os.path.join(save_dir, 'result_bcpdpp_autos.txt'))
            R = np.loadtxt(os.path.join(save_dir, 'result_bcpdpp_autoR.txt'))
            t = np.loadtxt(os.path.join(save_dir, 'result_bcpdpp_autot.txt'))
            v = np.loadtxt(os.path.join(save_dir, 'result_bcpdpp_autov.txt'))
            y_down = np.loadtxt(os.path.join(save_dir, 'result_bcpdpp_autoy.txt'))
            return s, R, t, v, y_down
        else:
            raise ValueError(f"Unknown method: {method}")
    except Exception as e:
        print(f"Error loading {method} params: {e}")
        return None, None, None, None, None

def interpolate_nonrigid_displacement(T0, y_down, v, beta=2.0):
    """把下采样位移 v(y_down) 插到原分辨率点 T0 上（Gaussian 权重，与你原逻辑一致）"""
    M = T0.shape[0]; M_down = y_down.shape[0]
    if M_down < M:
        G = np.exp(-cdist(T0, y_down, 'sqeuclidean') / (2 * beta**2))
        v_orig = np.zeros_like(T0)
        for d in range(3):
            v_orig[:, d] = G @ np.linalg.lstsq(G, v[:, d], rcond=None)[0]
    else:
        v_orig = v
    return v_orig

def compute_apex_displacement(apex, T0, v_orig, y_down, v, beta=2.0):
    """用与上面一致的 Gaussian/NN 思路为 apex 点插值非刚性位移"""
    if y_down.shape[0] < T0.shape[0]:  # 稀疏 → 稠密：Gaussian 权重
        diff = apex - y_down
        w = np.exp(-np.sum(diff**2, axis=1) / (2 * beta**2))
        s = w.sum()
        if s <= 0:   # 极端容错
            idx = np.argmin(np.sum((y_down - apex)**2, axis=1))
            return v[idx]
        w = w / s
        return (w[:, None] * v).sum(axis=0)
    else:
        # 已是稠密：直接从 v_orig 取最近邻
        tree = cKDTree(T0)
        _, idx = tree.query(apex[None, :], k=1)
        return v_orig[idx[0]]

def apply_transform_to_apex(apex, s, R, t, v_orig=None, method='BCPD++', T0=None, y_down=None, v=None, beta=2.0):
    """按各法把 apex 变换到目标空间"""
    if method in ['ICP', 'SICP', 'CPD_Affine', 'CPD_Rigid']:
        return s * apex @ R.T + t
    # BCPD++：非刚性 + 刚性
    v_apex = compute_apex_displacement(apex, T0, v_orig, y_down, v, beta)
    return s * (apex + v_apex) @ R.T + t

# ==================== CluReg：模型读取与应用 ====================
def load_clureg_model(patient):
    """
    在候选目录中依次寻找：
      - clureg_model.mat
      - {patient}_clureg_model.mat
    支持 v7 与 v7.3（优先 mat73）
    """
    candidates = []
    for tpl in clureg_dir_tpl_candidates:
        d = tpl.format(patient=patient)
        candidates.append(os.path.join(d, 'psr_clureg_model.mat'))
        candidates.append(os.path.join(d, f'{patient}_clureg_model.mat'))
    for pth in candidates:
        if os.path.exists(pth):
            try:
                # 1) mat73：v7.3 HDF5
                try:
                    import mat73
                    dd = mat73.loadmat(pth)
                    model = dd.get('model', dd)
                except Exception:
                    # 2) scipy：v7/v6
                    from scipy.io import loadmat
                    dd = loadmat(pth, squeeze_me=True, struct_as_record=False)
                    model = dd.get('model', dd)
                if model is not None:
                    return model
            except Exception as e:
                print(f"[CluReg] 读取 {pth} 失败: {e}")
    return None

def _get_field(st, name):
    if st is None: return None
    if hasattr(st, name): return getattr(st, name)
    if isinstance(st, dict) and name in st: return st[name]
    if hasattr(st, 'dtype') and getattr(st, 'dtype').names and name in st.dtype.names:
        return st[name]
    return None

def _to_ndarray(x):
    if x is None: return None
    if isinstance(x, np.ndarray): return x
    try: return np.array(x)
    except Exception: return None

def apply_clureg_points(Z_world, model):
    """
    用 clureg_model 对任意点 Z_world(Lx3) 做非刚性变换：
      Z' = Z + K_{Z,X} C + P_Z D（Laplacian L1 核）
    若 model.pre.center/scale 存在则自动(反)归一化
    """
    X   = _to_ndarray(_get_field(model, 'X'))
    C   = _to_ndarray(_get_field(model, 'C'))
    D   = _to_ndarray(_get_field(model, 'D'))
    mu  = _get_field(model, 'mu')
    use_poly = bool(_get_field(model, 'use_poly'))
    pre = _get_field(model, 'pre')

    if X is None or C is None or mu is None:
        raise ValueError("clureg_model.mat 缺少必要字段（X/C/mu）")

    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)
    mu = float(np.asarray(mu).item())
    D = None if D is None or (hasattr(D, 'size') and D.size == 0) else np.asarray(D, dtype=float)

    Z = np.asarray(Z_world, dtype=float)
    if Z.ndim == 1: Z = Z[None, :]
    L, d = Z.shape
    if X.shape[1] != d or C.shape[1] != d:
        raise ValueError(f"维度不一致：Z({d}) / X({X.shape[1]}) / C({C.shape[1]})")

    # 归一化（若有 pre）
    if pre is not None and _get_field(pre, 'center') is not None and _get_field(pre, 'scale') is not None:
        ctr = np.asarray(_get_field(pre, 'center'), dtype=float).ravel()
        scl = float(np.asarray(_get_field(pre, 'scale')).item())
        Zn  = (Z - ctr) / scl
        Xn  = X
    else:
        Zn = Z; Xn = X; ctr = scl = None

    # Laplacian 核
    K = np.exp(-mu * cdist(Zn, Xn, metric='cityblock'))   # [L, N]
    disp = K.dot(C)                                       # [L, d]
    if use_poly and D is not None:
        Pz = np.hstack([np.ones((L, 1)), Zn])             # [L, 1+d]
        disp += Pz.dot(D)

    Zp_norm = Zn + disp
    Zp = Zp_norm * scl + ctr if (isinstance(scl, float) and ctr is not None) else Zp_norm
    return Zp if Zp.shape[0] > 1 else Zp[0]


# ==================== FFD 非刚性复用 & 仿射近似（无迭代） ====================
def _bspline_basis(u):
    u2 = u*u; u3 = u2*u
    Bm1 = (1 - 3*u + 3*u2 - u3) / 6.0
    B0  = (4 - 6*u2 + 3*u3) / 6.0
    B1  = (1 + 3*u + 3*u2 - 3*u3) / 6.0
    B2  = u3 / 6.0
    return np.stack([Bm1, B0, B1, B2], axis=-1)

def _clamp_idx(i, n):
    return np.clip(i, 0, n-1)

def deform_points_by_ffd(pts, disp_grid, grid_axes):
    """三次B样条 FFD：对 Nx3 点直接应用已保存的 disp_grid / grid_axes"""
    gx,gy,gz = grid_axes
    nx,ny,nz = disp_grid.shape[:3]
    sx = (gx[-1]-gx[0])/(len(gx)-1)
    sy = (gy[-1]-gy[0])/(len(gy)-1)
    sz = (gz[-1]-gz[0])/(len(gz)-1)
    fx = (pts[:,0]-gx[0])/sx; fy=(pts[:,1]-gy[0])/sy; fz=(pts[:,2]-gz[0])/sz
    ix = np.floor(fx).astype(int); iy=np.floor(fy).astype(int); iz=np.floor(fz).astype(int)
    ux = fx-ix; uy=fy-iy; uz=fz-iz
    wx = _bspline_basis(ux); wy = _bspline_basis(uy); wz = _bspline_basis(uz)
    off = np.array([-1,0,1,2])
    D = disp_grid.reshape((-1,3))
    N = len(pts); disp = np.zeros((N,3), float)
    for a in range(4):
        ia = _clamp_idx(ix+off[a], nx); wa = wx[:,a][:,None]
        for b in range(4):
            jb = _clamp_idx(iy+off[b], ny); wb = wy[:,b][:,None]
            for c in range(4):
                kc = _clamp_idx(iz+off[c], nz); wc = wz[:,c][:,None]
                w = wa*wb*wc
                gind = ia*(ny*nz) + jb*nz + kc
                disp += w * D[gind]
    return pts + disp

def find_ffd_npz(save_dir):
    """在 save_dir 下查找 *_disp_grid.npz 或 ffd_disp_grid.npz"""
    import glob, os
    cands = sorted(glob.glob(os.path.join(save_dir, "*_disp_grid.npz")))
    if cands:
        return cands[0]
    alt = os.path.join(save_dir, "ffd_disp_grid.npz")
    return alt if os.path.exists(alt) else None

def fit_affine(A, B):
    """最小二乘拟合 4x4 仿射矩阵：A -> B"""
    N = A.shape[0]
    Ah = np.hstack([A, np.ones((N,1))])
    ATA = Ah.T @ Ah
    T = np.eye(4)
    for j in range(3):
        bj = B[:, j]
        rhs = Ah.T @ bj
        x = np.linalg.lstsq(ATA, rhs, rcond=None)[0]
        T[:, j] = x
    T[3,:] = [0,0,0,1]
    return T

def apply_affine(pts, T):
    Ph = np.hstack([pts, np.ones((len(pts),1))])
    Qh = Ph @ T.T
    return Qh[:, :3]

# ==================== 角度计算 ====================
def compute_angle(center, apex1, apex2):
    vec1 = apex1 - center
    vec2 = apex2 - center
    n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return None
    ct = np.clip(np.dot(vec1, vec2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(ct)))

# ==================== 主流程 ====================
def main():
    datas = []
    patient_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

    for patient in patient_dirs:
        print(f"\nProcessing patient: {patient}")
        cta_file   = rf"preprocess\name\{patient}\ijkcta.txt"
        spect_file = rf"preprocess\cloud_result212\{patient}\cu.txt"
        apex_file  = rf"{base_dir}\{patient}\apex_points.txt"
        save_dir   = rf"preprocess\cloud_result212\{patient}"

        cta_points   = load_point_cloud(cta_file)
        spect_points = load_point_cloud(spect_file)
        if cta_points is None or spect_points is None:
            print(f"Patient {patient}: Failed to load point clouds. Skipping.")
            continue

        cta_apex, spect_apex = load_apex_points(apex_file)
        if cta_apex is None or spect_apex is None:
            print(f"Patient {patient}: Failed to load apex points. Skipping.")
            continue

        cta_center = np.mean(cta_points, axis=0)
        patient_row = {'Patient ID': patient}

        for method in methods:
            if method == 'CluReg':
                model = load_clureg_model(patient)
                if model is None:
                    print(f"Patient {patient}, Method CluReg: clureg_model.mat not found. Skipping.")
                    continue
                try:
                    apex_transformed = apply_clureg_points(spect_apex, model)  # 世界坐标进出
                    angle = compute_angle(cta_center, cta_apex, apex_transformed)
                    if angle is None:
                        print(f"Patient {patient}, CluReg: Failed to compute angle.")
                        continue
                    patient_row[method] = angle
                    print(f"Patient {patient}, CluReg: Angle = {angle:.2f} degrees")
                except Exception as e:
                    print(f"Patient {patient}, CluReg failed: {e}")
                continue            # —— 原有方法 + FFD复用 / 仿射近似 —— #
            if method == 'FFD':
                ffd_npz = find_ffd_npz(save_dir)
                if ffd_npz is None:
                    print(f"Patient {patient}, Method FFD: disp_grid npz not found in {save_dir}. Skipping.")
                    continue
                data = np.load(ffd_npz, allow_pickle=True)
                disp_grid = data['disp_grid']; grid_axes = tuple(data['grid_axes'])
                apex_transformed = deform_points_by_ffd(spect_apex.reshape(1,3), disp_grid, grid_axes).reshape(3,)
            elif method == 'AFFINE_FROM_FFD':
                affine_path = os.path.join(save_dir, 'T_affine_from_ffd.txt')
                if os.path.exists(affine_path):
                    T = np.loadtxt(affine_path)
                else:
                    ffd_npz = find_ffd_npz(save_dir)
                    if ffd_npz is None:
                        print(f"Patient {patient}, Method AFFINE_FROM_FFD: disp_grid npz not found in {save_dir}. Skipping.")
                        continue
                    data = np.load(ffd_npz, allow_pickle=True)
                    disp_grid = data['disp_grid']; grid_axes = tuple(data['grid_axes'])
                    B = deform_points_by_ffd(spect_points, disp_grid, grid_axes)
                    T = fit_affine(spect_points, B)
                    np.savetxt(affine_path, T, fmt='%.8f')
                apex_transformed = apply_affine(spect_apex.reshape(1,3), T).reshape(3,)
            else:
                s, R, t, v, y_down = load_transform_params(method, save_dir)
                if s is None or R is None or t is None:
                    print(f"Patient {patient}, Method {method}: Failed to load transform params. Skipping.")
                    continue
                if method == 'BCPD++':
                    v_orig = interpolate_nonrigid_displacement(spect_points, y_down, v, beta=2.0)
                    apex_transformed = apply_transform_to_apex(
                        spect_apex, s, R, t, v_orig, method, spect_points, y_down, v, beta=2.0
                    )
                else:
                    # 其他方法仅刚/仿射
                    apex_transformed = s * spect_apex @ R.T + t

            angle = compute_angle(cta_center, cta_apex, apex_transformed)
            if angle is None:
                print(f"Patient {patient}, Method {method}: Failed to compute angle. Skipping.")
                continue
            patient_row[method] = angle
            print(f"Patient {patient}, Method {method}: Angle = {angle:.2f} degrees")

        datas.append(patient_row)

    if not datas:
        print("No valid data, exit.")
        return

    df = pd.DataFrame(datas)
    print("\nCollected data:")
    print(df)

    # 汇总统计
    stats = []
    for method in methods:
        if method in df.columns:
            angles = df[method].dropna()
            if len(angles) > 0:
                mean_angle = angles.mean()
                var_angle  = angles.var() if len(angles) > 1 else 0.0
                std_angle  = float(np.sqrt(var_angle)) if var_angle > 0 else 0.0
                stats.append({
                    'Method': method,
                    'Mean Angle (degrees)': mean_angle,
                    'Variance (degrees^2)': var_angle,
                    'Standard Deviation (degrees)': std_angle
                })

    # 保存 Excel
    try:
        df.to_excel(output_excel, index=False, engine='openpyxl')
        stats_df = pd.DataFrame(stats)
        stats_excel = output_excel.replace('.xlsx', '_stats.xlsx')
        stats_df.to_excel(stats_excel, index=False, engine='openpyxl')
        print(f"Saved: {output_excel}\nSaved: {stats_excel}")
    except Exception as e:
        print(f"Error saving Excel: {e}")

    # 绘图
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'black']
    for idx, method in enumerate(methods):
        if method in df.columns:
            plt.plot(df['Patient ID'], df[method], marker='o', label=method, color=colors[idx % len(colors)])
    # 标题含统计摘要
    stats_text = "\n".join([f"{s['Method']}: Mean={s['Mean Angle (degrees)']:.2f}, Std={s['Standard Deviation (degrees)']:.2f}" for s in stats])
    plt.title(f'Angle Deviations Across Patients by Method (+CluReg)\n{stats_text}', fontsize=14, pad=20)
    plt.xlabel('Patient ID', fontsize=12)
    plt.ylabel('Angle (degrees)', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    try:
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_plot}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.show()

if __name__ == "__main__":
    main()
