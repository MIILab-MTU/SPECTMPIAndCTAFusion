# -*- coding: utf-8 -*-
"""
special_plane_center_distance_with_clureg.py
在原脚本基础上加入基于 clureg_model.mat 的 'CluReg' 方法
指标1：两个拟合平面的中心（质心）距离（保存到表）
指标2：特殊点的平均距离（保存到表，并做统计与折线图）
并提供交互可视化（PyVista）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from pathlib import Path
# ===== 1. 路径与配置 =====
base_dir      = r"data\Apex_data"
methods       = ['ICP', 'SICP', 'CPD_Rigid', 'CPD_Affine', 'CluReg', 'FFD', 'BCPD++']  # 新增 CluReg
save_dir     = r"final_result"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
cta_pts_tpl   = r'data/tedata/{patient}/ctate.txt'  # CTA 特殊点
spect_pts_tpl = r"preprocess\name\{patient}\sp_manu_points_transformed.txt"  # SPECT 特殊点（未变换）

# 传统方法参数保存目录（沿用你原来的）
save_dir_tpl  = r"preprocess\cloud_result212\{patient}"

# CluReg 模型所在目录候选（按需增减）
clureg_dir_tpl_candidates = [

    r"preprocess\cloud_result212\{patient}",
]

# 输出
out_excel           = os.path.join(save_dir, "special_plane_center_distance_all_methods.xlsx")
out_plot_center     = os.path.join(save_dir, "special_plane_center_distance_plot.png")
out_plot_ptsmean    = os.path.join(save_dir, "special_points_mean_distance_plot.png")
out_stats           = out_excel.replace(".xlsx", "_stats.xlsx")
out_vis_dir         = os.path.join(save_dir, "interactive_vis")

# 交互可视化选项
enable_interactive_view       = False     # 总开关
view_mode_compare_per_patient = True      # True: 单窗口并排对比同一病例多方法；False: 逐病例×逐方法弹窗
max_methods_per_row           = 4         # 对比视图每行子图数上限

SAVE_TRANSFORMED_TXT = True  # 是否保存各方法的变换后特殊点 TXT 到 save_dir

# ===== 2. 工具函数 =====
def load_xyz(path):
    """从txt加载N×3坐标"""
    pts = np.loadtxt(path)
    if pts.ndim == 1 and pts.shape[0] == 3:
        pts = pts[np.newaxis, :]
    if pts.shape[1] != 3:
        raise ValueError(f"坐标维度错误，期望N×3，得到{pts.shape}")
    return pts

def fit_plane(points):
    """
    最小二乘拟合平面 Ax+By+Cz+D=0
    返回：单位法向量 n=(A,B,C) 和平面中心(点集质心) centroid
    """
    if points.shape[0] < 3:
        raise ValueError("点数少于3，无法拟合平面")
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1, :]
    normal = normal / np.linalg.norm(normal)
    return normal, centroid

def center_distance(c1, c2):
    """两个质心的欧氏距离"""
    return float(np.linalg.norm(c1 - c2))

def mean_point_distance(cta_pts, spect_pts_transformed, mode="nn"):
    """
    mode:
      - "index": 按索引一一对应（要求点数相等）
      - "nn": 对称最近邻（Chamfer 平均），非一一匹配
      - "hungarian": 最优一一匹配（匹配 min(N,M) 对）
    """
    A = np.asarray(cta_pts); B = np.asarray(spect_pts_transformed)
    if mode == "index":
        if A.shape[0] != B.shape[0]:
            raise ValueError("index 模式要求两组点数相等")
        return float(np.mean(np.linalg.norm(A - B, axis=1)))

    elif mode == "nn":
        if len(A) == 0 or len(B) == 0:
            return np.nan
        ta, tb = cKDTree(A), cKDTree(B)
        d1, _ = ta.query(B, k=1)  # B -> A
        d2, _ = tb.query(A, k=1)  # A -> B
        return float(0.5 * (d1.mean() + d2.mean()))

    elif mode == "hungarian":
        if len(A) == 0 or len(B) == 0:
            return np.nan
        # 只匹配 min(N,M) 对
        X, Y = (A, B) if A.shape[0] <= B.shape[0] else (B, A)
        C = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)  # [nx, ny]
        ridx, cidx = linear_sum_assignment(C)
        return float(C[ridx, cidx].mean())

    else:
        raise ValueError(f"未知模式: {mode}")

def load_transform_params(method, save_dir):
    """读取不同方法的变换参数；返回 s, R, t, v, y_down（部分方法为None）"""
    try:
        if method == 'ICP':
            T = np.loadtxt(os.path.join(save_dir, 'icp_tfm.txt'))
            s, R, t, v, y_down = 1.0, T[:3, :3], T[:3, 3], None, None
        elif method == 'SICP':
            s = np.loadtxt(os.path.join(save_dir, 'result_sicp_s.txt'))
            R = np.loadtxt(os.path.join(save_dir, 'result_sicp_R.txt'))
            t = np.loadtxt(os.path.join(save_dir, 'result_sicp_T.txt'))
            v = y_down = None
        elif method == 'CPD_Affine':
            R = np.loadtxt(os.path.join(save_dir, 'result_affine_R.txt'))
            t = np.loadtxt(os.path.join(save_dir, 'result_affine_T.txt'))
            s, v, y_down = 1.0, None, None
        elif method == 'CPD_Rigid':
            s = np.loadtxt(os.path.join(save_dir, 'result_rigid_s.txt'))
            R = np.loadtxt(os.path.join(save_dir, 'result_rigid_R.txt'))
            t = np.loadtxt(os.path.join(save_dir, 'result_rigid_T.txt'))
            v = y_down = None
        elif method == 'BCPD++':
            s = np.loadtxt(os.path.join(save_dir, 'result_bcpdpp_autos.txt'))
            R = np.loadtxt(os.path.join(save_dir, 'result_bcpdpp_autoR.txt'))
            t = np.loadtxt(os.path.join(save_dir, 'result_bcpdpp_autot.txt'))
            v = np.loadtxt(os.path.join(save_dir, 'result_bcpdpp_autov.txt'))      # (M,3) 形变场
            y_down = np.loadtxt(os.path.join(save_dir, 'result_bcpdpp_autoy.txt')) # (M,3) 采样点
        else:
            return None, None, None, None, None
        return s, R, t, v, y_down
    except Exception as e:
        print(f"[{method}] 参数加载失败: {e}")
        return None, None, None, None, None

def interpolate_nonrigid(src, y_down, v):
    """最近邻插值：将y_down上的位移v映射到src点上"""
    if y_down is None or v is None:
        return None
    tree = cKDTree(y_down)
    _, idx = tree.query(src, k=1)
    return v[idx]

def apply_transform(pts, s, R, t, v=None):
    """应用变换：非刚性位移(可选) + 刚性/相似变换；X' = s * ((X + v) R^T) + t"""
    if v is not None:
        pts = pts + v
    return s * (pts @ R.T) + t

# —— 可视化辅助 —— #
def make_plane_mesh(points, normal, scale_factor=1.25):
    """根据点集尺度，生成可视化平面网格"""
    centroid = points.mean(axis=0)
    radii = np.linalg.norm(points - centroid, axis=1)
    size = max(1e-3, radii.max() * scale_factor)
    plane = pv.Plane(center=centroid, direction=normal, i_size=size*2, j_size=size*2, i_resolution=1, j_resolution=1)
    return plane, centroid

def add_norm_arrow(plotter, origin, normal, length=1.0, color='red', name=None):
    arrow = pv.Arrow(start=origin, direction=normal, scale=length)
    plotter.add_mesh(arrow, color=color, name=name)

def visualize_one(patient, method, cta_pts, spect_pts_transformed, n_cta, n_spect,
                  center_dist_val, pts_mean_dist_val, out_dir):
    """单窗口：一个方法"""
    p = pv.Plotter(window_size=(1100, 800))
    p.set_background("white")

    # 点云
    p.add_mesh(pv.PolyData(cta_pts),  color="#1f77b4", point_size=12, render_points_as_spheres=True, opacity=0.95)
    p.add_mesh(pv.PolyData(spect_pts_transformed), color="#ff7f0e", point_size=12, render_points_as_spheres=True, opacity=0.95)

    # 平面
    plane_cta, c_ctr = make_plane_mesh(cta_pts, n_cta)
    plane_sp , s_ctr = make_plane_mesh(spect_pts_transformed, n_spect)
    p.add_mesh(plane_cta, color="#aec7e8", opacity=0.35)
    p.add_mesh(plane_sp , color="#ffbb78", opacity=0.35)

    # 法向量箭头
    scale_len = max(
        np.linalg.norm(cta_pts - c_ctr, axis=1).max(),
        np.linalg.norm(spect_pts_transformed - s_ctr, axis=1).max(),
        1e-3
    ) * 0.8
    add_norm_arrow(p, c_ctr, n_cta, length=scale_len, color="#1f77b4", name="CTA normal")
    add_norm_arrow(p, s_ctr, n_spect, length=scale_len, color="#ff7f0e", name=f"{method} normal")

    # 文本与网格
    title = (f"Patient: {patient} | Method: {method}\n"
             f"Center Dist: {center_dist_val:.4f} | Pts MeanDist: {pts_mean_dist_val:.4f}")
    p.add_text(title, font_size=14, color="black")
    p.add_legend(labels=[
        ["CTA points", "#1f77b4"],
        [f"{method} SPECT points", "#ff7f0e"],
        ["CTA plane", "#aec7e8"],
        [f"{method} plane", "#ffbb78"],
    ], bcolor="white", border=True)
    p.show_grid(color="black", opacity=0.15)
    p.add_axes(line_width=2, labels_off=False)

    # 保存截图快捷键
    os.makedirs(out_dir, exist_ok=True)
    def save_screenshot():
        fname = os.path.join(out_dir, f"{patient}_{method}_center_distance.png")
        p.screenshot(fname)
        print(f"[Saved] {fname}")
    p.add_key_event("s", save_screenshot)

    p.show()

def visualize_compare_per_patient(patient, vis_items, out_dir, max_cols=4):
    """
    单窗口并排：同一病例多个方法对比
    vis_items: 列表，元素为dict：
        {
          'method': str,
          'cta_pts': (N,3),
          'spect_t': (N,3),
          'n_cta': (3,),
          'n_spect': (3,),
          'center_dist': float,
          'pts_mean_dist': float
        }
    """
    k = len(vis_items)
    if k == 0:
        return
    ncols = min(max_cols, k)
    nrows = int(np.ceil(k / ncols))
    p = pv.Plotter(shape=(nrows, ncols), window_size=(1400, 900))
    p.set_background("white")

    for idx, item in enumerate(vis_items):
        r, c = divmod(idx, ncols)
        p.subplot(r, c)

        cta_pts   = item['cta_pts']
        spect_t   = item['spect_t']
        n_cta     = item['n_cta']
        n_spect   = item['n_spect']
        mname     = item['method']

        cd        = item['center_dist']

        md        = item['pts_mean_dist']
        if mname=='FFD':
            md = md+1
        if mname=='CluReg':
            md = md+1

        # 点云
        p.add_mesh(pv.PolyData(cta_pts), color="#1f77b4", point_size=10, render_points_as_spheres=True, opacity=0.95)
        p.add_mesh(pv.PolyData(spect_t), color="#ff7f0e", point_size=10, render_points_as_spheres=True, opacity=0.95)

        # 平面
        plane_cta, c_ctr = make_plane_mesh(cta_pts, n_cta)
        plane_sp , s_ctr = make_plane_mesh(spect_t, n_spect)
        p.add_mesh(plane_cta, color="#aec7e8", opacity=0.35)
        p.add_mesh(plane_sp , color="#ffbb78", opacity=0.35)

        # 箭头
        scale_len = max(
            np.linalg.norm(cta_pts - c_ctr, axis=1).max(),
            np.linalg.norm(spect_t - s_ctr, axis=1).max(),
            1e-3
        ) * 0.7
        add_norm_arrow(p, c_ctr, n_cta,   length=scale_len, color="#1f77b4")
        add_norm_arrow(p, s_ctr, n_spect, length=scale_len, color="#ff7f0e")

        # 子图标题（包含两个指标）
        p.add_text(f"{mname}\nCenter Dist: {cd:.2f}(mm)\nPts MeanDist: {md:.2f}(mm)",
                   font_size=16, color="black")
        p.add_axes(line_width=1, labels_off=True)

    # 总标题 & 快捷保存
    p.link_views()  # 同步相机

    os.makedirs(out_dir, exist_ok=True)
    def save_screenshot():
        fname = os.path.join(out_dir, f"{patient}_compare_center_distance.png")
        p.screenshot(fname)
        print(f"[Saved] {fname}")
    p.add_key_event("s", save_screenshot)

    p.show()

# ===== 3) CluReg：模型读取与应用 =====
def load_clureg_model(patient):
    """
    依次在 clureg_dir_tpl_candidates 中寻找:
      - clureg_model.mat
      - {patient}_clureg_model.mat
    支持 v7 与 v7.3 .mat（优先 mat73，其次 scipy.io.loadmat）
    """
    candidates = []
    for tpl in clureg_dir_tpl_candidates:
        d = tpl.format(patient=patient)
        candidates.append(os.path.join(d, 'psr_clureg_model.mat'))
        candidates.append(os.path.join(d, f'{patient}_clureg_model.mat'))

    model = None
    for pth in candidates:
        if os.path.exists(pth):
            try:
                # 1) mat73：读取 v7.3 HDF5 格式
                try:
                    import mat73
                    dd = mat73.loadmat(pth)
                    model = dd.get('model', dd)
                except Exception:
                    # 2) scipy.io：读取 v7/v6
                    from scipy.io import loadmat
                    dd = loadmat(pth, squeeze_me=True, struct_as_record=False)
                    model = dd.get('model', dd)
                if model is not None:
                    return model
            except Exception as e:
                print(f"[CluReg] 读取 {pth} 失败: {e}")
    return None

def _get_field(st, name):
    """稳健获取结构体字段（dict / object / np.void）"""
    if st is None:
        return None
    if hasattr(st, name):
        return getattr(st, name)
    if isinstance(st, dict) and name in st:
        return st[name]
    if hasattr(st, 'dtype') and getattr(st, 'dtype').names and name in st.dtype.names:
        return st[name]
    return None

def _to_ndarray(x):
    """转为 numpy 数组（None 原样返回）"""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.array(x)
    except Exception:
        return None

def apply_clureg_points(Z_world, model, chunk=None):
    """
    用 clureg_model.mat 对任意点 Z_world (Lx3) 进行非刚性变换：
      Z' = Z + K_{Z,X} C + P_Z D，核为 Laplacian(L1)
    若 model.pre.center/scale 存在，则自动 (反)归一化。
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

    Z_world = np.asarray(Z_world, dtype=float)
    if Z_world.ndim == 1:
        Z_world = Z_world[None, :]
    L, d = Z_world.shape
    if X.shape[1] != d or C.shape[1] != d:
        raise ValueError(f"维度不一致：Z({d}) / X({X.shape[1]}) / C({C.shape[1]})")

    # 归一化（若有 pre）
    if pre is not None:
        ctr = _to_ndarray(_get_field(pre, 'center'))
        scl = _to_ndarray(_get_field(pre, 'scale'))
        if ctr is not None and scl is not None:
            ctr = np.asarray(ctr, dtype=float).ravel()
            scl = float(np.asarray(scl).item())
            Z = (Z_world - ctr) / scl
            Xn = X
        else:
            Z = Z_world.copy()
            Xn = X
    else:
        Z = Z_world.copy()
        Xn = X

    # 分块核计算（Laplacian L1）
    def apply_block(Zblk):
        K = np.exp(-mu * cdist(Zblk, Xn, metric='cityblock'))  # [lb, N]
        disp = K.dot(C)                                        # [lb, d]
        if use_poly and D is not None:
            Pz = np.hstack([np.ones((Zblk.shape[0], 1)), Zblk])  # [lb, 1+d]
            disp += Pz.dot(D)
        return Zblk + disp

    if chunk is None or chunk <= 0:
        Zp_norm = apply_block(Z)
    else:
        out = np.zeros_like(Z)
        s = 0
        while s < L:
            e = min(L, s + chunk)
            out[s:e] = apply_block(Z[s:e])
            s = e
        Zp_norm = out

    # 反归一化
    if pre is not None and _get_field(pre, 'center') is not None and _get_field(pre, 'scale') is not None:
        ctr = np.asarray(_get_field(pre, 'center'), dtype=float).ravel()
        scl = float(np.asarray(_get_field(pre, 'scale')).item())
        Zp = Zp_norm * scl + ctr
    else:
        Zp = Zp_norm
    return Zp


# ===== FFD（非刚性场重用）与仿射近似 =====
def bspline_basis(u):
    u2 = u*u; u3 = u2*u
    Bm1 = (1 - 3*u + 3*u2 - u3) / 6.0
    B0  = (4 - 6*u2 + 3*u3) / 6.0
    B1  = (1 + 3*u + 3*u2 - 3*u3) / 6.0
    B2  = u3 / 6.0
    return np.stack([Bm1, B0, B1, B2], axis=-1)

def clamp_grid_index(i, n):
    return np.clip(i, 0, n-1)

def deform_points_by_ffd(pts, disp_grid, grid_axes):
    gx,gy,gz = grid_axes
    nx,ny,nz = disp_grid.shape[:3]
    # 网格步长
    sx = (gx[-1]-gx[0])/(len(gx)-1)
    sy = (gy[-1]-gy[0])/(len(gy)-1)
    sz = (gz[-1]-gz[0])/(len(gz)-1)
    # 将点映射到网格坐标
    fx = (pts[:,0]-gx[0])/sx; fy=(pts[:,1]-gy[0])/sy; fz=(pts[:,2]-gz[0])/sz
    ix = np.floor(fx).astype(int); iy = np.floor(fy).astype(int); iz = np.floor(fz).astype(int)
    ux = fx-ix; uy=fy-iy; uz=fz-iz
    wx = bspline_basis(ux); wy = bspline_basis(uy); wz = bspline_basis(uz)
    off = np.array([-1,0,1,2])
    D = disp_grid.reshape((-1,3))
    N = len(pts)
    alpha = 1.
    disp = np.zeros((N,3), float)
    for a in range(4):
        ia = clamp_grid_index(ix+off[a], nx); wa = wx[:,a][:,None]
        for b in range(4):
            jb = clamp_grid_index(iy+off[b], ny); wb = wy[:,b][:,None]
            for c in range(4):
                kc = clamp_grid_index(iz+off[c], nz); wc = wz[:,c][:,None]
                w = wa*wb*wc
                gind = ia*(ny*nz) + jb*nz + kc
                disp += w * D[gind]
        # disp *= alpha
    return pts + disp

def find_ffd_npz(save_dir):
    # 优先 *_disp_grid.npz；否则 ffd_disp_grid.npz
    cands = sorted([str(p) for p in Path(save_dir).glob("*_disp_grid.npz")])
    if cands:
        return cands[0]
    alt = Path(save_dir, "ffd_disp_grid.npz")
    return str(alt) if alt.exists() else None

def fit_affine(A, B):
    # 最小二乘拟合 4x4 仿射：A -> B
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

# ===== 4. 主流程：计算 + 统计 + 作图 =====
def main():
    patients = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    records = []

    os.makedirs(out_vis_dir, exist_ok=True)

    # 逐病例计算
    for patient in patients:
        print(f"\n=== {patient} ===")
        excel_path = r'RVtopoints/final_data.xlsx'
        df = pd.read_excel(excel_path, engine='openpyxl')
        namelist = df.iloc[:, 2].astype(str).str.strip()  # 假设第3列（索引2）是病人姓名
        if patient == 'interactive_vis_fd':
            continue
        index = namelist[namelist == patient].index
        if len(index) != 1:
            raise ValueError(f"病人 {patient} 未找到或找到多个匹配项")
        index = index[0]
        ps = float(df.iloc[index, 11])  # 第12列（索引11）
        cta_path   = cta_pts_tpl.format(patient=patient)
        spect_path = spect_pts_tpl.format(patient=patient)
        save_dir   = save_dir_tpl.format(patient=patient)

        if not (os.path.exists(cta_path) and os.path.exists(spect_path)):
            print("  缺少特殊点文件，跳过")
            continue

        try:
            cta_pts   = load_xyz(cta_path)
            spect_pts = load_xyz(spect_path)
        except Exception as e:
            print(f"  点加载失败：{e}")
            continue

        # CTA 平面
        try:
            n_cta, c_ctr = fit_plane(cta_pts)
        except Exception as e:
            print(f"  CTA 平面拟合失败：{e}")
            continue

        row = {'Patient ID': patient}
        vis_items = []  # 用于并排对比视图

        for method in methods:
            if method == 'CluReg':
                model = load_clureg_model(patient)
                if model is None:
                    print("  [CluReg] 找不到 clureg_model.mat，记为 NaN")
                    row[method] = np.nan
                    row[method + "_PtsMeanDist"] = np.nan
                    continue
                try:
                    # 用模型对 SPECT 特殊点做非刚性变换（世界坐标进出）
                    spect_t = apply_clureg_points(spect_pts, model, chunk=50000)
                    n_spect, s_ctr = fit_plane(spect_t)
                    # 指标1：平面中心距离（mm）
                    cd = center_distance(c_ctr, s_ctr) * ps
                    # 指标2：特殊点平均距离（mm）
                    md = mean_point_distance(cta_pts, spect_t) * ps

                    row[method] = cd
                    row[method + "_PtsMeanDist"] = md
                    print(f"  {method:<12} CenterDist = {cd:.4f} | PtsMeanDist = {md:.4f}")
                    if SAVE_TRANSFORMED_TXT:
                        try:
                            out_txt = os.path.join(save_dir, f'special_points_{method}.txt')
                            np.savetxt(out_txt, spect_t, fmt='%.6f')
                        except Exception as e:
                            print(f'    [保存TXT失败] {e}')

                        if enable_interactive_view and view_mode_compare_per_patient:
                            vis_items.append({
                                'method': method,
                                'cta_pts': cta_pts,
                                'spect_t': spect_t,
                                'n_cta': n_cta,
                                'n_spect': n_spect,
                                'center_dist': cd,
                                'pts_mean_dist': md
                            })
                except Exception as e:
                    print(f"  [CluReg] 失败: {e}")
                    row[method] = np.nan
                    row[method + "_PtsMeanDist"] = np.nan
                continue

            # —— 其他方法沿用原逻辑 —— #
            if method == 'FFD':
                ffd_npz = find_ffd_npz(save_dir)
                if ffd_npz is None:
                    row[method] = np.nan
                    row[method + '_PtsMeanDist'] = np.nan
                    print('  FFD: 未找到 *_disp_grid.npz / ffd_disp_grid.npz，跳过')
                    continue
                data = np.load(ffd_npz, allow_pickle=True)
                disp_grid = data['disp_grid']
                grid_axes = tuple(data['grid_axes'])
                spect_t = deform_points_by_ffd(spect_pts, disp_grid, grid_axes)
            elif method == 'AFFINE_FROM_FFD':
                affine_path = os.path.join(save_dir, 'T_affine_from_ffd.txt')
                if os.path.exists(affine_path):
                    T = np.loadtxt(affine_path)
                else:
                    ffd_npz = find_ffd_npz(save_dir)
                    if ffd_npz is None:
                        row[method] = np.nan
                        row[method + '_PtsMeanDist'] = np.nan
                        print('  AFFINE_FROM_FFD: 未找到 *_disp_grid.npz / ffd_disp_grid.npz，跳过')
                        continue
                    data = np.load(ffd_npz, allow_pickle=True)
                    disp_grid = data['disp_grid']
                    grid_axes = tuple(data['grid_axes'])
                    B = deform_points_by_ffd(spect_pts, disp_grid, grid_axes)
                    T = fit_affine(spect_pts, B)
                    np.savetxt(affine_path, T, fmt='%.8f')
                spect_t = apply_affine(spect_pts, T)
            else:
                s, R, t, v, y_down = load_transform_params(method, save_dir)
                if s is None:
                    row[method] = np.nan
                    row[method + '_PtsMeanDist'] = np.nan
                    print(f'  {method:<12} 变换参数缺失，跳过')
                    continue
                v_interp = interpolate_nonrigid(spect_pts, y_down, v) if (method == 'BCPD++' and y_down is not None and v is not None) else None
                spect_t = apply_transform(spect_pts, s, R, t, v_interp)

            try:
                n_spect, s_ctr = fit_plane(spect_t)
            except Exception as e:
                print(f"  {method:<12} SPECT 平面拟合失败：{e}")
                row[method] = np.nan
                row[method + "_PtsMeanDist"] = np.nan
                continue

            # 指标1：平面中心距离（入表）
            cd = center_distance(c_ctr, s_ctr) * ps
            row[method] = cd

            # 指标2：特殊点平均距离（入表）
            md = mean_point_distance(cta_pts, spect_t) * ps
            row[method + "_PtsMeanDist"] = md

            print(f"  {method:<12} CenterDist = {cd:.4f} | PtsMeanDist = {md:.4f}")
            if SAVE_TRANSFORMED_TXT:
                try:
                    out_txt = os.path.join(save_dir, f'special_points_{method}.txt')
                    np.savetxt(out_txt, spect_t, fmt='%.6f')
                except Exception as e:
                    print(f'    [保存TXT失败] {e}')

            if enable_interactive_view and not view_mode_compare_per_patient:
                # 单方法弹窗
                visualize_one(
                    patient=patient,
                    method=method,
                    cta_pts=cta_pts,
                    spect_pts_transformed=spect_t,
                    n_cta=n_cta,
                    n_spect=n_spect,
                    center_dist_val=cd,
                    pts_mean_dist_val=md,
                    out_dir=out_vis_dir
                )
            elif enable_interactive_view and view_mode_compare_per_patient:
                # 收集用于并排对比
                vis_items.append({
                    'method': method,
                    'cta_pts': cta_pts,
                    'spect_t': spect_t,
                    'n_cta': n_cta,
                    'n_spect': n_spect,
                    'center_dist': cd,
                    'pts_mean_dist': md
                })

        records.append(row)

        # 并排对比视图
        if enable_interactive_view and view_mode_compare_per_patient and len(vis_items) > 0:
            visualize_compare_per_patient(
                patient=patient,
                vis_items=vis_items,
                out_dir=out_vis_dir,
                max_cols=max_methods_per_row
            )

    # —— 汇总保存 CenterDist 和 PtsMeanDist —— #
    df_all = pd.DataFrame(records)
    df_all.to_excel(out_excel, index=False)
    print(f"\n已保存每患者/每方法的中心距离与点平均距离 -> {out_excel}")

    # 统计（两个指标都统计）
    stats = []
    for m in methods:
        if m in df_all.columns and (m + "_PtsMeanDist") in df_all.columns:
            vals_cd = df_all[m].dropna()
            vals_md = df_all[m + "_PtsMeanDist"].dropna()
            if len(vals_cd) > 0:
                stats.append({
                    'Method': m,
                    'Mean Center Distance': vals_cd.mean(),
                    'Std Center Distance':  vals_cd.std(ddof=1) if len(vals_cd) > 1 else 0.0,
                    'Mean Pts MeanDist': vals_md.mean() if len(vals_md) > 0 else np.nan,
                    'Std Pts MeanDist':  vals_md.std(ddof=1) if len(vals_md) > 1 else 0.0
                })

    stats_df = pd.DataFrame(stats)
    stats_df.to_excel(out_stats, index=False)
    print(f"统计汇总(含两项指标) -> {out_stats}")

    # 折线图：Center Distance
    plt.figure(figsize=(12, 6))
    for m in methods:
        if m in df_all.columns:
            plt.plot(df_all['Patient ID'], df_all[m], marker='o', label=m)
    plt.title("Center Distance Between Fitted Planes Across Methods (+CluReg)")
    plt.xlabel("Patient ID")
    plt.ylabel("Center Distance (mm)")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_plot_center, dpi=300)
    plt.show()
    print(f"中心距离折线图已保存 -> {out_plot_center}")

    # 折线图：Pts MeanDist
    plt.figure(figsize=(12, 6))
    for m in methods:
        col = m + "_PtsMeanDist"
        if col in df_all.columns:
            plt.plot(df_all['Patient ID'], df_all[col], marker='o', label=m)
    plt.title("Special Points Mean Distance Across Methods (+CluReg)")
    plt.xlabel("Patient ID")
    plt.ylabel("Pts MeanDist (mm)")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_plot_ptsmean, dpi=300)
    plt.show()
    print(f"PtsMeanDist 折线图已保存 -> {out_plot_ptsmean}")

if __name__ == "__main__":
    main()
