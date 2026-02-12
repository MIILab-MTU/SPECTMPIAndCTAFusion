import numpy as np  # 导入 numpy 用于处理点云数据和投影计算
import pyvista as pv  # 导入 pyvista 用于 3D 可视化
from sklearn.decomposition import PCA  # 导入 PCA 用于主成分分析
import os  # 导入 os 用于文件路径操作
from scipy.spatial import cKDTree

def load_point_cloud(file_path):
    """加载点云文件并验证格式

    Args:
        file_path (str): 点云文件路径（TXT 格式）

    Returns:
        np.ndarray: 点云数据，形状为 (n, 3)
    """
    try:
        points = np.loadtxt(file_path)  # 加载点云，预期每行 x, y, z 坐标
        if points.shape[1] != 3:
            raise ValueError(f"点云数据格式错误：形状 {points.shape}，预期每行 3 列 (x, y, z)")
        print(f"加载点云 {file_path}，形状：{points.shape}")
        return points
    except ValueError as e:
        print(f"加载点云文件失败：{e}")
        exit(1)

def compute_intersection_plane(left_points, right_points, distance_threshold=5.0):
    """计算左心和右心的交接平面，基于接触区域的最近点对

    Args:
        left_points (np.ndarray): 左心点云，形状 (n, 3)
        right_points (np.ndarray): 右心点云，形状 (m, 3)
        distance_threshold (float): 接触区域的距离阈值（默认 5.0）

    Returns:
        tuple: (normal, center)
               - 交接平面法向量 (3,)
               - 交接平面中心 (3,)
    """
    print(f"合并点云形状：{np.vstack((left_points, right_points)).shape}")

    # 使用 KDTree 找到左心和右心的最近点对
    left_tree = cKDTree(left_points)
    right_tree = cKDTree(right_points)

    # 计算左心点到右心点云的最近距离
    distances, indices = right_tree.query(left_points, k=1)
    contact_mask_left = distances < distance_threshold
    contact_points_left = left_points[contact_mask_left]

    # 计算右心点到左心点云的最近距离
    distances, indices = left_tree.query(right_points, k=1)
    contact_mask_right = distances < distance_threshold
    contact_points_right = right_points[contact_mask_right]

    # 合并接触区域的点
    if contact_points_left.shape[0] == 0 or contact_points_right.shape[0] == 0:
        print("警告：接触区域点数不足，退回到 PCA 方法")
        all_points = np.vstack((left_points, right_points))
        pca = PCA(n_components=3)
        pca.fit(all_points)
        normal = pca.components_[-1]
        center = np.mean(all_points, axis=0)
    else:
        contact_points = np.vstack((contact_points_left, contact_points_right))
        print(f"接触区域点云形状：{contact_points.shape}")

        # 对接触区域点云进行 PCA
        pca = PCA(n_components=3)
        pca.fit(contact_points)
        normal = pca.components_[-1]  # 最小方差方向
        center = np.mean(contact_points, axis=0)

    # 标准化法向量方向（假设法向量大致沿 x 轴正方向）
    if normal[0] < 0:
        normal = -normal
        print("法向量已翻转，使 x 分量为正")

    # 验证平面分隔效果
    left_projections = np.dot(left_points - center, normal)
    right_projections = np.dot(right_points - center, normal)
    left_side = np.mean(left_projections > 0)
    right_side = np.mean(right_projections < 0)
    print(f"左心点在平面正侧比例：{left_side:.2%}")
    print(f"右心点在平面负侧比例：{right_side:.2%}")

    if left_side < 0.7 or right_side < 0.7:
        print("警告：平面分隔效果不佳，可能仍倾斜，建议调整 distance_threshold 或检查点云对齐")

    return normal, center

def project_points_to_plane(points, normal, center):
    """将点云投影到指定平面上，并记录原始点索引

    Args:
        points (np.ndarray): 待投影点云，形状 (n, 3)
        normal (np.ndarray): 平面法向量，形状 (3,)
        center (np.ndarray): 平面中心，形状 (3,)

    Returns:
        tuple: (projected_points, indices) - 投影点云 (n, 3) 和原始点索引
    """
    projected_points = np.zeros_like(points)  # 初始化投影点数组
    indices = np.arange(len(points))  # 记录每个投影点对应的原始点索引
    for i, p in enumerate(points):
        d = np.dot(normal, p - center)  # 点到平面的带符号距离
        projected_points[i] = p - d * normal  # 投影点
    return projected_points, indices

def compute_major_axis(projected_points, original_points, indices, plane_normal, intersection_plane_center):
    """计算投影点云的长轴、短轴、短轴平面参数及调整后的交界点，并返回原始 3D 最低点

    Args:
        projected_points (np.ndarray): 左心投影点云，形状 (n, 3)
        original_points (np.ndarray): 原始左心点云，形状 (n, 3)
        indices (np.ndarray): 投影点对应的原始点索引
        plane_normal (np.ndarray): 交接平面法向量，形状 (3,)
        intersection_plane_center (np.ndarray): 交接平面中心，形状 (3,)

    Returns:
        tuple: (center, major_axis_direction, major_axis_points, minor_axis_direction, minor_axis_points, low_point, low_point_3d, short_plane_center, short_plane_normal, boundary_point_1, boundary_point_2)
               - 中心、长轴方向、长轴端点、短轴方向、短轴端点、投影最低点、原始 3D 最低点、短轴平面中心、短轴平面法向量、两个调整后的交界点
    """
    pca = PCA(n_components=2)  # 初始化 PCA，分析平面上的二维分布
    pca.fit(projected_points)  # 对投影点云进行 PCA
    center = np.mean(projected_points, axis=0)  # 计算中心（均值）
    major_axis_direction = pca.components_[0]  # 第一个主成分作为长轴方向
    minor_axis_direction = pca.components_[1]  # 第二个主成分作为短轴方向

    # 计算长轴的投影距离和端点
    major_projections = np.dot(projected_points - center, major_axis_direction)  # (p - c) · d
    max_idx = np.argmax(major_projections)  # 最大投影索引（最高点）
    min_idx = np.argmin(major_projections)  # 最小投影索引（最低点）
    high_point = projected_points[max_idx]  # 投影最高点
    low_point = projected_points[min_idx]  # 投影最低点
    major_axis_points = np.array([low_point, high_point])  # 长轴端点
    if high_point[1] < low_point[1]:
        high_point, low_point = low_point, high_point
        max_idx, min_idx = min_idx, max_idx
    if major_axis_direction[1] < 0:
        major_axis_direction = -major_axis_direction
        minor_axis_direction = -minor_axis_direction
    # 计算短轴的投影距离和端点
    minor_projections = np.dot(projected_points - center, minor_axis_direction)  # (p - c) · d_minor
    minor_max_idx = np.argmax(minor_projections)  # 短轴最大投影索引
    minor_min_idx = np.argmin(minor_projections)  # 短轴最小投影索引
    minor_high_point = projected_points[minor_max_idx]  # 短轴最大投影点
    minor_low_point = projected_points[minor_min_idx]  # 短轴最小投影点
    minor_axis_points = np.array([minor_low_point, minor_high_point])  # 短轴端点

    # 计算短轴平面参数
    short_plane_center = (high_point + low_point) / 2  # 长轴中点
    short_plane_normal = major_axis_direction / np.linalg.norm(major_axis_direction)  # 短轴平面法向量为长轴方向（归一化）

    # 验证短轴平面法向量与交接平面法向量垂直
    dot_product = np.dot(plane_normal, short_plane_normal)
    if abs(dot_product) > 1e-6:
        print(f"警告：短轴平面法向量与交接平面法向量不垂直！点积 = {dot_product}")

    # 计算交接平面与短轴平面的交线方向
    intersection_line_direction = np.cross(plane_normal, short_plane_normal)
    intersection_line_direction = intersection_line_direction / np.linalg.norm(intersection_line_direction)  # 归一化

    # 筛选在短轴平面附近且在交接平面上的候选点
    plane_tolerance = 1e-6  # 交接平面距离容差
    short_plane_distance_threshold = 1.0  # 短轴平面附近距离阈值
    candidate_indices = []
    short_plane_distances = []
    for i, point in enumerate(projected_points):
        # 检查是否在交接平面上
        plane_dist = abs(np.dot(point - intersection_plane_center, plane_normal))
        if plane_dist > plane_tolerance:
            continue
        # 计算到短轴平面的距离
        short_plane_dist = abs(np.dot(point - short_plane_center, short_plane_normal))
        if short_plane_dist <= short_plane_distance_threshold:
            candidate_indices.append(i)
            short_plane_distances.append(short_plane_dist)
        else:
            # 记录距离以备选
            short_plane_distances.append(short_plane_dist)

    if not candidate_indices:
        print(f"警告：没有点满足短轴平面附近约束（距离 <= {short_plane_distance_threshold}）。选择到短轴平面距离最小的两个点...")
        short_plane_distances = np.array(short_plane_distances)
        candidate_indices = np.argsort(short_plane_distances)[:2]

    # 从候选点中选择沿交线方向的最外部点
    candidate_points = projected_points[candidate_indices]
    line_projections = np.dot(candidate_points - short_plane_center, intersection_line_direction)
    max_line_idx = np.argmax(line_projections)
    min_line_idx = np.argmin(line_projections)
    boundary_point_1 = candidate_points[max_line_idx]  # 初始最外部交界点 1
    boundary_point_2 = candidate_points[min_line_idx]  # 初始最外部交界点 2

    # 计算点云范围以确定收缩步长
    x_range = projected_points[:, 0].max() - projected_points[:, 0].min()
    y_range = projected_points[:, 1].max() - projected_points[:, 1].min()
    z_range = projected_points[:, 2].max() - projected_points[:, 2].min()
    step_size = min(x_range, y_range, z_range) * 0.01  # 步长为点云范围的 1%
    point_cloud_distance_threshold = 1.0  # 左心点云附近距离阈值

    def compute_nearest_point_distance(point, points):
        """计算点到点云中最近点的距离"""
        distances = np.linalg.norm(points - point, axis=1)
        return np.min(distances)

    # 收缩交界点 1
    current_point = boundary_point_1
    projection_1 = np.dot(current_point - short_plane_center, intersection_line_direction)
    while True:
        nearest_dist = compute_nearest_point_distance(current_point, projected_points)  # 检查左心点云
        if nearest_dist <= point_cloud_distance_threshold:
            boundary_point_1 = current_point
            break
        # 沿 -intersection_line_direction 向内移动
        current_point = current_point - step_size * intersection_line_direction
        # 验证是否仍在交接平面上
        plane_dist = abs(np.dot(current_point - intersection_plane_center, plane_normal))
        if plane_dist > plane_tolerance:
            current_point = current_point + step_size * intersection_line_direction  # 回退
            break
        # 验证是否仍在短轴平面附近
        short_plane_dist = abs(np.dot(current_point - short_plane_center, short_plane_normal))
        if short_plane_dist > short_plane_distance_threshold:
            current_point = current_point + step_size * intersection_line_direction  # 回退
            break
        # 检查是否超过中心点
        new_projection = np.dot(current_point - short_plane_center, intersection_line_direction)
        if new_projection <= 0:  # 超过中心点
            current_point = short_plane_center
            break
    boundary_point_1 = current_point

    # 收缩交界点 2
    current_point = boundary_point_2
    projection_2 = np.dot(current_point - short_plane_center, intersection_line_direction)
    while True:
        nearest_dist = compute_nearest_point_distance(current_point, projected_points)  # 检查左心点云
        if nearest_dist <= point_cloud_distance_threshold:
            boundary_point_2 = current_point
            break
        # 沿 +intersection_line_direction 向内移动
        current_point = current_point + step_size * intersection_line_direction
        # 验证是否仍在交接平面上
        plane_dist = abs(np.dot(current_point - intersection_plane_center, plane_normal))
        if plane_dist > plane_tolerance:
            current_point = current_point - step_size * intersection_line_direction  # 回退
            break
        # 验证是否仍在短轴平面附近
        short_plane_dist = abs(np.dot(current_point - short_plane_center, short_plane_normal))
        if short_plane_dist > short_plane_distance_threshold:
            current_point = current_point - step_size * intersection_line_direction  # 回退
            break
        # 检查是否超过中心点
        new_projection = np.dot(current_point - short_plane_center, intersection_line_direction)
        if new_projection >= 0:  # 超过中心点
            current_point = short_plane_center
            break
    boundary_point_2 = current_point

    # 如果收缩失败（例如达到中心点或无法满足约束），选择到左心点云最近的候选点
    for point, label in [(boundary_point_1, "交界点 1"), (boundary_point_2, "交界点 2")]:
        nearest_dist = compute_nearest_point_distance(point, projected_points)
        if nearest_dist > point_cloud_distance_threshold:
            print(f"警告：{label} 收缩后未满足左心点云附近约束（距离 = {nearest_dist}）。选择到左心点云最近的候选点...")
            candidate_distances = [compute_nearest_point_distance(p, projected_points) for p in candidate_points]
            best_idx = np.argmin(candidate_distances)
            if label == "交界点 1":
                boundary_point_1 = candidate_points[best_idx]
            else:
                boundary_point_2 = candidate_points[best_idx]

    # 计算调整后交界点的距离
    boundary_dist_1 = abs(np.dot(boundary_point_1 - short_plane_center, short_plane_normal))
    boundary_dist_2 = abs(np.dot(boundary_point_2 - short_plane_center, short_plane_normal))
    plane_dist_1 = abs(np.dot(boundary_point_1 - intersection_plane_center, plane_normal))
    plane_dist_2 = abs(np.dot(boundary_point_2 - intersection_plane_center, plane_normal))
    nearest_dist_1 = compute_nearest_point_distance(boundary_point_1, projected_points)
    nearest_dist_2 = compute_nearest_point_distance(boundary_point_2, projected_points)

    # 找到最低点对应的原始 3D 点
    low_point_3d = original_points[indices[min_idx]]  # 原始 3D 最低点

    # 计算从最低点到交界点的直线长度
    line_length_1 = np.linalg.norm(low_point_3d - boundary_point_1)
    line_length_2 = np.linalg.norm(low_point_3d - boundary_point_2)

    print(f"长轴中心：{center}")
    print(f"长轴方向：{major_axis_direction}")
    print(f"短轴方向：{minor_axis_direction}")
    print(f"投影最低点：{low_point}")
    print(f"原始 3D 最低点：{low_point_3d}")
    print(f"长轴长度：{np.linalg.norm(high_point - low_point)}")
    print(f"短轴端点：{minor_low_point}, {minor_high_point}")
    print(f"短轴长度：{np.linalg.norm(minor_high_point - minor_low_point)}")
    print(f"短轴平面中心：{short_plane_center}")
    print(f"短轴平面法向量：{short_plane_normal}")
    print(f"交接平面法向量与短轴平面法向量点积：{dot_product}")
    print(f"交线方向：{intersection_line_direction}")
    print(f"调整后最外部交界点 1：{boundary_point_1}, 到短轴平面距离：{boundary_dist_1}, 到交接平面距离：{plane_dist_1}, 到左心点云最近距离：{nearest_dist_1}")
    print(f"调整后最外部交界点 2：{boundary_point_2}, 到短轴平面距离：{boundary_dist_2}, 到交接平面距离：{plane_dist_2}, 到左心点云最近距离：{nearest_dist_2}")
    print(f"从最低点到交界点 1 的直线长度：{line_length_1}")
    print(f"从最低点到交界点 2 的直线长度：{line_length_2}")
    return (
        center,
        major_axis_direction,
        major_axis_points,
        minor_axis_direction,
        minor_axis_points,
        low_point,
        low_point_3d,
        short_plane_center,
        short_plane_normal,
        boundary_point_1,
        boundary_point_2
    )

def create_plane(normal, center, all_points):
    """创建平面，动态调整大小

    Args:
        normal (np.ndarray): 平面法向量，形状 (3,)
        center (np.ndarray): 平面中心，形状 (3,)
        all_points (np.ndarray): 所有点云，用于计算范围

    Returns:
        pv.Plane: 平面对象
    """
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()  # x 范围
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()  # y 范围
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()  # z 范围
    print(f"点云范围：x=({x_min}, {x_max}), y=({y_min}, {y_max}), z=({z_min}, {z_max})")
    range_max = max(x_max - x_min, y_max - y_min, z_max - z_min)  # 最大范围
    plane_size = range_max * 1.5  # 平面大小为范围的 1.5 倍
    print(f"平面大小：{plane_size}")
    return pv.Plane(center=center, direction=normal, i_size=plane_size, j_size=plane_size)

def visualize_projected_points(projected_points, major_axis_points, minor_axis_points, low_point):
    """可视化投影到平面上的点云、长轴、短轴和最低点

    Args:
        projected_points (np.ndarray): 投影点云，形状 (n, 3)
        major_axis_points (np.ndarray): 长轴线段端点，形状 (2, 3)
        minor_axis_points (np.ndarray): 短轴线段端点，形状 (2, 3)
        low_point (np.ndarray): 投影最低点，形状 (3,)
    """
    plotter = pv.Plotter()  # 初始化可视化窗口
    projected_mesh = pv.PolyData(projected_points)  # 转换为点云对象
    plotter.add_mesh(projected_mesh, color='blue', opacity=0.5, point_size=5)  # 蓝色，半透明

    # 添加长轴线段
    major_axis_line = pv.Line(major_axis_points[0], major_axis_points[1])  # 创建长轴线段
    plotter.add_mesh(major_axis_line, color='red', line_width=3)  # 红色，线宽 3

    # 添加短轴线段
    minor_axis_line = pv.Line(minor_axis_points[0], minor_axis_points[1])  # 创建短轴线段
    plotter.add_mesh(minor_axis_line, color='green', line_width=3)  # 绿色，线宽 3

    # 添加投影最低点（橙色小球）
    low_sphere = pv.Sphere(radius=1.0, center=low_point)  # 半径 1.0
    plotter.add_mesh(low_sphere, color='orange')  # 橙色

    plotter.add_axes()  # 添加坐标轴
    plotter.show()  # 显示窗口

def visualize_full_scene(left_points, right_points, intersection_plane, short_plane, low_point_3d, boundary_point_1, boundary_point_2, patient, plane_normal, intersection_plane_center, short_plane_normal,save_dir,visual=True ):
    """可视化完整的 3D 场景，包括交接平面、短轴平面、放大的红色最低点、向交接平面弯曲的贝塞尔曲线及带颜色的均匀采样点（左边绿色，右边红色），并按从左到右排序输出点云，去重最低点

    Args:
        left_points (np.ndarray): 左心点云，形状 (n, 3)
        right_points (np.ndarray): 右心点云，形状 (m, 3)
        intersection_plane (pv.Plane): 交接平面
        short_plane (pv.Plane): 短轴平面
        low_point_3d (np.ndarray): 原始 3D 最低点，形状 (3,)
        boundary_point_1 (np.ndarray): 调整后最外部交界点 1，形状 (3,)
        boundary_point_2 (np.ndarray): 调整后最外部交界点 2，形状 (3,)
        patient (str): 患者名称，用于生成输出文件路径
        plane_normal (np.ndarray): 交接平面法向量，形状 (3,)
        intersection_plane_center (np.ndarray): 交接平面中心，形状 (3,)
        short_plane_normal (np.ndarray): 短轴平面法向量，形状 (3,)
    """
    plotter = pv.Plotter()  # 初始化可视化窗口
    left_mesh = pv.PolyData(left_points)  # 左心点云
    plotter.add_mesh(left_mesh, color='#d8654f', opacity=0.9, point_size=3)  # 蓝色，半透明
    right_mesh = pv.PolyData(right_points)  # 右心点云
    plotter.add_mesh(right_mesh, color='#80ae80', opacity=0.9, point_size=3)  # 紫色，半透明
    plotter.add_mesh(intersection_plane, color='yellow', opacity=0.7, show_edges=True, edge_color='black')  # 黄色交接平面，黑色边界
    # plotter.add_mesh(short_plane, color='green', opacity=0.7, show_edges=True, edge_color='black')  # 绿色短轴平面，黑色边界
    plotter.background_color = 'white'
    # 添加原始 3D 最低点（红色大球）
    low_sphere = pv.Sphere(radius=4.0, center=low_point_3d)  # 半径 2.0
    # plotter.add_mesh(low_sphere, color='#d8654f')  # 红色

    # 计算控制点并生成贝塞尔曲线
    plane_bend_factor = 0.2  # 控制向交接平面弯曲的幅度
    num_samples = 9  # 每条曲线采样 9 个点

    # 生成均匀的 t 值
    t_values = np.linspace(0.0, 1.0, num_samples)  # [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    print(f"均匀的 t 值：{t_values}")

    # 直线 1：从 low_point_3d 到 boundary_point_1（左边，绿色）
    line_length_1 = np.linalg.norm(boundary_point_1 - low_point_3d)
    mid_point_1 = (low_point_3d + boundary_point_1) / 2  # 中点
    vec_to_mid_1 = mid_point_1 - intersection_plane_center
    distance_to_plane_1 = np.dot(vec_to_mid_1, plane_normal)
    plane_direction_1 = plane_normal if distance_to_plane_1 < 0 else -plane_normal  # 向平面靠近
    plane_offset_1 = line_length_1 * plane_bend_factor
    control_point_1 = mid_point_1 + plane_offset_1 * plane_direction_1  # 控制点
    print(f"直线 1 控制点：{control_point_1}")

    # 生成贝塞尔曲线 1 的采样点
    curve_points_1 = []
    for t in t_values:
        t2 = t * t
        one_minus_t = 1 - t
        one_minus_t2 = one_minus_t * one_minus_t
        point = (one_minus_t2 * low_point_3d +
                 2 * one_minus_t * t * control_point_1 +
                 t2 * boundary_point_1)
        curve_points_1.append(point)
    curve_points_1 = np.array(curve_points_1)

    # 直线 2：从 low_point_3d 到 boundary_point_2（右边，红色）
    line_length_2 = np.linalg.norm(boundary_point_2 - low_point_3d)
    mid_point_2 = (low_point_3d + boundary_point_2) / 2  # 中点
    vec_to_mid_2 = mid_point_2 - intersection_plane_center
    distance_to_plane_2 = np.dot(vec_to_mid_2, plane_normal)
    plane_direction_2 = plane_normal if distance_to_plane_2 < 0 else -plane_normal  # 向平面靠近
    plane_offset_2 = line_length_2 * plane_bend_factor
    control_point_2 = mid_point_2 + plane_offset_2 * plane_direction_2  # 控制点
    print(f"直线 2 控制点：{control_point_2}")

    # 生成贝塞尔曲线 2 的采样点
    curve_points_2 = []
    for t in t_values:
        t2 = t * t
        one_minus_t = 1 - t
        one_minus_t2 = one_minus_t * one_minus_t
        point = (one_minus_t2 * low_point_3d +
                 2 * one_minus_t * t * control_point_2 +
                 t2 * boundary_point_2)
        curve_points_2.append(point)
    curve_points_2 = np.array(curve_points_2)

    # 为采样点分配颜色（曲线 1：绿色，曲线 2：红色）
    colors_1 = np.array([[0, 255, 0]] * num_samples)  # 绿色 [0, 255, 0]，(9, 3)
    colors_2 = np.array([[255, 0, 0]] * num_samples)  # 红色 [255, 0, 0]，(9, 3)

    # 打印原始采样点坐标及其颜色
    print("曲线 1 采样点（从最低点到交界点 1，绿色）：")
    for i, (point, color) in enumerate(zip(curve_points_1, colors_1)):
        print(f"  点 {i+1}: {point}, RGB: {color}")
    print("曲线 2 采样点（从最低点到交界点 2，红色）：")
    for i, (point, color) in enumerate(zip(curve_points_2, colors_2)):
        print(f"  点 {i+1}: {point}, RGB: {color}")

    # 合并采样点，去重最低点（保留曲线 1 的 low_point_3d，绿色）
    all_sampled_points = np.vstack((curve_points_1, curve_points_2[1:]))  # (9 + 8 = 17, 3)
    all_colors = np.vstack((colors_1, colors_2[1:]))  # (17, 3)

    # 计算交接平面与短轴平面的交线方向
    intersection_line_direction = np.cross(plane_normal, short_plane_normal)
    intersection_line_direction = intersection_line_direction / np.linalg.norm(intersection_line_direction)  # 归一化
    print(f"交线方向：{intersection_line_direction}")

    # 按交线方向从左到右排序
    short_plane_center = short_plane.center
    projections = np.dot(all_sampled_points - short_plane_center, intersection_line_direction)
    sort_indices = np.argsort(projections)
    all_sampled_points = all_sampled_points[sort_indices]
    all_colors = all_colors[sort_indices]

    # 打印排序后的输出点
    print("排序后的输出点（按交线方向从左到右）：")
    for i, (point, color) in enumerate(zip(all_sampled_points, all_colors)):
        print(f"  点 {i+1}: {point}, RGB: {color}")

    # 显示贝塞尔曲线（使用 Spline 绘制平滑曲线）
    curve_spline_1 = pv.Spline(curve_points_1, n_points=50)  # 增加点数以平滑
    plotter.add_mesh(curve_spline_1, color='red', line_width=3)  # 红色，线宽 3
    curve_spline_2 = pv.Spline(curve_points_2, n_points=50)  # 增加点数以平滑
    plotter.add_mesh(curve_spline_2, color='red', line_width=3)  # 红色，线宽 3

    # 显示采样点（曲线 1：绿色小球，曲线 2：红色小球）
    for point in curve_points_1:
        sample_sphere = pv.Sphere(radius=3, center=point)  # 半径 0.5
        plotter.add_mesh(sample_sphere, color='black')  # 绿色
    for point in curve_points_2:
        sample_sphere = pv.Sphere(radius=3, center=point)  # 半径 0.5
        plotter.add_mesh(sample_sphere, color='black')  # 红色

    # 保存采样点为带 RGB 的 TXT 点云格式
    output_dir = f"{save_dir}/{patient}/"
    output_file = os.path.join(output_dir, f"sampled_points_{patient}.txt")
    try:
        os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
        # 合并坐标和颜色
        output_data = np.hstack((all_sampled_points, all_colors))  # (17, 6)
        # 保存为 TXT，坐标为浮点数，RGB 为整数
        np.savetxt(output_file, output_data, fmt=['%.6f', '%.6f', '%.6f', '%d', '%d', '%d'], delimiter=' ')
        print(f"带 RGB 的排序后采样点（去重最低点）已保存到：{output_file}")
    except Exception as e:
        print(f"保存采样点失败：{e}")
    if visual:
        bounds = pv.PolyData(np.vstack([left_points, right_points])).bounds
        center = np.array([(bounds[0] + bounds[1]) / 2,
                           (bounds[2] + bounds[3]) / 2,
                           (bounds[4] + bounds[5]) / 2])

        # 计算一个合适的观察距离
        size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        eye = center + np.array([0, 0, 3.0 * size])  # 站在 -Z 方向
        up = [0, 1, 0]  # Y 轴向上

        plotter.camera_position = [eye, center, up]

        plotter.add_axes(line_width=2)  # 添加坐标轴
        # plotter.set_background('#f9f9f9')
        plotter.set_background('white')
        # plotter.show_bounds(grid=False, location='outer', color='black' )  # 添加边界框

        plotter.show()  # 显示窗口

def CTA_main(data_dir, p_name,re_LV=False, visual=True):
    """主函数，协调加载、投影、长轴、短轴、短轴平面、调整后交界点提取及可视化"""
    # 定义患者名称和点云路径
    patient = p_name
    if re_LV:
        left_sur_file = rf'LVtopoints/name/{p_name}/{p_name}_LV_transformed.txt'
    else:
        left_sur_file = rf"{data_dir}/{patient}/ijkcta.txt"  # 左心点云路径
    right_txt = rf"RVtopoints\name\{patient}\{patient}_RV_transformed.txt"  # 右心点云路径

    # 加载点云
    left_points = load_point_cloud(left_sur_file)
    right_points = load_point_cloud(right_txt)

    # 计算交接平面
    normal, center = compute_intersection_plane(left_points, right_points, 5)

    # 投影左心点到平面，并记录索引
    projected_points, indices = project_points_to_plane(left_points, normal, center)

    # 计算投影点云的长轴、短轴、短轴平面参数及调整后交界点，并获取原始 3D 最低点
    _, _, major_axis_points, _, minor_axis_points, low_point, low_point_3d, short_plane_center, short_plane_normal, boundary_point_1, boundary_point_2 = compute_major_axis(
        projected_points, left_points, indices, normal, center
    )

    # 创建交接平面
    all_points = np.vstack((left_points, right_points))
    intersection_plane = create_plane(normal, center, all_points)

    # 创建短轴平面
    short_plane = create_plane(short_plane_normal, short_plane_center, all_points)

    # 首先显示投影平面、长轴、短轴和投影最低点
    if visual:
        print("显示投影平面、长轴、短轴和投影最低点...")
        visualize_projected_points(projected_points, major_axis_points, minor_axis_points, low_point)

    # 随后显示完整 3D 场景，包括交接平面、短轴平面、放大的红色最低点、弯曲的贝塞尔曲线及带颜色的均匀采样点
    print("显示完整 3D 场景，包括交接平面、短轴平面、放大的红色最低点、弯曲的贝塞尔曲线及带颜色的均匀采样点（左边绿色，右边红色，输出按左到右排序，去重最低点）...")
    visualize_full_scene(left_points, right_points, intersection_plane, short_plane, low_point_3d, boundary_point_1, boundary_point_2, patient, normal, center, short_plane_normal,data_dir, visual)

if __name__ == "__main__":
    p_name = r'geze'
    CTA_main(p_name)  # 运行主函数