import os

import numpy as np
import pyvista as pv
from sklearn.cluster import DBSCAN
import scipy.io

# RANSAC 拟合圆，指定半径范围内包含点数最多，圆心在点云点上，圆周点数量必须为 0，圆周点范围可自定义
def ransac_circle(xy_points, num_iterations=1000, threshold_lower=0.01, threshold_upper=0.01, radius_range=(0, 0.7)):
    best_circle = None
    max_points_inside = 0  # 记录圆内最大点数
    max_radius = 0  # 记录最大半径
    n_points = len(xy_points)
    r_min, r_max = radius_range

    for _ in range(num_iterations):
        # 随机选择一个点作为圆心
        center_idx = np.random.choice(n_points)
        center = xy_points[center_idx]  # 圆心必须是点云中的点
        xc, yc = center[0], center[1]

        # 随机选择一个半径，在指定范围内
        r = np.random.uniform(r_min, r_max)

        # 计算所有点到圆心的距离
        distances = np.linalg.norm(xy_points - [xc, yc], axis=1)

        # 识别圆内点（距离 <= 半径）
        points_inside = distances <= r
        points_inside_count = np.sum(points_inside)

        # 识别圆周点（距离在 [r - threshold_lower, r + threshold_upper] 范围内）
        inliers = (distances >= r - threshold_lower) & (distances <= r + threshold_upper)

        # 约束：圆周点数量必须为 0
        if np.sum(inliers) != 0:
            continue  # 如果圆周点数量不为 0，跳过当前圆

        # 优先级：1. 圆内点数量最多；2. 圆内点数量相同时，半径最大
        if (points_inside_count > max_points_inside) or (points_inside_count == max_points_inside and r > max_radius):
            max_points_inside = points_inside_count
            max_radius = r
            best_circle = (xc, yc, r)

    if best_circle is None:
        raise ValueError("无法找到满足条件的圆（圆周点数量必须为 0）")

    return best_circle, np.array([])  # 返回空 inliers，确保圆周点数量为 0

# 1. 加载点云（按行判断：3 列为心肌点，6 列为室间沟点）
def load_point_cloud_from_txt(file_path):
    points_myocardium_list = []
    colors_myocardium_list = []
    points_groove_list = []
    colors_groove_list = []

    # 逐行读取文件
    with open(file_path, 'r') as file:
        for line in file:
            # 按空格分割，移除空字符串
            values = line.strip().split()
            # 转换为浮点数
            values = [float(v) for v in values]

            # 根据列数判断
            if len(values) == 3:  # 心肌点：x, y, z
                points_myocardium_list.append(values)
                colors_myocardium_list.append([1, 1, 1])  # 默认颜色：白色
            elif len(values) == 6 and values[3]==0:  # 心肌点：x, y, z
                points_myocardium_list.append(values[:3])
                colors_myocardium_list.append([1, 1, 1])  # 默认颜色：白色
            elif len(values) == 6:  # 室间沟点：x, y, z, r, g, b
                points_groove_list.append(values[:3])
                colors_groove_list.append(values[3:6])
            else:
                raise ValueError(f"行数据格式错误：{line}，必须有 3 列或 6 列")

    # 转换为 NumPy 数组
    points_myocardium = np.array(points_myocardium_list)
    colors_myocardium = np.array(colors_myocardium_list)  # 白色 [1, 1, 1]
    points_groove = np.array(points_groove_list) if points_groove_list else np.array([])
    colors_groove = np.array(colors_groove_list) if colors_groove_list else np.array([])

    # 如果有室间沟点，将 RGB 从 0-255 转换为 0-1（PyVista 要求）
    if colors_groove.size > 0:
        colors_groove = colors_groove / 255.0

    return points_myocardium, colors_myocardium, points_groove, colors_groove

# 2. 预处理点云：去噪（仅对心肌点）
def preprocess_point_cloud(points):
    # 统计滤波去噪（移除离群点）
    k = 20
    dists = []
    for i in range(len(points)):
        diffs = points - points[i]
        distances = np.linalg.norm(diffs, axis=1)
        distances = np.sort(distances)[1:k + 1]  # 排除自身，最近 k 个点
        dists.append(np.mean(distances))
    dists = np.array(dists)

    # 移除距离异常的点（均值 ± 2 倍标准偏差）
    mean_dist, std_dist = np.mean(dists), np.std(dists)
    inliers = (dists >= mean_dist - 2 * std_dist) & (dists <= mean_dist + 2 * std_dist)
    return points[inliers], inliers

# 3. 可视化 XY 平面投影
def visualize_xy_projection(unique_xy, xc, yc, r, inliers):
    # 将 2D 点扩展为 3D（Z 设为 0），以便 PyVista 可视化
    xy_points_3d = np.zeros((len(unique_xy), 3))
    xy_points_3d[:, :2] = unique_xy  # X, Y 坐标
    xy_points_3d[:, 2] = 0  # Z 坐标设为 0

    # 创建点云对象
    cloud = pv.PolyData(xy_points_3d)

    # 创建绘图对象
    plotter = pv.Plotter()
    plotter.add_points(cloud, color='blue', point_size=5, render_points_as_spheres=True, label='Projected Points')

    # 绘制圆心
    center = np.array([xc, yc, 0])
    center_sphere = pv.Sphere(radius=0.5, center=center)
    plotter.add_mesh(center_sphere, color='red', label='Circle Center')

    # 绘制圆周（用圆盘近似）
    circle = pv.Circle(radius=r, resolution=100)
    circle.translate([xc, yc, 0])
    plotter.add_mesh(circle, color='cyan', style='wireframe', label='Fitted Circle')

    # 添加图例
    plotter.add_legend()

    # 设置视角（俯视 XY 平面）
    plotter.view_xy()
    plotter.show_grid()
    plotter.set_background('gray')
    plotter.show()

# 4. 计算 XY 平面的圆心（加入去重逻辑和指定半径范围选择）
def find_xy_center(points, visual=True):
    # 投影到 XY 平面
    xy_points = points[:, :2]  # 取 x, y 坐标

    # 去重：如果 XY 值相同，只保留一个
    unique_xy = np.unique(xy_points, axis=0)
    print(f"投影到 XY 平面后点数: {len(xy_points)}")
    print(f"去重后点数: {len(unique_xy)}")

    # 使用 RANSAC 拟合圆，选择指定半径范围内包含点数最多的圆，且圆周点数量为 0
    (xc, yc, r), inliers = ransac_circle(unique_xy, num_iterations=10000, threshold_lower=0.01, threshold_upper=0.01, radius_range=(0, 0.7))

    # 重新计算圆内点数量（用于日志输出）
    distances = np.linalg.norm(unique_xy - [xc, yc], axis=1)
    points_inside = distances <= r
    points_inside_count = np.sum(points_inside)

    # 计算圆周点数量（用于验证）
    inliers = (distances >= r - 0.01) & (distances <= r + 0.01)

    print(f"XY 平面最佳圆心: (x: {xc:.2f}, y: {yc:.2f}), 半径: {r:.2f}")
    print(f"圆内点数量: {points_inside_count}")
    print(f"圆周点范围: [{r - 0.01:.4f}, {r + 0.01:.4f}]")
    print(f"圆周点数量: {np.sum(inliers)}（必须为 0）")
    if visual:

        # 单独可视化 XY 平面投影
        visualize_xy_projection(unique_xy, xc, yc, r, inliers)

    return xc, yc, r

# 5. 识别心尖（Apex）
def find_apex(points, xc, yc):
    # 心尖的 XY 坐标为最佳圆心
    # 找到 Z 轴最小的点，获取 Z 值
    z_values = points[:, 2]
    z_min = np.min(z_values)

    # 心尖坐标
    apex = np.array([xc, yc, z_min])

    # 使用 DBSCAN 聚类，确认 Z 最小值区域（避免噪声点）
    z_min_threshold = np.percentile(z_values, 5)  # 取 Z 轴前 5% 的点
    apex_points = points[z_values <= z_min_threshold]
    clustering = DBSCAN(eps=1.0, min_samples=5).fit(apex_points)
    labels = clustering.labels_
    apex_cluster_points = apex_points[labels != -1]  # 排除噪声点
    if len(apex_cluster_points) > 0:
        z_min = np.min(apex_cluster_points[:, 2])
        apex = np.array([xc, yc, z_min])
    return apex

# 6. 识别基底（Base）
def find_base(points, xc, yc):
    # 基底的 XY 坐标为最佳圆心
    # 找到 Z 轴最大的点，获取 Z 值
    z_values = points[:, 2]
    z_max = np.max(z_values)

    # 基底坐标
    base_center = np.array([xc, yc, z_max])

    # 拟合平面（最小二乘法，仅用于验证）
    z_max_threshold = np.percentile(z_values, 95)  # 取 Z 轴后 5% 的点
    base_points = points[z_values >= z_max_threshold]
    A = np.c_[base_points[:, 0], base_points[:, 1], np.ones(len(base_points))]
    C = base_points[:, 2]
    plane_model, _, _, _ = np.linalg.lstsq(A, C, rcond=None)
    return base_center, plane_model

# 7. 计算水平长轴（Horizontal Long Axis）
def compute_long_axis(apex, base_center):
    long_axis = base_center - apex  # 从心尖指向基底
    long_axis = long_axis / np.linalg.norm(long_axis)  # 归一化
    return long_axis

# 8. 计算短轴（Short Axis）切面
def compute_short_axes(apex, base_center, long_axis, num_slices=5):
    # 沿长轴从心尖到基底均匀切分
    distance = np.linalg.norm(base_center - apex)
    slice_positions = np.linspace(0, distance, num_slices + 2)[1:-1]  # 忽略首尾
    short_axes = []

    for pos in slice_positions:
        # 切面中心点
        center = apex + pos * long_axis
        # 短轴平面法向量为长轴方向
        short_axes.append((center, long_axis))
    return short_axes

def compute_contracted_planes(points, apex, base_center, long_axis):
    # 计算长轴中心
    long_axis_center = (apex + base_center) / 2

    # 计算基向量（选择 base1 作为初始法向量）
    if np.abs(long_axis[0]) < 0.9:  # 如果 long_axis 不太接近 X 轴
        base1 = np.cross(long_axis, np.array([1, 0, 0]))
    else:
        base1 = np.cross(long_axis, np.array([0, 1, 0]))
    base1 = base1 / np.linalg.norm(base1)  # 归一化
    projections = np.dot(points - long_axis_center, base1)
    max_projection = np.max(projections)
    # 计算基向量到长轴的垂直距离作为半径
    # 投影 base1 到长轴的垂直分量
    long_axis_norm = long_axis / np.linalg.norm(long_axis)
    parallel_component = np.dot(base1, long_axis_norm) * long_axis_norm
    perpendicular_component = base1 - parallel_component
    radius = np.linalg.norm(perpendicular_component)*4
    radius = max_projection*0.60
    if radius < 1e-6:
        radius = 5.0  # 默认半径，避免为 0

    # 生成旋转角度（-90° 到 90°，步长 5°）
    angles_deg = np.arange(0, 360, 5)  # 包括 90°，总共 37 个角度
    angles_rad = np.deg2rad(angles_deg)

    # 收缩参数
    distance_threshold = 0.8  # 覆盖点的距离阈值
    best_plane = None
    best_plane_points = None
    max_points_covered = 0
    best_angle = 0

    # 保存所有平面数据
    plane_centers = []
    plane_normals = []
    plane_points_list = []
    all_angles_deg = []

    # 对每个旋转角度生成平面
    for angle_rad, angle_deg in zip(angles_rad, angles_deg):
        # 绕长轴旋转基向量（使用罗德里格斯旋转公式）
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        k = long_axis_norm  # 旋转轴（长轴）
        K = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ])
        I = np.eye(3)
        rotation_matrix = I + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
        rotated_normal = np.dot(rotation_matrix, base1)
        rotated_normal = rotated_normal / np.linalg.norm(rotated_normal)  # 归一化

        # 计算平面中心（沿旋转方向偏移）
        plane_center = long_axis_center + radius * rotated_normal

        # 计算点到平面的距离
        plane_vec = points - plane_center
        distances_to_plane = np.abs(np.dot(plane_vec, rotated_normal))

        # 计算覆盖点数量（距离小于阈值）
        points_covered = np.sum(distances_to_plane <= distance_threshold)

        # 保存当前平面数据
        plane_centers.append(plane_center)
        plane_normals.append(rotated_normal)
        plane_points_mask = distances_to_plane <= distance_threshold
        plane_points_list.append(points[plane_points_mask])
        all_angles_deg.append(angle_deg)

        # 更新最佳平面（覆盖点数量最多）
        if points_covered > max_points_covered:
            max_points_covered = points_covered
            best_plane = (plane_center, rotated_normal)
            best_angle = angle_deg
            best_plane_points = points[plane_points_mask]

    if best_plane is None:
        raise ValueError("无法找到满足条件的收缩平面")

    plane_center, normal = best_plane
    print(f"最佳收缩平面 - 覆盖点数量: {max_points_covered}, 平面中心: {plane_center}")
    print(f"最佳法向量: {normal}, 旋转角度: {best_angle:.1f}°")

    # 返回最佳平面和所有平面数据
    return best_plane, best_plane_points, all_angles_deg, plane_centers, plane_normals, plane_points_list

def compute_contracted_plane(points, apex, base_center, long_axis):
    # 计算点云中心（用于收缩目标）
    cloud_center = (apex + base_center) / 2

    # 计算两个垂直于 long_axis 的基向量（通过叉积）
    if np.abs(long_axis[0]) < 0.9:  # 如果 long_axis 不太接近 X 轴
        base1 = np.cross(long_axis, np.array([1, 0, 0]))
    else:
        base1 = np.cross(long_axis, np.array([0, 1, 0]))
    base1 = base1 / np.linalg.norm(base1)  # 归一化
    base2 = np.cross(long_axis, base1)
    base2 = base2 / np.linalg.norm(base2)  # 归一化

    # 构造四个初始法向量（垂直于 long_axis）
    normals = [
        base1,   # 正向 base1
        -base1,  # 反向 base1
        base2,   # 正向 base2
        -base2   # 反向 base2
    ]

    # 收缩参数
    step_size = 0.5  # 收缩步长
    distance_threshold = 0.8  # 覆盖点的距离阈值
    best_plane = None
    best_plane_points = None
    max_points_covered = 0
    best_angle = 0  # 记录最佳旋转角度

    # 使用记住的参数：旋转角度范围
    angles_deg = np.arange(-20, 20, 5)  # 从 -10° 到 0°，步长 1°
    angles_rad = np.deg2rad(angles_deg)

    # 保存所有平面数据
    plane_centers = []
    plane_normals = []
    plane_points_list = []
    all_angles_deg = []  # 存储所有平面的角度

    # 对每个初始法向量进行收缩和旋转
    for normal in normals:
        # 计算点云在该法向量方向上的投影，确定边界
        projections = np.dot(points - cloud_center, normal)
        max_projection = np.max(projections)

        # 使用记住的参数：收缩范围
        start_distance = max_projection * 0.61  # 从 61% 开始
        end_distance = max_projection * 0.6     # 到 60% 结束

        # 从 start_distance 向 end_distance 收缩
        distance = start_distance
        while distance >= end_distance:
            # 计算平面中心（沿法向量移动）
            plane_center = cloud_center + distance * normal

            # 对每个旋转角度
            for angle_rad, angle_deg in zip(angles_rad, angles_deg):
                # 绕长轴旋转法向量（使用罗德里格斯旋转公式）
                cos_theta = np.cos(angle_rad)
                sin_theta = np.sin(angle_rad)
                k = long_axis  # 旋转轴（长轴）
                # 反对称矩阵 K
                K = np.array([
                    [0, -k[2], k[1]],
                    [k[2], 0, -k[0]],
                    [-k[1], k[0], 0]
                ])
                # 旋转矩阵：I + sin(θ)K + (1 - cos(θ))K^2
                I = np.eye(3)
                rotation_matrix = I + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
                # 应用旋转
                rotated_normal = np.dot(rotation_matrix, normal)
                rotated_normal = rotated_normal / np.linalg.norm(rotated_normal)  # 归一化

                # 计算点到平面的距离
                plane_vec = points - plane_center
                distances_to_plane = np.abs(np.dot(plane_vec, rotated_normal))

                # 计算覆盖点数量（距离小于阈值）
                points_covered = np.sum(distances_to_plane <= distance_threshold)

                # 保存当前平面数据
                plane_centers.append(plane_center)
                plane_normals.append(rotated_normal)
                plane_points_mask = distances_to_plane <= distance_threshold
                plane_points_list.append(points[plane_points_mask])
                all_angles_deg.append(angle_deg)  # 记录当前角度

                # 更新最佳平面（覆盖点数量最多）
                if points_covered > max_points_covered:
                    max_points_covered = points_covered
                    best_plane = (plane_center, rotated_normal)
                    best_angle = angle_deg
                    best_plane_points = points[plane_points_mask]

            distance -= step_size

    if best_plane is None:
        raise ValueError("无法找到满足条件的收缩平面")

    plane_center, normal = best_plane
    print(f"最佳收缩平面 - 覆盖点数量: {max_points_covered}, 平面中心: {plane_center}")
    print(f"最佳法向量: {normal}, 绕长轴旋转角度: {best_angle:.2f} 度")

    # 返回最佳平面和所有平面数据
    return best_plane, best_plane_points, all_angles_deg, plane_centers, plane_normals, plane_points_list

# 9. 计算最佳收缩平面（锁定收缩范围，绕长轴旋转）
def compute_contracted_plane(points, apex, base_center, long_axis):
    # 计算点云中心（用于收缩目标）
    cloud_center = (apex + base_center) / 2

    # 计算两个垂直于 long_axis 的基向量（通过叉积）
    if np.abs(long_axis[0]) < 0.9:  # 如果 long_axis 不太接近 X 轴
        base1 = np.cross(long_axis, np.array([1, 0, 0]))
    else:
        base1 = np.cross(long_axis, np.array([0, 1, 0]))
    base1 = base1 / np.linalg.norm(base1)  # 归一化
    base2 = np.cross(long_axis, base1)
    base2 = base2 / np.linalg.norm(base2)  # 归一化

    # 构造四个初始法向量（垂直于 long_axis）
    normals = [
        base1,   # 正向 base1
        -base1,  # 反向 base1
        base2,   # 正向 base2
        -base2   # 反向 base2
    ]

    # 收缩参数
    step_size = 0.5  # 收缩步长
    distance_threshold = 0.8  # 覆盖点的距离阈值
    best_plane = None
    best_plane_points = None
    max_points_covered = 0
    best_angle = 0  # 记录最佳旋转角度

    # 使用记住的参数：旋转角度范围
    angles_deg = np.arange(-10, 0, 1)  # 从 -10° 到 0°，步长 1°
    angles_rad = np.deg2rad(angles_deg)

    # 对每个初始法向量进行收缩和旋转
    for normal in normals:
        # 计算点云在该法向量方向上的投影，确定边界
        projections = np.dot(points - cloud_center, normal)
        max_projection = np.max(projections)

        # 使用记住的参数：收缩范围
        start_distance = max_projection * 0.91  # 从 61% 开始
        end_distance = max_projection * 0.6     # 到 60% 结束

        # 从 start_distance 向 end_distance 收缩
        distance = start_distance
        while distance >= end_distance:
            # 计算平面中心（沿法向量移动）
            plane_center = cloud_center + distance * normal

            # 对每个旋转角度
            for angle_rad in angles_rad:
                # 绕长轴旋转法向量（使用罗德里格斯旋转公式）
                cos_theta = np.cos(angle_rad)
                sin_theta = np.sin(angle_rad)
                k = long_axis  # 旋转轴（长轴）
                # 反对称矩阵 K
                K = np.array([
                    [0, -k[2], k[1]],
                    [k[2], 0, -k[0]],
                    [-k[1], k[0], 0]
                ])
                # 旋转矩阵：I + sin(θ)K + (1 - cos(θ))K^2
                I = np.eye(3)
                rotation_matrix = I + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
                # 应用旋转
                rotated_normal = np.dot(rotation_matrix, normal)
                rotated_normal = rotated_normal / np.linalg.norm(rotated_normal)  # 归一化

                # 计算点到平面的距离
                plane_vec = points - plane_center
                distances_to_plane = np.abs(np.dot(plane_vec, rotated_normal))

                # 计算覆盖点数量（距离小于阈值）
                points_covered = np.sum(distances_to_plane <= distance_threshold)

                # 更新最佳平面（覆盖点数量最多）
                if points_covered > max_points_covered:
                    max_points_covered = points_covered
                    best_plane = (plane_center, rotated_normal)
                    best_angle = np.degrees(angle_rad)
                    # 提取平面上的点
                    plane_points_mask = distances_to_plane <= distance_threshold
                    best_plane_points = points[plane_points_mask]

            distance -= step_size

    if best_plane is None:
        raise ValueError("无法找到满足条件的收缩平面")

    plane_center, normal = best_plane
    print(f"最佳收缩平面 - 覆盖点数量: {max_points_covered}, 平面中心: {plane_center}")
    print(f"最佳法向量: {normal}, 绕长轴旋转角度: {best_angle:.2f} 度")
    return best_plane, best_plane_points

# 10. 提取最佳收缩平面上的外圈点（仅用于计算，不显示）
def extract_outer_edge_points(plane_points, plane_center, plane_normal):
    if len(plane_points) == 0:
        return np.array([])

    # 构建平面坐标系的基向量
    if np.abs(plane_normal[0]) < 0.9:
        base1 = np.cross(plane_normal, np.array([1, 0, 0]))
    else:
        base1 = np.cross(plane_normal, np.array([0, 1, 0]))
    base1 = base1 / np.linalg.norm(base1)
    base2 = np.cross(plane_normal, base1)
    base2 = base2 / np.linalg.norm(base2)

    # 将点投影到平面坐标系
    vec_to_points = plane_points - plane_center
    coords_u = np.dot(vec_to_points, base1)
    coords_v = np.dot(vec_to_points, base2)

    # 计算每个点在平面上的距离（到平面中心的距离）
    distances = np.sqrt(coords_u**2 + coords_v**2)

    # 选择外圈点：距离最大的前 30%
    if len(distances) == 0:
        return np.array([])
    percentile_threshold = np.percentile(distances, 70)  # 取前 30% 最远的点
    outer_mask = distances >= percentile_threshold
    outer_points = plane_points[outer_mask]

    return outer_points

# 11. 计算两个平面的交线（返回交线方向和一个点）
def compute_plane_intersection(plane1_center, plane1_normal, plane2_center, plane2_normal):
    # 交线方向：两个法向量的叉积
    line_direction = np.cross(plane1_normal, plane2_normal)
    if np.linalg.norm(line_direction) < 1e-6:  # 平面平行
        raise ValueError("短轴平面和最佳收缩平面平行，无法计算交线")

    line_direction = line_direction / np.linalg.norm(line_direction)

    # 解线性方程组，找到交线上的一点
    # 平面方程：n1·(x - p1) = 0 和 n2·(x - p2) = 0
    # 设 x = p + t * d，求 p（一个与 d 垂直的点）
    A = np.array([plane1_normal, plane2_normal])
    b = np.array([np.dot(plane1_normal, plane1_center), np.dot(plane2_normal, plane2_center)])
    # 增加约束：p 与 line_direction 垂直
    A = np.vstack([A, line_direction])
    b = np.append(b, 0)

    # 解线性方程组
    p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return p, line_direction

# 12. 提取室间隔点（生成对称且向外弯曲的 V 字形，并向最佳收缩平面弯曲，同时采样新点）
def extract_septum_points(points_myocardium, apex, best_plane_center, best_plane_normal, long_axis, short_axes, sampling_method='random'):
    # Step 1: 提取内表面点（仅用于确定交线范围）
    vec_to_points = points_myocardium - apex
    projections = np.dot(vec_to_points, long_axis)
    points_on_axis = apex + np.outer(projections, long_axis)
    distances_to_axis = np.linalg.norm(points_myocardium - points_on_axis, axis=1)

    # 选择距离长轴最近的前 30% 点作为内表面点
    percentile_threshold = np.percentile(distances_to_axis, 30)  # 前 30%
    inner_mask = distances_to_axis <= percentile_threshold
    inner_points = points_myocardium[inner_mask]

    if len(inner_points) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([0, 0, 0])

    # Step 2: 计算心尖到最佳收缩平面的方向（作为抬升方向，仅用于验证）
    direction = best_plane_center - apex
    direction = direction / np.linalg.norm(direction)  # 归一化

    # Step 3: 获取最中间的短轴平面
    middle_index = len(short_axes) // 2
    short_axis_center, short_axis_normal = short_axes[middle_index]

    # Step 4: 计算短轴平面和最佳收缩平面的交线
    intersection_point, intersection_direction = compute_plane_intersection(
        short_axis_center, short_axis_normal, best_plane_center, best_plane_normal
    )

    # Step 5: 计算内表面点在交线上的投影范围
    vec_to_points = inner_points - intersection_point
    projections = np.dot(vec_to_points, intersection_direction)
    min_proj = np.min(projections)
    max_proj = np.max(projections)

    # 确定交线的两个端点
    left_end = intersection_point + min_proj * intersection_direction
    right_end = intersection_point + max_proj * intersection_direction

    # Step 6: 验证开口方向（仍平行于最佳收缩平面）
    # base2 沿交线方向（intersection_direction），它天然位于最佳收缩平面内
    base2 = intersection_direction
    base2 = base2 / np.linalg.norm(base2)  # 归一化

    # 验证 base2 垂直于 best_plane_normal（即平行于最佳收缩平面）
    dot_product = np.dot(base2, best_plane_normal)
    if abs(dot_product) > 1e-6:
        print(f"Warning: base2 is not perfectly perpendicular to best_plane_normal (dot product: {dot_product})")

    # Step 7: 计算对称轴（从 apex 到交线中点）
    mid_point = (left_end + right_end) / 2
    symmetry_axis = mid_point - apex
    symmetry_axis_norm = np.linalg.norm(symmetry_axis)
    if symmetry_axis_norm > 1e-6:
        symmetry_axis = symmetry_axis / symmetry_axis_norm  # 归一化
    else:
        symmetry_axis = np.zeros(3)  # 避免除以 0

    # 计算从 apex 到 left_end 和 right_end 的方向和距离
    left_direction = left_end - apex
    right_direction = right_end - apex
    left_distance = np.linalg.norm(left_direction)
    right_distance = np.linalg.norm(right_direction)
    if left_distance > 1e-6:
        left_direction = left_direction / left_distance  # 归一化
    else:
        left_direction = np.zeros(3)  # 避免除以 0
    if right_distance > 1e-6:
        right_direction = right_direction / right_distance  # 归一化
    else:
        right_direction = np.zeros(3)  # 避免除以 0

    # 计算中点
    left_mid_point = (apex + left_end) / 2
    right_mid_point = (apex + right_end) / 2

    # 计算偏移方向（对称且向外）
    # 偏移方向垂直于 left_direction 和 symmetry_axis
    left_perp = np.cross(left_direction, symmetry_axis)
    left_perp_norm = np.linalg.norm(left_perp)
    if left_perp_norm > 1e-6:
        left_perp = left_perp / left_perp_norm  # 归一化
    else:
        left_perp = np.zeros(3)  # 避免除以 0

    # 确保 right_perp 是 left_perp 的镜像（对称）
    left_vec = left_end - apex
    right_vec = right_end - apex
    left_proj = np.dot(left_vec, symmetry_axis)
    right_proj = np.dot(right_vec, symmetry_axis)
    left_perp_vec = left_vec - left_proj * symmetry_axis
    right_perp_vec = right_vec - right_proj * symmetry_axis

    # 确定向外方向
    left_perp = left_perp_vec / np.linalg.norm(left_perp_vec) if np.linalg.norm(left_perp_vec) > 1e-6 else np.zeros(3)
    right_perp = right_perp_vec / np.linalg.norm(right_perp_vec) if np.linalg.norm(right_perp_vec) > 1e-6 else np.zeros(3)

    # 确保对称：调整 right_perp 使之与 left_perp 镜像
    dot_product_perp = np.dot(left_perp, right_perp)
    if dot_product_perp > 0:  # 如果方向相同，说明不对称
        right_perp = -right_perp  # 反转方向

    # 向外扩张端点
    expand_factor = 0.1  # 扩张 10%
    left_expand_distance = left_distance * expand_factor
    right_expand_distance = right_distance * expand_factor
    left_end = left_end + left_expand_distance * left_perp
    right_end = right_end + right_expand_distance * right_perp

    # 重新计算中点（因为端点变了）
    left_mid_point = (apex + left_end) / 2
    right_mid_point = (apex + right_end) / 2

    # 偏移量（控制向外弯曲弧度），对称使用相同偏移量
    bend_factor = 0.2  # 可调整，值越大向外弯曲越明显
    # 使用更新后的距离计算偏移量
    left_distance = np.linalg.norm(left_end - apex)
    right_distance = np.linalg.norm(right_end - apex)
    offset = max(left_distance, right_distance) * bend_factor  # 使用最大距离确保一致

    # 控制点位置（向外弯曲）
    left_control_point = left_mid_point + offset * left_perp
    right_control_point = right_mid_point + offset * right_perp

    # Step 8: 向最佳收缩平面弯曲
    plane_bend_factor = 0.1  # 控制向平面弯曲的幅度（10%）
    plane_offset = max(left_distance, right_distance) * plane_bend_factor

    # 左边控制点
    vec_to_left_control = left_control_point - best_plane_center
    left_distance_to_plane = np.dot(vec_to_left_control, best_plane_normal)
    left_plane_direction = best_plane_normal if left_distance_to_plane < 0 else -best_plane_normal
    left_control_point += plane_offset * left_plane_direction

    # 右边控制点
    vec_to_right_control = right_control_point - best_plane_center
    right_distance_to_plane = np.dot(vec_to_right_control, best_plane_normal)
    right_plane_direction = best_plane_normal if right_distance_to_plane < 0 else -best_plane_normal
    right_control_point += plane_offset * right_plane_direction

    # Step 9: 使用二次贝塞尔曲线生成点
    # 二次贝塞尔曲线公式：B(t) = (1-t)^2 * P0 + 2 * (1-t) * t * P1 + t^2 * P2
    num_points = int(max(left_distance, right_distance) / 0.3) + 1  # 保持与之前步长一致
    t_values = np.linspace(0, 1, num_points)

    left_points_list = []
    right_points_list = []

    for t in t_values:
        # 左边曲线
        t2 = t * t
        one_minus_t = 1 - t
        one_minus_t2 = one_minus_t * one_minus_t
        left_point = (one_minus_t2 * apex +
                      2 * one_minus_t * t * left_control_point +
                      t2 * left_end)
        left_points_list.append(left_point)

        # 右边曲线
        right_point = (one_minus_t2 * apex +
                       2 * one_minus_t * t * right_control_point +
                       t2 * right_end)
        right_points_list.append(right_point)

    # Step 10: 将最后一个点替换为交线的两端（确保精确）
    left_points_list[-1] = left_end
    right_points_list[-1] = right_end

    # Step 11: 转换为 NumPy 数组
    left_points = np.array(left_points_list)
    right_points = np.array(right_points_list)

    # Step 12: 在每条曲线上采样 9 个点（根据 sampling_method 选择采样方式）
    num_samples = 9
    left_sampled_points = []
    right_sampled_points = []

    if sampling_method == 'uniform':
        # 均匀采样：直接取 9 个均匀的 t 值
        t_values = np.linspace(0, 1, num_samples)
        for t in t_values:
            # 左边曲线采样点
            t2 = t * t
            one_minus_t = 1 - t
            one_minus_t2 = one_minus_t * one_minus_t
            left_sampled_point = (one_minus_t2 * apex +
                                  2 * one_minus_t * t * left_control_point +
                                  t2 * left_end)
            left_sampled_points.append(left_sampled_point)

            # 右边曲线采样点
            right_sampled_point = (one_minus_t2 * apex +
                                   2 * one_minus_t * t * right_control_point +
                                   t2 * right_end)
            right_sampled_points.append(right_sampled_point)

    elif sampling_method == 'random':
        # 均匀随机采样：将 t 区间分为 9 个子区间，在每个子区间内随机采样
        t_intervals = np.linspace(0, 1, num_samples + 1)
        for i in range(num_samples):
            # 每个子区间内随机选择一个 t 值
            t_lower = t_intervals[i]
            t_upper = t_intervals[i + 1]
            t_left = np.random.uniform(t_lower, t_upper)
            t_right = np.random.uniform(t_lower, t_upper)

            # 左边曲线采样点
            t2 = t_left * t_left
            one_minus_t = 1 - t_left
            one_minus_t2 = one_minus_t * one_minus_t
            left_sampled_point = (one_minus_t2 * apex +
                                  2 * one_minus_t * t_left * left_control_point +
                                  t2 * left_end)
            left_sampled_points.append(left_sampled_point)

            # 右边曲线采样点
            t2 = t_right * t_right
            one_minus_t = 1 - t_right
            one_minus_t2 = one_minus_t * one_minus_t
            right_sampled_point = (one_minus_t2 * apex +
                                   2 * one_minus_t * t_right * right_control_point +
                                   t2 * right_end)
            right_sampled_points.append(right_sampled_point)
    else:
        raise ValueError("sampling_method 必须是 'uniform' 或 'random'")

    # 转换为 NumPy 数组
    left_sampled_points = np.array(left_sampled_points)
    right_sampled_points = np.array(right_sampled_points)

    return left_points, right_points, left_sampled_points, right_sampled_points, intersection_direction

# 13. 可视化结果（三维点云）并保存采样点
def visualize_results(manu_points, points_myocardium, colors_myocardium, points_groove, colors_groove, apex, base_center, long_axis, short_axes, best_plane, plane_points, sampling_method='random', output_file='sampled_points.txt', visual=True):
    # 计算 V 字形的室间隔点，并采样新点
    best_plane_center, best_plane_normal = best_plane
    left_points, right_points, left_sampled_points, right_sampled_points, intersection_direction = extract_septum_points(
        points_myocardium, apex, best_plane_center, best_plane_normal, long_axis, short_axes, sampling_method
    )
    print(f"V 字形左边点数量: {len(left_points)}, 右边点数量: {len(right_points)}")
    print(f"采样点 - 左边: {len(left_sampled_points)}, 右边: {len(right_sampled_points)}，采样方式: {sampling_method}")
    print(f"交线方向 (intersection_direction): {intersection_direction}")

    # 移除左边采样点的第一个点（apex），避免重复
    left_sampled_points = left_sampled_points[1:]  # 移除第一个点，剩余 8 个点

    # 合并采样点：右边 9 个点 + 左边 8 个点 = 17 个点
    sampled_points = np.vstack([right_sampled_points, left_sampled_points])
    # 右边 9 个点为绿色，左边 8 个点为红色
    right_colors = np.array([[0, 1, 0]] * len(right_sampled_points))  # 绿色
    left_colors = np.array([[1, 0, 0]] * len(left_sampled_points))  # 红色
    sampled_colors = np.vstack([right_colors, left_colors])

    # 确定排序轴（基于 intersection_direction 的主要分量）
    abs_direction = np.abs(intersection_direction)
    if abs_direction[0] > abs_direction[1]:  # X 分量较大，按 X 轴排序
        sort_axis = 0  # X 轴
        print("按 X 轴从大到小排序（从右往左）")
    else:  # Y 分量较大，按 Y 轴排序
        sort_axis = 1  # Y 轴
        print("按 Y 轴从大到小排序（从右往左）")

    # 按选定轴从大到小排序（从右往左）
    indices = np.argsort(-sampled_points[:, sort_axis])  # 负号表示从大到小
    sampled_points = sampled_points[indices]
    sampled_colors = sampled_colors[indices]

    # 保存采样点到文件（格式与输入点云一致：x, y, z, r, g, b）
    # 将颜色从 0-1 转换为 0-255
    sampled_colors_255 = (sampled_colors * 255).astype(int)  # [0, 1, 0] -> [0, 255, 0], [1, 0, 0] -> [255, 0, 0]
    # 合并坐标和颜色，生成 6 列数据
    sampled_data = np.hstack([sampled_points, sampled_colors_255])
    # 写入文件
    with open(output_file, 'w') as f:
        for point in sampled_data:
            # 每行：x y z r g b，空格分隔，浮点数保留 6 位小数
            line = f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(point[3])} {int(point[4])} {int(point[5])}\n"
            f.write(line)
    print(f"采样点已保存到文件: {output_file}，总点数: {len(sampled_points)}")

    # 将采样点添加到室间沟点中（用于可视化）
    if len(points_groove) > 0:
        points_groove = np.vstack([points_groove, sampled_points])
        colors_groove = np.vstack([colors_groove, sampled_colors])
    else:
        points_groove = sampled_points
        colors_groove = sampled_colors

    # 创建心肌点的 PyVista 点云对象
    cloud_myocardium = pv.PolyData(points_myocardium)
    cloud_myocardium['colors'] = colors_myocardium  # 心肌点默认白色

    cloud_manupoints = pv.PolyData(manu_points)

    # 创建室间沟点的 PyVista 点云对象（包括采样点）
    if len(points_groove) > 0:
        cloud_groove = pv.PolyData(points_groove)
        cloud_groove['colors'] = colors_groove  # 室间沟点使用其 RGB 颜色（包括红色和绿色采样点）
    else:
        cloud_groove = None

    if visual:

        # 创建绘图对象
        plotter = pv.Plotter()

        # 绘制心肌点（默认白色）
        plotter.add_points(cloud_myocardium, color='#d8654f', rgb=True, point_size=5, render_points_as_spheres=True, label='Heart Muscle')
        # plotter.add_points(cloud_manupoints, color='red', rgb=False, point_size=10, render_points_as_spheres=True)
        # 绘制室间沟点（包括采样点，按其 RGB 颜色显示）
        if cloud_groove is not None:
            plotter.add_points(cloud_groove, color='black', rgb=True, point_size=10, render_points_as_spheres=True, label='Interventricular Groove')

        # 绘制 V 字形线（左边）
        if len(left_points) > 1:  # 确保有足够点绘制线
            left_line = pv.lines_from_points(left_points)
            plotter.add_mesh(left_line, color='red', line_width=3, label='Interventricular Septum (Left)')

        # 绘制 V 字形线（右边）
        if len(right_points) > 1:  # 确保有足够点绘制线
            right_line = pv.lines_from_points(right_points)
            plotter.add_mesh(right_line, color='red', line_width=3, label='Interventricular Septum (Right)')

        # 绘制心尖
        # apex_sphere = pv.Sphere(radius=0.2, center=apex)
        # plotter.add_mesh(apex_sphere, color='red', label='Apex')

        # 绘制基底
        # base_sphere = pv.Sphere(radius=0.2, center=base_center)
        # plotter.add_mesh(base_sphere, color='green', label='Base')

        # 绘制水平长轴
        # line_points = np.array([apex, base_center])
        # line = pv.lines_from_points(line_points)
        # plotter.add_mesh(line, color='cyan', line_width=3, label='Long Axis')

        # 仅绘制长轴最中间的一个短轴平面
        # if short_axes:  # 确保 short_axes 不为空
        #     middle_index = len(short_axes) // 2
        #     center, normal = short_axes[middle_index]
        #     plane = pv.Plane(center=center, direction=normal, i_size=10, j_size=10)
        #     # plotter.add_mesh(plane, color='yellow', opacity=0.5, label='Short Axis')

        # 绘制最佳收缩平面
        contracted_plane = pv.Plane(center=best_plane_center, direction=best_plane_normal, i_size=25, j_size=25)
        plotter.add_mesh(contracted_plane, color='yellow', opacity=0.7, label='Best Contracted Plane',show_edges=True, edge_color='black')

        # 不显示外圈点（粉色点）

        # 添加图例
        # plotter.add_legend()

        # 设置视角和背景
        plotter.show_grid()
        # plotter.set_background('#f9f9f9')
        plotter.set_background('white')
        # plotter.show_bounds(grid=False, location='outer', color='black')
        plotter.show()


def visualize_all_contracted_plane(points, apex, base_center, long_axis, all_angles_deg, plane_centers, plane_normals, plane_points_list, step_by_step=False, current_index=1):
    # 计算点云中心
    cloud_center = (apex + base_center) / 2

    # 总平面数
    total_planes = len(plane_centers)
    if current_index > total_planes:
        current_index = total_planes

    # 逐步显示模式
    if step_by_step:
        for i in range(1, total_planes + 1):
            # 创建绘图对象
            plotter = pv.Plotter()
            plotter.set_background('gray')
            plotter.show_grid()

            # 绘制原始点云
            cloud = pv.PolyData(points)
            plotter.add_points(cloud, color='blue', point_size=2, render_points_as_spheres=True, label='Original Points')

            # 绘制从 1 到 current_index 的平面
            colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan']  # 循环使用颜色
            for j in range(i):
                center = plane_centers[j]
                normal = plane_normals[j]
                plane_points = plane_points_list[j]
                angle_deg = all_angles_deg[j]

                # 创建平面
                plane = pv.Plane(center=center, direction=normal, i_size=10, j_size=10)
                color = colors[j % len(colors)]  # 循环使用颜色
                plotter.add_mesh(plane, color=color, opacity=0.3, label=f'Plane Angle {angle_deg:.1f}°')

                # # 绘制平面中心
                # center_sphere = pv.Sphere(radius=0.5, center=center)
                # plotter.add_mesh(center_sphere, color=color, label=f'Center {angle_deg:.1f}°')

                # # 如果是新添加的平面，绘制其截取点
                # if j == i - 1 and len(plane_points) > 0:
                #     plane_cloud = pv.PolyData(plane_points)
                #     plotter.add_points(plane_cloud, color=color, point_size=5, render_points_as_spheres=True, label=f'New Points at {angle_deg:.1f}°')

            # 绘制心尖和基底
            apex_sphere = pv.Sphere(radius=0.5, center=apex)
            plotter.add_mesh(apex_sphere, color='red', label='Apex')
            base_sphere = pv.Sphere(radius=0.5, center=base_center)
            plotter.add_mesh(base_sphere, color='green', label='Base')

            # 绘制水平长轴
            line_points = np.array([apex, base_center])
            line = pv.lines_from_points(line_points)
            plotter.add_mesh(line, color='cyan', line_width=3, label='Long Axis')

            # 添加图例
            plotter.add_legend()

            # 设置视角
            plotter.view_isometric()

            # 显示当前窗口，用户关闭后进入下一循环
            plotter.show()
            if i == total_planes:
                break
    else:
        # 非逐步模式，显示所有平面
        plotter = pv.Plotter()
        plotter.set_background('gray')
        plotter.show_grid()

        # 绘制原始点云
        cloud = pv.PolyData(points)
        plotter.add_points(cloud, color='blue', point_size=2, render_points_as_spheres=True, label='Original Points')

        # 绘制所有平面
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan']  # 循环使用颜色
        for i, (center, normal, plane_points) in enumerate(zip(plane_centers, plane_normals, plane_points_list)):
            angle_deg = all_angles_deg[i]

            # 创建平面
            plane = pv.Plane(center=center, direction=normal, i_size=10, j_size=10)
            color = colors[i % len(colors)]  # 循环使用颜色
            plotter.add_mesh(plane, color=color, opacity=0.3, label=f'Plane Angle {angle_deg:.1f}°')

            # # 绘制平面上的点（覆盖点）
            # if len(plane_points) > 0:
            #     plane_cloud = pv.PolyData(plane_points)
            #     plotter.add_points(plane_cloud, color=color, point_size=5, render_points_as_spheres=True, label=f'Points at {angle_deg:.1f}°')

            # # 绘制平面中心
            # center_sphere = pv.Sphere(radius=0.5, center=center)
            # plotter.add_mesh(center_sphere, color=color, label=f'Center {angle_deg:.1f}°')

        # 绘制心尖和基底
        apex_sphere = pv.Sphere(radius=0.5, center=apex)
        plotter.add_mesh(apex_sphere, color='red', label='Apex')
        base_sphere = pv.Sphere(radius=0.5, center=base_center)
        plotter.add_mesh(base_sphere, color='green', label='Base')

        # 绘制水平长轴
        line_points = np.array([apex, base_center])
        line = pv.lines_from_points(line_points)
        plotter.add_mesh(line, color='cyan', line_width=3, label='Long Axis')

        # 添加图例
        plotter.add_legend()

        # 设置视角
        plotter.view_isometric()
        plotter.show()


def visualize_all_contracted_planes(points, apex, base_center, long_axis, all_angles_deg, plane_centers, plane_normals,
                                    plane_points_list, top_n=5):
    # 计算点云中心
    cloud_center = (apex + base_center) / 2

    # 创建绘图对象
    plotter = pv.Plotter()
    plotter.set_background('gray')
    plotter.show_grid()

    # 绘制原始点云
    cloud = pv.PolyData(points)
    plotter.add_points(cloud, color=[216 / 255.0, 101 / 255.0, 79 / 255.0], point_size=5, render_points_as_spheres=True, label='Original Points')

    # 计算每个平面的点数
    point_counts = [len(pts) for pts in plane_points_list]

    # 如果指定 top_n，排序并取前 n 个
    if top_n is not None and top_n < len(plane_centers):
        # 创建索引数组并按点数降序排序
        indices = sorted(range(len(point_counts)), key=lambda i: point_counts[i], reverse=True)
        selected_indices = indices[:top_n]
    else:
        selected_indices = range(len(plane_centers))  # 显示所有平面

    # 为每个选中的平面绘制
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan']  # 循环使用颜色
    for i in selected_indices:
        center = plane_centers[i]
        normal = plane_normals[i]
        plane_points = plane_points_list[i]
        angle_deg = all_angles_deg[i]

        # 创建平面
        plane = pv.Plane(center=center, direction=normal, i_size=10, j_size=10)
        color = colors[i % len(colors)]  # 循环使用颜色
        plotter.add_mesh(plane, color=color, opacity=0.8,
                         label=f'Plane Angle {angle_deg:.1f}° (Points: {point_counts[i]})')

        # # 绘制平面上的点（覆盖点）
        # if len(plane_points) > 0:
        #     plane_cloud = pv.PolyData(plane_points)
        #     plotter.add_points(plane_cloud, color=color, point_size=5, render_points_as_spheres=True,
        #                        label=f'Points at {angle_deg:.1f}°')

        # # 绘制平面中心
        # center_sphere = pv.Sphere(radius=0.5, center=center)
        # plotter.add_mesh(center_sphere, color=color, label=f'Center {angle_deg:.1f}°')


    # # 绘制心尖和基底
    # apex_sphere = pv.Sphere(radius=0.5, center=apex)
    # plotter.add_mesh(apex_sphere, color='red', label='Apex')
    # base_sphere = pv.Sphere(radius=0.5, center=base_center)
    # plotter.add_mesh(base_sphere, color='green', label='Base')

    # # 绘制水平长轴
    # line_points = np.array([apex, base_center])
    # line = pv.lines_from_points(line_points)
    # plotter.add_mesh(line, color='cyan', line_width=3, label='Long Axis')

    # 添加图例
    # plotter.add_legend()
    plotter.set_background('#f9f9f9')
    plotter.show_bounds(grid=False, location='outer', color='black')

    # 设置视角
    plotter.view_isometric()
    plotter.show()

def visualize_all_contracted_plane(points, apex, base_center, long_axis, all_angles_deg, plane_centers, plane_normals, plane_points_list):
    # 计算点云中心
    cloud_center = (apex + base_center) / 2

    # 创建绘图对象
    plotter = pv.Plotter()
    plotter.set_background('gray')
    plotter.show_grid()

    # 绘制原始点云
    cloud = pv.PolyData(points)
    plotter.add_points(cloud, color='blue', point_size=2, render_points_as_spheres=True, label='Original Points')

    # 为每个旋转角度绘制平面
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan']  # 循环使用颜色
    for i, (center, normal, plane_points) in enumerate(zip(plane_centers, plane_normals, plane_points_list)):
        # 创建平面
        plane = pv.Plane(center=center, direction=normal, i_size=10, j_size=10)
        color = colors[i % len(colors)]  # 循环使用颜色
        plotter.add_mesh(plane, color=color, opacity=0.3, label=f'Plane Angle {all_angles_deg[i]:.1f}°')

        # 绘制平面上的点（覆盖点）
        if len(plane_points) > 0:
            plane_cloud = pv.PolyData(plane_points)
            plotter.add_points(plane_cloud, color=color, point_size=5, render_points_as_spheres=True, label=f'Points at {all_angles_deg[i]:.1f}°')

        # 绘制平面中心
        center_sphere = pv.Sphere(radius=0.5, center=center)
        plotter.add_mesh(center_sphere, color=color, label=f'Center {all_angles_deg[i]:.1f}°')

    # 绘制心尖和基底
    apex_sphere = pv.Sphere(radius=0.5, center=apex)
    plotter.add_mesh(apex_sphere, color='red', label='Apex')
    base_sphere = pv.Sphere(radius=0.5, center=base_center)
    plotter.add_mesh(base_sphere, color='green', label='Base')

    # 绘制水平长轴
    line_points = np.array([apex, base_center])
    line = pv.lines_from_points(line_points)
    plotter.add_mesh(line, color='cyan', line_width=3, label='Long Axis')

    # 添加图例
    plotter.add_legend()

    # 设置视角
    plotter.view_isometric()
    plotter.show()


def mirror_point_cloud_x_center(input_file, center_type='mean'):
    """
    沿x轴中心镜像点云
    参数:
        input_file: 输入点云文件路径（TXT格式）
        output_file: 输出镜像点云文件路径
        center_type: 'mean'（使用x坐标平均值）或'midpoint'（使用x坐标最小最大值中点）
    """
    try:
        # 读取点云文件
        points = input_file

        # 计算x轴中心
        if center_type == 'mean':
            x_center = np.mean(points[:, 0])
        elif center_type == 'midpoint':
            x_center = (np.min(points[:, 0]) + np.max(points[:, 0])) / 2
        else:
            raise ValueError("center_type must be 'mean' or 'midpoint'")

        print(f"x轴中心: {x_center:.6f} (使用{center_type}方法)")

        # 沿x轴中心镜像：x_new = 2 * x_center - x
        points_mirrored = points.copy()
        points_mirrored[:, 0] = 2 * x_center - points[:, 0]


        return points_mirrored

    except Exception as e:
        print(f"处理点云文件时出错: {str(e)}")
        return None, None, None
# 主函数
def main(file_path, sampling_method, output_file,manu_path, visual=True):
    manu_points = scipy.io.loadmat(manu_path)
    manu_points = np.array(manu_points['Positions_SelectedPoints']).T

    # 验证 sampling_method 参数
    if sampling_method not in ['uniform', 'random']:
        raise ValueError("sampling_method 必须是 'uniform' 或 'random'")

    # 加载点云，分离心肌点和室间沟点（按行判断）
    points_myocardium, colors_myocardium, points_groove, colors_groove = load_point_cloud_from_txt(file_path)
    # points_myocardium = mirror_point_cloud_x_center(points_myocardium, center_type='mean')
    # points_myocardium[:, 0] = -points_myocardium[:, 0]

    print(f"加载点云，心肌点数: {len(points_myocardium)}，室间沟点数: {len(points_groove)}")

    # 预处理（仅对心肌点去噪）
    points_myocardium, inliers = preprocess_point_cloud(points_myocardium)
    colors_myocardium = colors_myocardium[inliers]  # 应用去噪掩码
    print(f"去噪后心肌点数: {len(points_myocardium)}")

    # 计算 XY 平面的最佳圆心，并可视化投影（仅使用心肌点）
    xc, yc, r = find_xy_center(points_myocardium, visual=visual)

    # 识别心尖（仅使用心肌点）
    apex = find_apex(points_myocardium, xc, yc)
    print(f"心尖 (Apex): {apex}")

    # 识别基底（仅使用心肌点）
    base_center, plane_model = find_base(points_myocardium, xc, yc)
    print(f"基底 (Base): {base_center}")

    # 计算水平长轴
    long_axis = compute_long_axis(apex, base_center)
    print(f"水平长轴 (Horizontal Long Axis): {long_axis}")

    # 计算短轴切面
    short_axes = compute_short_axes(apex, base_center, long_axis, num_slices=5)
    print("短轴切面 (Short Axes):")
    for i, (center, normal) in enumerate(short_axes):
        print(f"切面 {i + 1} - 中心: {center}, 法向量: {normal}")

    # # 计算最佳收缩平面和其上的点（仅使用心肌点）
    # best_plane, plane_points = compute_contracted_planes(points_myocardium, apex, base_center, long_axis)
    # 计算最佳收缩平面和其上的点（仅使用心肌点）
    best_plane, plane_points, angles_deg, plane_centers, plane_normals, plane_points_list = compute_contracted_planes(
        points_myocardium, apex, base_center, long_axis)

    if visual:
        # 可视化所有旋转面
        visualize_all_contracted_planes(points_myocardium, apex, base_center, long_axis, angles_deg, plane_centers,
                                        plane_normals, plane_points_list,top_n=5)


    # 可视化三维点云（包括心肌点、室间沟点和 V 字形室间隔线），并保存采样点
    visualize_results(manu_points, points_myocardium, colors_myocardium, points_groove, colors_groove, apex, base_center, long_axis, short_axes, best_plane, plane_points, sampling_method, output_file, visual=visual)

def SPECT_main(root_dir, p_name,visual=True):
    patient = p_name
    file_path = rf"{root_dir}\{patient}\ijkspect.txt"  # 替换为实际路径
    # 选择采样方式：'uniform' 或 'random'
    sampling_method = 'uniform'  # 或者 'random'



    manu_path = rf'data\SPECT\name_selectedpoints'
    for i in os.listdir(manu_path):
        if i.lower().replace(' ', '') == patient:
            manu_path = os.path.join(manu_path, i, 'SelectedPoints.mat')
    # 指定输出文件路径
    output_file = rf"{root_dir}\{patient}\sp_sampled_points.txt"  # 替换为实际路径
    main(file_path, sampling_method, output_file,manu_path, visual=visual)


if __name__ == "__main__":
    # 替换为你的 .txt 文件路径
    patient = r'guozuyu'
    SPECT_main(patient)

