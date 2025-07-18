import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA
import os
from scipy.spatial import cKDTree

def load_point_cloud(file_path):
    try:
        points = np.loadtxt(file_path)
        if points.shape[1] != 3:
            raise ValueError(f"Point cloud format error: shape {points.shape}, expected 3 columns (x, y, z)")
        print(f"Loaded point cloud {file_path}, shape: {points.shape}")
        return points
    except ValueError as e:
        print(f"Failed to load point cloud file: {e}")
        exit(1)

def compute_intersection_plane(left_points, right_points, distance_threshold=5.0):
    print(f"Merged point cloud shape: {np.vstack((left_points, right_points)).shape}")
    left_tree = cKDTree(left_points)
    right_tree = cKDTree(right_points)
    distances, indices = right_tree.query(left_points, k=1)
    contact_mask_left = distances < distance_threshold
    contact_points_left = left_points[contact_mask_left]
    distances, indices = left_tree.query(right_points, k=1)
    contact_mask_right = distances < distance_threshold
    contact_points_right = right_points[contact_mask_right]
    if contact_points_left.shape[0] == 0 or contact_points_right.shape[0] == 0:
        print("Warning: Insufficient contact points, falling back to PCA method")
        all_points = np.vstack((left_points, right_points))
        pca = PCA(n_components=3)
        pca.fit(all_points)
        normal = pca.components_[-1]
        center = np.mean(all_points, axis=0)
    else:
        contact_points = np.vstack((contact_points_left, contact_points_right))
        print(f"Contact region point cloud shape: {contact_points.shape}")
        pca = PCA(n_components=3)
        pca.fit(contact_points)
        normal = pca.components_[-1]
        center = np.mean(contact_points, axis=0)
    if normal[0] < 0:
        normal = -normal
        print("Normal flipped to make x component positive")
    left_projections = np.dot(left_points - center, normal)
    right_projections = np.dot(right_points - center, normal)
    left_side = np.mean(left_projections > 0)
    right_side = np.mean(right_projections < 0)
    print(f"Left points on positive side ratio: {left_side:.2%}")
    print(f"Right points on negative side ratio: {right_side:.2%}")
    if left_side < 0.7 or right_side < 0.7:
        print("Warning: Poor separation, consider adjusting distance_threshold or checking alignment")
    return normal, center

def project_points_to_plane(points, normal, center):
    projected_points = np.zeros_like(points)
    indices = np.arange(len(points))
    for i, p in enumerate(points):
        d = np.dot(normal, p - center)
        projected_points[i] = p - d * normal
    return projected_points, indices

def compute_major_axis(projected_points, original_points, indices, plane_normal, intersection_plane_center):
    pca = PCA(n_components=2)
    pca.fit(projected_points)
    center = np.mean(projected_points, axis=0)
    major_axis_direction = pca.components_[0]
    minor_axis_direction = pca.components_[1]
    major_projections = np.dot(projected_points - center, major_axis_direction)
    max_idx = np.argmax(major_projections)
    min_idx = np.argmin(major_projections)
    high_point = projected_points[max_idx]
    low_point = projected_points[min_idx]
    major_axis_points = np.array([low_point, high_point])
    if high_point[1] < low_point[1]:
        high_point, low_point = low_point, high_point
        max_idx, min_idx = min_idx, max_idx
    if major_axis_direction[1] < 0:
        major_axis_direction = -major_axis_direction
        minor_axis_direction = -minor_axis_direction
    minor_projections = np.dot(projected_points - center, minor_axis_direction)
    minor_max_idx = np.argmax(minor_projections)
    minor_min_idx = np.argmin(minor_projections)
    minor_high_point = projected_points[minor_max_idx]
    minor_low_point = projected_points[minor_min_idx]
    minor_axis_points = np.array([minor_low_point, minor_high_point])
    short_plane_center = (high_point + low_point) / 2
    short_plane_normal = major_axis_direction / np.linalg.norm(major_axis_direction)
    dot_product = np.dot(plane_normal, short_plane_normal)
    if abs(dot_product) > 1e-6:
        print(f"Warning: Short-axis plane normal not perpendicular to intersection plane normal! Dot product = {dot_product}")
    intersection_line_direction = np.cross(plane_normal, short_plane_normal)
    intersection_line_direction = intersection_line_direction / np.linalg.norm(intersection_line_direction)
    plane_tolerance = 1e-6
    short_plane_distance_threshold = 1.0
    candidate_indices = []
    short_plane_distances = []
    for i, point in enumerate(projected_points):
        plane_dist = abs(np.dot(point - intersection_plane_center, plane_normal))
        if plane_dist > plane_tolerance:
            continue
        short_plane_dist = abs(np.dot(point - short_plane_center, short_plane_normal))
        if short_plane_dist <= short_plane_distance_threshold:
            candidate_indices.append(i)
            short_plane_distances.append(short_plane_dist)
        else:
            short_plane_distances.append(short_plane_dist)
    if not candidate_indices:
        print(f"Warning: No points satisfy short-axis plane proximity (distance <= {short_plane_distance_threshold}). Selecting two closest points...")
        short_plane_distances = np.array(short_plane_distances)
        candidate_indices = np.argsort(short_plane_distances)[:2]
    candidate_points = projected_points[candidate_indices]
    line_projections = np.dot(candidate_points - short_plane_center, intersection_line_direction)
    max_line_idx = np.argmax(line_projections)
    min_line_idx = np.argmin(line_projections)
    boundary_point_1 = candidate_points[max_line_idx]
    boundary_point_2 = candidate_points[min_line_idx]
    x_range = projected_points[:, 0].max() - projected_points[:, 0].min()
    y_range = projected_points[:, 1].max() - projected_points[:, 1].min()
    z_range = projected_points[:, 2].max() - projected_points[:, 2].min()
    step_size = min(x_range, y_range, z_range) * 0.01
    point_cloud_distance_threshold = 1.0

    def compute_nearest_point_distance(point, points):
        distances = np.linalg.norm(points - point, axis=1)
        return np.min(distances)

    current_point = boundary_point_1
    projection_1 = np.dot(current_point - short_plane_center, intersection_line_direction)
    while True:
        nearest_dist = compute_nearest_point_distance(current_point, projected_points)
        if nearest_dist <= point_cloud_distance_threshold:
            boundary_point_1 = current_point
            break
        current_point = current_point - step_size * intersection_line_direction
        plane_dist = abs(np.dot(current_point - intersection_plane_center, plane_normal))
        if plane_dist > plane_tolerance:
            current_point = current_point + step_size * intersection_line_direction
            break
        short_plane_dist = abs(np.dot(current_point - short_plane_center, short_plane_normal))
        if short_plane_dist > short_plane_distance_threshold:
            current_point = current_point + step_size * intersection_line_direction
            break
        new_projection = np.dot(current_point - short_plane_center, intersection_line_direction)
        if new_projection <= 0:
            current_point = short_plane_center
            break
    boundary_point_1 = current_point
    current_point = boundary_point_2
    projection_2 = np.dot(current_point - short_plane_center, intersection_line_direction)
    while True:
        nearest_dist = compute_nearest_point_distance(current_point, projected_points)
        if nearest_dist <= point_cloud_distance_threshold:
            boundary_point_2 = current_point
            break
        current_point = current_point + step_size * intersection_line_direction
        plane_dist = abs(np.dot(current_point - intersection_plane_center, plane_normal))
        if plane_dist > plane_tolerance:
            current_point = current_point - step_size * intersection_line_direction
            break
        short_plane_dist = abs(np.dot(current_point - short_plane_center, short_plane_normal))
        if short_plane_dist > short_plane_distance_threshold:
            current_point = current_point - step_size * intersection_line_direction
            break
        new_projection = np.dot(current_point - short_plane_center, intersection_line_direction)
        if new_projection >= 0:
            current_point = short_plane_center
            break
    boundary_point_2 = current_point
    for point, label in [(boundary_point_1, "Boundary point 1"), (boundary_point_2, "Boundary point 2")]:
        nearest_dist = compute_nearest_point_distance(point, projected_points)
        if nearest_dist > point_cloud_distance_threshold:
            print(f"Warning: {label} after contraction does not satisfy left point cloud proximity (distance = {nearest_dist}). Selecting closest candidate point...")
            candidate_distances = [compute_nearest_point_distance(p, projected_points) for p in candidate_points]
            best_idx = np.argmin(candidate_distances)
            if label == "Boundary point 1":
                boundary_point_1 = candidate_points[best_idx]
            else:
                boundary_point_2 = candidate_points[best_idx]
    boundary_dist_1 = abs(np.dot(boundary_point_1 - short_plane_center, short_plane_normal))
    boundary_dist_2 = abs(np.dot(boundary_point_2 - short_plane_center, short_plane_normal))
    plane_dist_1 = abs(np.dot(boundary_point_1 - intersection_plane_center, plane_normal))
    plane_dist_2 = abs(np.dot(boundary_point_2 - intersection_plane_center, plane_normal))
    nearest_dist_1 = compute_nearest_point_distance(boundary_point_1, projected_points)
    nearest_dist_2 = compute_nearest_point_distance(boundary_point_2, projected_points)
    low_point_3d = original_points[indices[min_idx]]
    line_length_1 = np.linalg.norm(low_point_3d - boundary_point_1)
    line_length_2 = np.linalg.norm(low_point_3d - boundary_point_2)
    print(f"Major axis center: {center}")
    print(f"Major axis direction: {major_axis_direction}")
    print(f"Minor axis direction: {minor_axis_direction}")
    print(f"Projected lowest point: {low_point}")
    print(f"Original 3D lowest point: {low_point_3d}")
    print(f"Major axis length: {np.linalg.norm(high_point - low_point)}")
    print(f"Minor axis endpoints: {minor_low_point}, {minor_high_point}")
    print(f"Minor axis length: {np.linalg.norm(minor_high_point - minor_low_point)}")
    print(f"Short-axis plane center: {short_plane_center}")
    print(f"Short-axis plane normal: {short_plane_normal}")
    print(f"Dot product between intersection plane normal and short-axis plane normal: {dot_product}")
    print(f"Intersection line direction: {intersection_line_direction}")
    print(f"Adjusted outermost boundary point 1: {boundary_point_1}, distance to short-axis plane: {boundary_dist_1}, distance to intersection plane: {plane_dist_1}, nearest distance to left point cloud: {nearest_dist_1}")
    print(f"Adjusted outermost boundary point 2: {boundary_point_2}, distance to short-axis plane: {boundary_dist_2}, distance to intersection plane: {plane_dist_2}, nearest distance to left point cloud: {nearest_dist_2}")
    print(f"Line length from lowest point to boundary point 1: {line_length_1}")
    print(f"Line length from lowest point to boundary point 2: {line_length_2}")
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
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    print(f"Point cloud range: x=({x_min}, {x_max}), y=({y_min}, {y_max}), z=({z_min}, {z_max})")
    range_max = max(x_max - x_min, y_max - y_min, z_max - z_min)
    plane_size = range_max * 1.5
    print(f"Plane size: {plane_size}")
    return pv.Plane(center=center, direction=normal, i_size=plane_size, j_size=plane_size)

def visualize_projected_points(projected_points, major_axis_points, minor_axis_points, low_point):
    plotter = pv.Plotter()
    projected_mesh = pv.PolyData(projected_points)
    plotter.add_mesh(projected_mesh, color='blue', opacity=0.5, point_size=5)
    major_axis_line = pv.Line(major_axis_points[0], major_axis_points[1])
    plotter.add_mesh(major_axis_line, color='red', line_width=3)
    minor_axis_line = pv.Line(minor_axis_points[0], minor_axis_points[1])
    plotter.add_mesh(minor_axis_line, color='green', line_width=3)
    low_sphere = pv.Sphere(radius=1.0, center=low_point)
    plotter.add_mesh(low_sphere, color='orange')
    plotter.add_axes()
    plotter.show()

def visualize_full_scene(left_points, right_points, intersection_plane, short_plane, low_point_3d, boundary_point_1, boundary_point_2, patient, plane_normal, intersection_plane_center, short_plane_normal, save_dir, visual=True):
    plotter = pv.Plotter()
    left_mesh = pv.PolyData(left_points)
    plotter.add_mesh(left_mesh, color='#d8654f', opacity=0.9, point_size=2)
    right_mesh = pv.PolyData(right_points)
    plotter.add_mesh(right_mesh, color='#80ae80', opacity=0.9, point_size=2)
    plotter.add_mesh(intersection_plane, color='yellow', opacity=0.7, show_edges=True,
                     edge_color='black')
    plotter.add_mesh(short_plane, color='green', opacity=0.7, show_edges=True, edge_color='black')
    plotter.background_color = 'white'
    low_sphere = pv.Sphere(radius=4.0, center=low_point_3d)
    plotter.add_mesh(low_sphere, color='#d8654f')
    plane_bend_factor = 0.2
    num_samples = 9
    t_values = np.linspace(0.0, 1.0, num_samples)
    print(f"Uniform t values: {t_values}")
    line_length_1 = np.linalg.norm(boundary_point_1 - low_point_3d)
    mid_point_1 = (low_point_3d + boundary_point_1) / 2
    vec_to_mid_1 = mid_point_1 - intersection_plane_center
    distance_to_plane_1 = np.dot(vec_to_mid_1, plane_normal)
    plane_direction_1 = plane_normal if distance_to_plane_1 < 0 else -plane_normal
    plane_offset_1 = line_length_1 * plane_bend_factor
    control_point_1 = mid_point_1 + plane_offset_1 * plane_direction_1
    print(f"Line 1 control point: {control_point_1}")
    curve_points_1 = []
    for t in t_values:
        t2 = t * t
        one_minus_t = 1 - t
        one_minus_t2 = one_minus_t * one_minus_t
        point = (one_minus_t2 * low_point_3d + 2 * one_minus_t * t * control_point_1 + t2 * boundary_point_1)
        curve_points_1.append(point)
    curve_points_1 = np.array(curve_points_1)
    line_length_2 = np.linalg.norm(boundary_point_2 - low_point_3d)
    mid_point_2 = (low_point_3d + boundary_point_2) / 2
    vec_to_mid_2 = mid_point_2 - intersection_plane_center
    distance_to_plane_2 = np.dot(vec_to_mid_2, plane_normal)
    plane_direction_2 = plane_normal if distance_to_plane_2 < 0 else -plane_normal
    plane_offset_2 = line_length_2 * plane_bend_factor
    control_point_2 = mid_point_2 + plane_offset_2 * plane_direction_2
    print(f"Line 2 control point: {control_point_2}")
    curve_points_2 = []
    for t in t_values:
        t2 = t * t
        one_minus_t = 1 - t
        one_minus_t2 = one_minus_t * one_minus_t
        point = (one_minus_t2 * low_point_3d + 2 * one_minus_t * t * control_point_2 + t2 * boundary_point_2)
        curve_points_2.append(point)
    curve_points_2 = np.array(curve_points_2)
    colors_1 = np.array([[0, 255, 0]] * num_samples)
    colors_2 = np.array([[255, 0, 0]] * num_samples)
    print("Curve 1 sampled points (from lowest point to boundary point 1, green):")
    for i, (point, color) in enumerate(zip(curve_points_1, colors_1)):
        print(f"  Point {i+1}: {point}, RGB: {color}")
    print("Curve 2 sampled points (from lowest point to boundary point 2, red):")
    for i, (point, color) in enumerate(zip(curve_points_2, colors_2)):
        print(f"  Point {i+1}: {point}, RGB: {color}")
    all_sampled_points = np.vstack((curve_points_1, curve_points_2[1:]))
    all_colors = np.vstack((colors_1, colors_2[1:]))
    intersection_line_direction = np.cross(plane_normal, short_plane_normal)
    intersection_line_direction = intersection_line_direction / np.linalg.norm(intersection_line_direction)
    print(f"Intersection line direction: {intersection_line_direction}")
    short_plane_center = short_plane.center
    projections = np.dot(all_sampled_points - short_plane_center, intersection_line_direction)
    sort_indices = np.argsort(projections)
    all_sampled_points = all_sampled_points[sort_indices]
    all_colors = all_colors[sort_indices]
    print("Sorted output points (left to right along intersection line):")
    for i, (point, color) in enumerate(zip(all_sampled_points, all_colors)):
        print(f"  Point {i+1}: {point}, RGB: {color}")
    curve_spline_1 = pv.Spline(curve_points_1, n_points=50)
    plotter.add_mesh(curve_spline_1, color='red', line_width=3)
    curve_spline_2 = pv.Spline(curve_points_2, n_points=50)
    plotter.add_mesh(curve_spline_2, color='red', line_width=3)
    for point in curve_points_1:
        sample_sphere = pv.Sphere(radius=0.5, center=point)
        plotter.add_mesh(sample_sphere, color=[0, 255, 0])
    for point in curve_points_2:
        sample_sphere = pv.Sphere(radius=0.5, center=point)
        plotter.add_mesh(sample_sphere, color=[255, 0, 0])
    output_dir = f"{save_dir}/{patient}/"
    output_file = os.path.join(output_dir, f"sampled_points_{patient}.txt")
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_data = np.hstack((all_sampled_points, all_colors))
        np.savetxt(output_file, output_data, fmt=['%.6f', '%.6f', '%.6f', '%d', '%d', '%d'], delimiter=' ')
        print(f"Sorted sampled points with RGB (deduplicated lowest point) saved to: {output_file}")
    except Exception as e:
        print(f"Failed to save sampled points: {e}")
    if visual:
        plotter.show_bounds(grid=False, location='outer', color='black')
        plotter.add_axes()
        plotter.show()

def CTA_main(data_dir, p_name, re_LV=False, visual=True):
    patient = p_name
    if re_LV:
        left_sur_file = rf'../LVtopoints/name/{p_name}/{p_name}_LV_transformed.txt'
    else:
        left_sur_file = rf"{data_dir}/{patient}/ijkcta.txt"
    right_txt = rf"../RVtopoints/name/{patient}/{patient}_RV_transformed.txt"
    left_points = load_point_cloud(left_sur_file)
    right_points = load_point_cloud(right_txt)
    normal, center = compute_intersection_plane(left_points, right_points, 5)
    projected_points, indices = project_points_to_plane(left_points, normal, center)
    _, _, major_axis_points, _, minor_axis_points, low_point, low_point_3d, short_plane_center, short_plane_normal, boundary_point_1, boundary_point_2 = compute_major_axis(
        projected_points, left_points, indices, normal, center
    )
    all_points = np.vstack((left_points, right_points))
    intersection_plane = create_plane(normal, center, all_points)
    short_plane = create_plane(short_plane_normal, short_plane_center, all_points)
    if visual:
        print("Displaying projected plane, major axis, minor axis, and projected lowest point...")
        visualize_projected_points(projected_points, major_axis_points, minor_axis_points, low_point)
    print("Displaying full 3D scene including intersection plane, short-axis plane, enlarged red lowest point, curved Bezier curves, and colored uniform sampled points (green left, red right, sorted left to right, deduplicated lowest point)...")
    visualize_full_scene(left_points, right_points, intersection_plane, short_plane, low_point_3d, boundary_point_1, boundary_point_2, patient, normal, center, short_plane_normal, data_dir, visual)