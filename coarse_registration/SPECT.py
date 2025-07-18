import os
import numpy as np
import pyvista as pv
from sklearn.cluster import DBSCAN
import scipy.io

def ransac_circle(xy_points, num_iterations=1000, threshold_lower=0.01, threshold_upper=0.01, radius_range=(0, 0.7)):
    best_circle = None
    max_points_inside = 0
    max_radius = 0
    n_points = len(xy_points)
    r_min, r_max = radius_range
    for _ in range(num_iterations):
        center_idx = np.random.choice(n_points)
        center = xy_points[center_idx]
        xc, yc = center[0], center[1]
        r = np.random.uniform(r_min, r_max)
        distances = np.linalg.norm(xy_points - [xc, yc], axis=1)
        points_inside = distances <= r
        points_inside_count = np.sum(points_inside)
        inliers = (distances >= r - threshold_lower) & (distances <= r + threshold_upper)
        if np.sum(inliers) != 0:
            continue
        if (points_inside_count > max_points_inside) or (points_inside_count == max_points_inside and r > max_radius):
            max_points_inside = points_inside_count
            max_radius = r
            best_circle = (xc, yc, r)
    if best_circle is None:
        raise ValueError("Unable to find a circle meeting the conditions (no points on the circumference)")
    return best_circle, np.array([])

def load_point_cloud_from_txt(file_path):
    points_myocardium_list = []
    colors_myocardium_list = []
    points_groove_list = []
    colors_groove_list = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            values = [float(v) for v in values]
            if len(values) == 3:
                points_myocardium_list.append(values)
                colors_myocardium_list.append([1, 1, 1])
            elif len(values) == 6 and values[3]==0:
                points_myocardium_list.append(values[:3])
                colors_myocardium_list.append([1, 1, 1])
            elif len(values) == 6:
                points_groove_list.append(values[:3])
                colors_groove_list.append(values[3:6])
            else:
                raise ValueError(f"Invalid row data format: {line}, must have 3 or 6 columns")
    points_myocardium = np.array(points_myocardium_list)
    colors_myocardium = np.array(colors_myocardium_list)
    points_groove = np.array(points_groove_list) if points_groove_list else np.array([])
    colors_groove = np.array(colors_groove_list) if colors_groove_list else np.array([])
    if colors_groove.size > 0:
        colors_groove = colors_groove / 255.0
    return points_myocardium, colors_myocardium, points_groove, colors_groove

def preprocess_point_cloud(points):
    k = 20
    dists = []
    for i in range(len(points)):
        diffs = points - points[i]
        distances = np.linalg.norm(diffs, axis=1)
        distances = np.sort(distances)[1:k + 1]
        dists.append(np.mean(distances))
    dists = np.array(dists)
    mean_dist, std_dist = np.mean(dists), np.std(dists)
    inliers = (dists >= mean_dist - 2 * std_dist) & (dists <= mean_dist + 2 * std_dist)
    return points[inliers], inliers

def visualize_xy_projection(unique_xy, xc, yc, r, inliers):
    xy_points_3d = np.zeros((len(unique_xy), 3))
    xy_points_3d[:, :2] = unique_xy
    xy_points_3d[:, 2] = 0
    cloud = pv.PolyData(xy_points_3d)
    plotter = pv.Plotter()
    plotter.add_points(cloud, color='blue', point_size=5, render_points_as_spheres=True, label='Projected Points')
    center = np.array([xc, yc, 0])
    center_sphere = pv.Sphere(radius=0.5, center=center)
    plotter.add_mesh(center_sphere, color='red', label='Circle Center')
    circle = pv.Circle(radius=r, resolution=100)
    circle.translate([xc, yc, 0])
    plotter.add_mesh(circle, color='cyan', style='wireframe', label='Fitted Circle')
    plotter.add_legend()
    plotter.view_xy()
    plotter.show_grid()
    plotter.set_background('gray')
    plotter.show()

def find_xy_center(points, visual=True):
    xy_points = points[:, :2]
    unique_xy = np.unique(xy_points, axis=0)
    print(f"Number of points after projecting to XY plane: {len(xy_points)}")
    print(f"Number of points after deduplication: {len(unique_xy)}")
    (xc, yc, r), inliers = ransac_circle(unique_xy, num_iterations=10000, threshold_lower=0.01, threshold_upper=0.01, radius_range=(0, 0.7))
    distances = np.linalg.norm(unique_xy - [xc, yc], axis=1)
    points_inside = distances <= r
    points_inside_count = np.sum(points_inside)
    inliers = (distances >= r - 0.01) & (distances <= r + 0.01)
    print(f"Best circle center on XY plane: (x: {xc:.2f}, y: {yc:.2f}), radius: {r:.2f}")
    print(f"Number of points inside the circle: {points_inside_count}")
    print(f"Circumference point range: [{r - 0.01:.4f}, {r + 0.01:.4f}]")
    print(f"Number of circumference points: {np.sum(inliers)} (must be 0)")
    if visual:
        visualize_xy_projection(unique_xy, xc, yc, r, inliers)
    return xc, yc, r

def find_apex(points, xc, yc):
    z_values = points[:, 2]
    z_min = np.min(z_values)
    apex = np.array([xc, yc, z_min])
    z_min_threshold = np.percentile(z_values, 5)
    apex_points = points[z_values <= z_min_threshold]
    clustering = DBSCAN(eps=1.0, min_samples=5).fit(apex_points)
    labels = clustering.labels_
    apex_cluster_points = apex_points[labels != -1]
    if len(apex_cluster_points) > 0:
        z_min = np.min(apex_cluster_points[:, 2])
        apex = np.array([xc, yc, z_min])
    return apex

def find_base(points, xc, yc):
    z_values = points[:, 2]
    z_max = np.max(z_values)
    base_center = np.array([xc, yc, z_max])
    z_max_threshold = np.percentile(z_values, 95)
    base_points = points[z_values >= z_max_threshold]
    A = np.c_[base_points[:, 0], base_points[:, 1], np.ones(len(base_points))]
    C = base_points[:, 2]
    plane_model, _, _, _ = np.linalg.lstsq(A, C, rcond=None)
    return base_center, plane_model

def compute_long_axis(apex, base_center):
    long_axis = base_center - apex
    long_axis = long_axis / np.linalg.norm(long_axis)
    return long_axis

def compute_short_axes(apex, base_center, long_axis, num_slices=5):
    distance = np.linalg.norm(base_center - apex)
    slice_positions = np.linspace(0, distance, num_slices + 2)[1:-1]
    short_axes = []
    for pos in slice_positions:
        center = apex + pos * long_axis
        short_axes.append((center, long_axis))
    return short_axes

def compute_contracted_planes(points, apex, base_center, long_axis):
    long_axis_center = (apex + base_center) / 2
    if np.abs(long_axis[0]) < 0.9:
        base1 = np.cross(long_axis, np.array([1, 0, 0]))
    else:
        base1 = np.cross(long_axis, np.array([0, 1, 0]))
    base1 = base1 / np.linalg.norm(base1)
    projections = np.dot(points - long_axis_center, base1)
    max_projection = np.max(projections)
    long_axis_norm = long_axis / np.linalg.norm(long_axis)
    parallel_component = np.dot(base1, long_axis_norm) * long_axis_norm
    perpendicular_component = base1 - parallel_component
    radius = np.linalg.norm(perpendicular_component)*4
    radius = max_projection*0.60
    if radius < 1e-6:
        radius = 5.0
    angles_deg = np.arange(0, 360, 5)
    angles_rad = np.deg2rad(angles_deg)
    distance_threshold = 0.8
    best_plane = None
    best_plane_points = None
    max_points_covered = 0
    best_angle = 0
    plane_centers = []
    plane_normals = []
    plane_points_list = []
    all_angles_deg = []
    for angle_rad, angle_deg in zip(angles_rad, angles_deg):
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        k = long_axis_norm
        K = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ])
        I = np.eye(3)
        rotation_matrix = I + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
        rotated_normal = np.dot(rotation_matrix, base1)
        rotated_normal = rotated_normal / np.linalg.norm(rotated_normal)
        plane_center = long_axis_center + radius * rotated_normal
        plane_vec = points - plane_center
        distances_to_plane = np.abs(np.dot(plane_vec, rotated_normal))
        points_covered = np.sum(distances_to_plane <= distance_threshold)
        plane_centers.append(plane_center)
        plane_normals.append(rotated_normal)
        plane_points_mask = distances_to_plane <= distance_threshold
        plane_points_list.append(points[plane_points_mask])
        all_angles_deg.append(angle_deg)
        if points_covered > max_points_covered:
            max_points_covered = points_covered
            best_plane = (plane_center, rotated_normal)
            best_angle = angle_deg
            best_plane_points = points[plane_points_mask]
    if best_plane is None:
        raise ValueError("Unable to find a valid contracted plane")
    plane_center, normal = best_plane
    print(f"Best contracted plane - Number of points covered: {max_points_covered}, Plane center: {plane_center}")
    print(f"Best normal vector: {normal}, Rotation angle: {best_angle:.1f}°")
    return best_plane, best_plane_points, all_angles_deg, plane_centers, plane_normals, plane_points_list

def compute_contracted_plane(points, apex, base_center, long_axis):
    cloud_center = (apex + base_center) / 2
    if np.abs(long_axis[0]) < 0.9:
        base1 = np.cross(long_axis, np.array([1, 0, 0]))
    else:
        base1 = np.cross(long_axis, np.array([0, 1, 0]))
    base1 = base1 / np.linalg.norm(base1)
    base2 = np.cross(long_axis, base1)
    base2 = base2 / np.linalg.norm(base2)
    normals = [
        base1,
        -base1,
        base2,
        -base2
    ]
    step_size = 0.5
    distance_threshold = 0.8
    best_plane = None
    best_plane_points = None
    max_points_covered = 0
    best_angle = 0
    angles_deg = np.arange(-20, 20, 5)
    angles_rad = np.deg2rad(angles_deg)
    plane_centers = []
    plane_normals = []
    plane_points_list = []
    all_angles_deg = []
    for normal in normals:
        projections = np.dot(points - cloud_center, normal)
        max_projection = np.max(projections)
        start_distance = max_projection * 0.61
        end_distance = max_projection * 0.6
        distance = start_distance
        while distance >= end_distance:
            plane_center = cloud_center + distance * normal
            for angle_rad, angle_deg in zip(angles_rad, angles_deg):
                cos_theta = np.cos(angle_rad)
                sin_theta = np.sin(angle_rad)
                k = long_axis
                K = np.array([
                    [0, -k[2], k[1]],
                    [k[2], 0, -k[0]],
                    [-k[1], k[0], 0]
                ])
                I = np.eye(3)
                rotation_matrix = I + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
                rotated_normal = np.dot(rotation_matrix, normal)
                rotated_normal = rotated_normal / np.linalg.norm(rotated_normal)
                plane_vec = points - plane_center
                distances_to_plane = np.abs(np.dot(plane_vec, rotated_normal))
                points_covered = np.sum(distances_to_plane <= distance_threshold)
                plane_centers.append(plane_center)
                plane_normals.append(rotated_normal)
                plane_points_mask = distances_to_plane <= distance_threshold
                plane_points_list.append(points[plane_points_mask])
                all_angles_deg.append(angle_deg)
                if points_covered > max_points_covered:
                    max_points_covered = points_covered
                    best_plane = (plane_center, rotated_normal)
                    best_angle = angle_deg
                    best_plane_points = points[plane_points_mask]
            distance -= step_size
    if best_plane is None:
        raise ValueError("Unable to find a valid contracted plane")
    plane_center, normal = best_plane
    print(f"Best contracted plane - Number of points covered: {max_points_covered}, Plane center: {plane_center}")
    print(f"Best normal vector: {normal}, Rotation angle around long axis: {best_angle:.2f} degrees")
    return best_plane, best_plane_points, all_angles_deg, plane_centers, plane_normals, plane_points_list

def extract_outer_edge_points(plane_points, plane_center, plane_normal):
    if len(plane_points) == 0:
        return np.array([])
    if np.abs(plane_normal[0]) < 0.9:
        base1 = np.cross(plane_normal, np.array([1, 0, 0]))
    else:
        base1 = np.cross(plane_normal, np.array([0, 1, 0]))
    base1 = base1 / np.linalg.norm(base1)
    base2 = np.cross(plane_normal, base1)
    base2 = base2 / np.linalg.norm(base2)
    vec_to_points = plane_points - plane_center
    coords_u = np.dot(vec_to_points, base1)
    coords_v = np.dot(vec_to_points, base2)
    distances = np.sqrt(coords_u**2 + coords_v**2)
    if len(distances) == 0:
        return np.array([])
    percentile_threshold = np.percentile(distances, 70)
    outer_mask = distances >= percentile_threshold
    outer_points = plane_points[outer_mask]
    return outer_points

def compute_plane_intersection(plane1_center, plane1_normal, plane2_center, plane2_normal):
    line_direction = np.cross(plane1_normal, plane2_normal)
    if np.linalg.norm(line_direction) < 1e-6:
        raise ValueError("Short axis plane and best contracted plane are parallel, cannot compute intersection")
    line_direction = line_direction / np.linalg.norm(line_direction)
    A = np.array([plane1_normal, plane2_normal])
    b = np.array([np.dot(plane1_normal, plane1_center), np.dot(plane2_normal, plane2_center)])
    A = np.vstack([A, line_direction])
    b = np.append(b, 0)
    p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return p, line_direction

def extract_septum_points(points_myocardium, apex, best_plane_center, best_plane_normal, long_axis, short_axes, sampling_method='random'):
    vec_to_points = points_myocardium - apex
    projections = np.dot(vec_to_points, long_axis)
    points_on_axis = apex + np.outer(projections, long_axis)
    distances_to_axis = np.linalg.norm(points_myocardium - points_on_axis, axis=1)
    percentile_threshold = np.percentile(distances_to_axis, 30)
    inner_mask = distances_to_axis <= percentile_threshold
    inner_points = points_myocardium[inner_mask]
    if len(inner_points) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([0, 0, 0])
    direction = best_plane_center - apex
    direction = direction / np.linalg.norm(direction)
    middle_index = len(short_axes) // 2
    short_axis_center, short_axis_normal = short_axes[middle_index]
    intersection_point, intersection_direction = compute_plane_intersection(
        short_axis_center, short_axis_normal, best_plane_center, best_plane_normal
    )
    vec_to_points = inner_points - intersection_point
    projections = np.dot(vec_to_points, intersection_direction)
    min_proj = np.min(projections)
    max_proj = np.max(projections)
    left_end = intersection_point + min_proj * intersection_direction
    right_end = intersection_point + max_proj * intersection_direction
    base2 = intersection_direction
    base2 = base2 / np.linalg.norm(base2)
    dot_product = np.dot(base2, best_plane_normal)
    if abs(dot_product) > 1e-6:
        print(f"Warning: base2 is not perfectly perpendicular to best_plane_normal (dot product: {dot_product})")
    mid_point = (left_end + right_end) / 2
    symmetry_axis = mid_point - apex
    symmetry_axis_norm = np.linalg.norm(symmetry_axis)
    if symmetry_axis_norm > 1e-6:
        symmetry_axis = symmetry_axis / symmetry_axis_norm
    else:
        symmetry_axis = np.zeros(3)
    left_direction = left_end - apex
    right_direction = right_end - apex
    left_distance = np.linalg.norm(left_direction)
    right_distance = np.linalg.norm(right_direction)
    if left_distance > 1e-6:
        left_direction = left_direction / left_distance
    else:
        left_direction = np.zeros(3)
    if right_distance > 1e-6:
        right_direction = right_direction / right_distance
    else:
        right_direction = np.zeros(3)
    left_mid_point = (apex + left_end) / 2
    right_mid_point = (apex + right_end) / 2
    left_perp = np.cross(left_direction, symmetry_axis)
    left_perp_norm = np.linalg.norm(left_perp)
    if left_perp_norm > 1e-6:
        left_perp = left_perp / left_perp_norm
    else:
        left_perp = np.zeros(3)
    left_vec = left_end - apex
    right_vec = right_end - apex
    left_proj = np.dot(left_vec, symmetry_axis)
    right_proj = np.dot(right_vec, symmetry_axis)
    left_perp_vec = left_vec - left_proj * symmetry_axis
    right_perp_vec = right_vec - right_proj * symmetry_axis
    left_perp = left_perp_vec / np.linalg.norm(left_perp_vec) if np.linalg.norm(left_perp_vec) > 1e-6 else np.zeros(3)
    right_perp = right_perp_vec / np.linalg.norm(right_perp_vec) if np.linalg.norm(right_perp_vec) > 1e-6 else np.zeros(3)
    dot_product_perp = np.dot(left_perp, right_perp)
    if dot_product_perp > 0:
        right_perp = -right_perp
    expand_factor = 0.1
    left_expand_distance = left_distance * expand_factor
    right_expand_distance = right_distance * expand_factor
    left_end = left_end + left_expand_distance * left_perp
    right_end = right_end + right_expand_distance * right_perp
    left_mid_point = (apex + left_end) / 2
    right_mid_point = (apex + right_end) / 2
    bend_factor = 0.2
    left_distance = np.linalg.norm(left_end - apex)
    right_distance = np.linalg.norm(right_end - apex)
    offset = max(left_distance, right_distance) * bend_factor
    left_control_point = left_mid_point + offset * left_perp
    right_control_point = right_mid_point + offset * right_perp
    plane_bend_factor = 0.1
    plane_offset = max(left_distance, right_distance) * plane_bend_factor
    vec_to_left_control = left_control_point - best_plane_center
    left_distance_to_plane = np.dot(vec_to_left_control, best_plane_normal)
    left_plane_direction = best_plane_normal if left_distance_to_plane < 0 else -best_plane_normal
    left_control_point += plane_offset * left_plane_direction
    vec_to_right_control = right_control_point - best_plane_center
    right_distance_to_plane = np.dot(vec_to_right_control, best_plane_normal)
    right_plane_direction = best_plane_normal if right_distance_to_plane < 0 else -best_plane_normal
    right_control_point += plane_offset * right_plane_direction
    num_points = int(max(left_distance, right_distance) / 0.3) + 1
    t_values = np.linspace(0, 1, num_points)
    left_points_list = []
    right_points_list = []
    for t in t_values:
        t2 = t * t
        one_minus_t = 1 - t
        one_minus_t2 = one_minus_t * one_minus_t
        left_point = (one_minus_t2 * apex +
                      2 * one_minus_t * t * left_control_point +
                      t2 * left_end)
        left_points_list.append(left_point)
        right_point = (one_minus_t2 * apex +
                       2 * one_minus_t * t * right_control_point +
                       t2 * right_end)
        right_points_list.append(right_point)
    left_points_list[-1] = left_end
    right_points_list[-1] = right_end
    left_points = np.array(left_points_list)
    right_points = np.array(right_points_list)
    num_samples = 9
    left_sampled_points = []
    right_sampled_points = []
    if sampling_method == 'uniform':
        t_values = np.linspace(0, 1, num_samples)
        for t in t_values:
            t2 = t * t
            one_minus_t = 1 - t
            one_minus_t2 = one_minus_t * one_minus_t
            left_sampled_point = (one_minus_t2 * apex +
                                  2 * one_minus_t * t * left_control_point +
                                  t2 * left_end)
            left_sampled_points.append(left_sampled_point)
            right_sampled_point = (one_minus_t2 * apex +
                                   2 * one_minus_t * t * right_control_point +
                                   t2 * right_end)
            right_sampled_points.append(right_sampled_point)
    elif sampling_method == 'random':
        t_intervals = np.linspace(0, 1, num_samples + 1)
        for i in range(num_samples):
            t_lower = t_intervals[i]
            t_upper = t_intervals[i + 1]
            t_left = np.random.uniform(t_lower, t_upper)
            t_right = np.random.uniform(t_lower, t_upper)
            t2 = t_left * t_left
            one_minus_t = 1 - t_left
            one_minus_t2 = one_minus_t * one_minus_t
            left_sampled_point = (one_minus_t2 * apex +
                                  2 * one_minus_t * t_left * left_control_point +
                                  t2 * left_end)
            left_sampled_points.append(left_sampled_point)
            t2 = t_right * t_right
            one_minus_t = 1 - t_right
            one_minus_t2 = one_minus_t * one_minus_t
            right_sampled_point = (one_minus_t2 * apex +
                                   2 * one_minus_t * t_right * right_control_point +
                                   t2 * right_end)
            right_sampled_points.append(right_sampled_point)
    else:
        raise ValueError("sampling_method must be 'uniform' or 'random'")
    left_sampled_points = np.array(left_sampled_points)
    right_sampled_points = np.array(right_sampled_points)
    return left_points, right_points, left_sampled_points, right_sampled_points, intersection_direction

def visualize_results(manu_points, points_myocardium, colors_myocardium, points_groove, colors_groove, apex, base_center, long_axis, short_axes, best_plane, plane_points, sampling_method='random', output_file='sampled_points.txt', visual=True):
    best_plane_center, best_plane_normal = best_plane
    left_points, right_points, left_sampled_points, right_sampled_points, intersection_direction = extract_septum_points(
        points_myocardium, apex, best_plane_center, best_plane_normal, long_axis, short_axes, sampling_method
    )
    print(f"V-shape left points count: {len(left_points)}, right points count: {len(right_points)}")
    print(f"Sampled points - Left: {len(left_sampled_points)}, Right: {len(right_sampled_points)}, Sampling method: {sampling_method}")
    print(f"Intersection direction: {intersection_direction}")
    left_sampled_points = left_sampled_points[1:]
    sampled_points = np.vstack([right_sampled_points, left_sampled_points])
    right_colors = np.array([[0, 1, 0]] * len(right_sampled_points))
    left_colors = np.array([[1, 0, 0]] * len(left_sampled_points))
    sampled_colors = np.vstack([right_colors, left_colors])
    abs_direction = np.abs(intersection_direction)
    if abs_direction[0] > abs_direction[1]:
        sort_axis = 0
        print("Sorting by X-axis from large to small (right to left)")
    else:
        sort_axis = 1
        print("Sorting by Y-axis from large to small (right to left)")
    indices = np.argsort(-sampled_points[:, sort_axis])
    sampled_points = sampled_points[indices]
    sampled_colors = sampled_colors[indices]
    sampled_colors_255 = (sampled_colors * 255).astype(int)
    sampled_data = np.hstack([sampled_points, sampled_colors_255])
    with open(output_file, 'w') as f:
        for point in sampled_data:
            line = f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(point[3])} {int(point[4])} {int(point[5])}\n"
            f.write(line)
    print(f"Sampled points saved to file: {output_file}, Total points: {len(sampled_points)}")
    if len(points_groove) > 0:
        points_groove = np.vstack([points_groove, sampled_points])
        colors_groove = np.vstack([colors_groove, sampled_colors])
    else:
        points_groove = sampled_points
        colors_groove = sampled_colors
    cloud_myocardium = pv.PolyData(points_myocardium)
    cloud_myocardium['colors'] = colors_myocardium
    if manu_points is not None:
        cloud_manupoints = pv.PolyData(manu_points)
    else: cloud_manupoints = None
    if len(points_groove) > 0:
        cloud_groove = pv.PolyData(points_groove)
        cloud_groove['colors'] = colors_groove
    else:
        cloud_groove = None
    if visual:
        plotter = pv.Plotter()
        plotter.add_points(cloud_myocardium, color='#d8654f', rgb=True, point_size=3, render_points_as_spheres=True, label='Heart Muscle')
        if manu_points is not None:
            plotter.add_points(cloud_manupoints, color='red', rgb=False, point_size=10, render_points_as_spheres=True)

        if cloud_groove is not None:
            plotter.add_points(cloud_groove, scalars='colors', rgb=True, point_size=10, render_points_as_spheres=True, label='Interventricular Groove')

        if len(left_points) > 1:
            left_line = pv.lines_from_points(left_points)
            plotter.add_mesh(left_line, color='red', line_width=3, label='Interventricular Septum (Left)')

        if len(right_points) > 1:
            right_line = pv.lines_from_points(right_points)
            plotter.add_mesh(right_line, color='red', line_width=3, label='Interventricular Septum (Right)')

        apex_sphere = pv.Sphere(radius=0.2, center=apex)
        plotter.add_mesh(apex_sphere, color='red', label='Apex')
        base_sphere = pv.Sphere(radius=0.2, center=base_center)
        plotter.add_mesh(base_sphere, color='green', label='Base')
        line_points = np.array([apex, base_center])
        line = pv.lines_from_points(line_points)
        plotter.add_mesh(line, color='cyan', line_width=3, label='Long Axis')

        if short_axes:
            middle_index = len(short_axes) // 2
            center, normal = short_axes[middle_index]
            plane = pv.Plane(center=center, direction=normal, i_size=10, j_size=10)
            plotter.add_mesh(plane, color='yellow', opacity=0.5, label='Short Axis')

        contracted_plane = pv.Plane(center=best_plane_center, direction=best_plane_normal, i_size=10, j_size=10)
        plotter.add_mesh(contracted_plane, color='purple', opacity=0.5, label='Best Contracted Plane')

        plotter.add_legend()

        plotter.show_grid()
        plotter.set_background('white')
        plotter.show()

def visualize_all_contracted_plane(points, apex, base_center, long_axis, all_angles_deg, plane_centers, plane_normals, plane_points_list, step_by_step=False, current_index=1):
    cloud_center = (apex + base_center) / 2
    total_planes = len(plane_centers)
    if current_index > total_planes:
        current_index = total_planes
    if step_by_step:
        for i in range(1, total_planes + 1):
            plotter = pv.Plotter()
            plotter.set_background('gray')
            plotter.show_grid()
            cloud = pv.PolyData(points)
            plotter.add_points(cloud, color='blue', point_size=2, render_points_as_spheres=True, label='Original Points')
            colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan']
            for j in range(i):
                center = plane_centers[j]
                normal = plane_normals[j]
                plane_points = plane_points_list[j]
                angle_deg = all_angles_deg[j]
                plane = pv.Plane(center=center, direction=normal, i_size=10, j_size=10)
                color = colors[j % len(colors)]
                plotter.add_mesh(plane, color=color, opacity=0.3, label=f'Plane Angle {angle_deg:.1f}°')
            apex_sphere = pv.Sphere(radius=0.5, center=apex)
            plotter.add_mesh(apex_sphere, color='red', label='Apex')
            base_sphere = pv.Sphere(radius=0.5, center=base_center)
            plotter.add_mesh(base_sphere, color='green', label='Base')
            line_points = np.array([apex, base_center])
            line = pv.lines_from_points(line_points)
            plotter.add_mesh(line, color='cyan', line_width=3, label='Long Axis')
            plotter.add_legend()
            plotter.view_isometric()
            plotter.show()
            if i == total_planes:
                break
    else:
        plotter = pv.Plotter()
        plotter.set_background('gray')
        plotter.show_grid()
        cloud = pv.PolyData(points)
        plotter.add_points(cloud, color='blue', point_size=2, render_points_as_spheres=True, label='Original Points')
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan']
        for i, (center, normal, plane_points) in enumerate(zip(plane_centers, plane_normals, plane_points_list)):
            angle_deg = all_angles_deg[i]
            plane = pv.Plane(center=center, direction=normal, i_size=10, j_size=10)
            color = colors[i % len(colors)]
            plotter.add_mesh(plane, color=color, opacity=0.3, label=f'Plane Angle {angle_deg:.1f}°')
        apex_sphere = pv.Sphere(radius=0.5, center=apex)
        plotter.add_mesh(apex_sphere, color='red', label='Apex')
        base_sphere = pv.Sphere(radius=0.5, center=base_center)
        plotter.add_mesh(base_sphere, color='green', label='Base')
        line_points = np.array([apex, base_center])
        line = pv.lines_from_points(line_points)
        plotter.add_mesh(line, color='cyan', line_width=3, label='Long Axis')
        plotter.add_legend()
        plotter.view_isometric()
        plotter.show()

def visualize_all_contracted_planes(points, apex, base_center, long_axis, all_angles_deg, plane_centers, plane_normals, plane_points_list, top_n=1):
    cloud_center = (apex + base_center) / 2
    plotter = pv.Plotter()
    plotter.set_background('gray')
    plotter.show_grid()
    cloud = pv.PolyData(points)
    plotter.add_points(cloud, color='blue', point_size=3, render_points_as_spheres=True, label='Original Points')
    point_counts = [len(pts) for pts in plane_points_list]
    if top_n is not None and top_n < len(plane_centers):
        indices = sorted(range(len(point_counts)), key=lambda i: point_counts[i], reverse=True)
        selected_indices = indices[:top_n]
    else:
        selected_indices = range(len(plane_centers))
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan']
    for i in selected_indices:
        center = plane_centers[i]
        normal = plane_normals[i]
        plane_points = plane_points_list[i]
        angle_deg = all_angles_deg[i]
        plane = pv.Plane(center=center, direction=normal, i_size=10, j_size=10)
        color = colors[i % len(colors)]
        plotter.add_mesh(plane, color=color, opacity=0.8,
                         label=f'Plane Angle {angle_deg:.1f}° (Points: {point_counts[i]})')
    apex_sphere = pv.Sphere(radius=0.5, center=apex)
    plotter.add_mesh(apex_sphere, color='red', label='Apex')
    base_sphere = pv.Sphere(radius=0.5, center=base_center)
    plotter.add_mesh(base_sphere, color='green', label='Base')
    line_points = np.array([apex, base_center])
    line = pv.lines_from_points(line_points)
    plotter.add_mesh(line, color='cyan', line_width=3, label='Long Axis')
    plotter.add_legend()
    plotter.view_isometric()
    plotter.show()

def main(file_path, sampling_method, output_file, manu_path, visual=True):
    if manu_path is not None:
        manu_points = scipy.io.loadmat(manu_path)
        manu_points = np.array(manu_points['Positions_SelectedPoints']).T
    else: manu_points = None
    if sampling_method not in ['uniform', 'random']:
        raise ValueError("sampling_method must be 'uniform' or 'random'")
    points_myocardium, colors_myocardium, points_groove, colors_groove = load_point_cloud_from_txt(file_path)
    print(f"Loaded point cloud, myocardium points: {len(points_myocardium)}, groove points: {len(points_groove)}")
    points_myocardium, inliers = preprocess_point_cloud(points_myocardium)
    colors_myocardium = colors_myocardium[inliers]
    print(f"Myocardium points after denoising: {len(points_myocardium)}")
    xc, yc, r = find_xy_center(points_myocardium, visual=visual)
    apex = find_apex(points_myocardium, xc, yc)
    print(f"Apex: {apex}")
    base_center, plane_model = find_base(points_myocardium, xc, yc)
    print(f"Base: {base_center}")
    long_axis = compute_long_axis(apex, base_center)
    print(f"Horizontal Long Axis: {long_axis}")
    short_axes = compute_short_axes(apex, base_center, long_axis, num_slices=5)
    print("Short Axes:")
    for i, (center, normal) in enumerate(short_axes):
        print(f"Slice {i + 1} - Center: {center}, Normal: {normal}")
    best_plane, plane_points, angles_deg, plane_centers, plane_normals, plane_points_list = compute_contracted_planes(
        points_myocardium, apex, base_center, long_axis)
    visualize_results(manu_points, points_myocardium, colors_myocardium, points_groove, colors_groove, apex, base_center, long_axis, short_axes, best_plane, plane_points, sampling_method, output_file, visual=visual)

def SPECT_main(root_dir, p_name, visual=True, showmanu=False):
    patient = p_name
    file_path = rf"{root_dir}/{patient}/ijkspect.txt"
    sampling_method = 'uniform'
    if showmanu:
        manu_path = rf'path/to/manu_selectedpoints'
        for i in os.listdir(manu_path):
            if i.lower().replace(' ', '') == patient:
                manu_path = os.path.join(manu_path, i, 'SelectedPoints.mat')
    else: manu_path = None
    output_file = rf"{root_dir}/{patient}/sp_sampled_points.txt"
    main(file_path, sampling_method, output_file, manu_path, visual=visual)

