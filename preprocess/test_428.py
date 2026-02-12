import numpy as np
import open3d as o3d
import pyvista as pv

import copy
import os

def _to_numpy(pc):
    """统一把输入转成 Nx3 numpy 数组"""
    if hasattr(pc, 'points'):          # open3d 点云
        return np.asarray(pc.points)
    elif isinstance(pc, np.ndarray):   # 已是 numpy
        return pc
    else:
        raise TypeError("点云必须是 open3d.geometry.PointCloud 或 Nx3 numpy.ndarray")

def visualize_with_axes(pc1, pc2,
                        point_size=5,
                        pc1_color='red',
                        pc2_color='blue',
                        axes_digits=3):
    """
    可视化两个点云，并在坐标轴上显示带数字的标尺网格。

    参数
    ----
    pc1, pc2 : open3d.geometry.PointCloud 或 Nx3 numpy.ndarray
    point_size : int, 点的大小
    pc1_color, pc2_color : str, 颜色名字或 RGB 三元组
    axes_digits : int, 标尺数字保留的小数位
    """

    # 转 numpy
    pts1 = _to_numpy(pc1)
    pts2 = _to_numpy(pc2)

    # PyVista PolyData
    cloud1 = pv.PolyData(pts1)
    cloud2 = pv.PolyData(pts2)

    # 绘图
    plotter = pv.Plotter()
    plotter.add_mesh(cloud1, color=[216 / 255.0, 101 / 255.0, 79 / 255.0], point_size=point_size, render_points_as_spheres=True)
    plotter.add_mesh(cloud2, color=[128 / 255.0, 174 / 255.0, 128 / 255.0], point_size=point_size, render_points_as_spheres=True)



    # 可选：添加方向小坐标轴
    plotter.add_axes()
    plotter.background_color = 'white'
    plotter.show_bounds(grid=False, location='outer', color='black')
    plotter.show()


# 目前使用终版10_29

def array_2_pcd(array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array[:, 0:3])
    # pcd.colors = o3d.utility.Vector3dVector(array[:, 3:])
    return pcd

def TEST_main(input_dir, output_dir, patient, visual=True):
    patientname = patient

    # ijkctargb = np.loadtxt('E:/ME/Study/配准数据处理/左心/CTA/final/'+patientname+'/ijkcta.txt')
    ijkctargb = np.loadtxt(rf'{input_dir}/{patientname}/ijkcta.txt')
    cta_pcd = array_2_pcd(ijkctargb)
    # ijksprgb = np.loadtxt('E:/ME/Study/配准数据处理/左心/SPECT/id_final/Patient{:02d}/ijkspect.txt'.format(patientid))
    ijksprgb = np.loadtxt(rf'{input_dir}/{patientname}/ijkspect.txt')
    smspect_pcd = array_2_pcd(ijksprgb)
    # 计算协方差，缩小CTA和SPECT大小差距
    xc = np.mean(ijkctargb[:, 0:3], axis=0)
    yc = np.mean(ijksprgb[:, 0:3], axis=0)
    ctapts = ijkctargb[:, 0:3]
    spectpts = ijksprgb[:, 0:3]
    Mx = np.cov(ctapts.T)
    My = np.cov(spectpts.T)
    xa = np.linalg.eigvals(Mx)
    yb = np.linalg.eigvals(My)
    xa = np.sort(xa)
    yb = np.sort(yb)
    s = []
    for i in range(0, 3):
        s.append(np.sqrt(xa[i] / yb[i]))
    s0 = np.mean(np.array(s))
    trans_s = np.array([[s0, 0, 0, 0],
                    [0, s0, 0, 0],
                    [0, 0, s0, 0],
                    [0, 0, 0, 1]])
    print(trans_s)
    print("")
    if not os.path.exists(f'{output_dir}/'+patientname):
        os.mkdir(f'{output_dir}/'+patientname)
    np.savetxt(f'{output_dir}/'+patientname+'/cu_s.txt', trans_s)
    smspect_pcd.paint_uniform_color([216 / 255.0, 101 / 255.0, 79 / 255.0])
    cta_pcd.paint_uniform_color([128 / 255.0, 174 / 255.0, 128 / 255.0])
    if visual:

        visualize_with_axes(smspect_pcd, cta_pcd)

    # o3d.visualization.draw_geometries([smspect_pcd, cta_pcd])
    spect_pcd = smspect_pcd.transform(trans_s) #放大SPECT
    spect_yuan = copy.deepcopy(spect_pcd) #复制一份

    # np.savetxt(f'{output_dir}/' + patientname + '/cu.txt', np.asarray(spect_pcd.points))

    # o3d.visualization.draw_geometries([spect_pcd, cta_pcd])
    if visual:
        visualize_with_axes(spect_pcd, cta_pcd)
    # 加载特殊点
    ctate = np.loadtxt(rf'{input_dir}/{patientname}/sampled_points_{patientname}.txt')
    spte = np.loadtxt(rf'{input_dir}/{patientname}/sp_sampled_points.txt')
    # if not os.path.isfile(r'F:\cta-spect\CTA-SPECT\data_process\tedata/' + patientname + '/ctate.txt') or not os.path.isfile(r'F:\cta-spect\CTA-SPECT\data_process\tedata/' + patientname + '/nsptexin.txt'):
    #     return 0
    # ctate = np.loadtxt(r'F:\cta-spect\CTA-SPECT\data_process\tedata/' + patientname + '/ctate.txt')
    # spte = np.loadtxt(r'F:\cta-spect\CTA-SPECT\data_process\tedata/' + patientname + '/nsptexin.txt')


    ctte_pcd = o3d.geometry.PointCloud()
    ctte_pcd.points = o3d.utility.Vector3dVector(ctate[:, 0:3])
    if ctate.shape[1] > 3:
        ctte_pcd.colors = o3d.utility.Vector3dVector(ctate[:, 3:6] / 255.0)
    else:
        ctte_pcd.paint_uniform_color([0, 0, 1.0]) # 如果没有颜色，保持蓝色
    spte_pcd = o3d.geometry.PointCloud()
    spte_pcd.points = o3d.utility.Vector3dVector(spte[:, 0:3])
    if spte.shape[1] > 3:
        spte_pcd.colors = o3d.utility.Vector3dVector(spte[:, 3:6] / 255.0)
    else:
        spte_pcd.paint_uniform_color([1.0, 0, 0]) # 如果没有颜色，保持红色
    spte_pcd.transform(trans_s) #给SPECT特殊点也放大
    # transformed_spte = np.asarray(spte_pcd.points)
    # np.savetxt(f'{output_dir}/{patientname}/sp_manu_points_transformed.txt', transformed_spte)

    if visual:
        # o3d.visualization.draw_geometries([spte_pcd, ctte_pcd])
        visualize_with_axes(spte_pcd, ctte_pcd)
    corr = np.zeros((len(spte), 2)) #对应关系是编号
    corr[:, 0] = np.arange(len(spte))
    corr[:, 1] = np.arange(len(spte))
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(spte_pcd, ctte_pcd,
                                            o3d.utility.Vector2iVector(corr))
    print(trans_init) #打印 变换矩阵
    print("")
    spte_pcd.transform(trans_init) # SPECT特殊点应用变换矩阵
    if visual:
        # o3d.visualization.draw_geometries([spte_pcd, ctte_pcd])
        visualize_with_axes(spte_pcd,ctte_pcd)

    np.savetxt(f'{output_dir}/'+patientname+'/cu_tfm.txt', trans_init)

    spect_pcd.transform(trans_init) #SPECT应用变换矩阵
    if visual:
        spect_pcd.paint_uniform_color([216/255.0, 101/255.0, 79/255.0])
        cta_pcd.paint_uniform_color([128/255.0, 174/255.0, 128/255.0])
        # o3d.visualization.draw_geometries([spect_pcd, cta_pcd])
        visualize_with_axes(spect_pcd, cta_pcd)
    cu = np.asarray(spect_pcd.points)
    # print(cu)

    np.savetxt(f'{output_dir}/'+patientname+'/cu.txt', cu)
    # # # 保存变换后的SPECT特殊点
    # transformed_spte = np.asarray(spte_pcd.points)
    # np.savetxt(f'{output_dir}/{patientname}/sp_manu_points_transformed.txt', transformed_spte)
