import numpy as np
import open3d as o3d
import copy
import os

def array_2_pcd(array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array[:, 0:3])
    return pcd

def TEST_main(input_dir, output_dir, patient, visual=True):
    patientname = patient
    ijkctargb = np.loadtxt(rf'{input_dir}/{patientname}/ijkcta.txt')
    cta_pcd = array_2_pcd(ijkctargb)
    ijksprgb = np.loadtxt(rf'{input_dir}/{patientname}/ijkspect.txt')
    smspect_pcd = array_2_pcd(ijksprgb)
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
    o3d.visualization.draw_geometries([smspect_pcd, cta_pcd])
    spect_pcd = smspect_pcd.transform(trans_s)
    spect_yuan = copy.deepcopy(spect_pcd)
    o3d.visualization.draw_geometries([spect_pcd, cta_pcd])
    ctate = np.loadtxt(rf'{input_dir}/{patientname}/sampled_points_{patientname}.txt')
    spte = np.loadtxt(rf'{input_dir}/{patientname}/sp_sampled_points.txt')
    ctte_pcd = o3d.geometry.PointCloud()
    ctte_pcd.points = o3d.utility.Vector3dVector(ctate[:, 0:3])
    if ctate.shape[1] > 3:
        ctte_pcd.colors = o3d.utility.Vector3dVector(ctate[:, 3:6] / 255.0)
    else:
        ctte_pcd.paint_uniform_color([0, 0, 1.0])
    spte_pcd = o3d.geometry.PointCloud()
    spte_pcd.points = o3d.utility.Vector3dVector(spte[:, 0:3])
    if spte.shape[1] > 3:
        spte_pcd.colors = o3d.utility.Vector3dVector(spte[:, 3:6] / 255.0)
    else:
        spte_pcd.paint_uniform_color([1.0, 0, 0])
    spte_pcd.transform(trans_s)
    if visual:
        o3d.visualization.draw_geometries([spte_pcd, ctte_pcd])
    corr = np.zeros((len(spte), 2))
    corr[:, 0] = np.arange(len(spte))
    corr[:, 1] = np.arange(len(spte))
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(spte_pcd, ctte_pcd,
                                            o3d.utility.Vector2iVector(corr))
    print(trans_init)
    print("")
    spte_pcd.transform(trans_init)
    if visual:
        o3d.visualization.draw_geometries([spte_pcd, ctte_pcd])
    np.savetxt(f'{output_dir}/'+patientname+'/cu_tfm.txt', trans_init)
    spect_pcd.transform(trans_init)
    if visual:
        o3d.visualization.draw_geometries([spect_pcd, cta_pcd])
    cu = np.asarray(spect_pcd.points)
    np.savetxt(f'{output_dir}/'+patientname+'/cu.txt', cu)