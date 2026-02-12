import numpy as np
import open3d as o3d
import copy
import os

# 目前使用终版10_29

def array_2_pcd(array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array[:, 0:3])
    # pcd.colors = o3d.utility.Vector3dVector(array[:, 3:])
    return pcd


patientname = 'zhangxiliang'
patientid = 72
ijkctargb = np.loadtxt('E:/ME/Study/配准数据处理/左心/CTA/final/'+patientname+'/ijkcta.txt')
cta_pcd = array_2_pcd(ijkctargb)
ijksprgb = np.loadtxt('E:/ME/Study/配准数据处理/左心/SPECT/id_final/Patient{:02d}/ijkspect.txt'.format(patientid))
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

#os.mkdir('./test10_29/'+patientname)
np.savetxt('./test10_29/'+patientname+'/cu_s.txt', trans_s)

spect_pcd = smspect_pcd.transform(trans_s) #放大SPECT
spect_yuan = copy.deepcopy(spect_pcd) #复制一份

# 加载特殊点
ctate = np.loadtxt('E:/ME/Study/配准数据处理/左心/tedata/'+patientname+'/ctate.txt')
spte = np.loadtxt('E:/ME/Study/配准数据处理/左心/tedata/'+patientname+'/nsptexin.txt')
ctte_pcd = o3d.geometry.PointCloud()
ctte_pcd.points = o3d.utility.Vector3dVector(ctate)
ctte_pcd.paint_uniform_color([0, 0, 1.0])
spte_pcd = o3d.geometry.PointCloud()
spte_pcd.points = o3d.utility.Vector3dVector(spte)
spte_pcd.paint_uniform_color([1.0, 0, 0])
spte_pcd.transform(trans_s) #给SPECT特殊点也放大
o3d.visualization.draw_geometries([spte_pcd, ctte_pcd])

corr = np.zeros((len(spte), 2)) #对应关系是编号
corr[:, 0] = np.arange(len(spte))
corr[:, 1] = np.arange(len(spte))
p2p = o3d.registration.TransformationEstimationPointToPoint()
trans_init = p2p.compute_transformation(spte_pcd, ctte_pcd,
                                        o3d.utility.Vector2iVector(corr))
print(trans_init) #打印 变换矩阵
print("")
spte_pcd.transform(trans_init) # SPECT特殊点应用变换矩阵
o3d.visualization.draw_geometries([spte_pcd, ctte_pcd])

np.savetxt('./test10_29/'+patientname+'/cu_tfm.txt', trans_init)

spect_pcd.transform(trans_init) #SPECT应用变换矩阵
o3d.visualization.draw_geometries([spect_pcd, cta_pcd])

cu = np.asarray(spect_pcd.points)
# print(cu)

np.savetxt('./test10_29/'+patientname+'/cu.txt', cu)
