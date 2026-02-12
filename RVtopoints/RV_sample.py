import pandas as pd
import pyvista as pv
import numpy as np
import os
def RV_main(patient, CTA_path):

    # 路径和病人姓名
    excel_path = r'RVtopoints/finaldata.xlsx'


    patient_name = patient

    for i in os.listdir(CTA_path):
        s = i.replace(' ', '')
        if patient_name in s:
            stl_path = rf'{CTA_path}/{i}/mySeg/Segmentation_RV.stl'
    print(stl_path)


    # p_name = patient_name.replace(" ", "")
    output_txt = rf'RVtopoints/name/{patient_name}/{patient_name}_RV_transformed.txt'


    os.makedirs(rf'RVtopoints/name/{patient_name}', exist_ok=True)
    target_points = 5200  # 目标点数

    # 1. 读取Excel文件
    df = pd.read_excel(excel_path, engine='openpyxl')

    # 2. 查找病人
    namelist = df.iloc[:, 2].astype(str).str.strip()  # 假设第3列（索引2）是病人姓名
    for i in range(len(namelist)):
        namelist[i] = namelist[i].replace(' ', '')

    index = namelist[namelist == patient_name].index


    if len(index) != 1:
        raise ValueError(f"病人 {patient_name} 未找到或找到多个匹配项")
    index = index[0]

    # 3. 提取变换参数
    l0 = float(df.iloc[index, 8])  # 第9列（索引8）
    p0 = float(df.iloc[index, 9])  # 第10列（索引9）
    s0 = float(df.iloc[index, 10]) # 第11列（索引10）
    ps = float(df.iloc[index, 11]) # 第12列（索引11）
    as_ = float(df.iloc[index, 12]) # 第13列（索引12）

    # 4. 构建变换矩阵
    transf = np.array([
        [ps, 0, 0, l0],
        [0, ps, 0, p0],
        [0, 0, as_, s0],
        [0, 0, 0, 1]
    ])
    transf_inv = np.linalg.inv(transf)

    # 5. 加载STL文件并提取点云
    mesh = pv.read(stl_path)
    points = mesh.points  # 获取顶点坐标（N x 3）

    # 6. 降采样到5200个点
    if points.shape[0] > target_points:
        indices = np.random.choice(points.shape[0], target_points, replace=False)
        points_downsampled = points[indices]
    else:
        points_downsampled = points

    # 7. 转换为齐次坐标
    ones = np.ones((points_downsampled.shape[0], 1))
    lps = np.hstack((points_downsampled, ones))  # N x 4

    # 8. 应用逆变换
    ijk = transf_inv @ lps.T  # 4 x N
    ijk = ijk[:3, :].T  # N x 3，提取x, y, z坐标

    # 9. 保存为TXT文件
    np.savetxt(output_txt, ijk, fmt='%.6f', delimiter=' ')
    print(f"变换并降采样后的点云已保存到 {output_txt}，点数：{ijk.shape[0]}")


if __name__ == '__main__':
    patient = r'guozuyu'
    RV_main(patient)