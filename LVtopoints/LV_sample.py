import pandas as pd
import pyvista as pv
import numpy as np
import os

def LV_main(patient):
    excel_path = r'path/to/all_data_info.xlsx'
    patient_name = patient
    for i in os.listdir(r'CTA/PATH'):
        s = i.replace(' ', '')
        if patient_name in s:
            for j in os.listdir(os.path.join(r'CTA/PATH', i, r'mySeg')):
                if 'LV.stl' in j:
                    stl_path = rf'CTA/PATH/{i}/mySeg/{j}'
    print(stl_path)
    output_txt = rf'id/{patient_name}/{patient_name}_LV_transformed.txt'
    if not os.path.exists(rf'id/{patient_name}'):
        os.makedirs(rf'id/{patient_name}')
    target_points = 5200
    df = pd.read_excel(excel_path, engine='openpyxl')
    namelist = df.iloc[:, 2].astype(str).str.strip()
    for i in range(len(namelist)):
        namelist[i] = namelist[i].replace(' ', '')
    index = namelist[namelist == patient_name].index
    if len(index) != 1:
        raise ValueError(f"Patient {patient_name} not found or multiple matches found")
    index = index[0]
    l0 = float(df.iloc[index, 8])
    p0 = float(df.iloc[index, 9])
    s0 = float(df.iloc[index, 10])
    ps = float(df.iloc[index, 11])
    as_ = float(df.iloc[index, 12])
    transf = np.array([
        [ps, 0, 0, l0],
        [0, ps, 0, p0],
        [0, 0, as_, s0],
        [0, 0, 0, 1]
    ])
    transf_inv = np.linalg.inv(transf)
    mesh = pv.read(stl_path)
    points = mesh.points
    if points.shape[0] > target_points:
        indices = np.random.choice(points.shape[0], target_points, replace=False)
        points_downsampled = points[indices]
    else:
        points_downsampled = points
    ones = np.ones((points_downsampled.shape[0], 1))
    lps = np.hstack((points_downsampled, ones))
    ijk = transf_inv @ lps.T
    ijk = ijk[:3, :].T
    np.savetxt(output_txt, ijk, fmt='%.6f', delimiter=' ')
    print(f"Transformed and downsampled point cloud saved to {output_txt}, points: {ijk.shape[0]}")

