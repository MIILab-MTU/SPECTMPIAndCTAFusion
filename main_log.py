import os
import matlab.engine
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matlab.engine
from preprocess.cta_pointcloud_pc import CTA_main
from preprocess.spect_pointcloud_pc import SPECT_main
from preprocess.test_428 import TEST_main
from to_matlab.methods import reg_matlab
from to_matlab.eval_log import eval_matlab
from pathlib import Path
import ast
from mat_code.demo_txt_nricp import main as demo_main


def write_list_to_txt(lst, txt_path, encoding="utf-8", newline="\n"):
    """
    把列表写入txt：每个元素一行（用 str()）
    """
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding=encoding, newline=newline) as f:
        for x in lst:
            f.write(f"{x}\n")

def read_txt_to_list_str(txt_path, encoding="utf-8"):
    """
    从txt读取：每行一个元素，返回字符串列表（去掉行尾换行）
    """
    txt_path = Path(txt_path)
    with txt_path.open("r", encoding=encoding) as f:
        return [line.rstrip("\n\r") for line in f]

def read_txt_to_list_typed(txt_path, encoding="utf-8"):
    """
    从txt读取：每行尝试用 literal_eval 还原类型（int/float/list/dict/tuple/str等）
    读不回就当字符串。
    """
    txt_path = Path(txt_path)
    out = []
    with txt_path.open("r", encoding=encoding) as f:
        for line in f:
            s = line.strip()
            if s == "":
                out.append("")  # 空行就当空字符串
                continue
            try:
                out.append(ast.literal_eval(s))
            except Exception:
                out.append(s)
    return out

def get_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                filename = os.path.splitext(file)[0]
                csv_files.append(filename)
    return csv_files

def check_files(patient_name, cloud_dir):
    stl_path = r'null'
    for i in os.listdir(r'data\CTA'):
        s = i.replace(' ', '')
        if patient_name in s and os.path.exists(rf'data\CTA\{i}\mySeg\Segmentation_RV.stl'):
            stl_path = rf'data\CTA\{i}\mySeg\Segmentation_RV.stl'
    if stl_path == 'null':
        print(rf'{patient_name}:没有右心文件，跳过')
        return False
    elif not os.path.exists(rf'preprocess\name\{patient_name}\ijkcta.txt'):
        print(rf'{patient_name}:没有cta点云文件，跳过')
        return False
    elif not os.path.exists(rf'preprocess\name\{patient_name}\ijkspect.txt'):
        print(rf'{patient_name}:没有spect点云文件，跳过')
        return False
    # elif not os.path.exists(rf'{cloud_dir}\{patient_name}\cu.txt'):
    #     print(rf'{patient_name}:没有cu文件，跳过')
    #     return False
    else:
        print(rf'{patient_name}:检查完成')
        return True



if __name__ == '__main__':
    spect_path = r'data/SPECT/id_final'
    cta_path = r'data/CTA'
    patient_all = get_csv_files(spect_path)


    xlsx_dir = r'RVtopoints/final_data.xlsx'

    results_file = r'preprocess/matric_212/eval_results.csv'
    os.makedirs(r'preprocess/matric_212', exist_ok=True)

    root_dir = r'preprocess/name'
    cloud_dir = r'preprocess/cloud_result212'
    os.makedirs(cloud_dir, exist_ok=True)



    visual = False
    # 记录被跳过的患者
    skipped_result = {
        'patient_name': None,
        'timestamp': None,
        'ori': None,
        'icp_mse': None, 'sicp_mse': None, 'rigid_mse': None, 'affine_mse': None, 'bcpdpp_mse': None,
        'icp_juli': None, 'sicp_juli': None, 'rigid_juli': None, 'affine_juli': None, 'bcpdpp_juli': None,
        'icp_jl': None, 'sicp_jl': None, 'rigid_jl': None, 'affine_jl': None, 'bcpdpp_jl': None,
        'error': None
    }

    for patients in tqdm(patient_all):
        print(f"\nProcessing patient: {patients}")
        if check_files(patients, cloud_dir):
            eng = matlab.engine.start_matlab()
            mat_root_dir = r'mat_code'
            eng.addpath(eng.genpath(mat_root_dir), nargout=0)
            eng.parallel.gpu.enableCUDAForwardCompatibility(True)


            # RV_main(patients, cta_path)
            # LV_main(patients, cta_path)
            CTA_main(root_dir, patients, re_LV=False, visual=visual)
            SPECT_main(root_dir, patients, visual=visual)
            TEST_main(root_dir, cloud_dir, patients, visual=False)
            demo_main(patients, cloud_dir)


            reg_matlab(root_dir, cloud_dir, patients, eng, False)
            #
            eval_matlab(root_dir, cloud_dir, patients, xlsx_dir, eng, False, results_file)

            print(f"Completed processing for {patients}")

        else:
            # 记录跳过的患者
            skipped_result.update({
                'patient_name': patients,
                'timestamp': datetime.now().isoformat(),
                'ori': 'N/A',
                'error': 'Skipped due to missing files'
            })
            result_df = pd.DataFrame([skipped_result])
            if os.path.exists(results_file):
                result_df.to_csv(results_file, mode='a', header=False, index=False)
            else:
                result_df.to_csv(results_file, mode='w', header=True, index=False)
            print(f"Skipped {patients} and recorded in {results_file}")
    eng.quit()
    print(f"\nAll results saved to {results_file}")