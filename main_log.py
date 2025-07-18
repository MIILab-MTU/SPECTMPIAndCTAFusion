import os
import matlab.engine
import numpy as np
import pandas as pd
from datetime import datetime
from coarse_registration.CTA import CTA_main
from coarse_registration.SPECT import SPECT_main
from RVtopoints.sample import RV_main
from coarse_registration.test_428 import TEST_main
from LVtopoints.LV_sample import LV_main
from to_matlab.methods import reg_matlab
from to_matlab.eval_log import eval_matlab, eval_gt_matlab

def get_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                filename = os.path.splitext(file)[0]
                csv_files.append(filename)
    return csv_files

def check_files(patient_name):
    stl_path = r'null'
    for i in os.listdir(r'CTA/PATH'):
        s = i.replace(' ', '')
        if patient_name in s and os.path.exists(rf'CTA/PATH/{i}/mySeg/Segmentation_RV.stl'):
            stl_path = rf'CTA/PATH/{i}/mySeg/Segmentation_RV.stl'
    if stl_path == 'null':
        print(f'{patient_name}: No right ventricle file, skipping')
        return False
    elif not os.path.exists(rf'fileprocess\id\{patient_name}\ijkcta.txt'):
        print(f'{patient_name}: No CTA point cloud file, skipping')
        return False
    elif not os.path.exists(rf'fileprocess\id\{patient_name}\ijkspect.txt'):
        print(f'{patient_name}: No SPECT point cloud file, skipping')
        return False
    elif not os.path.exists(rf'PATH/to/manu/{patient_name}/cu.txt'):
        print(f'{patient_name}: No cu file, skipping')
        return False
    else:
        print(f'{patient_name}: Check completed')
        return True

if __name__ == '__main__':
    patient_all = get_csv_files(r'path/to/all_data_info.xlsx')
    xlsx_dir = r'path/to/all_data_info.xlsx'
    results_file = r'file_process/matric/eval_results_bcpdpp.csv'
    visual = True
    root_dir = r'file_process\id'
    cloud_dir = r'file_process\cloudpoints_result'
    skipped_result = {
        'patient_name': None,
        'timestamp': None,
        'ori': None,
        'icp_mse': None, 'sicp_mse': None, 'rigid_mse': None, 'affine_mse': None, 'bcpdpp_mse': None,
        'icp_juli': None, 'sicp_juli': None, 'rigid_juli': None, 'affine_juli': None, 'bcpdpp_juli': None,
        'icp_jl': None, 'sicp_jl': None, 'rigid_jl': None, 'affine_jl': None, 'bcpdpp_jl': None,
        'error': None
    }
    for patients in patient_all:
        print(f"\nProcessing patient: {patients}")
        if check_files(patients):
            RV_main(patients)
            # LV_main(patients)
            CTA_main(root_dir, patients, re_LV=False, visual=visual)
            SPECT_main(root_dir, patients, visual=visual, showmanu=False)
            TEST_main(root_dir, cloud_dir, patients, visual=visual)
            reg_matlab(root_dir, cloud_dir, patients, False)
            # reg_matlab(patients, True)
            eval_matlab(root_dir, cloud_dir, patients, xlsx_dir, False, results_file)
            # eval_matlab(patients, xlsx_dir, patients, xlsx_dir,True, results_file)
            # eval_gt_matlab(patients, xlsx_dir, False, results_file)
            print(f"Completed processing for {patients}")
        else:
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
    print(f"\nAll results saved to {results_file}")