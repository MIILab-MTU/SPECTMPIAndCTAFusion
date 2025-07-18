import os
import matlab.engine
import numpy as np
import pandas as pd
from datetime import datetime

def eval_matlab(base_dir, cloud_dir, name, xlsx_dir, ori=False, results_file="eval_results.csv"):
    eng = matlab.engine.start_matlab()
    mat_root_dir = r'../methods'
    eng.addpath(eng.genpath(mat_root_dir), nargout=0)
    patientname = str(name)
    result = {
        'patient_name': patientname,
        'timestamp': datetime.now().isoformat(),
        'ori': 'manual' if ori else 'auto',
        'icp_mse': None, 'sicp_mse': None, 'rigid_mse': None, 'affine_mse': None, 'bcpdpp_mse': None,
        'icp_juli': None, 'sicp_juli': None, 'rigid_juli': None, 'affine_juli': None, 'bcpdpp_juli': None,
        'icp_jl': None, 'sicp_jl': None, 'rigid_jl': None, 'affine_jl': None, 'bcpdpp_jl': None,
        'error': None
    }
    try:
        rmse, juli, jl = eng.eval_python(xlsx_dir, base_dir, cloud_dir, patientname, ori, nargout=3)
        rmse = np.array(rmse).flatten()
        juli = np.array(juli).flatten()
        jl = np.array(jl).flatten()
        result.update({
            'icp_mse': rmse[0], 'sicp_mse': rmse[1], 'rigid_mse': rmse[2], 'affine_mse': rmse[3], 'bcpdpp_mse': rmse[4],
            'icp_juli': juli[0], 'sicp_juli': juli[1], 'rigid_juli': juli[2], 'affine_juli': juli[3], 'bcpdpp_juli': juli[4],
            'icp_jl': jl[0], 'sicp_jl': jl[1], 'rigid_jl': jl[2], 'affine_jl': jl[3], 'bcpdpp_jl': jl[4]
        })
        print(f"{'Manual' if ori else 'Automatic'} labeled points result for {patientname}:")
        print("Mean Square Error (MSE):")
        print(f"  ICP MSE: {result['icp_mse']}")
        print(f"  SICP MSE: {result['sicp_mse']}")
        print(f"  Rigid MSE: {result['rigid_mse']}")
        print(f"  Affine MSE: {result['affine_mse']}")
        print(f"  Bcpdpp MSE: {result['bcpdpp_mse']}")
        print("Average Distance (juli):")
        print(f"  ICP juli: {result['icp_juli']}")
        print(f"  SICP juli: {result['sicp_juli']}")
        print(f"  Rigid juli: {result['rigid_juli']}")
        print(f"  Affine juli: {result['affine_juli']}")
        print(f"  Bcpdpp juli: {result['bcpdpp_juli']}")
        print("Correction Distance (jl):")
        print(f"  ICP jl: {result['icp_jl']}")
        print(f"  SICP jl: {result['sicp_jl']}")
        print(f"  Rigid jl: {result['rigid_jl']}")
        print(f"  Affine jl: {result['affine_jl']}")
        print(f"  Bcpdpp jl: {result['bcpdpp_jl']}")
    except matlab.engine.MatlabExecutionError as e:
        print(f"MATLAB execution error for {patientname}: {e}")
        result['error'] = str(e)
    except Exception as e:
        print(f"Other error for {patientname}: {e}")
        result['error'] = str(e)
    finally:
        eng.quit()
    result_df = pd.DataFrame([result])
    if os.path.exists(results_file):
        result_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        result_df.to_csv(results_file, mode='w', header=True, index=False)
    print(f"Result saved to {results_file}")
    return result['icp_mse'], result['sicp_mse'], result['rigid_mse'], result['affine_mse'], result['bcpdpp_mse']

def eval_gt_matlab(name, xlsx_dir, ori=False, results_file="eval_results.csv"):
    eng = matlab.engine.start_matlab()
    mat_root_dir = r'../methods'
    eng.addpath(eng.genpath(mat_root_dir), nargout=0)
    base_dir = r'../file_process'
    patientname = str(name)
    result = {
        'patient_name': patientname,
        'timestamp': datetime.now().isoformat(),
        'ori': 'manual' if ori else 'auto',
        'icp_mse': None, 'sicp_mse': None, 'rigid_mse': None, 'affine_mse': None, 'bcpdpp_mse': None,
        'icp_juli': None, 'sicp_juli': None, 'rigid_juli': None, 'affine_juli': None, 'bcpdpp_juli': None,
        'icp_jl': None, 'sicp_jl': None, 'rigid_jl': None, 'affine_jl': None, 'bcpdpp_jl': None,
        'error': None
    }
    try:
        rmse, juli, jl = eng.eval_gt_python(xlsx_dir, base_dir, patientname, ori, nargout=3)
        rmse = np.array(rmse).flatten()
        juli = np.array(juli).flatten()
        jl = np.array(jl).flatten()
        result.update({
            'icp_mse': rmse[0], 'sicp_mse': rmse[1], 'rigid_mse': rmse[2], 'affine_mse': rmse[3], 'bcpdpp_mse': rmse[4],
            'icp_juli': juli[0], 'sicp_juli': juli[1], 'rigid_juli': juli[2], 'affine_juli': juli[3], 'bcpdpp_juli': juli[4],
            'icp_jl': jl[0], 'sicp_jl': jl[1], 'rigid_jl': jl[2], 'affine_jl': jl[3], 'bcpdpp_jl': jl[4]
        })
        print(f"{'Manual' if ori else 'Automatic'} labeled points result for {patientname}:")
        print("Mean Square Error (MSE):")
        print(f"  ICP MSE: {result['icp_mse']}")
        print(f"  SICP MSE: {result['sicp_mse']}")
        print(f"  Rigid MSE: {result['rigid_mse']}")
        print(f"  Affine MSE: {result['affine_mse']}")
        print(f"  Bcpdpp MSE: {result['bcpdpp_mse']}")
        print("Average Distance (juli):")
        print(f"  ICP juli: {result['icp_juli']}")
        print(f"  SICP juli: {result['sicp_juli']}")
        print(f"  Rigid juli: {result['rigid_juli']}")
        print(f"  Affine juli: {result['affine_juli']}")
        print(f"  Bcpdpp juli: {result['bcpdpp_juli']}")
        print("Correction Distance (jl):")
        print(f"  ICP jl: {result['icp_jl']}")
        print(f"  SICP jl: {result['sicp_jl']}")
        print(f"  Rigid jl: {result['rigid_jl']}")
        print(f"  Affine jl: {result['affine_jl']}")
        print(f"  Bcpdpp jl: {result['bcpdpp_jl']}")
    except matlab.engine.MatlabExecutionError as e:
        print(f"MATLAB execution error for {patientname}: {e}")
        result['error'] = str(e)
    except Exception as e:
        print(f"Other error for {patientname}: {e}")
        result['error'] = str(e)
    finally:
        eng.quit()
    result_df = pd.DataFrame([result])
    if os.path.exists(results_file):
        result_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        result_df.to_csv(results_file, mode='w', header=True, index=False)
    print(f"Result saved to {results_file}")
    return result['icp_mse'], result['sicp_mse'], result['rigid_mse'], result['affine_mse'], result['bcpdpp_mse']