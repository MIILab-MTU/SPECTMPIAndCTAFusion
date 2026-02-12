import os
import matlab.engine
import numpy as np
import pandas as pd
from datetime import datetime

def eval_matlab(base_dir, cloud_dir, name, xlsx_dir,eng, ori=False, results_file="eval_results.csv"):
    # 启动 MATLAB 引擎


    # 设置基础目录
    # base_dir = r'E:/proj/peizhun/file_process'

    # 确保患者名称是字符串
    patientname = str(name)

    # 初始化结果字典
    result = {
        'patient_name': patientname,
        'timestamp': datetime.now().isoformat(),
        'ori': 'manual' if ori else 'auto',
        'icp_mse': None, 'sicp_mse': None, 'rigid_mse': None, 'affine_mse': None,'bcpdpp_mse': None,'psr_mse': None,'ffd_mse': None,
        'icp_juli': None, 'sicp_juli': None, 'rigid_juli': None, 'affine_juli': None,'bcpdpp_juli': None,'psr_juli': None,'ffd_juli': None,
        'icp_jl': None, 'sicp_jl': None, 'rigid_jl': None, 'affine_jl': None,'bcpdpp_jl': None,'psr_jl': None,'ffd_jl': None,
        'error': None
    }

    try:
        # 调用 MATLAB 函数 compute_error
        rmse, juli, jl = eng.eval_python(xlsx_dir, base_dir, cloud_dir, patientname, ori, nargout=3)

        # 转换为 NumPy 数组
        rmse = np.array(rmse).flatten()
        juli = np.array(juli).flatten()
        jl = np.array(jl).flatten()

        # 填充结果
        result.update({
            'icp_mse': rmse[0], 'sicp_mse': rmse[1], 'rigid_mse': rmse[2], 'affine_mse': rmse[3],'bcpdpp_mse': rmse[4],'psr_mse': rmse[5],'ffd_mse': rmse[6],
            'icp_juli': juli[0], 'sicp_juli': juli[1], 'rigid_juli': juli[2], 'affine_juli': juli[3],'bcpdpp_juli': juli[4],'psr_juli': juli[5],'ffd_juli': juli[6],
            'icp_jl': jl[0], 'sicp_jl': jl[1], 'rigid_jl': jl[2], 'affine_jl': jl[3], 'bcpdpp_jl': jl[4], 'psr_jl': jl[5],'ffd_jl': jl[6],
        })

        # 打印结果
        print(f"{'手动' if ori else '自动'}标记点结果 for {patientname}:")
        print("均方误差 (MSE):")
        print(f"  ICP MSE: {result['icp_mse']}")
        print(f"  SICP MSE: {result['sicp_mse']}")
        print(f"  Rigid MSE: {result['rigid_mse']}")
        print(f"  Affine MSE: {result['affine_mse']}")
        print(f"  Bcpdpp MSE: {result['bcpdpp_mse']}")
        print(f"  Psr MSE: {result['psr_mse']}")
        print(f"  FFD MSE: {result['ffd_mse']}")
        print("平均距离 (juli):")
        print(f"  ICP juli: {result['icp_juli']}")
        print(f"  SICP juli: {result['sicp_juli']}")
        print(f"  Rigid juli: {result['rigid_juli']}")
        print(f"  Affine juli: {result['affine_juli']}")
        print(f"  Bcpdpp juli: {result['bcpdpp_juli']}")
        print(f"  Psr juli: {result['psr_juli']}")
        print(f"  FFD juli: {result['ffd_juli']}")
        print("校正距离 (jl):")
        print(f"  ICP jl: {result['icp_jl']}")
        print(f"  SICP jl: {result['sicp_jl']}")
        print(f"  Rigid jl: {result['rigid_jl']}")
        print(f"  Affine jl: {result['affine_jl']}")
        print(f"  Bcpdpp jl: {result['bcpdpp_jl']}")
        print(f"  Psr jl: {result['psr_jl']}")
        print(f"  FFD jl: {result['ffd_jl']}")
        eng.quit()

    except matlab.engine.MatlabExecutionError as e:
        print(f"MATLAB 执行错误 for {patientname}: {e}")
        result['error'] = str(e)
    except Exception as e:
        print(f"其他错误 for {patientname}: {e}")
        result['error'] = str(e)
    finally:
        # 关闭 MATLAB 引擎
        print('final')

    # 记录结果到 CSV
    result_df = pd.DataFrame([result])
    if os.path.exists(results_file):
        result_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        result_df.to_csv(results_file, mode='w', header=True, index=False)
    print(f"Result saved to {results_file}")

    return result['icp_mse'], result['sicp_mse'], result['rigid_mse'], result['affine_mse'], result['bcpdpp_mse'], result['psr_mse'], result['ffd_mse']



def eval_gt_matlab(name, xlsx_dir, ori=False, results_file="eval_results.csv"):
    # 启动 MATLAB 引擎
    eng = matlab.engine.start_matlab()
    mat_root_dir = r'F:/cta-spect/CTA-SPECT/code/code'
    eng.addpath(eng.genpath(mat_root_dir), nargout=0)

    # 设置基础目录
    base_dir = r'E:/proj/peizhun/file_process'

    # 确保患者名称是字符串
    patientname = str(name)

    # 初始化结果字典
    result = {
        'patient_name': patientname,
        'timestamp': datetime.now().isoformat(),
        'ori': 'manual' if ori else 'auto',
        'icp_mse': None, 'sicp_mse': None, 'rigid_mse': None, 'affine_mse': None,'bcpdpp_mse': None,
        'icp_juli': None, 'sicp_juli': None, 'rigid_juli': None, 'affine_juli': None,'bcpdpp_juli': None,
        'icp_jl': None, 'sicp_jl': None, 'rigid_jl': None, 'affine_jl': None,'bcpdpp_jl': None,
        'error': None
    }

    try:
        # 调用 MATLAB 函数 compute_error
        rmse, juli, jl = eng.eval_gt_python(xlsx_dir, base_dir, patientname, ori, nargout=3)

        # 转换为 NumPy 数组
        rmse = np.array(rmse).flatten()
        juli = np.array(juli).flatten()
        jl = np.array(jl).flatten()

        # 填充结果
        result.update({
            'icp_mse': rmse[0], 'sicp_mse': rmse[1], 'rigid_mse': rmse[2], 'affine_mse': rmse[3],'bcpdpp_mse': rmse[4],
            'icp_juli': juli[0], 'sicp_juli': juli[1], 'rigid_juli': juli[2], 'affine_juli': juli[3],'bcpdpp_juli': juli[4],
            'icp_jl': jl[0], 'sicp_jl': jl[1], 'rigid_jl': jl[2], 'affine_jl': jl[3], 'bcpdpp_jl': jl[4]
        })

        # 打印结果
        print(f"{'手动' if ori else '自动'}标记点结果 for {patientname}:")
        print("均方误差 (MSE):")
        print(f"  ICP MSE: {result['icp_mse']}")
        print(f"  SICP MSE: {result['sicp_mse']}")
        print(f"  Rigid MSE: {result['rigid_mse']}")
        print(f"  Affine MSE: {result['affine_mse']}")
        print(f"  Bcpdpp MSE: {result['bcpdpp_mse']}")
        print("平均距离 (juli):")
        print(f"  ICP juli: {result['icp_juli']}")
        print(f"  SICP juli: {result['sicp_juli']}")
        print(f"  Rigid juli: {result['rigid_juli']}")
        print(f"  Affine juli: {result['affine_juli']}")
        print(f"  Bcpdpp juli: {result['bcpdpp_juli']}")
        print("校正距离 (jl):")
        print(f"  ICP jl: {result['icp_jl']}")
        print(f"  SICP jl: {result['sicp_jl']}")
        print(f"  Rigid jl: {result['rigid_jl']}")
        print(f"  Affine jl: {result['affine_jl']}")
        print(f"  Bcpdpp jl: {result['bcpdpp_jl']}")

    except matlab.engine.MatlabExecutionError as e:
        print(f"MATLAB 执行错误 for {patientname}: {e}")
        result['error'] = str(e)
    except Exception as e:
        print(f"其他错误 for {patientname}: {e}")
        result['error'] = str(e)
    finally:
        # 关闭 MATLAB 引擎
        eng.quit()

    # 记录结果到 CSV
    result_df = pd.DataFrame([result])
    if os.path.exists(results_file):
        result_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        result_df.to_csv(results_file, mode='w', header=True, index=False)
    print(f"Result saved to {results_file}")

    return result['icp_mse'], result['sicp_mse'], result['rigid_mse'], result['affine_mse'], result['bcpdpp_mse']