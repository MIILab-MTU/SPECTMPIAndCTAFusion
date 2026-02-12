import os
import matlab.engine
import numpy as np


def eval_matlab(name, xlsx_dir, ori=False):
    # 启动 MATLAB 引擎
    eng = matlab.engine.start_matlab()
    mat_root_dir = r'F:/cta-spect/CTA-SPECT/code/code'
    eng.addpath(eng.genpath(mat_root_dir), nargout=0)

    # 设置基础目录
    base_dir = r'E:/proj/peizhun/file_process'

    # 确保患者名称是字符串
    patientname = str(name)

    try:
        # 调用 MATLAB 函数 compute_error
        # 返回 rmse, juli, jl（每个是长度为 4 的数组）
        rmse, juli, jl = eng.eval_python(xlsx_dir, base_dir, patientname,ori, nargout=3)

        # 转换为 NumPy 数组
        rmse = np.array(rmse).flatten()  # [icp_mse, sicp_mse, rigid_mse, affine_mse]
        juli = np.array(juli).flatten()  # 平均距离
        jl = np.array(jl).flatten()      # 校正距离

        # 提取 MSE 值
        icp_mse = rmse[0]
        sicp_mse = rmse[1]
        rigid_mse = rmse[2]
        affine_mse = rmse[3]

        # 提取平均距离
        icp_juli = juli[0]
        sicp_juli = juli[1]
        rigid_juli = juli[2]
        affine_juli = juli[3]

        # 提取校正距离
        icp_jl = jl[0]
        sicp_jl = jl[1]
        rigid_jl = jl[2]
        affine_jl = jl[3]
        if ori:
            print('-------手动标记点的结果-------')
        else:
            print('-------自动标记点的结果-------')

        # 打印所有指标
        print("均方误差 (MSE):")
        print(f"  ICP MSE: {icp_mse}")
        print(f"  SICP MSE: {sicp_mse}")
        print(f"  Rigid MSE: {rigid_mse}")
        print(f"  Affine MSE: {affine_mse}")
        print("平均距离 (juli):")
        print(f"  ICP juli: {icp_juli}")
        print(f"  SICP juli: {sicp_juli}")
        print(f"  Rigid juli: {rigid_juli}")
        print(f"  Affine juli: {affine_juli}")
        print("校正距离 (jl):")
        print(f"  ICP jl: {icp_jl}")
        print(f"  SICP jl: {sicp_jl}")
        print(f"  Rigid jl: {rigid_jl}")
        print(f"  Affine jl: {affine_jl}")

        # 返回 MSE 值（保持原函数签名）
        return icp_mse, sicp_mse, rigid_mse, affine_mse

    except matlab.engine.MatlabExecutionError as e:
        print(f"MATLAB 执行错误: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"其他错误: {e}")
        return None, None, None, None
    finally:
        # 关闭 MATLAB 引擎
        eng.quit()


# 示例调用（可选）
if __name__ == "__main__":
    patient_name = "zengyuming"
    xlsx_dir = r"E:/proj/peizhun/data/patient_data.xlsx"
    icp_mse, sicp_mse, rigid_mse, affine_mse = eval_matlab(patient_name, xlsx_dir)