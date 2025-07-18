import os
import matlab.engine
import numpy as np

def reg_matlab(input_dir, cloud_dir, patients, ori=False):
    eng = matlab.engine.start_matlab()
    name = patients
    mat_root_dir = r'../methods'
    eng.addpath(eng.genpath(mat_root_dir), nargout=0)
    save_dir = rf'{cloud_dir}/{name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_affine, affine_s2 = eng.cpd_affine_python(input_dir, cloud_dir, name, ori, nargout=2)
    result_rigid, rigid_s2 = eng.cpd_rigid_python(input_dir, cloud_dir, name, ori, nargout=2)
    result_icp, icp_err = eng.icp_python(input_dir, cloud_dir, name, ori, nargout=2)
    result_sicp, sicp_err = eng.sicp_python(input_dir, cloud_dir, name, ori, nargout=2)
    result_bcpdpp = eng.bcpdpp_python(input_dir, name, ori, save_dir, nargout=1)
    result_icp = np.array(result_icp).T
    result_sicp = np.array(result_sicp)
    if ori:
        np.savetxt(os.path.join(save_dir, 'result_affine_manu.txt'), result_affine)
        np.savetxt(os.path.join(save_dir, 'result_rigid_manu.txt'), result_rigid)
        np.savetxt(os.path.join(save_dir, 'result_icp_manu.txt'), result_icp)
        np.savetxt(os.path.join(save_dir, 'result_sicp_manu.txt'), result_sicp)
        bcpdpp_txt = os.path.join(save_dir, 'result_bcpdpp_manuy.txt')
    else:
        np.savetxt(os.path.join(save_dir, 'result_affine_auto.txt'), result_affine)
        np.savetxt(os.path.join(save_dir, 'result_rigid_auto.txt'), result_rigid)
        np.savetxt(os.path.join(save_dir, 'result_icp_auto.txt'), result_icp)
        np.savetxt(os.path.join(save_dir, 'result_sicp_auto.txt'), result_sicp)
        bcpdpp_txt = os.path.join(save_dir, 'result_bcpdpp_autoy.txt')
    eng.quit()
