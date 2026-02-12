import os
import matlab.engine
import numpy as np
from pathlib import Path


def reg_matlab(input_dir, cloud_dir, patients, eng, ori=False):

    name = patients
    save_dir = rf'{cloud_dir}/{name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result_affine,affine_R, affine_T, affine_s2 = eng.cpd_affine_python(input_dir, cloud_dir, name,ori, nargout=4) #shoudong True, zidong False
    result_rigid, rigid_s, rigid_R, rigid_T, rigid_s2 = eng.cpd_rigid_python(input_dir, cloud_dir, name,ori, nargout=5)

    result_icp, icp_tfm, icp_err = eng.icp_python(input_dir, cloud_dir, name,ori, nargout=3)
    result_sicp,sicp_s,sicp_R,sicp_T, sicp_err = eng.sicp_python(input_dir, cloud_dir, name,ori, nargout=5)
    result_bcpdpp = eng.bcpdpp_python(input_dir, name, ori, save_dir, nargout=1)  # shoudong True, zidong False
    # save_dir = Path(save_dir).expanduser().resolve()
    eng.psr_python2(input_dir, name, save_dir, nargout=0)

    rigid_s = np.array([rigid_s])
    result_icp = np.array(result_icp).T
    result_sicp = np.array(result_sicp)

    if ori:
        np.savetxt(os.path.join(save_dir, 'result_affine_manu.txt'), result_affine)
        # np.savetxt(os.path.join(save_dir, 'result_rigid_manu.txt'), result_rigid)
        # np.savetxt(os.path.join(save_dir, 'result_rigid_s.txt'), sicp_s)
        # np.savetxt(os.path.join(save_dir, 'result_rigid_R.txt'), sicp_R)
        # np.savetxt(os.path.join(save_dir, 'result_rigid_T.txt'), sicp_T)
        # np.savetxt(os.path.join(save_dir, 'result_icp_manu.txt'), result_icp)
        # np.savetxt(os.path.join(save_dir, 'result_sicp_manu.txt'), result_sicp)
        # np.savetxt(os.path.join(save_dir, 'result_sicp_s.txt'), sicp_s)
        # np.savetxt(os.path.join(save_dir, 'result_sicp_R.txt'), sicp_R)
        # np.savetxt(os.path.join(save_dir, 'result_sicp_T.txt'), sicp_T)


        # bcpdpp_txt = os.path.join(save_dir, 'result_bcpdpp_manuy.txt')
        # if not os.path.exists(bcpdpp_txt):
        #     np.savetxt(bcpdpp_txt, result_bcpdpp)
    else:
        # print( )
        np.savetxt(os.path.join(save_dir, 'result_affine_auto.txt'), result_affine)
        np.savetxt(os.path.join(save_dir, 'result_affine_R.txt'), affine_R)
        np.savetxt(os.path.join(save_dir, 'result_affine_T.txt'), affine_T)
        np.savetxt(os.path.join(save_dir, 'result_rigid_auto.txt'), result_rigid)
        np.savetxt(os.path.join(save_dir, 'result_rigid_s.txt'), rigid_s)
        np.savetxt(os.path.join(save_dir, 'result_rigid_R.txt'), rigid_R)
        np.savetxt(os.path.join(save_dir, 'result_rigid_T.txt'), rigid_T)
        np.savetxt(os.path.join(save_dir, 'result_icp_auto.txt'), result_icp)
        np.savetxt(os.path.join(save_dir, 'result_sicp_auto.txt'), result_sicp)
        np.savetxt(os.path.join(save_dir, 'result_sicp_s.txt'), sicp_s)
        np.savetxt(os.path.join(save_dir, 'result_sicp_R.txt'), sicp_R)
        np.savetxt(os.path.join(save_dir, 'result_sicp_T.txt'), sicp_T)
        np.savetxt(os.path.join(save_dir, 'icp_tfm.txt'), icp_tfm)
        # bcpdpp_txt = os.path.join(save_dir, 'result_bcpdpp_autoy.txt')
        # if not os.path:
        #     np.savetxt(bcpdpp_txt, result_bcpdpp)
    # eng.quit()


if __name__ == '__main__':
    eng = matlab.engine.start_matlab()
    cpd_root_dir = r'F:/cta-spect/CTA-SPECT/code/code/CPDori'
    icp_root_dir = r'F:/cta-spect/CTA-SPECT/code/code/ICP'
    sicp_root_dir = r'F:/cta-spect/CTA-SPECT/code/code/register3D'
    bcpd_root_dir = r'F:/cta-spect/CTA-SPECT/code/code/bcpd-master'
    psr_root_dir = r'F:/cta-spect/CTA-SPECT/code/code/CVPR24_PointSetReg-main'
    eng.addpath(eng.genpath(cpd_root_dir), nargout=0)
    eng.addpath(eng.genpath(icp_root_dir), nargout=0)
    eng.addpath(eng.genpath(sicp_root_dir), nargout=0)
    eng.addpath(eng.genpath(bcpd_root_dir), nargout=0)
    eng.addpath(eng.genpath(psr_root_dir), nargout=0)
    eng.parallel.gpu.enableCUDAForwardCompatibility(True)


    root_dir = r'E:\proj\peizhun\file_process\name'
    cloud_dir = r'E:\proj\peizhun\file_process\cloud_result0507'
    name = 'zengyuming'
    reg_matlab(root_dir, cloud_dir,name,eng)
    print(name)