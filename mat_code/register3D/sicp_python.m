function [out,s,R,T,e] = sicp_python(input_dir,cloud_dir, name,isori)

patientname = name;
load(sprintf('%s/%s/ijkcta.txt',input_dir,patientname));
if isori
    load(sprintf('F:/cta-spect/CTA-SPECT/result/%s/cu.txt',patientname));
else
    load(sprintf('%s/%s/cu.txt',cloud_dir,patientname));
end


X = ijkcta;
Y = cu;
% spect=pointss(5201:end,:);
% cta=changsuping_LV(3437:end,:);

plot_3d_3(X, Y); % 显示出当前两个点集
[out,s,R,T,e]=fSICP2D(X, Y);
% plot_3d_3(OCLVPoints, Segmentation_LV_jiang); % 显示出当前两个点集

%%保存变换矩阵
sicp_s = s;
sicp_R = R;
sicp_T = T;
% save(sprintf('E:/ME/Study/dianyunpeizhun/Open3Dmanual/test10_29/%s/sicp_s.mat',patientname),'sicp_s');
% save(sprintf('E:/ME/Study/dianyunpeizhun/Open3Dmanual/test10_29/%s/sicp_R.mat',patientname),'sicp_R');
% save(sprintf('E:/ME/Study/dianyunpeizhun/Open3Dmanual/test10_29/%s/sicp_T.mat',patientname),'sicp_T');

end

