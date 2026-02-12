function [nrigidresult, cpdr_s, cpdr_R, cpdr_t, sigma2] = cpd_rigid_python(input_dir,cloud_dir, name, isori)

patientname = name;
load(sprintf('%s/%s/ijkcta.txt',input_dir,patientname));
if isori
    load(sprintf('F:/cta-spect/CTA-SPECT/result/%s/cu.txt',patientname));
else
    load(sprintf('%s/%s/cu.txt',cloud_dir,patientname));
end


%;
% cta=ctapoints(5112:end,:);
% spect=spectpoints(5201:end,:);

opt.method=['rigid'];
%opt.method=['affine'];
% opt.method=['nonrigid'];
% opt.method=['nonrigid_lowrank'];
opt.fgt=0;
opt.viz=1;

[Transform, C, sigma2]=cpd_register(ijkcta, cu, opt);
figure,cpd_plot_iter(ijkcta, cu); title('Before');
figure,cpd_plot_iter(ijkcta, Transform.Y);  title('After registering Y to X');

nrigidresult = Transform.Y;
% ResultsDir=strcat(sprintf('E:/ME/Study/dianyunpeizhun/result/%s/',patientname)); %strcat()水平串联字符串
% if ~exist(ResultsDir,'dir')
%     mkdir(ResultsDir);
% end
% save(sprintf('%s/nrigidresult.mat',ResultsDir),'nrigidresult');

%%保存结果
% save(sprintf('E:/ME/Study/dianyunpeizhun/Open3Dmanual/test10_29/%s/nrigidresult.mat',patientname),'nrigidresult');

cpdr_s = Transform.s;
cpdr_R = Transform.R;
cpdr_t = Transform.t;
%%保存变换矩阵
% save(sprintf('E:/ME/Study/dianyunpeizhun/Open3Dmanual/test10_29/%s/cpdr_s.mat',patientname),'cpdr_s');
% save(sprintf('E:/ME/Study/dianyunpeizhun/Open3Dmanual/test10_29/%s/cpdr_R.mat',patientname),'cpdr_R');
% save(sprintf('E:/ME/Study/dianyunpeizhun/Open3Dmanual/test10_29/%s/cpdr_t.mat',patientname),'cpdr_t');





