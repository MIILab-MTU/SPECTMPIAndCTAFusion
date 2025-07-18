function [nrigidresult, sigma2] = cpd_rigid_python(input_dir,cloud_dir, name, isori)

patientname = name;
load(sprintf('%s/%s/ijkcta.txt',input_dir,patientname));
if isori
    load(sprintf('path/to/manu/%s/cu.txt',patientname));
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

cpdr_s = Transform.s;
cpdr_R = Transform.R;
cpdr_t = Transform.t;




