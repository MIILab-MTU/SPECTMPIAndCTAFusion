function [out,e] = sicp_python(input_dir,cloud_dir, name,isori)

patientname = name;
load(sprintf('%s/%s/ijkcta.txt',input_dir,patientname));
if isori
    load(sprintf('path/to/manu/%s/cu.txt',patientname));
else
    load(sprintf('%s/%s/cu.txt',cloud_dir,patientname));
end


X = ijkcta;
Y = cu;
% spect=pointss(5201:end,:);
% cta=changsuping_LV(3437:end,:);

plot_3d_3(X, Y);
[out,s,R,T,e]=fSICP2D(X, Y);
% plot_3d_3(OCLVPoints, Segmentation_LV_jiang);


sicp_s = s;
sicp_R = R;
sicp_T = T;

end

