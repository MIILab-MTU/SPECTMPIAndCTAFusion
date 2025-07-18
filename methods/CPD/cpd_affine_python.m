function [out, sigma2]=cpd_affine_python(input_dir, cloud_dir, name,isori)
patientname = name;
load(sprintf('%s/%s/ijkcta.txt',input_dir,patientname));


if isori
    load(sprintf('path/to/manu/%s/cu.txt',patientname));
else
    load(sprintf('%s/%s/cu.txt',cloud_dir,patientname));
end

X = ijkcta;
Y = cu;




opt.method='affine'; % use rigid registration
opt.viz=1;          % show every iteration
opt.outliers=0.6;   % use 0.6 noise weight to add robustness 

opt.normalize=1;    % normalize to unit variance and zero mean before registering (default)
opt.scale=1;        % estimate global scaling too (default)
opt.rot=1;          % estimate strictly rotational matrix (default)
opt.corresp=0;      % do not compute the correspondence vector at the end of registration (default)

opt.max_it=250;     % max number of iterations
opt.tol=1e-8;       % tolerance


% registering Y to X
[Transform, Correspondence, sigma2]=cpd_register(X,Y,opt);

figure,cpd_plot_iter(X, Y); title('Before');

% X(Correspondence,:) corresponds to Y
figure,cpd_plot_iter(X, Transform.Y);  title('After registering Y to X');
out=Transform.Y;

