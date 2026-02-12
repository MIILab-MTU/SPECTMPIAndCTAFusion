patientname = 'yangwenzhen';
isorg = false;
% patientname = 'yangwenzhen';
% load(sprintf('E:/proj/peizhun/file_process/name/%s/ijkcta.txt',patientname));
% load(sprintf('E:/proj/peizhun/file_process/cloud_result0507/%s/cu.txt',patientname));
save_dir=sprintf('E:/proj/peizhun/file_process/cloud_result0507/%s',patientname);
x = sprintf('E:/proj/peizhun/file_process/name/%s/ijkcta.txt', patientname);
if isorg
    y = sprintf('F:/cta-spect/CTA-SPECT/result/%s/cu_wrong.txt', patientname);
else
    y = sprintf('E:/proj/peizhun/file_process/cloud_result0507/%s/cu_wrong.txt', patientname);
end

fnm = sprintf('%s/../../bcpd', pwd);
fnw = 'F:/cta-spect/CTA-SPECT/code/code/bcpd-master/win/bcpd.exe';
if (ispc)
    bcpd = fnw;
else
    bcpd = fnm;
end

%% parameters
omg = '0.0';
bet = '2.0';
lmd = '50';
gma = '10';
K = '70';
J = '300';
f = '0.3';
c = '1e-6';
n = '500';
L = '100';
nrm = 'e';
dwn = 'B,5000,0.01';

%% execution
prm1 = sprintf('-w%s -b%s -l%s -g%s', omg, bet, lmd, gma);
prm2 = sprintf('-J%s -K%s -p -f%s -u%s -D%s', J, K, f, nrm, dwn);
prm3 = sprintf('-c%s -n%s -h -r1', c, n);

% 创建输出目录
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% BCPD 命令
if isorg
    cmd = sprintf('%s -x%s -y%s -o%s -s T,v %s %s %s', bcpd, x, y, sprintf('%s/result_bcpdpp_manu', save_dir), prm1, prm2, prm3);
else
    cmd = sprintf('%s -x%s -y%s -o%s -s T,v %s %s %s', bcpd, x, y, sprintf('%s/result_bcpdpp_auto', save_dir), prm1, prm2, prm3);
end
system(cmd);

%% 加载结果
X0 = load(x);  % 目标点云
T0 = load(y);  % 原始源点云
if isorg
    T1 = load(sprintf('%s/result_bcpdpp_manuy.txt', save_dir));  % 降采样配准点云
    s = load(sprintf('%s/result_bcpdpp_manus.txt', save_dir));
    R = load(sprintf('%s/result_bcpdpp_manuR.txt', save_dir));
    t = load(sprintf('%s/result_bcpdpp_manut.txt', save_dir));
    v = load(sprintf('%s/result_bcpdpp_manuv.txt', save_dir));
    Y_down = load(sprintf('%s/result_bcpdpp_manuy.txt', save_dir));  % 降采样点云
else
    T1 = load(sprintf('%s/result_bcpdpp_autoy.txt', save_dir));
    s = load(sprintf('%s/result_bcpdpp_autos.txt', save_dir));
    R = load(sprintf('%s/result_bcpdpp_autoR.txt', save_dir));
    t = load(sprintf('%s/result_bcpdpp_autot.txt', save_dir));
    v = load(sprintf('%s/result_bcpdpp_autov.txt', save_dir));
    Y_down = load(sprintf('%s/result_bcpdpp_autoy.txt', save_dir));
end

%% 插值非刚性位移
beta = 2.0;  % BCPD++ 核参数
M = size(T0, 1);
M_down = size(Y_down, 1);
if M_down < M
    G = zeros(M, M_down);
    for i = 1:M
        for j = 1:M_down
            G(i,j) = exp(-sum((T0(i,:) - Y_down(j,:)).^2) / (2 * beta^2));
        end
    end
    v_orig = zeros(M, 3);
    for d = 1:3
        v_orig(:,d) = G * (G \ v(:,d));
    end
else
    v_orig = v;
end

%% 应用变换
Y_transformed = s * (T0 + v_orig) * R' + t';

%% 保存转换后的点云
if isorg
    save_path = sprintf('%s/result_bcpdpp_manu_transformed.txt', save_dir);
else
    save_path = sprintf('%s/result_bcpdpp_auto_transformed.txt', save_dir);
end
save(save_path, 'Y_transformed', '-ascii');

%% 打印转换矩阵
fprintf('缩放因子 (s): %.4f\n', s);
fprintf('旋转矩阵 (R):\n');
disp(R);
fprintf('平移向量 (t):\n');
disp(t);
fprintf('非刚性位移 (v, 前5行):\n');
disp(v_orig(1:min(5, size(v_orig, 1)), :));

%% 保存转换矩阵
save(sprintf('%s/transform.mat', save_dir), 's', 'R', 't', 'v_orig');

%% 可视化
f2 = figure('Name', 'wrong Before/After Registration', 'NumberTitle', 'off');
subplot(1, 2, 1);
plot3(T0(:,1), T0(:,2), T0(:,3), '.r', 'MarkerSize', 3); hold on;
plot3(X0(:,1), X0(:,2), X0(:,3), '.b', 'MarkerSize', 3); daspect([1 1 1]); grid on;
title('Before Registration', 'FontSize', 18);
subplot(1, 2, 2);
plot3(Y_transformed(:,1), Y_transformed(:,2), Y_transformed(:,3), '.r', 'MarkerSize', 3); hold on;
plot3(X0(:,1), X0(:,2), X0(:,3), '.b', 'MarkerSize', 3); daspect([1 1 1]); grid on;
title('After Registration (Transformed)', 'FontSize', 18);