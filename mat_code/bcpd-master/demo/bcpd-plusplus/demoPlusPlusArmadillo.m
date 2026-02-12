close all; clear;
addpath('..');

%% input files
patientname = 'chenhongming';
x = sprintf('E:/proj/peizhun/file_process/name/%s/ijkcta.txt', patientname);
y = sprintf('E:/proj/peizhun/file_process/cloud_result0507/%s/cu.txt', patientname);
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
dwn = 'B,5000,0.05';

%% execution
prm1 = sprintf('-w%s -b%s -l%s -g%s', omg, bet, lmd, gma);
prm2 = sprintf('-J%s -K%s -p -f%s -u%s -D%s', J, K, f, nrm, dwn);
prm3 = sprintf('-c%s -n%s -h -r1', c, n);

% 创建输出目录
output_dir = 'E:/proj/peizhun/file_process/cloud_result0507/chenhongming';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% BCPD 命令
cmd = sprintf('%s -x%s -y%s -o%s -s T,v %s %s %s', bcpd, x, y, 'E:/proj/peizhun/file_process/cloud_result0507/chenhongming/%s/result_bcpdpp_manu', prm1, prm2, prm3);
system(cmd);

%% 加载结果
X0 = load(x);  % 目标点云
T0 = load(y);  % 原始源点云
T1 = load('E:/proj/peizhun/file_process/cloud_result0507/chenhongming/out_y.txt');  % 降采样配准点云
s = load('E:/proj/peizhun/file_process/cloud_result0507/chenhongming/out_s.txt');  % 缩放因子
R = load('E:/proj/peizhun/file_process/cloud_result0507/chenhongming/out_R.txt');  % 旋转矩阵
t = load('E:/proj/peizhun/file_process/cloud_result0507/chenhongming/out_t.txt');  % 平移向量
v = load('E:/proj/peizhun/file_process/cloud_result0507/chenhongming/out_v.txt');  % 非刚性位移

%% 插值非刚性位移
beta = 2.0;  % BCPD++ 核参数
M = size(T0, 1);
M_down = size(T1, 1);
if M_down < M
    G = zeros(M, M_down);
    for i = 1:M
        for j = 1:M_down
            G(i,j) = exp(-sum((T0(i,:) - T1(j,:)).^2) / (2 * beta^2));
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
save('E:/proj/peizhun/file_process/cloud_result0507/chenhongming/transformed_y.txt', 'Y_transformed', '-ascii');

%% 打印转换矩阵
fprintf('缩放因子 (s): %.4f\n', s);
fprintf('旋转矩阵 (R):\n');
disp(R);
fprintf('平移向量 (t):\n');
disp(t);
fprintf('非刚性位移 (v, 前5行):\n');
disp(v_orig(1:min(5, size(v_orig, 1)), :));

%% 保存转换矩阵
save('E:/proj/peizhun/file_process/cloud_result0507/chenhongming/transform.mat', 's', 'R', 't', 'v_orig');

%% 可视化
f2 = figure('Name', 'Before/After Registration', 'NumberTitle', 'off');
subplot(1, 2, 1);
plot3(T0(:,1), T0(:,2), T0(:,3), '.r', 'MarkerSize', 3); hold on;
plot3(X0(:,1), X0(:,2), X0(:,3), '.b', 'MarkerSize', 3); daspect([1 1 1]); grid on;
title('Before Registration', 'FontSize', 18);
subplot(1, 2, 2);
plot3(Y_transformed(:,1), Y_transformed(:,2), Y_transformed(:,3), '.r', 'MarkerSize', 3); hold on;
plot3(X0(:,1), X0(:,2), X0(:,3), '.b', 'MarkerSize', 3); daspect([1 1 1]); grid on;
title('After Registration (Transformed)', 'FontSize', 18);