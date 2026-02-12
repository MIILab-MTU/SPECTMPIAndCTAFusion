close all; clear; clc;
addpath('..');

%% --------- 输入文件 ----------
x   = sprintf('%s/../../data/armadillo-g-y.txt', pwd);   % source (待配准)
y   = sprintf('%s/../../data/armadillo-g-x.txt', pwd);   % target (参考)
fnm = sprintf('%s/../../bcpd',         pwd);             % Linux/Mac 可执行
fnw = sprintf('%s/../../win/bcpd.exe', pwd);             % Windows 可执行
if ispc, bcpd = fnw; else, bcpd = fnm; end

%% --------- 主要参数（与原稿一致） ----------
omg = '0.0';
bet = '1.0';
lmd = '50';
gma = '.1';
K   = '200';
J   = '300';
c   = '1e-6';
n   = '500';
L   = '100';
nrm = 'x';
dwn = 'B,10000,0.02';
tau = '1';         % geodesic 核的衰减系数（无网格模式下仍然有效）

%% --------- 选择 kernel 模式 ----------
use_geodesic_knn = true;   % =false 则用欧式核（不加 -G）

% 无网格 geodesic：使用 kNN 图构建测地距离
% 经验值：k=8~12, rad=点云尺度的 10%~30%
k   = '10';
rad = '0.15';      % 半径（与点云尺度相关，单位=归一化后坐标）

if use_geodesic_knn
    % geodesic, tau, k, radius  —— 注意这里不再需要 triangles.txt
    kern = sprintf('geodesic,%s,%s,%s', tau, k, rad);
    optG = sprintf('-G%s', kern);
else
    % 欧式核（不指定 -G，退回到普通 CPD/GBCPD 的高斯核）
    optG = '';
end

%% --------- 拼装命令 ----------
prm1 = sprintf('-w%s -b%s -l%s -g%s', omg, bet, lmd, gma);
prm2 = sprintf('-J%s -K%s -p -u%s -D%s', J, K, nrm, dwn);
prm3 = sprintf('-c%s -n%s -h -r1', c, n);
cmd  = sprintf('%s -x%s -y%s %s %s %s -ux %s', bcpd, x, y, prm1, prm2, prm3, optG);

fprintf('Running:\n%s\n\n', cmd);
status = system(cmd);
if status ~= 0
    error('BCPD/GBCPD 调用失败（返回码=%d）。请检查可执行文件路径与依赖。', status);
end

%% --------- 载入结果并可视化 ----------
X0 = load(x);     % source
T0 = load(y);     % target
if ~isfile('output_y.txt')
    error('未找到 output_y.txt（程序没有写出结果）。');
end
T1 = load('output_y.txt'); % 变换后的 source

figure('Color','w'); 
subplot(1,3,1); scatter3(X0(:,1),X0(:,2),X0(:,3),2,'.'); axis equal off; title('Source (X)');
subplot(1,3,2); scatter3(T0(:,1),T0(:,2),T0(:,3),2,'.'); axis equal off; title('Target (Y)');
subplot(1,3,3); hold on;
scatter3(T0(:,1),T0(:,2),T0(:,3),2,'.'); 
scatter3(T1(:,1),T1(:,2),T1(:,3),2,'.');
axis equal off; legend({'Target','X \rightarrow Y (after)'}, 'Location','southoutside');
title('Registration Result');

% 简单误差统计（与 target 的最近邻距离均值/中位数）
[idx, D] = knnsearch(T0, T1);
fprintf('NN 距离：mean = %.4g, median = %.4g\n', mean(D), median(D));
