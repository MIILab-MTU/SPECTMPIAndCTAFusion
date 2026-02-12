%********************
% Correspondence-Free Non-Rigid Point Set Registration Using Unsupervised
% Clustering Analysis, CVPR 2024   Mingyang Zhao & Jingen Jiang
% Initialized on May 27th, 2024, Hong Kong
%********************
close all; clc; clear;

addpath('./src');
addpath('./utils/');

%% ====== 配置区 ======
% 支持 .txt 或 .ply；.txt 默认为每行 x y z（允许多余列与注释 #/%//）
% src_path = "../data/tr_reg_059.ply";   % 也可改成 "../data/tr_reg_059.ply"
% tgt_path = "../data/tr_reg_057.ply";   % 也可改成 "../data/tr_reg_057.ply"
patientname = 'guozuyu';
src_path = sprintf('F:/cta-spect/CTA-SPECT/data_process/cta/final/%s/ijkcta.txt',patientname);   % 也可改成 "../data/tr_reg_059.ply"
tgt_path = sprintf('E:/proj/peizhun/file_process/cloud_result0507/%s/cu.txt',patientname);   % 也可改成 "../data/tr_reg_057.ply"

gridStep = 0.03;       % 下采样体素尺寸（可按密度调整）
useGPU   = [];         % [] 自动检测；true 强制GPU；false 仅CPU

%% ====== 读取点云（自动识别 .txt / .ply）======
src_pt = read_point_cloud(src_path);
tgt_pt = read_point_cloud(tgt_path);

rng(42);  % 可复现实验；如不需要固定随机数，可删掉

% 对点云添加极小随机扰动，幅度按数据尺度自适应
src_pt = add_tiny_jitter(src_pt, 1e-6);  % 相对尺度 1e-6
tgt_pt = add_tiny_jitter(tgt_pt, 1e-6);
% ---- 继续执行你的 data_normalize_input 等流程 ----

assert(size(src_pt,2) >= 3 && size(tgt_pt,2) >= 3, '点云需至少包含 xyz 三列。');

% 只取前三列为坐标
src_pt = double(src_pt(:,1:3));
tgt_pt = double(tgt_pt(:,1:3));

%% ====== 归一化（调用你已有的工具函数）======
[src_pt_normal, src_pre_normal] = data_normalize_input(src_pt);
[tgt_pt_normal, tgt_pre_normal] = data_normalize_input(tgt_pt);

%% ====== 下采样至 ~5000 点以内（可按需关闭）======
src_pc = pointCloud(src_pt_normal);
tgt_pc = pointCloud(tgt_pt_normal);

src_pc = pcdownsample(src_pc, 'gridAverage', gridStep);
tgt_pc = pcdownsample(tgt_pc, 'gridAverage', gridStep);

src_pt_normal = double(src_pc.Location);
tgt_pt_normal = double(tgt_pc.Location);

%% ====== 可视化：归一化后的点云 ======
figure('Name','Normalized Point Clouds');
subplot(1,2,1);
scatter3(src_pt_normal(:,1), src_pt_normal(:,2), src_pt_normal(:,3), 6, 'filled'); axis equal; title('source (normalized)'); grid on;
subplot(1,2,2);
scatter3(tgt_pt_normal(:,1), tgt_pt_normal(:,2), tgt_pt_normal(:,3), 6, 'filled'); axis equal; title('target (normalized)'); grid on;

%% ====== 设备选择（GPU / CPU）======
if isempty(useGPU)
    useGPU = (gpuDeviceCount > 0);
end
if useGPU
    fprintf('[Info] 使用 GPU 进行配准。\n');
    src_in = gpuArray(src_pt_normal);
    tgt_in = gpuArray(tgt_pt_normal);
else
    fprintf('[Info] 使用 CPU 进行配准。\n');
    src_in = src_pt_normal;
    tgt_in = tgt_pt_normal;
end

%% ====== 非刚性配准（你的 fuzzy_cluster_reg 接口保持不变）======
[alpha, T_deformed] = fuzzy_cluster_reg(src_in, tgt_in); %#ok<ASGLU>

% 若在 GPU 上，取回到内存
if isa(T_deformed, 'gpuArray')
    T_deformed = gather(T_deformed);
end

%% ====== 反归一化到原始尺度 ======
T_deformed_denormal = denormalize(tgt_pre_normal, T_deformed);
tgt_pt_denormal     = denormalize(tgt_pre_normal, tgt_pt_normal);

%% ====== 可视化：原始点云 ======
figure('Name','Original Point Clouds');
subplot(1,2,1);
scatter3(src_pt(:,1), src_pt(:,2), src_pt(:,3), 6, 'filled'); axis equal; title('source (original)'); grid on;
subplot(1,2,2);
scatter3(tgt_pt(:,1), tgt_pt(:,2), tgt_pt(:,3), 6, 'filled'); axis equal; title('target (original)'); grid on;

%% ====== 可视化：目标与变形后结果对齐效果 ======
figure('Name','Registration Result');
hold on; grid on; axis equal;
scatter3(tgt_pt_denormal(:,1), tgt_pt_denormal(:,2), tgt_pt_denormal(:,3), 6, 'filled');
scatter3(T_deformed_denormal(:,1), T_deformed_denormal(:,2), T_deformed_denormal(:,3), 6, 'filled');
legend('Target (denormalized)','Deformed (denormalized)');
title('Registration');
hold off;

%% ======（可选）简单的误差评估：最近邻均方误差 ======
try
    % 将变形后的点到目标点求最近邻距离的RMSE（粗略指标）
    Mdl = KDTreeSearcher(tgt_pt_denormal);
    [idx, D] = knnsearch(Mdl, T_deformed_denormal); %#ok<ASGLU>
    rmse = sqrt(mean(D.^2));
    fprintf('[Info] NN-RMSE (Deformed -> Target) = %.6f\n', rmse);
catch
    % 没有 Statistics/NN 工具箱也没关系
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 本脚本内的本地函数（R2016b+ 支持脚本尾部定义局部函数）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function P = read_point_cloud(fp)
%READ_POINT_CLOUD 读取 .txt 或 .ply 点云为 Nx3(+) 数组
%  - .txt: 支持分隔符空白/逗号，注释符 # / % / //
%  - .ply: 使用 pcread 读取 Location
    arguments
        fp (1,1) string
    end
    ext = lower(string(fp));
    if endsWith(ext, ".ply")
        pc = pcread(fp);
        P = double(pc.Location);
        P = sanitize_points(P);
        return;
    end

    % 默认按文本表格读（允许逗号或空白，忽略注释行）
    if endsWith(ext, ".txt") || endsWith(ext, ".csv") || endsWith(ext, ".tsv")
        opts = detectImportOptions(fp, 'FileType','text');
        % 注释风格与分隔符设置（尽量鲁棒）
        try
            opts = setvaropts(opts, opts.VariableNames, 'WhitespaceRule','trim');
        catch
        end
        try
            opts.CommentStyle = {'#','%','//'};
        catch
        end
        % 若 detectImportOptions 失败时回退 readmatrix
        try
            T = readmatrix(fp, 'FileType','text');  % 自动识别分隔符
        catch
            % 再次尝试 textscan 兜底
            fid = fopen(fp,'r');
            C = textscan(fid, '%f%f%f%f%f%f', 'Delimiter',{' ', ',', '\t'}, ...
                         'MultipleDelimsAsOne',true, 'CommentStyle',{'#','%','//'});
            fclose(fid);
            T = [];
            for i=1:numel(C)
                if ~isempty(C{i})
                    T = [T, C{i}]; %#ok<AGROW>
                end
            end
        end
        if isempty(T)
            error('无法从文本文件读取数据：%s', fp);
        end
        P = double(T);
        P = sanitize_points(P);
        return;
    end

    error('不支持的文件扩展名：%s（仅支持 .txt/.csv/.tsv/.ply）', fp);
end

function Q = sanitize_points(P)
%SANITIZE_POINTS 仅保留数值型、有限值，去掉异常行
    if ~isnumeric(P)
        error('读取的数据不是数值矩阵。');
    end
    % 去除非有限/NaN 行
    good = all(isfinite(P), 2);
    Q = P(good, :);
    if size(Q,2) < 3
        error('数据列不足：需要至少3列 (x,y,z)。');
    end
end

function Pj = add_tiny_jitter(P, rel)
% 对点云 P 添加极小随机扰动
% rel: 相对扰动比例（相对于点云包围盒对角线长度），默认 1e-6 ~ 1e-5
    if nargin < 2 || isempty(rel), rel = 1e-6; end
    P = double(P);
    mins = min(P, [], 1);
    maxs = max(P, [], 1);
    bbox = maxs - mins;
    diag_len = norm(bbox);
    if ~isfinite(diag_len) || diag_len == 0
        % 极端情况：所有点完全一致或尺度为0，则用数据绝对值尺度兜底
        diag_len = max(1, norm(max(abs(P),[],1)));
    end

    % 扰动幅度（均匀分布），范围 [-eps, +eps]
    eps_abs = rel * diag_len;
    jitter = (rand(size(P)) - 0.5) * 2 * eps_abs;  % uniform
    % 如果你更喜欢高斯噪声，可用：randn(size(P)) * (eps_abs/3)

    Pj = P + jitter;
end

