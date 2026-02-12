function psr_python(tgt_path, patientname, out_dir)
%RUN_CLUREG_REGISTRATION Correspondence-Free Non-Rigid Point Set Registration
%   src_path   : 源点云路径（支持 .txt/.csv/.tsv/.ply）
%   tgt_path   : 目标点云路径（同上）
%   patientname: 样本名/患者名，用于输出文件命名
%   out_dir    : 输出目录
%
% 输出：
%   - out_dir/patientname_reg.txt  去归一化的配准结果点云
%   - out_dir/patientname_clureg_model.mat  保存的模型
%
% 依赖：
%   fuzzy_cluster_reg.m
%   extract_pair_model_from_pairs.m
%   apply_clureg_model.m
%
% 作者: 你

    addpath('./src');
    addpath('./utils/');

    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    gridStep      = 0.03;
    force_cpu     = false;
    rng(42);

    %% 读取点云
    src_path = sprintf('F:/cta-spect/CTA-SPECT/result/%s/cu.txt', patientname);
    tgt_path = sprintf('%s/%s/ijkcta.txt', tgt_path, patientname);
    src_pt = read_point_cloud(src_path);
    tgt_pt = read_point_cloud(tgt_path);
    fprintf('[Info] Read src=%d pts, tgt=%d pts\n', size(src_pt,1), size(tgt_pt,1));

    src_pt = sanitize_points(src_pt);
    tgt_pt = sanitize_points(tgt_pt);

    %% 归一化
    [src_pt_normal, src_pre_normal] = safe_normalize_input(src_pt);
    [tgt_pt_normal, tgt_pre_normal] = safe_normalize_input(tgt_pt);

    %% 下采样
    src_pc = pointCloud(src_pt_normal);
    tgt_pc = pointCloud(tgt_pt_normal);
    src_pc = pcdownsample(src_pc, 'gridAverage', gridStep);
    tgt_pc = pcdownsample(tgt_pc, 'gridAverage', gridStep);
    src_pt_normal = sanitize_points(src_pc.Location);
    tgt_pt_normal = sanitize_points(tgt_pc.Location);

    %% GPU / CPU
    useGPU = (~force_cpu) && (gpuDeviceCount > 0);
    if useGPU
        fprintf('[Info] 使用 GPU 进行配准。\n');
        src_in = gpuArray(src_pt_normal);
        tgt_in = gpuArray(tgt_pt_normal);
    else
        fprintf('[Info] 使用 CPU 进行配准。\n');
        src_in = src_pt_normal;
        tgt_in = tgt_pt_normal;
    end

    %% 核心配准
    t_start = tic;
    [alpha, T_deformed] = fuzzy_cluster_reg(src_in, tgt_in); %#ok<ASGLU>
    if isa(T_deformed, 'gpuArray'), T_deformed = gather(T_deformed); end
    fprintf('[Info] Registration time: %.3f s\n', toc(t_start));

    %% 去归一化
    T_deformed_denormal = denormalize(tgt_pre_normal, T_deformed);
    tgt_pt_denormal     = denormalize(tgt_pre_normal, tgt_pt_normal);
    T_deformed_denormal = sanitize_points(T_deformed_denormal);
    tgt_pt_denormal = sanitize_points(tgt_pt_denormal)

    %% 提取模型
    opts.mu       = [];
    opts.lambda   = 1e-8;
    opts.use_poly = true;
    [model, stats] = extract_pair_model_from_pairs( ...
        src_pt_normal, T_deformed_denormal, src_pre_normal, tgt_pre_normal, opts);
    fprintf('[Pair-Model] RMSE at source (world coords) = %.6g\n', stats.rmse_world_on_src);

    %% 保存结果
    reg_out = fullfile(out_dir, sprintf('%s_reg.txt', patientname));
    writematrix(T_deformed_denormal, reg_out, 'Delimiter',' ');
    fprintf('[OK] Saved deformed point cloud -> %s\n', reg_out);

    model_out = fullfile(out_dir, sprintf('%s_clureg_model.mat', patientname));
    save(model_out, 'model');
    fprintf('[OK] Saved model -> %s\n', model_out);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 本地函数区（R2016b+脚本尾部函数）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function P = read_point_cloud(fp)
%READ_POINT_CLOUD 读取 .txt/.csv/.tsv（x y z [+...]）或 .ply 为 Nx3(+) 数组
    arguments, fp (1,1) string, end
    ext = lower(string(fp));
    if endsWith(ext, ".ply")
        pc = pcread(fp);
        P = double(pc.Location);
        P = sanitize_points(P);
        return;
    end
    if endsWith(ext, ".txt") || endsWith(ext, ".csv") || endsWith(ext, ".tsv")
        % readmatrix 自适应分隔，忽略注释
        try
            P = readmatrix(fp, 'FileType','text', 'NumHeaderLines',0);
        catch
            % 兜底 textscan
            fid = fopen(fp,'r');
            C = textscan(fid, '%f%f%f%f%f%f', 'Delimiter',{' ', '\t', ','}, ...
                         'MultipleDelimsAsOne',true, 'CommentStyle',{'#','%','//'});
            fclose(fid);
            P = [];
            for i=1:numel(C)
                if ~isempty(C{i}), P = [P, C{i}]; end %#ok<AGROW>
            end
        end
        if isempty(P), error('无法从文本文件读取数据：%s', fp); end
        P = double(P);
        P = sanitize_points(P);
        return;
    end
    error('不支持的扩展：%s（仅支持 .ply/.txt/.csv/.tsv）', fp);
end

function Q = sanitize_points(P)
%SANITIZE_POINTS 只保留前三列 xyz；行全为有限值；去极端爆点
    if ~isnumeric(P), error('点云不是数值矩阵。'); end
    if size(P,2) < 3, error('需要至少 3 列 (x,y,z)。'); end
    Q = double(P(:,1:3));
    mask = all(isfinite(Q), 2);
    Q = Q(mask, :);
    if isempty(Q), return; end
    % 宽松离群剔除（防止极端大值导致尺度溢出）
    med = median(Q,1);
    iq = iqr(Q);
    iq(iq==0) = 1; % 防除0
    z = abs((Q - med) ./ iq);
    Q = Q(all(z < 1e6, 2), :);
end

function Pu = dedup_points(P, tol)
%DEDUP_POINTS 以公差 tol 去重（量化到网格再 unique）
    if nargin < 2 || isempty(tol), tol = 1e-8; end
    if tol <= 0
        Pu = unique(P, 'rows', 'stable'); 
        return;
    end
    G = round(P / tol) * tol;
    [~, ia] = unique(G, 'rows', 'stable');
    Pu = P(ia, :);
end

function Pj = add_tiny_jitter(P, rel)
%ADD_TINY_JITTER 给点云加极小扰动，避免 0 距离/0 方差
    if nargin < 2 || isempty(rel), rel = 1e-6; end
    mins = min(P, [], 1); maxs = max(P, [], 1);
    diag_len = norm(maxs - mins);
    if ~isfinite(diag_len) || diag_len < 1e-12
        diag_len = max(1e-6, norm(std(P,0,1,'omitnan')));
        if ~isfinite(diag_len) || diag_len < 1e-12, diag_len = 1.0; end
    end
    eps_abs = rel * diag_len;
    Pj = P + (rand(size(P)) - 0.5) * 2 * eps_abs; % 均匀噪声
    % 若偏好高斯噪声：P + randn(size(P)) * (eps_abs/3);
end

function [Pn, pre] = safe_normalize_input(P)
%SAFE_NORMALIZE_INPUT 归一化：中心化 + 全局尺度对角线；稳定防除0
    P = double(P);
    P = sanitize_points(P);
    ctr = mean(P, 1, 'omitnan');
    Pc  = bsxfun(@minus, P, ctr);

    mins = min(P, [], 1);
    maxs = max(P, [], 1);
    diag_len = norm(maxs - mins);
    if ~isfinite(diag_len) || diag_len < 1e-12
        diag_len = max(1e-6, norm(std(P,0,1,'omitnan')));
        if ~isfinite(diag_len) || diag_len < 1e-12
            diag_len = 1.0;
        end
    end
    scale = diag_len;

    Pn = Pc ./ scale;

    pre.center = ctr;
    pre.scale  = scale;
end

function P = denormalize(pre, Pn)
%DENORMALIZE 与 safe_normalize_input 配套反变换
    P = Pn .* pre.scale + pre.center;
end

function D = nearest_neighbor_distances(Q, R)
%NEAREST_NEIGHBOR_DISTANCES Q 中各点到 R 的最近邻距离（稳健实现）
% 优先使用 knnsearch；若不可用，退化到朴素 O(NM) 实现
    D = [];
    if isempty(Q) || isempty(R), return; end
    Q = sanitize_points(Q);
    R = sanitize_points(R);
    try
        % Statistics and Machine Learning Toolbox
        Mdl = KDTreeSearcher(R);
        [~, d] = knnsearch(Mdl, Q);
        D = double(d);
        return;
    catch
        % 无工具箱则朴素实现（小规模可用）
        nQ = size(Q,1); nR = size(R,1);
        D  = zeros(nQ,1);
        for i = 1:nQ
            di = sqrt(sum((R - Q(i,:)).^2, 2));
            D(i) = min(di);
        end
    end
end

