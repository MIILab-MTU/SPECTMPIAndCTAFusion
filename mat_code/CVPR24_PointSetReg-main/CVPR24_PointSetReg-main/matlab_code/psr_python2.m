function psr_python2(tgt_path, patientname, out_dir)
    %% ====== 配置区 ======
    % 支持 .txt 或 .ply；.txt 默认为每行 x y z（允许多余列与注释 #/%//）
    % src_path = "../data/tr_reg_059.ply";   % 也可改成 "../data/tr_reg_059.ply"
    % tgt_path = "../data/tr_reg_057.ply";   % 也可改成 "../data/tr_reg_057.ply"
    % patientname = 'guozuyu';
    src_path = sprintf('%s/cu.txt',out_dir);   % 也可改成 "../data/tr_reg_059.ply"
    tgt_path = sprintf('%s/%s/ijkcta.txt', tgt_path, patientname);   % 也可改成 "../data/tr_reg_057.ply"



    gridStep      = 0.03;
    dedup_tol     = 1e-4;
    jitter_ratio  = 5e-2;
    force_cpu     = false;

    rng(42);

    %% ========= 读取 & 初步清洗（剔除NaN/Inf，保留前三列） =========
    src_pt = read_point_cloud(src_path);
    tgt_pt = read_point_cloud(tgt_path);

    fprintf('[Info] Read src=%d pts, tgt=%d pts\n', size(src_pt,1), size(tgt_pt,1));

    src_pt = sanitize_points(src_pt);
    tgt_pt = sanitize_points(tgt_pt);

    %% ========= 去重（基于体素化/量化），再加微量抖动 =========
    src_pt = dedup_points(src_pt, dedup_tol);
    tgt_pt = dedup_points(tgt_pt, dedup_tol);
    
    src_pt = add_tiny_jitter(src_pt, jitter_ratio);
    tgt_pt = add_tiny_jitter(tgt_pt, jitter_ratio);
    % 
    % fprintf('[Info] After dedup+jitter: src=%d, tgt=%d\n', size(src_pt,1), size(tgt_pt,1));

    %% ========= 安全归一化（中心化 + 全局尺度；避免除0） =========
    [src_pt_normal, src_pre_normal] = safe_normalize_input(src_pt);
    [tgt_pt_normal, tgt_pre_normal] = safe_normalize_input(tgt_pt);

    %% ========= 下采样到 ~5000 内并二次清洗 =========
    src_pc = pointCloud(src_pt_normal);
    tgt_pc = pointCloud(tgt_pt_normal);

%     src_pc = pcdownsample(src_pc, 'gridAverage', gridStep);
%     tgt_pc = pcdownsample(tgt_pc, 'gridAverage', gridStep);

    src_pt_normal = sanitize_points(src_pc.Location);
    tgt_pt_normal = sanitize_points(tgt_pc.Location);

    % 防空回退
    if isempty(src_pt_normal)
        warning('src 下采样为空，回退到原始归一化前的 src_pt。');
        [src_pt_normal, src_pre_normal] = safe_normalize_input(sanitize_points(src_pt));
    end
    if isempty(tgt_pt_normal)
        warning('tgt 下采样为空，回退到原始归一化前的 tgt_pt。');
        [tgt_pt_normal, tgt_pre_normal] = safe_normalize_input(sanitize_points(tgt_pt));
    end

    fprintf('[Info] After downsample: src=%d, tgt=%d\n', size(src_pt_normal,1), size(tgt_pt_normal,1));



    %% ========= 可视化：归一化后的点云 =========
%     figure('Name','Normalized Point Clouds');
%     subplot(1,2,1);
%     scatter3(src_pt_normal(:,1), src_pt_normal(:,2), src_pt_normal(:,3), 6, 'filled'); axis equal; title('source (normalized)'); grid on;
%     subplot(1,2,2);
%     scatter3(tgt_pt_normal(:,1), tgt_pt_normal(:,2), tgt_pt_normal(:,3), 6, 'filled'); axis equal; title('target (normalized)'); grid on;

    %% ========= 设备选择（GPU/CPU） =========
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

    %% ========= 非刚性配准（核心算法：fuzzy_cluster_reg） =========
    t_start = tic;
    [alpha, T_deformed] = fuzzy_cluster_reg(src_in, tgt_in); %#ok<ASGLU>
    opts.mu = [];         % 让函数自动估，或你填论文/代码里的默认值
    opts.zeta = 1e-3;     % 可按论文区间调
    opts.use_poly = false; % 如需仿射/常数项，改 true



    % model = extract_clureg_model(src_pt_normal, tgt_pt_normal, alpha, opts);
    % 
    % % 验证：用模型重算一遍源点
    % T_model = apply_clureg_model(src_pt_normal, model);
    % 
    % % 反归一化
    % T_model_den = denormalize(tgt_pre_normal, T_model);
    % T_model_den = sanitize_points(T_model_den)
    % save('clureg_model.mat','model');  % 以后直接 load 后 apply 即可




    el = toc(t_start);
    fprintf('[Info] Registration time: %.3f s\n', el);

    % 若 GPU，取回内存
    if isa(T_deformed, 'gpuArray'), T_deformed = gather(T_deformed); end

    % 数值健康检查
    if ~isnumeric(T_deformed) || size(T_deformed,2) < 3
        error('T_deformed 维度异常（期望 Nx3+）。');
    end

    n_bad = sum(~all(isfinite(T_deformed),2));
    if n_bad > 0
        warning('T_deformed 含 NaN/Inf（%.2f%%）。尝试 CPU 回退重跑或增大 jitter_ratio。', 100*n_bad/max(1,size(T_deformed,1)));
        % 尝试 CPU 回退一次
        if useGPU
            [alpha, T2] = fuzzy_cluster_reg(double(src_pt_normal), double(tgt_pt_normal));
            if isa(T2, 'gpuArray'), T2 = gather(T2); end
            if sum(~all(isfinite(T2),2)) < n_bad
                T_deformed = T2;
                fprintf('[Info] CPU 回退后，坏点数减少。\n');
            end
        end
        % 再次剪除非有限值
        T_deformed = T_deformed(all(isfinite(T_deformed),2), :);
    end

    %% ========= 反归一化（回到原始尺度） =========

    T_deformed_denormal = denormalize(tgt_pre_normal, T_deformed);
    tgt_pt_denormal     = denormalize(tgt_pre_normal, tgt_pt_normal);

    T_deformed_denormal = sanitize_points(T_deformed_denormal);
    tgt_pt_denormal     = sanitize_points(tgt_pt_denormal);
    % T_deformed_denormal = T_deformed;
    % tgt_pt_denormal = tgt_pt_normal;

    [model, stats] = extract_pair_model_from_pairs( ...
        src_pt_normal, T_deformed_denormal, src_pre_normal, tgt_pre_normal, opts);

    fprintf('[Pair-Model] RMSE at source (world coords) = %.6g\n', stats.rmse_world_on_src);

    % ——验证：用模型直接重算源点并与原 T_deformed_denormal 对比
    Zp_from_model = apply_clureg_model( (src_pt_normal.*src_pre_normal.scale + src_pre_normal.center), ...
                                        model, 'return_normalized', false );
    fprintf('[Pair-Model] Check RMSE again = %.6g\n', sqrt(mean(sum((Zp_from_model - T_deformed_denormal).^2,2))));
    
%     save('F:/cta-spect/CTA-SPECT/code/code/CVPR24_PointSetReg-main/CVPR24_PointSetReg-main/matlab_code/clureg_model1.mat','model');


    %% ========= 可视化：原始点云 & 配准结果 =========
%     figure('Name','Original Point Clouds');
%     subplot(1,2,1);
%     scatter3(src_pt(:,1), src_pt(:,2), src_pt(:,3), 6, 'filled'); axis equal; title('source (original)'); grid on;
%     subplot(1,2,2);
%     scatter3(tgt_pt(:,1), tgt_pt(:,2), tgt_pt(:,3), 6, 'filled'); axis equal; title('target (original)'); grid on;
% 
%     figure('Name','Registration Result');
%     hold on; grid on; axis equal;
%     scatter3(tgt_pt_denormal(:,1), tgt_pt_denormal(:,2), tgt_pt_denormal(:,3), 6, 'filled');
%     scatter3(Zp_from_model(:,1), Zp_from_model(:,2), Zp_from_model(:,3), 6, 'filled');
%     legend('Target (denormalized)','Deformed (denormalized)');
%     title('Registration');
%     hold off;

    %% ========= 稳健最近邻 RMSE 评估（过滤 NaN/Inf/空集） =========
    if isempty(T_deformed_denormal) || isempty(tgt_pt_denormal)
        warning('评估集为空：跳过 RMSE。');
    else
        D = nearest_neighbor_distances(T_deformed_denormal, tgt_pt_denormal);
        D = D(isfinite(D));
        if isempty(D)
            warning('NN 距离为空或非有限，RMSE 跳过。');
        else
            rmse = sqrt(mean(D.^2));
            fprintf('[Info] NN-RMSE (Deformed -> Target) = %.6f\n', rmse);
        end
    end
    %% 保存结果
    reg_out = fullfile(out_dir, 'result_psr.txt');
    writematrix(T_deformed_denormal, reg_out, 'Delimiter',' ');
    fprintf('[OK] Saved deformed point cloud -> %s\n', reg_out);

    model_out = fullfile(out_dir,'psr_clureg_model.mat');
    save(model_out, 'model');
    fprintf('[OK] Saved model -> %s\n', model_out);
    %% ========= 调试输出（方便定位 NaN 源头） =========
    fprintf('Sizes | src:%d tgt:%d | srcN:%d tgtN:%d | Tdef:%d x %d | Tdef_den:%d | tgt_den:%d\n', ...
        size(src_pt,1), size(tgt_pt,1), size(src_pt_normal,1), size(tgt_pt_normal,1), ...
        size(T_deformed,1), size(T_deformed,2), size(T_deformed_denormal,1), size(tgt_pt_denormal,1));
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





