function [model, stats] = extract_pair_model_from_pairs(src_pt_normal, Tdef_world, src_pre, tgt_pre, opts)
% 从“源的归一化点 src_pt_normal (NxD)”和“对应的配准后结果(世界坐标) Tdef_world (NxD)”
% 提取一个闭式形变模型：Z' = Z + K_{ZX} C + P_Z D
%
% src_pt_normal : 源点（已按 src_pre 归一化） NxD
% Tdef_world    : 与 src 一一对应的“配准结果”的世界坐标 NxD
% src_pre       : struct with fields: center(1xD), scale(scalar) —— 源的归一化信息
% tgt_pre       : struct with fields: center, scale —— 目标的归一化信息（这里只用于一致性检查/可选）
% opts          : .mu(核带宽, 若空则自动估), .lambda(正则, 默认1e-8), .use_poly(true/false, 默认true)
%
% 输出：
% model.X, C, D, mu, use_poly, pre(=src_pre) —— 与 apply_clureg_model 兼容
% stats.rmse_world_on_src —— 用模型作用于“源(世界坐标)”在世界坐标下与 Tdef_world 的RMSE
%
% 注意：本函数不依赖 U/alpha，直接用成对点拟合，源点处基本可实现“零误差重现”。

    if nargin < 5, opts = struct; end
    if ~isfield(opts,'lambda') || isempty(opts.lambda), opts.lambda = 1e-8; end
    if ~isfield(opts,'use_poly') || isempty(opts.use_poly), opts.use_poly = true; end

    Xn = double(src_pt_normal);                      % 源（归一化）
    [N,d] = size(Xn);
    Tw = double(Tdef_world);                         % 目标（世界）
    if size(Tw,1) ~= N || size(Tw,2) ~= d
        error('Tdef_world size mismatch. Expected %dx%d, got %dx%d.', N,d,size(Tw,1),size(Tw,2));
    end

    % ——把 Tdef_world 转回 “源的归一化坐标系”
    ctrS  = double(src_pre.center(:))';
    sclS  = double(src_pre.scale);
    if ~isscalar(sclS) || ~isfinite(sclS) || sclS==0, error('src_pre.scale invalid'); end
    Tn    = (Tw - ctrS) ./ sclS;                    % 目标结果在“源归一化系”下
    Delta = Tn - Xn;                                 % 想拟合的位移（在源归一化系）

    % ——估计核带宽 mu（L1-Laplacian）
    if ~isfield(opts,'mu') || isempty(opts.mu)
        if N >= 2
            Dnn = pdist2(Xn, Xn, 'cityblock');      % L1
            Dnn(1:N+1:end) = inf;
            kn   = max(1, min(20, N-1));
            nn   = mink(Dnn, kn, 2);
            dmed = median(nn(:));
            if ~isfinite(dmed) || dmed <= 0
                Dall = Dnn(isfinite(Dnn)); dmed = iff_empty(Dall, 1.0, median(Dall));
            end
            mu = 1 / max(dmed, 1e-8);
        else
            mu = 1.0;
        end
    else
        mu = double(opts.mu);
    end

    % ——核矩阵 & 多项式基
    Kxx = exp(-mu * pdist2(Xn, Xn, 'cityblock'));   % N x N （Laplacian核）
    if opts.use_poly
        P = [ones(N,1), Xn];                        % N x (1+d)，可表达到仿射
        Zblk = zeros(size(P,2));
    else
        P = [];
    end

    % ——解线性系统 [K+λI, P; P', 0] * [C;D] = [Delta; 0]
    A11 = Kxx + opts.lambda * eye(N);
    if opts.use_poly
        A   = [A11, P; P', Zblk];
        RHS = [Delta; zeros(size(P,2), d)];
        sol = A \ RHS;
        C = sol(1:N, :);
        D = sol(N+1:end, :);
    else
        C = A11 \ Delta;
        D = zeros(0, d);
    end

    % ——打包模型（注意：pre 绑定“源”的归一化；apply时输入世界坐标即可）
    model.X        = Xn;
    model.C        = C;
    model.D        = D;
    model.mu       = mu;
    model.use_poly = logical(opts.use_poly);
    model.pre      = src_pre;    % 关键：将来 apply_clureg_model 会用它来自动做(反)归一化
    model.zeta     = opts.lambda; % 仅记录

    % ——在源点上自检：应≈0误差
    % 用 apply_clureg_model 走一遍（世界坐标进/出）
    Zp_world = apply_clureg_model( (Xn.*sclS + ctrS), model, 'return_normalized', false );
    dif = Zp_world - Tw;
    stats.rmse_world_on_src = sqrt(mean(sum(dif.^2, 2)));
end

function out = iff_empty(vec, default_val, agg_val)
    if isempty(vec), out = default_val; else, out = agg_val; end
end
