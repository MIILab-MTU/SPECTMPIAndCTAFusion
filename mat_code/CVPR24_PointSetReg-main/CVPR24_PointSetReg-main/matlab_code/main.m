% ===========================
% apply_model_main.m
% 批处理：载入模型 -> 应用到任意点云 -> 可视化(源vs配准) -> 保存结果
% 依赖：apply_clureg_model.m
% ===========================
clear; clc; close all;

addpath('./src');
addpath('./utils/');
%% ===== 配置 =====
patientname = 'zengyuming';
model_path   = 'psr_clureg_model.mat';               % 提取好的模型（包含 model 结构）
input_path   = sprintf('E:/proj/peizhun/file_process/cloud_result0507/%s/cu.txt',patientname);   % 任意 .txt/.csv/.tsv/.ply
output_path  = '';                                % 留空则自动以 _reg 命名
chunk_size   = 2e5;                               % 大点云可调大/小
return_norm  = false;                             % 若 model.pre 存在：false=返回世界坐标
tgt_path = sprintf('E:/proj/peizhun/file_process/name/%s/ijkcta.txt',patientname);
%% ===== 载入模型 =====
S = load(model_path);
if ~isfield(S, 'model')
    error('未在 %s 中找到变量 "model"。请先用 extract_clureg_model 生成。', model_path);
end
model = S.model;

% 可选：把归一化信息也塞进 model（若你之前单独保存了 pre）
if isfield(S, 'pre') && ~isfield(model, 'pre')
    model.pre = S.pre; %#ok<STRNU>
end

%% ===== 读取源点云 Z =====
Z = read_point_cloud(input_path);    % Lx3
fprintf('[Info] Loaded source points: %d\n', size(Z,1));

tgt = read_point_cloud(tgt_path)

% Z = sanitize_points(Z);
% [Z, Z1] = safe_normalize_input(Z)

%% ===== 应用模型得到配准结果 Zp =====
Zp = apply_clureg_model(Z, model, ...
    'chunk', chunk_size, ...
    'return_normalized', return_norm);

fprintf('[Info] Transformed points: %d\n', size(Zp,1));

%% ===== 可视化：源 vs 配准后 =====
figure('Name','apply_clureg_model: Visualization','Color','w','Position',[100 100 1200 520]);

subplot(1,2,1);
scatter3(Z(:,1), Z(:,2), Z(:,3), 4, 'filled');
axis equal; grid on; view(3);
title('Original (Source)','FontWeight','bold');

subplot(1,2,2); hold on;
scatter3(tgt(:,1),  tgt(:,2),  tgt(:,3), 4, 'filled');              % 源
scatter3(Zp(:,1), Zp(:,2), Zp(:,3), 4, 'filled');             % 配准后
axis equal; grid on; view(3);
legend({'Source','Transformed'}, 'Location','northeast');
title('Overlay: Source vs. Transformed','FontWeight','bold');
hold off;

%% ===== 自动命名输出并保存 =====
if isempty(output_path)
    [p, n, ext] = fileparts(input_path);
    if strcmpi(ext, '.ply')
        output_path = fullfile(p, [n '_reg.ply']);
    else
        output_path = fullfile(p, [n '_reg.txt']);
    end
end

save_point_cloud(Zp, output_path);
fprintf('[OK] Saved transformed points to: %s\n', output_path);

%% ======== 本地函数 ========
function P = read_point_cloud(fp)
% READ_POINT_CLOUD 读取 .txt/.csv/.tsv（x y z [+...]）或 .ply
    arguments, fp (1,1) string, end
    [~,~,ext] = fileparts(fp);
    ext = lower(ext);
    switch ext
        case '.ply'
            pc = pcread(fp);
            P  = double(pc.Location);
        otherwise
            % 文本：自动分隔，忽略非数值，保留前三列
            try
                P = readmatrix(fp, 'FileType','text');
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
    end
    if isempty(P), error('无法从文件读取点云：%s', fp); end
    % 基本清洗：仅取 xyz，去除非有限
    P = double(P(:,1:3));
    P = P(all(isfinite(P),2), :);
    if isempty(P), error('点云为空或全为非有限值：%s', fp); end
end

function save_point_cloud(P, fp)
% SAVE_POINT_CLOUD 保存为 .ply 或 .txt（按扩展名决定）
    [p,n,ext] = fileparts(fp);
    ext = lower(ext);
    switch ext
        case '.ply'
            try
                pc = pointCloud(P);
                pcwrite(pc, fullfile(p,[n ext]));
            catch ME
                warning('pcwrite 失败(%s)，改为保存 txt。', ME.message);
                writematrix(P, fullfile(p,[n '.txt']), 'Delimiter',' ');
            end
        otherwise
            writematrix(P, fullfile(p,[n '.txt']), 'Delimiter',' ');
    end
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
