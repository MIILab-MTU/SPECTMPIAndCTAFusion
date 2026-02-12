function [rmse, juli, jl] = eval_python(xlsx_dir, base_dir, cloud_dir, patientname,ori)
    % 读取 Excel 文件
    if ~exist(xlsx_dir, 'file')
        error('Excel 文件 %s 不存在', xlsx_dir);
    end
    data = readcell(xlsx_dir);
    namelist = data(:, 3);
    nlnum = length(namelist);

    % 查找患者索引
    index = find(cellfun(@(x) isequal(strrep(x, ' ', ''), patientname), namelist), 1);
    if isempty(index)
        error('未找到患者 %s', patientname);
    end

    % 提取比例因子
    ps = cell2mat(data(index, 12));

    % 加载点云数据
    if ori
        files = {sprintf('%s/%s/ijkcta.txt', base_dir, patientname), ...
             sprintf('%s/%s/result_icp_manu.txt', cloud_dir, patientname), ...
             sprintf('%s/%s/result_sicp_manu.txt', cloud_dir, patientname), ...
             sprintf('%s/%s/result_rigid_manu.txt', cloud_dir, patientname), ...
             sprintf('%s/%s/result_affine_manu.txt', cloud_dir, patientname), ...
             sprintf('%s/%s/result_bcpdpp_manuy.txt', cloud_dir, patientname)};
    else
        files = {sprintf('%s/%s/ijkcta.txt', base_dir, patientname), ...
             sprintf('%s/%s/result_icp_auto.txt', cloud_dir, patientname), ...
             sprintf('%s/%s/result_sicp_auto.txt', cloud_dir, patientname), ...
             sprintf('%s/%s/result_rigid_auto.txt', cloud_dir, patientname), ...
             sprintf('%s/%s/result_affine_auto.txt', cloud_dir, patientname), ...
             sprintf('%s/%s/result_bcpdpp_autoy.txt', cloud_dir, patientname), ...
             sprintf('%s/%s/result_psr.txt', cloud_dir, patientname), ...
             sprintf('%s/%s/ffd_result.txt', cloud_dir, patientname)};
             
    end
    
    for i = 1:length(files)
        if ~exist(files{i}, 'file')
            error('文件 %s 不存在', files{i});
        end
    end

    X = load(files{1}); % CTA 点云
    Y = {load(files{2}), load(files{3}), load(files{4}), load(files{5}), load(files{6}), load(files{7}), load(files{8})}; % SPECT 点云

    % 初始化输出
    rmse = zeros(1, 7);
    juli = zeros(1, 7);
    jl = zeros(1, 7);

    % 计算误差
    for i = 1:7
        Yi = Y{i};
        % 检查数据完整性
        if any(isnan(X(:))) || any(isinf(X(:)))
            warning('CTA 点云包含 NaN 或 Inf，已替换为 0');
            X(isnan(X) | isinf(X)) = 0;
        end
        if any(isnan(Yi(:))) || any(isinf(Yi(:)))
            warning('SPECT 点云 %d 包含 NaN 或 Inf，已替换为 0', i);
            Yi(isnan(Yi) | isinf(Yi)) = 0;
        end

        % 最近点搜索（使用 knnsearch 替代 delaunayn 和 dsearchn）
        [ki, disti] = knnsearch(Yi, X);
        % 计算误差
        dist2i = dot(disti, disti);
        rmse(i) = sqrt(dist2i / size(X, 1));
        juli(i) = mean(disti);
        jl(i) = juli(i) * ps;
    end
end