function [rmse, juli, jl] = eval_gt_python(xlsx_dir, base_dir, patientname,ori)
    if ~exist(xlsx_dir, 'file')
        error('Excel  %s not exist', xlsx_dir);
    end
    data = readcell(xlsx_dir);
    namelist = data(:, 3);
    nlnum = length(namelist);

    index = find(cellfun(@(x) isequal(strrep(x, ' ', ''), patientname), namelist), 1);
    if isempty(index)
        error('can not find %s', patientname);
    end

    ps = cell2mat(data(index, 12));

    if ori
        files = {sprintf('%s/ground_truth/%s/transformed_y.txt', base_dir, patientname), ...
             sprintf('%s/cloud_result0507/%s/result_icp_manu.txt', base_dir, patientname), ...
             sprintf('%s/cloud_result0507/%s/result_sicp_manu.txt', base_dir, patientname), ...
             sprintf('%s/cloud_result0507/%s/result_rigid_manu.txt', base_dir, patientname), ...
             sprintf('%s/cloud_result0507/%s/result_affine_manu.txt', base_dir, patientname), ...
             sprintf('%s/cloud_result0507/%s/result_bcpdpp_manuy.txt', base_dir, patientname)};
    else
        files = {sprintf('%s/ground_truth/%s/transformed_y.txt', base_dir, patientname), ...
             sprintf('%s/cloud_result0507/%s/result_icp_auto.txt', base_dir, patientname), ...
             sprintf('%s/cloud_result0507/%s/result_sicp_auto.txt', base_dir, patientname), ...
             sprintf('%s/cloud_result0507/%s/result_rigid_auto.txt', base_dir, patientname), ...
             sprintf('%s/cloud_result0507/%s/result_affine_auto.txt', base_dir, patientname), ...
             sprintf('%s/cloud_result0507/%s/result_bcpdpp_autoy.txt', base_dir, patientname)};
             
    end
    
    for i = 1:length(files)
        if ~exist(files{i}, 'file')
            error('file %s not exist', files{i});
        end
    end

    X = load(files{1});
    Y = {load(files{2}), load(files{3}), load(files{4}), load(files{5}), load(files{6})}; % SPECT 点云

    rmse = zeros(1, 5);
    juli = zeros(1, 5);
    jl = zeros(1, 5);

    for i = 1:5
        Yi = Y{i};
        if any(isnan(X(:))) || any(isinf(X(:)))
            warning('NaN or Inf');
            X(isnan(X) | isinf(X)) = 0;
        end
        if any(isnan(Yi(:))) || any(isinf(Yi(:)))
            warning('NaN or Inf', i);
            Yi(isnan(Yi) | isinf(Yi)) = 0;
        end

        [ki, disti] = knnsearch(Yi, X);
        dist2i = dot(disti, disti);
        rmse(i) = sqrt(dist2i / size(X, 1));
        juli(i) = mean(disti);
        jl(i) = juli(i) * ps;
    end
end