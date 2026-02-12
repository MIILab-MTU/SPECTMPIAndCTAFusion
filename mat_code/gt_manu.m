classdef gt_manu < matlab.apps.AppBase
    properties (Access = public)
        UIFigure
        UIAxes
        TxSlider, TySlider, TzSlider
        RxSlider, RySlider, RzSlider
        SSlider
        ResetButton, SaveButton
        ErrorLabel
        X, Y, Y_transformed
        s, R, t
    end

    methods (Access = private)
        function startupFcn(app)
            % 加载点云
            try
                app.X = dlmread('E:/proj/peizhun/file_process/name/chenhongming/ijkcta.txt', ' ', 0, 0);
                app.Y = dlmread('E:/proj/peizhun/file_process/cloud_result0507/chenhongming/cu.txt', ' ', 0, 0);
            catch e
                error('加载点云失败: %s', e.message);
            end
            
            % 打印维度
            fprintf('X 维度: %s\n', mat2str(size(app.X)));
            fprintf('Y 维度: %s\n', mat2str(size(app.Y)));
            
            % 确保点云是 M x 3
            if size(app.X, 2) ~= 3 || size(app.Y, 2) ~= 3
                error('点云必须是 M x 3 矩阵，X: %s, Y: %s', mat2str(size(app.X)), mat2str(size(app.Y)));
            end
            
            % 初始化变换
            app.s = 1;
            app.R = eye(3);
            app.t = [0, 0, 0];
            
            % 打印变换参数维度
            fprintf('s 维度: %s\n', mat2str(size(app.s)));
            fprintf('R 维度: %s\n', mat2str(size(app.R)));
            fprintf('t 维度: %s\n', mat2str(size(app.t)));
            
            % 检查变换
            app.Y_transformed = app.apply_transform(app.Y, app.s, app.R, app.t);
            
            % 显示点云
            app.update_plot();
            
            % 初始化滑块
            app.TxSlider.Value = 0;
            app.TySlider.Value = 0;
            app.TzSlider.Value = 0;
            app.RxSlider.Value = 0;
            app.RySlider.Value = 0;
            app.RzSlider.Value = 0;
            app.SSlider.Value = 1;
        end

        function Y_transformed = apply_transform(~, Y, s, R, t)
            % 确保 Y 是 M x 3
            if size(Y, 2) ~= 3
                error('Y 必须是 M x 3 矩阵，当前维度: %s', mat2str(size(Y)));
            end
            % 确保 R 是 3 x 3
            if ~isequal(size(R), [3 3])
                error('R 必须是 3 x 3 矩阵，当前维度: %s', mat2str(size(R)));
            end
            % 确保 t 是 1 x 3
            if ~isequal(size(t), [1 3])
                t = reshape(t, 1, 3);
            end
            % 应用变换
            Y_transformed = s * Y * R' + t;
        end

        function update_plot(app)
            cla(app.UIAxes);
            plot3(app.UIAxes, app.Y_transformed(:,1), app.Y_transformed(:,2), app.Y_transformed(:,3), '.r', 'MarkerSize', 5); hold(app.UIAxes, 'on');
            plot3(app.UIAxes, app.X(:,1), app.X(:,2), app.X(:,3), '.b', 'MarkerSize', 5);
            daspect(app.UIAxes, [1 1 1]); grid(app.UIAxes, 'on');
            title(app.UIAxes, 'Manual Point Cloud Registration');
            
            % 计算误差
            try
                D = pdist2(app.Y_transformed, app.X, 'euclidean');
                err = mean(min(D, [], 2));
                app.ErrorLabel.Text = sprintf('Hausdorff Error: %.4f', err);
            catch
                app.ErrorLabel.Text = 'Hausdorff Error: N/A';
            end
        end

        function SliderValueChanged(app, ~)
            % 获取滑块值
            tx = app.TxSlider.Value;
            ty = app.TySlider.Value;
            tz = app.TzSlider.Value;
            rx = deg2rad(app.RxSlider.Value);
            ry = deg2rad(app.RySlider.Value);
            rz = deg2rad(app.RzSlider.Value);
            s = app.SSlider.Value;

            % 计算旋转矩阵
            Rx = [1 0 0; 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)];
            Ry = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)];
            Rz = [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1];
            app.R = Rz * Ry * Rx;
            app.t = [tx, ty, tz];
            app.s = s;

            % 更新变换
            app.Y_transformed = app.apply_transform(app.Y, app.s, app.R, app.t);
            app.update_plot();
        end

        function ResetButtonPushed(app, ~)
            % 重置变换
            app.s = 1;
            app.R = eye(3);
            app.t = [0, 0, 0];
            app.TxSlider.Value = 0;
            app.TySlider.Value = 0;
            app.TzSlider.Value = 0;
            app.RxSlider.Value = 0;
            app.RySlider.Value = 0;
            app.RzSlider.Value = 0;
            app.SSlider.Value = 1;
            app.Y_transformed = app.apply_transform(app.Y, app.s, app.R, app.t);
            app.update_plot();
        end

        function SaveButtonPushed(app, ~)
            % 确保输出目录存在
            output_dir = 'E:/proj/peizhun/file_process/ground_truth/chenhongming';
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end
            % 保存变换和点云
            s = app.s;
            R = app.R;
            t = app.t;
            save('E:/proj/peizhun/file_process/ground_truth/chenhongming/transform.mat', 's', 'R', 't');
            save('E:/proj/peizhun/file_process/ground_truth/chenhongming/transformed_y.txt', 'app.Y_transformed', '-ascii');
            msgbox('Transformation saved!');
        end
    end

    methods (Access = public)
        function app = gt_manu
            % 创建 GUI
            app.UIFigure = uifigure('Name', 'Manual Point Cloud Registration', 'Position', [100, 100, 600, 450]);
            app.UIAxes = uiaxes(app.UIFigure, 'Position', [10, 10, 400, 400]);

            % 平移滑块
            app.TxSlider = uislider(app.UIFigure, 'Position', [420, 400, 150, 3], 'Limits', [-50, 50], 'ValueChangedFcn', @(src, event) app.SliderValueChanged());
            uilabel(app.UIFigure, 'Position', [420, 420, 50, 22], 'Text', 'Tx');
            app.TySlider = uislider(app.UIFigure, 'Position', [420, 350, 150, 3], 'Limits', [-50, 50], 'ValueChangedFcn', @(src, event) app.SliderValueChanged());
            uilabel(app.UIFigure, 'Position', [420, 370, 50, 22], 'Text', 'Ty');
            app.TzSlider = uislider(app.UIFigure, 'Position', [420, 300, 150, 3], 'Limits', [-50, 50], 'ValueChangedFcn', @(src, event) app.SliderValueChanged());
            uilabel(app.UIFigure, 'Position', [420, 320, 50, 22], 'Text', 'Tz');

            % 旋转滑块
            app.RxSlider = uislider(app.UIFigure, 'Position', [420, 250, 150, 3], 'Limits', [-180, 180], 'ValueChangedFcn', @(src, event) app.SliderValueChanged());
            uilabel(app.UIFigure, 'Position', [420, 270, 50, 22], 'Text', 'Rx');
            app.RySlider = uislider(app.UIFigure, 'Position', [420, 200, 150, 3], 'Limits', [-180, 180], 'ValueChangedFcn', @(src, event) app.SliderValueChanged());
            uilabel(app.UIFigure, 'Position', [420, 220, 50, 22], 'Text', 'Ry');
            app.RzSlider = uislider(app.UIFigure, 'Position', [420, 150, 150, 3], 'Limits', [-180, 180], 'ValueChangedFcn', @(src, event) app.SliderValueChanged());
            uilabel(app.UIFigure, 'Position', [420, 170, 50, 22], 'Text', 'Rz');

            % 缩放滑块
            app.SSlider = uislider(app.UIFigure, 'Position', [420, 100, 150, 3], 'Limits', [0.5, 1.5], 'Value', 1, 'ValueChangedFcn', @(src, event) app.SliderValueChanged());
            uilabel(app.UIFigure, 'Position', [420, 120, 50, 22], 'Text', 'Scale');

            % 按钮
            app.ResetButton = uibutton(app.UIFigure, 'Position', [420, 60, 100, 22], 'Text', 'Reset', 'ButtonPushedFcn', @(src, event) app.ResetButtonPushed());
            app.SaveButton = uibutton(app.UIFigure, 'Position', [420, 20, 100, 22], 'Text', 'Save', 'ButtonPushedFcn', @(src, event) app.SaveButtonPushed());

            % 误差显示
            app.ErrorLabel = uilabel(app.UIFigure, 'Position', [420, 450, 150, 22], 'Text', 'Hausdorff Error: 0');

            % 初始化
            app.startupFcn();
        end
    end
end