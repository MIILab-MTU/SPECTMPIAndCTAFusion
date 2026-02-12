classdef gt_cusor < matlab.apps.AppBase
    properties (Access = public)
        UIFigure
        UIAxes
        RegistrationModeToggle
        StretchModeToggle
        ResetButton, SaveButton
        ErrorLabel
        X, Y, Y_transformed
        sx, sy, sz
        R, t
        centroid
        isRegistrationMode
        isStretchMode
        isDragging
        dragMode
        lastMousePos
        stretchPoint
    end

    methods (Access = private)
        function startupFcn(app)
            try
                app.X = dlmread('E:/proj/peizhun/file_process/name/chenhongming/ijkcta.txt', ' ', 0, 0);
                app.Y = load('E:/proj/peizhun/file_process/cloud_result0507/chenhongming/result_bcpdpp_autoy.txt');
            catch e
                error('加载点云失败: %s', e.message);
            end
            
            fprintf('X 维度: %s\n', mat2str(size(app.X)));
            fprintf('Y 维度: %s\n', mat2str(size(app.Y)));
            
            if size(app.X, 2) ~= 3 || size(app.Y, 2) ~= 3
                error('点云必须是 M x 3 矩阵，X: %s, Y: %s', mat2str(size(app.X)), mat2str(size(app.Y)));
            end
            
            app.sx = 1;
            app.sy = 1;
            app.sz = 1;
            app.R = eye(3);
            app.t = [0, 0, 0];
            app.centroid = mean(app.Y);
            app.isRegistrationMode = false;
            app.isStretchMode = false;
            app.isDragging = false;
            app.dragMode = '';
            app.lastMousePos = [];
            app.stretchPoint = [];
            
            fprintf('sx, sy, sz 维度: [%s, %s, %s]\n', mat2str(size(app.sx)), mat2str(size(app.sy)), mat2str(size(app.sz)));
            fprintf('R 维度: %s\n', mat2str(size(app.R)));
            fprintf('t 维度: %s\n', mat2str(size(app.t)));
            fprintf('质心: %s\n', mat2str(app.centroid));
            
            app.Y_transformed = app.apply_transform(app.Y, app.sx, app.sy, app.sz, app.R, app.t, app.centroid);
            
            app.update_plot();
            
            set(app.RegistrationModeToggle, 'Value', 0);
            set(app.StretchModeToggle, 'Value', 0);
        end

        function Y_transformed = apply_transform(~, Y, sx, sy, sz, R, t, centroid)
            if size(Y, 2) ~= 3
                error('Y 必须是 M x 3 矩阵，当前维度: %s', mat2str(size(Y)));
            end
            if ~isequal(size(R), [3 3])
                error('R 必须是 3 x 3 矩阵，当前维度: %s', mat2str(size(R)));
            end
            if ~isequal(size(t), [1 3])
                t = reshape(t, 1, 3);
            end
            if ~isequal(size(centroid), [1 3])
                centroid = reshape(centroid, 1, 3);
            end
            S = diag([sx, sy, sz]);
            Y_transformed = (Y - centroid) * S * R' + centroid + t;
        end

        function update_plot(app)
            cla(app.UIAxes);
            plot3(app.UIAxes, app.Y_transformed(:,1), app.Y_transformed(:,2), app.Y_transformed(:,3), '.r', 'MarkerSize', 5); hold(app.UIAxes, 'on');
            plot3(app.UIAxes, app.X(:,1), app.X(:,2), app.X(:,3), '.b', 'MarkerSize', 5);
            daspect(app.UIAxes, [1 1 1]); grid(app.UIAxes, 'on');
            title(app.UIAxes, 'Manual Point Cloud Registration');
            
            try
                D = pdist2(app.Y_transformed, app.X, 'euclidean');
                err = mean(min(D, [], 2));
                set(app.ErrorLabel, 'String', sprintf('Hausdorff Error: %.4f', err));
            catch
                set(app.ErrorLabel, 'String', 'Hausdorff Error: N/A');
            end
        end

        function RegistrationModeToggled(app, src, ~)
            app.isRegistrationMode = get(src, 'Value');
            if app.isRegistrationMode
                set(app.UIAxes, 'Interactions', {});
                set(app.RegistrationModeToggle, 'String', '配准模式 (锁定)');
            else
                set(app.UIAxes, 'Interactions', 'rotate');
                set(app.RegistrationModeToggle, 'String', '视角模式');
                app.isDragging = false;
                app.dragMode = '';
                app.lastMousePos = [];
            end
        end

        function StretchModeToggled(app, src, ~)
            app.isStretchMode = get(src, 'Value');
            if app.isStretchMode
                set(app.StretchModeToggle, 'String', '拉伸模式 (启用)');
            else
                set(app.StretchModeToggle, 'String', '拉伸模式');
                app.stretchPoint = [];
            end
        end

        function MouseDown(app, src, event)
            if ~app.isRegistrationMode
                return;
            end
            if strcmp(get(src, 'SelectionType'), 'normal')
                if app.isStretchMode
                    mousePos = get(app.UIAxes, 'CurrentPoint');
                    mousePos = mousePos(1, 1:3);
                    D = pdist2(mousePos, app.Y_transformed);
                    [~, idx] = min(D);
                    app.stretchPoint = app.Y_transformed(idx, :);
                    app.dragMode = 'stretch';
                else
                    app.dragMode = 'translate';
                end
            elseif strcmp(get(src, 'SelectionType'), 'alt')
                app.dragMode = 'rotate';
            else
                return;
            end
            app.isDragging = true;
            app.lastMousePos = get(app.UIAxes, 'CurrentPoint');
            app.lastMousePos = app.lastMousePos(1, 1:2);
        end

        function MouseMotion(app, ~, ~)
            if ~app.isRegistrationMode || ~app.isDragging
                return;
            end
            currentPos = get(app.UIAxes, 'CurrentPoint');
            currentPos = currentPos(1, 1:2);
            delta = currentPos - app.lastMousePos;
            
            if strcmp(app.dragMode, 'translate')
                app.t(1) = app.t(1) + delta(1) * 0.1;
                app.t(2) = app.t(2) + delta(2) * 0.1;
            elseif strcmp(app.dragMode, 'rotate')
                rx = deg2rad(delta(2) * 0.5);
                ry = deg2rad(delta(1) * 0.5);
                Rx = [1 0 0; 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)];
                Ry = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)];
                app.R = app.R * (Ry * Rx);
            elseif strcmp(app.dragMode, 'stretch')
                app.sx = app.sx + delta(1) * 0.01;
                app.sy = app.sy + delta(2) * 0.01;
                app.sx = max(0.5, min(1.5, app.sx));
                app.sy = max(0.5, min(1.5, app.sy));
            end
            
            app.Y_transformed = app.apply_transform(app.Y, app.sx, app.sy, app.sz, app.R, app.t, app.centroid);
            app.update_plot();
            app.lastMousePos = currentPos;
        end

        function MouseUp(app, ~, ~)
            app.isDragging = false;
            app.dragMode = '';
            app.lastMousePos = [];
        end

        function MouseScroll(app, src, event)
            if ~app.isRegistrationMode
                return;
            end
            delta = event.VerticalScrollCount;
            scaleFactor = 0.01;
            app.sx = app.sx * (1 - delta * scaleFactor);
            app.sy = app.sy * (1 - delta * scaleFactor);
            app.sz = app.sz * (1 - delta * scaleFactor);
            app.sx = max(0.5, min(1.5, app.sx));
            app.sy = max(0.5, min(1.5, app.sy));
            app.sz = max(0.5, min(1.5, app.sz));
            
            app.Y_transformed = app.apply_transform(app.Y, app.sx, app.sy, app.sz, app.R, app.t, app.centroid);
            app.update_plot();
        end

        function ResetButtonPushed(app, ~)
            app.sx = 1;
            app.sy = 1;
            app.sz = 1;
            app.R = eye(3);
            app.t = [0, 0, 0];
            app.Y_transformed = app.apply_transform(app.Y, app.sx, app.sy, app.sz, app.R, app.t, app.centroid);
            app.update_plot();
        end

        function SaveButtonPushed(app, ~)
            output_dir = 'E:/proj/peizhun/file_process/ground_truth/chenhongming';
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end
            sx = app.sx;
            sy = app.sy;
            sz = app.sz;
            R = app.R;
            t = app.t;
            result = app.Y_transformed;
            save('E:/proj/peizhun/file_process/ground_truth/chenhongming/transform.mat', 'sx', 'sy', 'sz', 'R', 't');
            save('E:/proj/peizhun/file_process/ground_truth/chenhongming/transformed_y.txt', 'result', '-ascii');
            msgbox('Transformation saved!');
        end
    end

    methods (Access = public)
        function app = gt_cusor
            app.UIFigure = figure('Name', 'Manual Point Cloud Registration', 'Position', [100, 100, 600, 450]);
            app.UIAxes = axes('Parent', app.UIFigure, 'Position', [10, 10, 400, 400]/600);

            app.RegistrationModeToggle = uicontrol('Parent', app.UIFigure, 'Style', 'togglebutton', 'Position', [420, 400, 150, 22], 'String', '视角模式', 'Callback', @(src, event) app.RegistrationModeToggled(src, event));
            
            app.StretchModeToggle = uicontrol('Parent', app.UIFigure, 'Style', 'togglebutton', 'Position', [420, 350, 150, 22], 'String', '拉伸模式', 'Callback', @(src, event) app.StretchModeToggled(src, event));

            app.ResetButton = uicontrol('Parent', app.UIFigure, 'Style', 'pushbutton', 'Position', [420, 60, 100, 22], 'String', 'Reset', 'Callback', @(src, event) app.ResetButtonPushed());
            app.SaveButton = uicontrol('Parent', app.UIFigure, 'Style', 'pushbutton', 'Position', [420, 20, 100, 22], 'String', 'Save', 'Callback', @(src, event) app.SaveButtonPushed());

            app.ErrorLabel = uicontrol('Parent', app.UIFigure, 'Style', 'text', 'Position', [420, 450, 150, 22], 'String', 'Hausdorff Error: 0');

            set(app.UIFigure, 'WindowButtonDownFcn', @(src, event) app.MouseDown(src, event));
            set(app.UIFigure, 'WindowButtonMotionFcn', @(src, event) app.MouseMotion(src, event));
            set(app.UIFigure, 'WindowButtonUpFcn', @(src, event) app.MouseUp(src, event));
            set(app.UIFigure, 'WindowScrollWheelFcn', @(src, event) app.MouseScroll(src, event));

            app.startupFcn();
        end
    end
end