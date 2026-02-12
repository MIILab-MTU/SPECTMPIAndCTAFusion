function result = init(data1,data2)
pointx = data1{1};
X = data1{2};
pointy = data2{1};
Y = data2{2};
%mean(x,2)求每一行均值的列向量
xc = mean(X,2); %点集1的中心即均值
yc = mean(Y,2); %点集2的中心即均值
%B = repmat(A,[m n])将A复制m*n块，即B由m*n块A平铺而成
x1 = X - repmat(xc,[1 pointx]);
Mx =x1 * x1';

y1 = Y - repmat(yc,[1 pointy]);
My = y1 * y1';
%求特征向量和特征值
[Vx,Dx] = eig(Mx,'nobalance')
[Vy,Dy] = eig(My,'nobalance')

%s的初始值
sq = sum(sqrt(Dy/Dx));
% sq = sqrt(Dy/Dx); %我自己认为应该是这样的
s = sum(sq)/3;
%I
I = [min(sq),max(sq)];

% %R存在问题
% %判断特征向量矩阵方向
% p1 = Vx(:,1);
% p2 = Vx(:,2);
% q1 = Vy(:,1);
% q2 = Vy(:,2);
% q3 = Vy(:,3);
% f = 0.8;   %f是阀值,为啥这样？
% if dot(p1,q1) < f
%     p1 = -p1;
% end
% if dot(p2,q2)<f
%    p2 = -p2;
% end
% p3 = cross(p1,p2);
% R = [q1,q2,q3]*inv([p1,p2,p3]);

Sx = Vx/sum(Vx); %归一化特征向量
Sy = Vy/sum(Vy);
p1 = Sx(:,1);
p2 = Sx(:,2);
p3 = Sx(:,3);
q1 = Sy(:,1);
q2 = Sy(:,2);
q3 = Sy(:,3);
R = [q1,q2,q3]*inv([p1,p2,p3]);

%T重新求
%T = (yc - xc);
xc2 = mean(s*R*X,2);
T = (yc - xc2);
result = cell({R;T;s;I});

