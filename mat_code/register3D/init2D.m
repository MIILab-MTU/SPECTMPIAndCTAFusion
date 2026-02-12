function result = init(data1,data2)
pointx = data1{1};
X0 = data1{2};
X = X0(1:2,:);%SJTang
pointy = data2{1};
Y0 = data2{2};
Y = Y0(1:2,:);%SJTang

xc = mean(X,2);
yc = mean(Y,2);

x1 = X - repmat(xc,[1 pointx]);
Mx = x1 * x1';

y1 = Y - repmat(yc,[1 pointy]);
My = y1 * y1';

[Vx,Dx] = eig(Mx,'nobalance');
[Vy,Dy] = eig(My,'nobalance');

%s
sq = sum(sqrt(Dy/Dx));
% s = sum(sq)/3;
s = [sq 1];%SJTang


%I
I = [min(sq),max(sq)];

%R存在问题
%判断特征向量矩阵方向
p1 = Vx(:,1);
p2 = Vx(:,2);
q1 = Vy(:,1);
q2 = Vy(:,2);
% q3 = Vy(:,3);
f = 0.1;   %f是阀值
if dot(p1,q1) < f
    p1 = -p1;
end
if dot(p2,q2)<f
   p2 = -p2;
end
% p3 = cross(p1,p2);
% R = [q1,q2,q3]*inv([p1,p2,p3]);
R = [q1,q2]*inv([p1,p2]);%SJTang
R(3,3) = 1;%SJTang

%T重新求
yc = mean(Y0,2);%SJTang
%T = (yc - xc);
% xc2 = mean(s*R*X0,2);
xc2 = mean(diag(s)*R*X0,2);%SJTang
T = (yc - xc2);
result = cell({R;T;s;I});

