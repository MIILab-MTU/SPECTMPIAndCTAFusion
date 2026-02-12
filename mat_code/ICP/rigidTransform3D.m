%% 计算两个点集p，q的刚性变换参数，p和q的大小要一致
function [R, t] = rigidTransform3D(p, q)
n = cast(size(p, 1), 'like', p);
m = cast(size(q, 1), 'like', q);
% 去中心化
pmean = sum(p,1)/n;
p2 = bsxfun(@minus, p, pmean);
qmean = sum(q,1)/m;
q2 = bsxfun(@minus, q, qmean);
% 对协方差矩阵进行SVD分解
C = p2'*q2;
[U,~,V] = svd(C);
R = V*diag([1 1 sign(det(U*V'))])*U';
t = qmean' - R*pmean';
end
