% Computes the Mean Absolute Difference (MAD) for thegiven two blocks（对给定的两个块计算最小绝对差）

% Input

%??????currentBlk : The block for which we are finding the MAD（当前块）

%?????? refBlk :the block w.r.t. which the MAD is being computed（参考块）

%?????? n : theside of the two square blocks

%

% Output

%?????? cost :The MAD for the two blocks（两个块的最小绝对差）

%

% Written by Aroh Barjatya

% 定义函数文件costFuncMAD.m，currentBlk、refBlk、 n为传入参数，cost为返回参数

function cost = costFuncMAD(currentBlk,refBlk, n)

% 补充下面程序

cost=sum(sum(abs(currentBlk-refBlk)))/(n*n);

