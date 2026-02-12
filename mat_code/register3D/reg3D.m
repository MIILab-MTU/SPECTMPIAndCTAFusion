function [s,R,T,e,it]=reg3D(file1,file2)

%后续--处理下文件名的约束
% data1 = ascread(file1);%40097points
% data2 = ascread(file2);%40256points
[row1 col1]=size(file1);data1{1}=col1;data1{2}=file1;
[row2 col2]=size(file2);data2{1}=col2;data2{2}=file2;

X = data1{2};
Y = data2{2};

%初始化
initD = init3D(data1,data2);%初始化的R0有些问题,基本解决
R0 = initD{1};
T0 = initD{2};
s0 = initD{3};
I = initD{4};
o = 0.001;
%构造Y的delaunay曲面
Yo = delaunayn(Y');


tic; %记录当前命令执行的时间
%第1组数据
c0 = Solvecircle3D(s0,R0,T0,I,X,Y,Yo);%得到下一组数据
Rn = c0{1};
Tn = c0{2};
sn = c0{3};
etemp = c0{5};
q=1;
flag = 2;
while (q>o)
    c = Solvecircle3D(sn,Rn,Tn,I,X,Y,Yo);
    Rn = c{1};
    Tn = c{2};
    sn = c{3};
    en = c{5};
    q = 1 - en/etemp;
    etemp = en;
    flag = flag + 1; 
end 
s = sn;
R = Rn;
T = Tn;
e = en;
it = flag - 1;
toc 
end

