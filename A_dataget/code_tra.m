clear
close all
%sampletime = 0.05;
time = 5000000;
ts = 1;
z = tf('z', ts);

%% 随机的u(t)输入

seed = 17;
ran_min = 5;
ran_max = 20;
sam_time = 40; %输入切换周期

%% 初始信号机

numerator_0 = 1;
denominator_0 = [1 2 1];
p0_s = tf(numerator_0, denominator_0);
p0_z = c2d(p0_s, ts, 'tustin');

[num0_d, den0_d] = tfdata(p0_z, 'V');
[A0, B0, C0, D0] = tf2ss(num0_d, den0_d);

%% 线性离散控制对象（双线性变换）

A = [0.7, 0.2; 0, 0.5];
B = [1; 0.5];
C = [0.2, 0.8];
D = [0];
[numerator, denominator] = ss2tf(A, B, C, D);
p_z = tf(numerator, denominator, ts);

%% 联合理想概强正实（绝对正实稳定）模型

num_aspr = [1, -0.6, 0.61];
den_aspr = [1, -1.2, 0.35];
p_aspr_z = tf(num_aspr, den_aspr, ts);

[num_aspr_d, den_aspr_d] = tfdata(p_aspr_z, 'V');
[A_aspr, B_aspr, C_aspr, D_aspr] = tf2ss(num_aspr_d, den_aspr_d); %如有需要可获取整体状态空间

%% 运行模型

sim("dataget2.slx");

%% 转格式为csv

% load('simout3', 'simout3');
% csvwrite('simout3.csv', simout3.');

load('simout2pfc', 'simout2pfc');
csvwrite('simout2pfc.csv', simout2pfc.');