%%The top-left smallest model is used to confirming the accuracy of NNPFC,
%%The bottom-left model is used to the simulation of Non-linear cases,
%%The right biggest model is used to the simulation of Linear ceases,
%%causing the complex causal-possiable algorithm, while the non-linear model mustn't. 

%%The parameters of adaptive controller's iterate law in non-linear cases
%%should be adjusted in the simulink board, in linear cases it should be
%%adjusted in this matlab dashboard by "sigma" and "gamma".

%%For the paramaters of robust feedback, it can be setted in this board by
%%"rho" and "sig" either in linear or non-linear, espically the "sig"
%%denotes the parameter "$\tau$" in the atricle.

clear
close all

time = 400;%the total running time
ts = 0.005;%the discrete sampling time

omega = 3.14159265*5; % the angle frequency when the disturbance is sine wave
wav = 0.7;% the amplitude of disturbance in the sine wave or random step

z = tf('z', ts);
%swi = 1; %是否使用神经网络前馈控制
%swi_2 = 1; %是否使用对外乱输入的模拟误差
%kkk = 0;
%ep = 0.4;
%swi_rob = 1;%是否使用复杂稳健项
rho = 0.101; %the robust parameter "\rho"
sig = 0.001; %the robust parameter "\tau"
%ep_0 = 0.1;
%alp = 0.98;
%gam = 0.5;
%pp = 0.35;
%qq = 0.1;
%ml = 0;

%% 自适应控制器内部参数

sigma = 10; %误差权重e  % the adaptive law's parameter "\sigma"
gamma = 2; %前位权重1  %the adaptive law's parameter "\gamma"

%wuca = 100;
%qian = 0.1;

%hig_1 = 1e3;
%hig_2 = 2;
% the_e = 1e7;
% the_1 = 1;

%% 随机的r(t)输入 % the random step signal , which to be the source of ture input signal

seed = 17; %默认17  %the seed of random input source
ran_min = 5;  %the min of random input
ran_max = 20;  % the  max of random input
sam_time = 40; %输入切换周期 % the switching time of random input get its sample

%% 初始随机信号机 % the signal generator, using the random step singal the make the true r(t)

numerator_0 = 1;
denominator_0 = [1 2 1];
p0_s = tf(numerator_0, denominator_0);
p0_z = c2d(p0_s, ts, 'tustin');

[num0_d, den0_d] = tfdata(p0_z, 'V');
[A0, B0, C0, D0] = tf2ss(num0_d, den0_d);

% A0 = [0, 1; -0.36, 1.2];
% B0 = [0; 0.5];
% C0 = [1, 0];
% D0 = [0];
% [numerator_0, denominator_0] = ss2tf(A0, B0, C0, D0);
% p0_z = tf(numerator_0, denominator_0, ts);

%% 线性离散控制对象（双线性变换） % the assumed non-ASPR linear system (plant)

% numerator = 131.2;
% denominator = [1 6.339 30.22 128.6];
% p_s = tf(numerator, denominator);
% p_z = c2d(p_s, ts, 'tustin');
% [num_d, den_d] = tfdata(p_z, 'V');
% [A, B, C, D] = tf2ss(num_d, den_d);

A = [0.7, 0.2; 0, 0.5];
B = [1; 0.5];
C = [0.2, 0.8];
D = [0];
[numerator, denominator] = ss2tf(A, B, C, D);
p_z = tf(numerator, denominator, ts);

%% 线性连续时间控制对象  % the continuous system equal to the present disceret system (plant)

AT = [0,1;-9,-1.2];
BT = [0;1];
CT = [22.5,9];
DT = 0;

%% 联合理想概强正实（绝对正实稳定）模型 % the assumed ideal ASPR model 

num_aspr = [1, -0.6, 0.61];
den_aspr = [1, -1.2, 0.35];
p_aspr_z = tf(num_aspr, den_aspr, ts);

[num_aspr_d, den_aspr_d] = tfdata(p_aspr_z, 'V');
[A_aspr, B_aspr, C_aspr, D_aspr] = tf2ss(num_aspr_d, den_aspr_d); %如有需要可获取整体状态空间

%% 理想前馈补偿器模型 % the ideal PFC model for linear plant

p_pfc_z = p_aspr_z - p_z;

[num_pfc_d, den_pfc_d] = tfdata(p_pfc_z, 'V');
[AA, BB, CC, DD] = tf2ss(num_pfc_d, den_pfc_d);

%% 调用读取神经网络参数并运行伪链模型  % the operation board

getNN_simu; % get the NNFF  or NNROB's matrix parameter
getPFC_simu; % get the NNPFC's .......
% sim("causality_0.slx");
sim("fin_simu.slx"); % finally running 
% sim("xxx.slx");

load('linear', 'linear'); % download the data for figure in the atricle , no essential.
csvwrite('linear.csv', linear.');