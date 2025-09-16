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
%swi = 1; %ニューラルネットワーク前馈制御を使用するかどうか
%swi_2 = 1; %外乱入力のシミュレーション誤差を使用するかどうか
%kkk = 0;
%ep = 0.4;
%swi_rob = 1;%複雑なロバスト項を使用するかどうか
rho = 0.101; %the robust parameter "\rho"
sig = 0.001; %the robust parameter "\tau"
%ep_0 = 0.1;
%alp = 0.98;
%gam = 0.5;
%pp = 0.35;
%qq = 0.1;
%ml = 0;

%% 適応制御器内部パラメータ

sigma = 10; %誤差重みe  % the adaptive law's parameter "\sigma"
gamma = 2; %前位重み1  %the adaptive law's parameter "\gamma"

%wuca = 100;
%qian = 0.1;

%hig_1 = 1e3;
%hig_2 = 2;
% the_e = 1e7;
% the_1 = 1;

%% ランダムなr(t)入力 % the random step signal , which to be the source of ture input signal

seed = 17; %デフォルト17  %the seed of random input source
ran_min = 5;  %the min of random input
ran_max = 20;  % the  max of random input
sam_time = 40; %入力切り替え周期 % the switching time of random input get its sample

%% 初期ランダム信号生成器 % the signal generator, using the random step singal the make the true r(t)

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

%% 線形離散制御対象（双線形変換） % the assumed non-ASPR linear system (plant)

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

%% 線形連続時間制御対象  % the continuous system equal to the present disceret system (plant)

AT = [0,1;-9,-1.2];
BT = [0;1];
CT = [22.5,9];
DT = 0;

%% 結合理想概強正実（絶対正実安定）モデル % the assumed ideal ASPR model 

num_aspr = [1, -0.6, 0.61];
den_aspr = [1, -1.2, 0.35];
p_aspr_z = tf(num_aspr, den_aspr, ts);

[num_aspr_d, den_aspr_d] = tfdata(p_aspr_z, 'V');
[A_aspr, B_aspr, C_aspr, D_aspr] = tf2ss(num_aspr_d, den_aspr_d); %必要に応じて全体状態空間を取得可能

%% 理想前馈補償器モデル % the ideal PFC model for linear plant

p_pfc_z = p_aspr_z - p_z;

[num_pfc_d, den_pfc_d] = tfdata(p_pfc_z, 'V');
[AA, BB, CC, DD] = tf2ss(num_pfc_d, den_pfc_d);

%% ニューラルネットワークパラメータ読み取りと疑似チェーンモデル実行の呼び出し  % the operation board

getNN_simu; % get the NNFF  or NNROB's matrix parameter
getPFC_simu; % get the NNPFC's .......
% sim("causality_0.slx");
sim("fin_simu.slx"); % finally running 
% sim("xxx.slx");

load('linear', 'linear'); % 論文図表用データダウンロード、必須ではない
csvwrite('linear.csv', linear.');

%% グラフ出力処理
data = linear.';  % データを転置して行列形式に変換
time_data = data(:,1);  % 時間データ（1列目）
signals = data(:,2:end);  % 信号データ（2列目以降）

% 図1: 全信号の時間応答
figure(1);
plot(time_data, signals);
grid on;
title('シミュレーション結果 - 全信号時間応答');
xlabel('時間 [秒]');
ylabel('信号値');
legend('制御入力 u(t)', '出力 y(t)', '参照信号 r(t)', '外乱 d(t)', '制御誤差 e(t)', 'Location', 'best');

% 図2: 制御入力と出力の比較
figure(2);
subplot(2,1,1);
plot(time_data, signals(:,1), 'b-', 'LineWidth', 1.5);
grid on;
title('制御入力 u(t)');
xlabel('時間 [秒]');
ylabel('制御入力');

subplot(2,1,2);
plot(time_data, signals(:,2), 'r-', 'LineWidth', 1.5);
grid on;
title('プラント出力 y(t)');
xlabel('時間 [秒]');
ylabel('出力');

% 図3: 参照信号と出力の追従性能
if size(signals, 2) >= 3
    figure(3);
    plot(time_data, signals(:,3), 'g--', 'LineWidth', 2, 'DisplayName', '参照信号 r(t)');
    hold on;
    plot(time_data, signals(:,2), 'r-', 'LineWidth', 1.5, 'DisplayName', '出力 y(t)');
    grid on;
    title('追従性能 - 参照信号と出力の比較');
    xlabel('時間 [秒]');
    ylabel('信号値');
    legend('show', 'Location', 'best');
    hold off;
end

% 図4: 制御誤差（最後の列を誤差と仮定）
if size(signals, 2) >= 5
    figure(4);
    plot(time_data, signals(:,end), 'm-', 'LineWidth', 1.5);
    grid on;
    title('制御誤差 e(t)');
    xlabel('時間 [秒]');
    ylabel('誤差');
end

% グラフをPNGファイルとして保存
saveas(figure(1), 'simulation_results_all.png');
saveas(figure(2), 'control_input_output.png');
if size(signals, 2) >= 3
    saveas(figure(3), 'tracking_performance.png');
end
if size(signals, 2) >= 5
    saveas(figure(4), 'control_error.png');
end

fprintf('グラフが正常に生成され、PNGファイルとして保存されました。\n');