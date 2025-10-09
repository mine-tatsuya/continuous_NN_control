%% 左上の最小モデルはNNPFCの精度確認に使用
%% 左下のモデルは非線形ケースのシミュレーションに使用
%% 右の最大モデルは線形ケースのシミュレーションに使用
%% 複雑な因果可能性アルゴリズムを引き起こし、非線形モデルでは必要ない

%% 非線形ケースの適応制御器の反復法則のパラメータは
%% Simulinkボードで調整する必要があり、線形ケースでは
%% このMATLABダッシュボードで"sigma"と"gamma"によって調整する

clear
close all

time = 400; % 総実行時間
Td = 0.005; % 離散サンプリング時間

omega = 3.14159265*0.1; % 外乱が正弦波の時の角周波数
wav = 1.5; % 正弦波またはランダムステップでの外乱の振幅

rho = -2; % ロバストパラメータ"ρ"

 %% ログ設定（テキスト + MAT）
% if ~exist('logs', 'dir')
%     mkdir('logs');
% end
 timestamp = datestr(now, 'yyyymmdd_HHMMSS');
% log_txt = fullfile('logs', ['sim_log_' timestamp '.txt']);
% log_mat = fullfile('logs', ['workspace_' timestamp '.mat']);
% diary(log_txt); diary on;
fprintf('=== Simulation Log (%s) ===\n', timestamp);
% 最小限の出力: ABCD と伝達関数のみ

%% 適応制御器内部パラメータ

sigma = 0.1; % 誤差重みe - 適応法則のパラメータ"σ"
gamma = 10000; % 前位重み1 - 適応法則のパラメータ"γ"

%% ランダムなr(t)入力 - 真の入力信号の源となるランダムステップ信号

seed = 1;%randi(2^31-2); % デフォルト17 - ランダム入力源のシード
ran_min = 5; % ランダム入力の最小値
ran_max = 20; % ランダム入力の最大値
sam_time = 40; % 入力切り替え周期 - ランダム入力がサンプルを取得する切り替え時間

% %% 初期ランダム信号生成器 - ランダムステップ信号を使用して真のr(t)を作る信号生成器
% 
% % 安定な2次システム（減衰振動）に変更
% % 伝達関数: G(s) = 1/(s^2 + 2*0.7*s + 1) = 1/(s^2 + 1.4s + 1)
% % 減衰比: ζ = 0.7, 固有角周波数: ωn = 1 rad/s
% % 連続系として定義
% A0 = [0, 1; -1, -1.4];
% B0 = [0; 1];
% C0 = [1, 0];
% D0 = [0];
% [numerator_0, denominator_0] = ss2tf(A0, B0, C0, D0);
% p0_s = tf(numerator_0, denominator_0); % 連続系
% fprintf('\n[Generator] A0=\n'); disp(A0);
% fprintf('[Generator] B0=\n'); disp(B0);
% fprintf('[Generator] C0=\n'); disp(C0);
% fprintf('[Generator] D0=\n'); disp(D0);
% fprintf('[Generator] Transfer function (p0_s - continuous):\n');
% fprintf('%s', evalc('disp(p0_s)'));
% 
% %% ========== 連続系用パラメータ（fin_simu_continuous.slx用） ==========
% %% code_simu.mから連続系パラメータを使用
% 
% %% 連続系プラント（code_simu.mの連続時間制御対象）
% AT = [0, 1; -9, -1.2];
% BT = [0; 1];
% CT = [22.5, 9];
% DT = 0;
% [numerator_T, denominator_T] = ss2tf(AT, BT, CT, DT);
% p_s = tf(numerator_T, denominator_T); % 連続系プラント伝達関数
% fprintf('\n[Continuous Plant] AT=\n'); disp(AT);
% fprintf('[Continuous Plant] BT=\n'); disp(BT);
% fprintf('[Continuous Plant] CT=\n'); disp(CT);
% fprintf('[Continuous Plant] DT=\n'); disp(DT);
% fprintf('[Continuous Plant] Transfer function (p_s):\n');
% fprintf('%s', evalc('disp(p_s)'));
% 
% %% 連続系ASPRモデル（連続系プラントをベースに設計）
% % 簡易的なASPRモデル: 1次系 G_aspr(s) = (0.1s + 1)/(s + 1)
% num_aspr_s = [0.1, 1];
% den_aspr_s = [1, 1];
% p_aspr_s = tf(num_aspr_s, den_aspr_s);
% [num_aspr_s_d, den_aspr_s_d] = tfdata(p_aspr_s, 'V');
% [A_aspr_s, B_aspr_s, C_aspr_s, D_aspr_s] = tf2ss(num_aspr_s_d, den_aspr_s_d);
% fprintf('\n[Continuous ASPR Model] A_aspr_s=\n'); disp(A_aspr_s);
% fprintf('[Continuous ASPR Model] B_aspr_s=\n'); disp(B_aspr_s);
% fprintf('[Continuous ASPR Model] C_aspr_s=\n'); disp(C_aspr_s);
% fprintf('[Continuous ASPR Model] D_aspr_s=\n'); disp(D_aspr_s);
% fprintf('[Continuous ASPR Model] Transfer function (p_aspr_s):\n');
% fprintf('%s', evalc('disp(p_aspr_s)'));

% %% 連続系PFCモデル（ASPR - プラント）
% p_pfc_s = p_aspr_s - p_s;
% [num_pfc_s_d, den_pfc_s_d] = tfdata(p_pfc_s, 'V');
% [AA_s, BB_s, CC_s, DD_s] = tf2ss(num_pfc_s_d, den_pfc_s_d);
% fprintf('\n[Continuous PFC Model] AA_s=\n'); disp(AA_s);
% fprintf('[Continuous PFC Model] BB_s=\n'); disp(BB_s);
% fprintf('[Continuous PFC Model] CC_s=\n'); disp(CC_s);
% fprintf('[Continuous PFC Model] DD_s=\n'); disp(DD_s);
% fprintf('[Continuous PFC Model] Transfer function (p_pfc_s):\n');
% fprintf('%s', evalc('disp(p_pfc_s)'));

%% ニューラルネットワークパラメータ読み取りと疑似チェーンモデル実行の呼び出し - 操作ボード

getNN_simu; % NNFFまたはNNROBの行列パラメータを取得
%getPFC_simu; % NNPFCのパラメータを取得
sim("fin_simu_continuous.slx"); % 最終実行

%% ログ保存と終了処理
%save(log_mat);
%fprintf('\nSaved log text to: %s\n', log_txt);
%fprintf('Saved workspace to: %s\n', log_mat);
fprintf('シミュレーションは正常に 終了しました。\n')
%diary off;
