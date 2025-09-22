%% 左上の最小モデルはNNPFCの精度確認に使用
%% 左下のモデルは非線形ケースのシミュレーションに使用
%% 右の最大モデルは線形ケースのシミュレーションに使用
%% 複雑な因果可能性アルゴリズムを引き起こし、非線形モデルでは必要ない

%% 非線形ケースの適応制御器の反復法則のパラメータは
%% Simulinkボードで調整する必要があり、線形ケースでは
%% このMATLABダッシュボードで"sigma"と"gamma"によって調整する

%% ロバストフィードバックのパラメータは、線形または非線形のいずれでも
%% このボードで"rho"と"sig"によって設定でき、特に"sig"は
%% 論文のパラメータ"τ"を表す

clear
close all

time = 400; % 総実行時間
ts = 0.005; % 離散サンプリング時間

omega = 3.14159265*5; % 外乱が正弦波の時の角周波数
wav = 0.7; % 正弦波またはランダムステップでの外乱の振幅

% 連続系設計のため、離散用のzは不要

rho = 0.5; % ロバストパラメータ"ρ"
sig = 0.001; % ロバストパラメータ"τ"

%% ログ設定（テキスト + MAT）
if ~exist('logs', 'dir')
    mkdir('logs');
end
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
log_txt = fullfile('logs', ['sim_log_' timestamp '.txt']);
log_mat = fullfile('logs', ['workspace_' timestamp '.mat']);
diary(log_txt); diary on;
fprintf('=== Simulation Log (%s) ===\n', timestamp);
% 最小限の出力: ABCD と伝達関数のみ

%% 適応制御器内部パラメータ

sigma = 10; % 誤差重みe - 適応法則のパラメータ"σ"
gamma = 2; % 前位重み1 - 適応法則のパラメータ"γ"

%% ランダムなr(t)入力 - 真の入力信号の源となるランダムステップ信号

seed = 17; % デフォルト17 - ランダム入力源のシード
ran_min = 5; % ランダム入力の最小値
ran_max = 20; % ランダム入力の最大値
sam_time = 40; % 入力切り替え周期 - ランダム入力がサンプルを取得する切り替え時間

%% 初期ランダム信号生成器 - ランダムステップ信号を使用して真のr(t)を作る信号生成器

% 安定な2次システム（減衰振動）に変更
% 伝達関数: G(s) = 1/(s^2 + 2*0.7*s + 1) = 1/(s^2 + 1.4s + 1)
% 減衰比: ζ = 0.7, 固有角周波数: ωn = 1 rad/s
A0 = [0, 1; -1, -1.4];
B0 = [0; 1];
C0 = [1, 0];
D0 = [0];
[numerator_0, denominator_0] = ss2tf(A0, B0, C0, D0);
p0_z = tf(numerator_0, denominator_0, ts);
fprintf('\n[Generator] A0=\n'); disp(A0);
fprintf('[Generator] B0=\n'); disp(B0);
fprintf('[Generator] C0=\n'); disp(C0);
fprintf('[Generator] D0=\n'); disp(D0);
fprintf('[Generator] Transfer function (p0_z):\n');
fprintf('%s', evalc('disp(p0_z)'));

%% 連続時間プラント（SISO）

A = [-1, 0; 0, -2];
B = [1; 0];
C = [1, 1];
D = [0];
p_s = ss(A, B, C, D); % 連続時間プラント
fprintf('\n[Plant] A=\n'); disp(A);
fprintf('[Plant] B=\n'); disp(B);
fprintf('[Plant] C=\n'); disp(C);
fprintf('[Plant] D=\n'); disp(D);
fprintf('[Plant] Transfer function (p_s):\n');
fprintf('%s', evalc('disp(tf(p_s))'));

%% 理想ASPRモデル（連続系）の自動設計
% SISO連続系でのSPRを満たす簡潔な一階モデル:
%   M(s) = d0 + alpha / (s + a)
% 条件: a > 0, d0 >= 0, alpha > 0 -> SPR（周波数軸上で実部が正）
% さらにDCゲインをプラントに近づけるため alpha = a * (gP - d0) とする
% ただし gP <= d0 となる場合は d0 を小さくし、必要なら目標DCゲイン一致を緩和

% プラントのDCゲインを評価
try
    gP = dcgain(p_s); % 連続系のDCゲイン
catch
    gP = 1.0;
end

% 設計パラメータ（必要に応じてチューニング可能）
a_aspr = 1.0;     % 極の位置 (>0)
d0_aspr = 0.1;    % 直達項 (>=0)

% DCゲイン整合を優先してalphaを決定
alpha_aspr = a_aspr * (gP - d0_aspr);

% 正実性のためにalpha>0を保証（gP<=d0のときは一致を緩和）
if ~(alpha_aspr > 0)
    d0_aspr = max(0.01, 0.1 * (1 + abs(gP))); % 小さめの正の直達項に再設定
    alpha_aspr = max(0.1, a_aspr * (d0_aspr)); % 常に正になるよう確保
end
% 連続時間ASPRモデルの状態空間表現（一階）
A_aspr = -a_aspr;
B_aspr = 1;
C_aspr = alpha_aspr;
D_aspr = d0_aspr;
p_aspr_s = ss(A_aspr, B_aspr, C_aspr, D_aspr);
fprintf('\n[ASPR Model] A_aspr= %g\n', A_aspr);
fprintf('[ASPR Model] B_aspr= %g\n', B_aspr);
fprintf('[ASPR Model] C_aspr= %g\n', C_aspr);
fprintf('[ASPR Model] D_aspr= %g\n', D_aspr);
fprintf('[ASPR Model] Transfer function (p_aspr_s):\n');
fprintf('%s', evalc('disp(tf(p_aspr_s))'));

%% 理想前馈補償器モデル（連続系） - PFC(s) = M(s) - P(s)

p_pfc_s = p_aspr_s - p_s;
[AA, BB, CC, DD] = ssdata(p_pfc_s);
fprintf('\n[PFC] AA=\n'); disp(AA);
fprintf('[PFC] BB=\n'); disp(BB);
fprintf('[PFC] CC=\n'); disp(CC);
fprintf('[PFC] DD=\n'); disp(DD);
fprintf('[PFC] Transfer function (p_pfc_s):\n');
fprintf('%s', evalc('disp(tf(p_pfc_s))'));

%% ニューラルネットワークパラメータ読み取りと疑似チェーンモデル実行の呼び出し - 操作ボード

getNN_simu; % NNFFまたはNNROBの行列パラメータを取得
getPFC_simu; % NNPFCのパラメータを取得
sim("fin_simu_continuous.slx"); % 最終実行

%% ログ保存と終了処理
save(log_mat);
fprintf('\nSaved log text to: %s\n', log_txt);
fprintf('Saved workspace to: %s\n', log_mat);
diary off;
