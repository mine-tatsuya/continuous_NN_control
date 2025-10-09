% ===== ここだけ変更してください =====
timestamp_folder = '20251008_160631';  % 使用するタイムスタンプフォルダ
% ====================================

% スクリプトの場所を基準にプロジェクトルートを取得
script_path = fileparts(mfilename('fullpath'));  % getNN_simu.mのあるフォルダ
project_root = fullfile(script_path, '..', '..');  % プロジェクトルート
nn_folder = fullfile(project_root, 'result_NNFF&ROB', timestamp_folder);

% --- 重みとバイアス（64-64-1構成）を「自動サイズで」読み込み ---
w0 = readmatrix(fullfile(nn_folder, 'z2_weights_layer_0.csv')); w0 = w0(~all(isnan(w0),2), ~all(isnan(w0),1));  % [4x64]
b0 = readmatrix(fullfile(nn_folder, 'z2_biases_layer_0.csv'));  b0 = b0(~all(isnan(b0),2), :);                 % [64x1]

w1 = readmatrix(fullfile(nn_folder, 'z2_weights_layer_1.csv')); w1 = w1(~all(isnan(w1),2), ~all(isnan(w1),1));  % [64x64]
b1 = readmatrix(fullfile(nn_folder, 'z2_biases_layer_1.csv'));  b1 = b1(~all(isnan(b1),2), :);                 % [64x1]

w2 = readmatrix(fullfile(nn_folder, 'z2_weights_layer_2.csv')); w2 = w2(~all(isnan(w2),2), :);                  % [64x1]
b2 = readmatrix(fullfile(nn_folder, 'z2_biases_layer_2.csv'));  b2 = b2(1);                                     % [1x1]

% --- 形の検証（念のため） ---
assert(all(size(w0)==[4,64]) && numel(b0)==64, 'layer0 shape mismatch');
assert(all(size(w1)==[64,64]) && numel(b1)==64, 'layer1 shape mismatch');
assert(all(size(w2)==[64,1])  && numel(b2)==1,  'layer2 shape mismatch');

% --- 入力Xの標準化パラメータ（mu_x/sig_x という名前で用意） ---
Tstd = readtable(fullfile(nn_folder, 'standardize_X_params.csv')); 
need = {'y','yd','ydd','yddd'};
[ok, idx] = ismember(need, Tstd.feature);
if ~all(ok)
    error('standardize_X_params.csv に必要な特徴 %s が見つかりません。', strjoin(need(~ok), ', '));
end
mu_x  = Tstd.mean(idx);   mu_x  = mu_x(:);   % [4x1]
sig_x = Tstd.std(idx);    sig_x = sig_x(:);  % [4x1]
sig_x(sig_x==0) = 1;  % 念のため

% --- 出力(=r) 側の標準化パラメータ（mu_y/sig_y という名前で用意） ---
Ty   = readtable(fullfile(nn_folder, 'standardize_y_params.csv')); % columns: target, mean, std
mu_y  = Ty.mean(1);
sig_y = Ty.std(1);
if sig_y == 0, sig_y = 1; end

disp('>> NN params loaded: w0,b0,w1,b1,w2,b2, mu_x,sig_x, mu_y,sig_y');
