% ===== ここだけ変更してください =====
timestamp_folder = '20251009_154125';  % 使用するタイムスタンプフォルダ
% ====================================

% スクリプトの場所を基準にプロジェクトルートを取得
script_path = fileparts(mfilename('fullpath'));  % getPFC_simu.mのあるフォルダ
project_root = fullfile(script_path, '..', '..');  % プロジェクトルート
nnpfc_folder = fullfile(project_root, 'result_NNPFC', timestamp_folder);

% --- 重みとバイアス（64-64-1構成）を「自動サイズで」読み込み ---
w0pfc = readmatrix(fullfile(nnpfc_folder, 'z2pfc_weights_layer_0.csv')); w0pfc = w0pfc(~all(isnan(w0pfc),2), ~all(isnan(w0pfc),1));  % [4x64]
b0pfc = readmatrix(fullfile(nnpfc_folder, 'z2pfc_biases_layer_0.csv'));  b0pfc = b0pfc(~all(isnan(b0pfc),2), :);                     % [64x1]

w1pfc = readmatrix(fullfile(nnpfc_folder, 'z2pfc_weights_layer_1.csv')); w1pfc = w1pfc(~all(isnan(w1pfc),2), ~all(isnan(w1pfc),1));  % [64x64]
b1pfc = readmatrix(fullfile(nnpfc_folder, 'z2pfc_biases_layer_1.csv'));  b1pfc = b1pfc(~all(isnan(b1pfc),2), :);                     % [64x1]

w2pfc = readmatrix(fullfile(nnpfc_folder, 'z2pfc_weights_layer_2.csv')); w2pfc = w2pfc(~all(isnan(w2pfc),2), :);                      % [64x1]
b2pfc = readmatrix(fullfile(nnpfc_folder, 'z2pfc_biases_layer_2.csv'));  b2pfc = b2pfc(1);                                           % [1x1]

% --- 形の検証（念のため） ---
assert(all(size(w0pfc)==[4,64]) && numel(b0pfc)==64, 'PFC layer0 shape mismatch');
assert(all(size(w1pfc)==[64,64]) && numel(b1pfc)==64, 'PFC layer1 shape mismatch');
assert(all(size(w2pfc)==[64,1])  && numel(b2pfc)==1,  'PFC layer2 shape mismatch');

% --- 入力Xの標準化パラメータ（mu_x_pfc/sig_x_pfc という名前で用意） ---
Tstd_pfc = readtable(fullfile(nnpfc_folder, 'standardize_X_params.csv')); 
need_pfc = {'ud','udd','yad','yadd'};
[ok_pfc, idx_pfc] = ismember(need_pfc, Tstd_pfc.feature);
if ~all(ok_pfc)
    error('standardize_X_params.csv に必要な特徴 %s が見つかりません。', strjoin(need_pfc(~ok_pfc), ', '));
end
mu_x_pfc  = Tstd_pfc.mean(idx_pfc);   mu_x_pfc  = mu_x_pfc(:);   % [4x1]
sig_x_pfc = Tstd_pfc.std(idx_pfc);    sig_x_pfc = sig_x_pfc(:);  % [4x1]
sig_x_pfc(sig_x_pfc==0) = 1;  % 念のため

% --- 出力(=ya-y) 側の標準化パラメータ（mu_y_pfc/sig_y_pfc という名前で用意） ---
Ty_pfc   = readtable(fullfile(nnpfc_folder, 'standardize_y_params.csv')); % columns: target, mean, std
mu_y_pfc  = Ty_pfc.mean(1);
sig_y_pfc = Ty_pfc.std(1);
if sig_y_pfc == 0, sig_y_pfc = 1; end

disp('>> NNPFC params loaded: w0pfc,b0pfc,w1pfc,b1pfc,w2pfc,b2pfc, mu_x_pfc,sig_x_pfc, mu_y_pfc,sig_y_pfc');
