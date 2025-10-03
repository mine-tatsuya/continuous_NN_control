# -*- coding: utf-8 -*-
"""
NN-based FF (inverse-plant) training script
- Inputs : y, yd, ydd, yddd
- Target : r  (プラント入力; あなたのデータ仕様を尊重)
Recipe:
  1) AdamW 段階 (ReduceLROnPlateau, checkpoint)
  2) SGD(M, nesterov=True) 段階に引き継ぎ
     - 超低LRスタート + 線形ウォームアップ(3–5ep)
     - コサイン減衰で冷却（例: 80ep）
     - （任意）SWAで終盤エポックを平均化
  3) 入出力ともに標準化（train統計でfit）、rは学習時標準化→評価時は逆変換
  4) L2正則化は層側に付与（1e-4）
"""

import os
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers, callbacks, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ===== tensorflow-addons (AdamW, SWA) =====
try:
    import tensorflow_addons as tfa
except Exception as e:
    raise ImportError(
        "tensorflow-addons が必要です。'pip install tensorflow-addons' を実行してください。"
    ) from e

# ===== 0) 再現性（任意） =====
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ===== 1) 結果保存フォルダ =====
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
ROOT = Path(__file__).resolve().parent.parent
result_dir = ROOT / 'result_NNFF&ROB' / timestamp
os.makedirs(result_dir, exist_ok=True)
print(f'[INFO] Results will be saved to: {result_dir}')

# ===== 2) データ読み込み =====
csv_path = ROOT / 'NN_dataget_continuous' / 'dataget.csv'
df = pd.read_csv(csv_path)

# 期待ヘッダー（あなたの仕様を尊重：r = プラント入力）
expected_cols = ['t','y','yd','ydd','yddd','r']
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f'CSVに想定ヘッダーが見つかりません: {missing}\n実ヘッダー: {list(df.columns)}')

# 入力と教師
x_cols = ['y','yd','ydd','yddd']  # 実機では y* 系に差し替える運用も可
y_col  = 'r'                      # 目標（入力）= 逆モデルの出力

X = df[x_cols].values.astype(np.float32)
y = df[y_col].values.astype(np.float32).reshape(-1, 1)

# ===== 3) 時系列分割（70/15/15, シャッフル無し） =====
N = len(X)
i_tr = int(N * 0.70)
i_va = int(N * 0.85)

X_tr, y_tr = X[:i_tr], y[:i_tr]
X_va, y_va = X[i_tr:i_va], y[i_tr:i_va]
X_te, y_te = X[i_va:], y[i_va:]

# ===== 4) 標準化（train統計でfit） =====
# 入力X
x_mean = X_tr.mean(axis=0, keepdims=True)
x_std  = X_tr.std(axis=0, keepdims=True)
x_std_safe = np.where(x_std == 0, 1.0, x_std)

X_tr_s = (X_tr - x_mean) / x_std_safe
X_va_s = (X_va - x_mean) / x_std_safe
X_te_s = (X_te - x_mean) / x_std_safe

# 出力y(=r) も標準化して学習（後で逆変換用に保存）
y_mean = y_tr.mean(axis=0, keepdims=True)
y_std  = y_tr.std(axis=0, keepdims=True)
y_std_safe = np.where(y_std == 0, 1.0, y_std)

y_tr_s = (y_tr - y_mean) / y_std_safe
y_va_s = (y_va - y_mean) / y_std_safe
y_te_s = (y_te - y_mean) / y_std_safe

# 標準化パラメータ保存
pd.DataFrame({'feature': x_cols,
              'mean': x_mean.flatten(),
              'std':  x_std_safe.flatten()}) \
  .to_csv(result_dir / 'standardize_X_params.csv', index=False)

pd.DataFrame({'target': [y_col],
              'mean':  [float(y_mean)],
              'std':   [float(y_std_safe)]}) \
  .to_csv(result_dir / 'standardize_y_params.csv', index=False)

print(f'[INFO] Saved standardization params to: {result_dir}')

# ===== 5) モデル定義（L2正則化つき） =====
def build_model(input_dim: int) -> tf.keras.Model:
    l2 = regularizers.l2(1e-4)
    model = Sequential([
        Dense(64, activation='relu', kernel_regularizer=l2, input_shape=(input_dim,)),
        Dense(64, activation='relu', kernel_regularizer=l2),
        Dense(1,  activation='linear')  # 標準化後の r を出力
    ])
    return model

model = build_model(X_tr_s.shape[1])

# ===== 6) AdamW 段階 =====
adamw = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
model.compile(optimizer=adamw, loss='mse', metrics=['mae'])

ckpt_adam = callbacks.ModelCheckpoint(
    filepath=str(result_dir / 'best_adamw.keras'),
    monitor='val_loss', save_best_only=True, mode='min', verbose=0
)
plateau = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0
)
early_adam = callbacks.EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True, verbose=0
)

BATCH_SIZE = 32  # ここは環境に合わせてOK（大きめで高速になることが多い）
EPOCHS_ADAM = 1000

print('\n[INFO] === AdamW phase ===')
history_adam = model.fit(
    X_tr_s, y_tr_s,
    validation_data=(X_va_s, y_va_s),
    epochs=EPOCHS_ADAM,
    batch_size=BATCH_SIZE,
    shuffle=False,
    verbose=1,
    callbacks=[ckpt_adam, plateau, early_adam]
)

# AdamWベストでのVal評価（標準化空間）
adam_val = model.evaluate(X_va_s, y_va_s, verbose=0)
print(f'[INFO] AdamW best (val) -> loss: {adam_val:.6f}')

# ===== 7) SGD(M) 段階に引き継ぎ =====
# ベストAdamW重みをロード
model.load_weights(str(result_dir / 'best_adamw.keras'))

# --- SGD(M) 基本設定 ---
LR_START  = 1e-6   # 超低LRスタート
LR_TARGET = 1e-4   # ウォームアップ後の目標LR（LR Range Testで調整可）
LR_MIN    = 1e-6   # コサイン後の最小LR

WARMUP_EPOCHS = 5
EPOCHS_SGD    = 80

# （任意）SWA 設定
USE_SWA = True
SWA_AVG_EPOCHS = 30                        # 終盤30エポックを平均化
SWA_START = max(0, EPOCHS_SGD - SWA_AVG_EPOCHS)

# ベースSGD
sgd_base = optimizers.SGD(learning_rate=LR_START, momentum=0.9, nesterov=True)

# SWAを使う場合はラップ
if USE_SWA:
    opt_sgd = tfa.optimizers.SWA(
        optimizer=sgd_base,
        start_averaging=SWA_START,   # epoch番号（0始まり）
        average_period=1
    )
else:
    opt_sgd = sgd_base

model.compile(optimizer=opt_sgd, loss='mse', metrics=['mae'])

# 学習率スケジューラ（線形ウォームアップ → コサイン減衰）
def lr_schedule(epoch, lr):
    if epoch < WARMUP_EPOCHS:
        # 線形ウォームアップ（epoch: 0..WARMUP_EPOCHS-1）
        return LR_START + (LR_TARGET - LR_START) * (epoch + 1) / WARMUP_EPOCHS
    # コサイン減衰（epoch: WARMUP_EPOCHS..EPOCHS_SGD-1）
    t = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS_SGD - WARMUP_EPOCHS)
    return LR_MIN + 0.5 * (LR_TARGET - LR_MIN) * (1 + math.cos(math.pi * t))

sched = callbacks.LearningRateScheduler(lr_schedule, verbose=0)

ckpt_sgd = callbacks.ModelCheckpoint(
    filepath=str(result_dir / 'best_sgd.keras'),
    monitor='val_loss', save_best_only=True, mode='min', verbose=0
)
# SGD段階では ReduceLROnPlateau は使わず、スケジューラ単独で統制
early_sgd = callbacks.EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True, verbose=0
)

print('\n[INFO] === SGD(M) phase (with warmup + cosine) ===')
history_sgd = model.fit(
    X_tr_s, y_tr_s,
    validation_data=(X_va_s, y_va_s),
    epochs=EPOCHS_SGD,
    batch_size=BATCH_SIZE,
    shuffle=False,
    verbose=1,
    callbacks=[sched, ckpt_sgd, early_sgd]
)

# SWAの平均重みを適用（USE_SWA=Trueの場合）
if USE_SWA:
    # 平均化済み重みをモデルに反映
    opt = model.optimizer
    if isinstance(opt, tfa.optimizers.SWA):
        opt.assign_average_vars(model.variables)
        print('[INFO] SWA average weights have been assigned.')

# ===== 8) テスト評価（物理単位へ逆変換） =====
# 標準化空間での予測
y_pred_te_s = model.predict(X_te_s, batch_size=BATCH_SIZE, verbose=0)

# 逆標準化（物理単位）
y_pred_te = y_pred_te_s * y_std_safe + y_mean
y_te_real = y_te_s      * y_std_safe + y_mean

mse_test = np.mean((y_pred_te - y_te_real) ** 2)
mae_test = np.mean(np.abs(y_pred_te - y_te_real))

print(f'\n[INFO] Test MSE (in target units): {mse_test:.6f}')
print(f'[INFO] Test MAE (in target units): {mae_test:.6f}')

# ===== 9) モデル保存（最終：現在の重み） =====
final_model_path = str(result_dir / 'nn_ff_rob_best.keras')
model.save(final_model_path)
print(f'[INFO] Model saved to: {final_model_path}')

# ===== 10) 重み/バイアスのCSVエクスポート =====
print('\n[INFO] Exporting weights and biases to CSV...')
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if len(weights) == 2:
        w, b = weights
        np.savetxt(str(result_dir / f'z2_weights_layer_{i}.csv'), w, delimiter=',')
        np.savetxt(str(result_dir / f'z2_biases_layer_{i}.csv'),  b, delimiter=',')
        print(f'[SAVE] Layer {i}: weights {w.shape}, biases {b.shape}')

print(f'\n[INFO] Export completed. All files saved to: {result_dir}')
