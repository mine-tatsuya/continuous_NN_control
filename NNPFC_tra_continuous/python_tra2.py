# ============================================
# PFC推定NN 学習スクリプト（時系列対応・Adam→SGDM仕上げ）
# ============================================
import os
import math
import random
import pickle
import numpy as np
import pandas as pd

# 再現性（必要な場合）
import tensorflow as tf
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

# -----------------------------
# 設定
# -----------------------------
CSV_PATH = 'simout2pfc.csv'   # 入力CSV
USE_COLUMN_NAMES = False      # True: 列名で抽出 / False: ilocで抽出（元コード互換）
# 列名で抽出する場合は下を実データに合わせて書き換えてください
X_COLS = ['x_n3', 'x_n2', 'x_n1', 'x_n0']  # 例
Y_COL  = 'u'                                # 例

# ilocで抽出する場合（あなたの元コード準拠：6,5,4,3列目→入力、2列目→出力）
# 0始まり: x1=5列, x2=4列, x3=3列, x4=2列, y=1列
ILOC_IDXS_X = [5, 4, 3, 2]
ILOC_IDX_Y  = 1

# 学習ハイパーパラメータ
BATCH_SIZE = 32   # = 手順のM
EPOCHS_ADAM = 1000
EPOCHS_SGDM = 1000
LR_ADAM = 1e-3
LR_SGDM = 3e-4   # 仕上げなので小さめ
PATIENCE_EARLY = 20
PATIENCE_RLROP = 8
MIN_LR = 1e-6

# 出力ファイル
BEST_ADAM_PATH = 'pfc_best_adam.keras'
BEST_SGDM_PATH = 'pfc_best_sgdm.keras'
FINAL_MODEL_PATH = 'pfc_nn_best.keras'
SCALER_X_PATH = 'pfc_scaler_X.pkl'
SCALER_Y_PATH = 'pfc_scaler_Y.pkl'
LOG_ADAM_LASTBATCH_CSV = 'pfc_adam_last_group_losses.csv'
LOG_SGDM_LASTBATCH_CSV = 'pfc_sgdm_last_group_losses.csv'

# -----------------------------
# データ読み込み
# -----------------------------
df = pd.read_csv(CSV_PATH)

if USE_COLUMN_NAMES:
    # 列名で安全に抽出（推奨）
    X = df[X_COLS].values
    y = df[Y_COL].values
else:
    # ilocで抽出（元コード互換）
    x_list = [df.iloc[:, idx].values for idx in ILOC_IDXS_X]
    X = np.column_stack(x_list)
    y = df.iloc[:, ILOC_IDX_Y].values

# NaN等の簡易チェック（必要なら詳細に）
if np.isnan(X).any() or np.isnan(y).any():
    raise ValueError("XまたはyにNaNが含まれています。入力CSVをご確認ください。")

# -----------------------------
# 時系列のまま train/val/test に分割（70/15/15）
# -----------------------------
N = len(X)
i_tr = int(N * 0.70)
i_va = int(N * 0.85)

X_tr, y_tr = X[:i_tr], y[:i_tr]
X_va, y_va = X[i_tr:i_va], y[i_tr:i_va]
X_te, y_te = X[i_va:], y[i_va:]

# -----------------------------
# 標準化（trainでfit→他にtransform）
# -----------------------------
scX = StandardScaler()
scY = StandardScaler()

X_tr = scX.fit_transform(X_tr)
y_tr = scY.fit_transform(y_tr.reshape(-1, 1)).ravel()
X_va = scX.transform(X_va)
y_va = scY.transform(y_va.reshape(-1, 1)).ravel()
X_te = scX.transform(X_te)
y_te = scY.transform(y_te.reshape(-1, 1)).ravel()

# -----------------------------
# モデル構築
# -----------------------------
def build_model(input_dim: int) -> Sequential:
    model = Sequential()
    model.add(Dense(30, input_dim=input_dim, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

model = build_model(X_tr.shape[1])

# -----------------------------
# 最後のグループ（=最後のバッチ）損失を記録するコールバック
# 手順8の "group[n]" に相当
# -----------------------------
class LastBatchLossLogger(Callback):
    def __init__(self, steps_per_epoch: int):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.last_group_losses = []

    def on_train_batch_end(self, batch, logs=None):
        # 最後のバッチ（group[n]）の損失を記録
        if (batch + 1) == self.steps_per_epoch:
            loss_val = float((logs or {}).get('loss', np.nan))
            self.last_group_losses.append(loss_val)

# steps/epoch を計算（端数グループも自動で処理される）
steps_per_epoch_tr = math.ceil(len(X_tr) / BATCH_SIZE)

# -----------------------------
# Adam 段階
# -----------------------------
adam = Adam(learning_rate=LR_ADAM)
model.compile(optimizer=adam, loss='mse')

logger_adam = LastBatchLossLogger(steps_per_epoch_tr)
ckpt_adam = ModelCheckpoint(BEST_ADAM_PATH, monitor='val_loss', save_best_only=True, verbose=0)
early = EarlyStopping(monitor='val_loss', patience=PATIENCE_EARLY, restore_best_weights=True, verbose=0)
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE_RLROP, min_lr=MIN_LR, verbose=0)

history_adam = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=EPOCHS_ADAM,
    batch_size=BATCH_SIZE,
    shuffle=False,          # 時系列では必須
    verbose=1,              # 学習の様子を見たい場合は1、静かにしたい場合は0
    callbacks=[ckpt_adam, early, rlrop, logger_adam]
)

# Adam 段階の最後のグループ損失を保存（任意）
pd.DataFrame({'last_group_loss': logger_adam.last_group_losses}).to_csv(LOG_ADAM_LASTBATCH_CSV, index=False)

# ベストのAdam重みへ復帰（明示）
model.load_weights(BEST_ADAM_PATH)

# -----------------------------
# SGDM 段階（仕上げの微調整）
# -----------------------------
sgdm = SGD(learning_rate=LR_SGDM, momentum=0.9, nesterov=False)
model.compile(optimizer=sgdm, loss='mse')

logger_sgdm = LastBatchLossLogger(steps_per_epoch_tr)
ckpt_sgdm = ModelCheckpoint(BEST_SGDM_PATH, monitor='val_loss', save_best_only=True, verbose=0)
early2 = EarlyStopping(monitor='val_loss', patience=PATIENCE_EARLY, restore_best_weights=True, verbose=0)
rlrop2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE_RLROP, min_lr=MIN_LR, verbose=0)

history_sgdm = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=EPOCHS_SGDM,
    batch_size=BATCH_SIZE,
    shuffle=False,
    verbose=1,
    callbacks=[ckpt_sgdm, early2, rlrop2, logger_sgdm]
)

# SGDM 段階の最後のグループ損失を保存（任意）
pd.DataFrame({'last_group_loss': logger_sgdm.last_group_losses}).to_csv(LOG_SGDM_LASTBATCH_CSV, index=False)

# ベストのSGDM重みへ復帰（明示）
model.load_weights(BEST_SGDM_PATH)

# -----------------------------
# テスト評価
# -----------------------------
test_mse = model.evaluate(X_te, y_te, verbose=0)
print(f"[INFO] Test MSE (scaled-space): {test_mse:.6f}")

# -----------------------------
# 保存（モデル + スケーラー）
# -----------------------------
model.save(FINAL_MODEL_PATH)
with open(SCALER_X_PATH, 'wb') as fx:
    pickle.dump(scX, fx)
with open(SCALER_Y_PATH, 'wb') as fy:
    pickle.dump(scY, fy)

print(f"[SAVE] Model: {FINAL_MODEL_PATH}")
print(f"[SAVE] Scaler X: {SCALER_X_PATH}")
print(f"[SAVE] Scaler Y: {SCALER_Y_PATH}")

# -----------------------------
# （任意）係数をCSVに吐きたい場合の例
# ※ 通常は model.save を推奨。下は既存環境への埋め込み用途などに。
# -----------------------------
EXPORT_WEIGHTS = False
if EXPORT_WEIGHTS:
    for i, layer in enumerate(model.layers):
        w_b = layer.get_weights()
        if len(w_b) == 2:
            w, b = w_b
            np.savetxt(f'pfc_weights_layer_{i}.csv', w, delimiter=',')
            np.savetxt(f'pfc_biases_layer_{i}.csv', b, delimiter=',')
    print("[SAVE] CSV weights/biases exported.")
