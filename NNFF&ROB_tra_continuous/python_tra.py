import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

# ===== 0) 結果保存用フォルダの作成 =====
# 実行日時でフォルダ名を作成（例: 20251002_153045）
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
result_dir = Path(__file__).resolve().parent.parent / 'result_NNFF&ROB' / timestamp
os.makedirs(result_dir, exist_ok=True)
print(f'[INFO] Results will be saved to: {result_dir}')

# ===== 1) CSV読み込み（同階層の兄弟フォルダ：NN_dataget_continuous/dataget.csv） =====
# このスクリプトファイルの親ディレクトリ/NN_dataget_continuous/dataget.csv を指す
csv_path = (Path(__file__).resolve().parent.parent / 'NN_dataget_continuous' / 'dataget.csv')
df = pd.read_csv(csv_path)

# 期待ヘッダー名
expected_cols = ['t','y','yd','ydd','yddd','r']
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f'CSVに想定ヘッダーが見つかりません: {missing}\n実ヘッダー: {list(df.columns)}')

# 入力: 2~5列目 → y, yd, ydd, yddd / 出力: 6列目 → r
x_cols = ['y','yd','ydd','yddd']
y_col  = 'r'

X = df[x_cols].values
y = df[y_col].values

# ===== 2) 時系列のまま分割（70/15/15） =====
N = len(X)
i_tr = int(N*0.70)
i_va = int(N*0.85)
X_tr, y_tr = X[:i_tr], y[:i_tr]
X_va, y_va = X[i_tr:i_va], y[i_tr:i_va]
X_te, y_te = X[i_va:], y[i_va:]

# ===== 3) モデル =====
model = Sequential([
    Dense(30, input_dim=X_tr.shape[1], activation='relu'),
    Dense(20, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='linear')
])

# ===== 4) group[n]の損失と検証損失の記録 =====
class LastBatchLossLogger(Callback):
    def __init__(self, steps_per_epoch):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.last_batch_losses = []  # group[n]
        self.val_losses = []         # 検証損失
    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) == self.steps_per_epoch:
            self.last_batch_losses.append(float(logs.get('loss', np.nan)))
    def on_epoch_end(self, epoch, logs=None):
        self.val_losses.append(float(logs.get('val_loss', np.nan)))

#Mでバッチサイズを設定
M = 32
steps_per_epoch = int(np.ceil(len(X_tr)/M))
logger_adam = LastBatchLossLogger(steps_per_epoch)

# ===== 5) Adam 段階 =====
adam = Adam(learning_rate=1e-3)
model.compile(optimizer=adam, loss='mse')

# ModelCheckpointのパスを結果フォルダに設定
ckpt_adam_path = str(result_dir / 'best_adam.keras')
ckpt_adam = ModelCheckpoint(ckpt_adam_path, monitor='val_loss', save_best_only=True, verbose=0)
early = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)

history_adam = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=1000,
    batch_size=M,
    shuffle=False,  # 時系列
    verbose=1,
    callbacks=[ckpt_adam, early, rlrop, logger_adam]
)

# ===== 6) Adamのベストで検証評価（任意） =====
adam_val = model.evaluate(X_va, y_va, verbose=0)

# ===== 7) SGDM 段階（Adamベストから再開） =====
model.load_weights(ckpt_adam_path)
sgdm = SGD(learning_rate=3e-4, momentum=0.9, nesterov=False)
model.compile(optimizer=sgdm, loss='mse')

logger_sgdm = LastBatchLossLogger(steps_per_epoch)
ckpt_sgdm_path = str(result_dir / 'best_sgdm.keras')
ckpt_sgdm = ModelCheckpoint(ckpt_sgdm_path, monitor='val_loss', save_best_only=True, verbose=0)
early2 = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
rlrop2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)

history_sgdm = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=1000,
    batch_size=M,
    shuffle=False,
    verbose=1,
    callbacks=[ckpt_sgdm, early2, rlrop2, logger_sgdm]
)

# ===== 8) テスト評価 =====
test_mse = model.evaluate(X_te, y_te, verbose=1)
print('Test MSE:', test_mse)

# ===== 9) 保存 =====
final_model_path = str(result_dir / 'nn_ff_rob_best.keras')
model.save(final_model_path)
print(f'[INFO] Model saved to: {final_model_path}')

# ===== 10) 重みとバイアスをCSVにエクスポート =====
print('\n[INFO] Exporting weights and biases to CSV...')
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if len(weights) == 2:  # 重みとバイアスがある層
        w, b = weights
        
        # CSV保存（結果フォルダ内）
        w_path = result_dir / f'z2_weights_layer_{i}.csv'
        b_path = result_dir / f'z2_biases_layer_{i}.csv'
        
        np.savetxt(str(w_path), w, delimiter=',')
        np.savetxt(str(b_path), b, delimiter=',')
        
        print(f'[SAVE] Layer {i}: weights {w.shape} -> {w_path}')
        print(f'[SAVE] Layer {i}: biases  {b.shape} -> {b_path}')

print(f'\n[INFO] Export completed successfully!')
print(f'[INFO] All files saved to: {result_dir}')
