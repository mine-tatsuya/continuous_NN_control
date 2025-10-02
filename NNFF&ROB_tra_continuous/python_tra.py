import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

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

# ===== 3) 標準化（trainでfit→他をtransform） =====
from sklearn.preprocessing import StandardScaler
scX, scY = StandardScaler(), StandardScaler()
X_tr = scX.fit_transform(X_tr)
y_tr = scY.fit_transform(y_tr.reshape(-1,1)).ravel()
X_va = scX.transform(X_va)
y_va = scY.transform(y_va.reshape(-1,1)).ravel()
X_te = scX.transform(X_te)
y_te = scY.transform(y_te.reshape(-1,1)).ravel()

# ===== 4) モデル =====
model = Sequential([
    Dense(30, input_dim=X_tr.shape[1], activation='relu'),
    Dense(20, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='linear')
])

# ===== 5) group[n]の損失と検証損失の記録 =====
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

M = 32
steps_per_epoch = int(np.ceil(len(X_tr)/M))
logger_adam = LastBatchLossLogger(steps_per_epoch)

# ===== 6) Adam 段階 =====
adam = Adam(learning_rate=1e-3)
model.compile(optimizer=adam, loss='mse')

ckpt_adam = ModelCheckpoint('best_adam.keras', monitor='val_loss', save_best_only=True, verbose=0)
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

# ===== 7) Adamのベストで検証評価（任意） =====
adam_val = model.evaluate(X_va, y_va, verbose=0)

# ===== 8) SGDM 段階（Adamベストから再開） =====
model.load_weights('best_adam.keras')
sgdm = SGD(learning_rate=3e-4, momentum=0.9, nesterov=False)
model.compile(optimizer=sgdm, loss='mse')

logger_sgdm = LastBatchLossLogger(steps_per_epoch)
ckpt_sgdm = ModelCheckpoint('best_sgdm.keras', monitor='val_loss', save_best_only=True, verbose=0)
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

# ===== 9) テスト評価 =====
test_mse = model.evaluate(X_te, y_te, verbose=1)
print('Test MSE:', test_mse)

# ===== 10) 保存 =====
model.save('nn_ff_rob_best.keras')
# （必要なら標準化器も保存：joblib.dump(scX, 'scX.joblib'); joblib.dump(scY, 'scY.joblib')）
