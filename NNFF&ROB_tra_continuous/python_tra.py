import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

# 1) データ読み込み（列名で安全に）
df = pd.read_csv('simout2.csv')
# 例: 列名を仮定（必要なら実データに合わせて変更）
# x_{n-3}, x_{n-2}, x_{n-1}, x_{n}, u といった列名を想定
x_cols = ['x_n3','x_n2','x_n1','x_n0']  # ←あなたのCSVの実列名に合わせて修正
y_col  = 'u'

# もし列名が無い/不明なら、ilocで確定させる（あなたの元コード準拠）
# x1 = df.iloc[:,5].values; x2 = df.iloc[:,4].values; x3 = df.iloc[:,3].values; x4 = df.iloc[:,2].values; y = df.iloc[:,1].values
# X = np.column_stack((x1,x2,x3,x4))

X = df[x_cols].values
y = df[y_col].values

# 2) 時系列のまま分割（例: 70/15/15）
N = len(X)
i_tr = int(N*0.70)
i_va = int(N*0.85)
X_tr, y_tr = X[:i_tr], y[:i_tr]
X_va, y_va = X[i_tr:i_va], y[i_tr:i_va]
X_te, y_te = X[i_va:], y[i_va:]

# 3) 標準化（trainでfit→他にtransform）
scX, scY = StandardScaler(), StandardScaler()
X_tr = scX.fit_transform(X_tr)
y_tr = scY.fit_transform(y_tr.reshape(-1,1)).ravel()
X_va = scX.transform(X_va)
y_va = scY.transform(y_va.reshape(-1,1)).ravel()
X_te = scX.transform(X_te)
y_te = scY.transform(y_te.reshape(-1,1)).ravel()

# 4) モデル
model = Sequential([
    Dense(30, input_dim=X_tr.shape[1], activation='relu'),
    Dense(20, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='linear')
])

# 5) “group[n]の損失”と“検証損失”の記録（最後のバッチを拾う簡易版）
class LastBatchLossLogger(Callback):
    def __init__(self, steps_per_epoch):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.last_batch_losses = []  # group[n]
        self.val_losses = []         # 検証損失

    def on_train_batch_end(self, batch, logs=None):
        # 最後のバッチ（= group[n]）の損失を記録
        if (batch + 1) == self.steps_per_epoch:
            self.last_batch_losses.append(float(logs.get('loss', np.nan)))

    def on_epoch_end(self, epoch, logs=None):
        self.val_losses.append(float(logs.get('val_loss', np.nan)))

# ミニバッチM（あなたのMに合わせて設定）
M = 32
steps_per_epoch = int(np.ceil(len(X_tr)/M))
logger_adam = LastBatchLossLogger(steps_per_epoch)

# 6) Adam 段階
adam = Adam(learning_rate=1e-3)
model.compile(optimizer=adam, loss='mse')

ckpt_adam = ModelCheckpoint('best_adam.keras', monitor='val_loss', save_best_only=True, verbose=0)
early = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)

history_adam = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=1000,             # 充分大 → 早期終了で止まる
    batch_size=M,
    shuffle=False,           # 時系列では必須
    verbose=0,
    callbacks=[ckpt_adam, early, rlrop, logger_adam]
)

# 7) Adamのベスト重みで評価（任意）
adam_val = model.evaluate(X_va, y_va, verbose=0)

# 8) SGDM 段階（Adamベストから再開、LR小さめ）
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
    verbose=0,
    callbacks=[ckpt_sgdm, early2, rlrop2, logger_sgdm]
)

# 9) テスト評価
test_mse = model.evaluate(X_te, y_te, verbose=0)

# 10) 保存（配布用途はSavedModel/keras形式推奨）
model.save('nn_ff_rob_best.keras')
# スケーラーも保存して実運用時の前処理を一致させる（joblib/pickleを使用）
