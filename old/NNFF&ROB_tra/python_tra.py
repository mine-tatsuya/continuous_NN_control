import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD

# CSVファイルを読み取り、ファイル名を'xxx.csv'と仮定
# 第2列はx1、第3列はx2、第4列はy
data = pd.read_csv('simout2.csv')

# x1, x2, yを抽出、iloc関数は0から開始することに注意
x1 = data.iloc[:, 5].values  # 第六列 (n-3)
x2 = data.iloc[:, 4].values  # 第五列 (n-2)
x3 = data.iloc[:, 3].values  # 第四列（n-1）
x4 = data.iloc[:, 2].values  # 第三列（n-0）
y = data.iloc[:, 1].values   # 第二列 (u)

# x1とx2を1つの入力行列に結合
X = np.column_stack((x1, x2, x3, x4))

# モデルを構築
model = Sequential()

# 入力層から第1隠れ層、30個のニューロン、活性化関数はReLU
model.add(Dense(30, input_dim=X.shape[1], activation='relu'))

# 第2隠れ層、20個のニューロン
model.add(Dense(20, activation='relu'))

# 第3隠れ層、10個のニューロン
model.add(Dense(10, activation='relu'))

# 出力層、出力は1つの値と仮定
model.add(Dense(1, activation='linear'))

adam_optimizer = Adam(learning_rate=0.001)

# モデルをコンパイル、損失関数として平均二乗誤差を使用、Adamオプティマイザー
model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

# モデルを訓練
model.fit(X, y, epochs=8, batch_size=32, verbose=1)

sgdm_optimizer = SGD(learning_rate=0.001, momentum=0.9)
model.compile(optimizer=sgdm_optimizer, loss='mean_squared_error')
model.fit(X, y, epochs=5, batch_size=10, verbose=1)


# モデルの重みをテキストファイルに保存
with open('za2_model_weights.txt', 'w') as f:
    for layer_num, layer in enumerate(model.layers):
        weights = layer.get_weights()
        f.write(f'Layer {layer_num + 1} weights:\n')
        f.write(f'Weights: {weights[0]}\n')
        f.write(f'Biases: {weights[1]}\n\n')

print("1. モデルの重みが'model_weights.txt'に保存されました")

# モデルのすべての重みとバイアスを取得
weights, biases = [], []
for layer in model.layers:
    weights.append(layer.get_weights()[0])
    biases.append(layer.get_weights()[1])

# 重みとバイアスをCSVファイルに保存
for i, (w, b) in enumerate(zip(weights, biases)):
    np.savetxt(f'z2_weights_layer_{i}.csv', w, delimiter=',')
    np.savetxt(f'z2_biases_layer_{i}.csv', b, delimiter=',')

print("2. モデルの重みが各CSVファイルに保存されました")