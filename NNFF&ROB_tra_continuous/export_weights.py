import numpy as np
from tensorflow.keras.models import load_model

# 訓練済みモデルを読み込み
model = load_model('nn_ff_rob_best.keras')

print('[INFO] Exporting weights and biases to CSV...')

# 各層の重みとバイアスをCSV出力
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if len(weights) == 2:  # 重みとバイアスがある層
        w, b = weights
        
        # 出力先（親フォルダの NNFF&ROB_tra/）
        w_path = f'../NNFF&ROB_tra_continuous/z2_weights_layer_{i}.csv'
        b_path = f'../NNFF&ROB_tra_continuous/z2_biases_layer_{i}.csv'
        
        # CSV保存
        np.savetxt(w_path, w, delimiter=',')
        np.savetxt(b_path, b, delimiter=',')
        
        print(f'[SAVE] Layer {i}: weights {w.shape} -> {w_path}')
        print(f'[SAVE] Layer {i}: biases  {b.shape} -> {b_path}')

print('[INFO] Export completed successfully!') 