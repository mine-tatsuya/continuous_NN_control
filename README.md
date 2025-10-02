# 連続時間適応制御システム - ニューラルネットワークベース

## 📋 プロジェクト概要

このプロジェクトは、**ニューラルネットワーク（NN）を用いた連続時間適応制御システム**のシミュレーション環境です。ASPR（Almost Strictly Positive Real）モデルベースの適応制御理論を応用し、以下の制御手法を実装しています：

- **NNFF（Neural Network FeedForward）**: フィードフォワード制御
- **NNROB（Neural Network Robust）**: ロバストフィードバック制御
- **NNPFC（Neural Network Parallel FeedForward Compensator）**: 並列フィードフォワード補償器

### 主な特徴

- ✅ 離散系から連続系への変換対応
- ✅ 時系列データを用いたNN学習（Adam → SGDM 2段階最適化）
- ✅ MATLAB/Simulinkによる制御シミュレーション
- ✅ Python（TensorFlow/Keras）によるNN訓練
- ✅ ロバストパラメータ調整による外乱対応

---

## 🗂️ ディレクトリ構造

```
gao 連続化/
│
├── 📁 NN_dataget_continuous/       # [ステップ1] 訓練データ生成
│   ├── code_tra.m                  # データ生成用MATLABスクリプト
│   ├── dataget_continuous.slx      # データ生成用Simulinkモデル
│   ├── dataget.csv                 # 生成された訓練データ（CSV形式）
│   └── dataget.mat                 # 生成された訓練データ（MAT形式）
│
├── 📁 NNFF&ROB_tra_continuous/     # [ステップ2] NNFF/ROB訓練
│   ├── python_tra.py               # NN訓練スクリプト（FF/ROB用）
│   └── (学習後に重み/バイアスCSVが生成される)
│
├── 📁 NNPFC_tra_continuous/        # [ステップ2] NNPFC訓練
│   ├── python_tra2.py              # NN訓練スクリプト（PFC用）
│   └── (学習後に重み/バイアスCSVが生成される)
│
├── 📁 NNFF&ROB_tra/                # 訓練済みモデル（NNFF/ROB）
│   ├── z2_weights_layer_*.csv      # 各層の重み行列
│   └── z2_biases_layer_*.csv       # 各層のバイアスベクトル
│
├── 📁 NNPFC_tra/                   # 訓練済みモデル（NNPFC）
│   ├── z2pfc_weights_layer_*.csv   # 各層の重み行列
│   └── z2pfc_biases_layer_*.csv    # 各層のバイアスベクトル
│
├── 📁 A_dataget/                   # 旧データ生成コード（参考用）
│
├── 📁 old/                         # 旧バージョン（離散系など）
│
├── 📄 code_simu_continuous_simplified.m  # [ステップ3] メインシミュレーション
├── 📄 fin_simu_continuous.slx      # 最終Simulinkモデル
├── 📄 getNN_simu.m                 # NN重み読み込みスクリプト（FF/ROB）
├── 📄 getPFC_simu.m                # NN重み読み込みスクリプト（PFC）
├── 📄 requirements.txt             # Python依存パッケージ
└── 📄 README.md                    # このファイル
```

---

## 🛠️ 必要な環境

### ソフトウェア要件

| ソフトウェア | バージョン | 用途 |
|------------|----------|------|
| **MATLAB** | R2020b以降推奨 | データ生成・シミュレーション |
| **Simulink** | R2020b以降推奨 | 制御系モデリング |
| **Python** | 3.10 または 3.11 | NN訓練（**3.13は非推奨**） |
| **TensorFlow** | 2.13以降 | ディープラーニング |

### Pythonパッケージ

以下のパッケージが必要です（`requirements.txt`に記載）：

- `numpy` - 数値計算
- `pandas` - データ処理
- `scikit-learn` - 標準化処理
- `tensorflow` - ニューラルネットワーク

---

## ⚙️ 環境構築（初回のみ）

### 1. Pythonのセットアップ

#### 推奨：Python 3.11のインストール

**⚠️ 重要**: Python 3.13では TensorFlow のインストールに問題が発生します。**Python 3.10 または 3.11** を使用してください。

1. [Python公式サイト](https://www.python.org/downloads/)から **Python 3.11** をダウンロード
2. インストール時に「**Add Python to PATH**」に必ずチェック
3. インストール完了後、PowerShellで確認：
   ```powershell
   python --version
   # Python 3.11.x と表示されればOK
   ```

### 2. 仮想環境の作成

プロジェクトフォルダ（`gao 連続化`）で以下を実行：

```powershell
# プロジェクトフォルダに移動
cd "C:\path\to\gao 連続化"

# 仮想環境を作成
python -m venv venv

# 仮想環境を有効化
.\venv\Scripts\Activate.ps1
```

**注意**: 初回実行時に「信頼されていない発行元」と表示された場合は **[A] 常に実行する** を選択してください。

### 3. Pythonパッケージのインストール

仮想環境を有効化した状態で：

```powershell
# pipをアップグレード
python -m pip install --upgrade pip

# 必要なパッケージをインストール
pip install -r requirements.txt
```

---

## 🚀 使用方法

### 全体のワークフロー

```
[ステップ1] データ生成（MATLAB）(slxファイルはMATLAB2022aを用いて作成されています。)
    ↓
[ステップ2] NN訓練（Python）
    ↓
[ステップ3] シミュレーション実行（MATLAB）(slxファイルはMATLAB2022aを用いて作成されています。)
```

---

### ステップ1: 訓練データの生成

MATLABで訓練用のデータセットを生成します。

```matlab
% MATLABのコマンドウィンドウで実行
cd 'NN_dataget_continuous'
run code_tra.m
```

**出力ファイル**:
- `dataget.csv` - CSVフォーマットのデータ（Python訓練用）
- `dataget.mat` - MATフォーマットのデータ

**データ構造**:
| 列 | 変数名 | 説明 |
|----|--------|------|
| 1 | `t` | 時刻 |
| 2 | `y` | システム出力 |
| 3 | `yd` | 出力の1サンプル前 |
| 4 | `ydd` | 出力の2サンプル前 |
| 5 | `yddd` | 出力の3サンプル前 |
| 6 | `r` | 制御入力（目標値） |

---

### ステップ2: ニューラルネットワークの訓練

Pythonで2種類のNNを訓練します。

#### 2-1. NNFF/ROB の訓練

```powershell
# 仮想環境を有効化（未起動の場合）
.\venv\Scripts\Activate.ps1

# NNFF&ROB訓練フォルダへ移動
cd NNFF&ROB_tra_continuous

# 訓練実行
python python_tra.py
```

**出力**: 
- `nn_ff_rob_best.keras` - 訓練済みモデル
- `best_adam.keras`, `best_sgdm.keras` - 中間モデル

**訓練後の処理**:
訓練したモデルの重みを `NNFF&ROB_tra/` フォルダ内の CSV ファイルとして保存する必要があります（現状は手動対応が必要）。

#### 2-2. NNPFC の訓練

```powershell
# NNPFCフォルダへ移動
cd ..\NNPFC_tra_continuous

# 訓練実行
python python_tra2.py
```

**出力**:
- `pfc_nn_best.keras` - 訓練済みモデル
- `pfc_scaler_X.pkl`, `pfc_scaler_Y.pkl` - 標準化器

---

### ステップ3: 制御シミュレーションの実行

訓練済みのNNを使用して制御シミュレーションを実行します。

```matlab
% MATLABのコマンドウィンドウで実行
cd 'C:\path\to\gao 連続化'
run code_simu_continuous_simplified.m
```

**処理の流れ**:
1. システムパラメータ設定（連続系プラント、ASPRモデルなど）
2. `getNN_simu.m` - NNFF/ROB の重みを読み込み
3. `getPFC_simu.m` - NNPFC の重みを読み込み
4. `fin_simu_continuous.slx` - Simulinkモデル実行
5. 結果を `logs/` フォルダに保存

**出力ファイル**:
- `logs/sim_log_YYYYMMDD_HHMMSS.txt` - シミュレーションログ
- `logs/workspace_YYYYMMDD_HHMMSS.mat` - 全変数保存

---

## 🎛️ パラメータ調整

`code_simu_continuous_simplified.m` 内で調整可能な主要パラメータ：

### 基本パラメータ

```matlab
time = 400;      % シミュレーション時間 [秒]
Td = 0.005;      % サンプリング時間 [秒]
```

### 外乱パラメータ

```matlab
omega = 3.14159265*5;  % 正弦波外乱の角周波数 [rad/s]
wav = 0.7;             % 外乱の振幅
```

### ロバスト制御パラメータ

```matlab
rho = 0.5;     % ロバストパラメータ ρ（論文の ρ に対応）
sig = 0.1;     % ロバストパラメータ τ（論文の τ に対応）
```

### 適応制御パラメータ

```matlab
sigma = 0.1;   % 誤差重み σ
gamma = 10;    % 前位重み γ
```

### ランダム入力パラメータ

```matlab
seed = 17;         % 乱数シード（再現性確保）
ran_min = 5;       % ランダム入力の最小値
ran_max = 20;      % ランダム入力の最大値
sam_time = 40;     % 入力切り替え周期 [秒]
```

---

## 📊 出力の確認

シミュレーション実行後、以下の方法で結果を確認できます：

### MATLABでの可視化

```matlab
% ログファイルから読み込み
load('logs/workspace_YYYYMMDD_HHMMSS.mat');

% 時系列プロット（例）
figure;
plot(tout, yout);
xlabel('Time [s]');
ylabel('Output');
title('System Response');
grid on;
```

### Simulinkスコープ

`fin_simu_continuous.slx` を開き、各ブロックのスコープで波形を確認できます。

---

## ⚠️ トラブルシューティング

### Python環境関連

#### ❌ TensorFlowのインストールエラー（Long Path Error）

**原因**: Python 3.13を使用している、またはWindowsの長いパス名制限

**解決策**:
- **推奨**: Python 3.11 に切り替える（上記「環境構築」参照）
- または Windows Long Path Support を有効化（管理者権限必要）

#### ❌ `ModuleNotFoundError: No module named 'tensorflow'`

**原因**: 仮想環境が有効化されていない

**解決策**:
```powershell
.\venv\Scripts\Activate.ps1
```

### MATLAB関連

#### ❌ `Unable to resolve the name 'dataget.csv'`

**原因**: データファイルが生成されていない

**解決策**:
```matlab
cd 'NN_dataget_continuous'
run code_tra.m
```

#### ❌ `Index exceeds matrix dimensions` in getNN_simu.m

**原因**: NN訓練が完了していない、またはCSVファイルが存在しない

**解決策**: ステップ2の訓練を完了させてください

---

## 📝 制御理論の背景

### ASPRモデルベース適応制御

このシステムは **ASPR（Almost Strictly Positive Real）** 特性を利用した適応制御を実装しています。

- **プラント**: 連続時間2次系
- **ASPRモデル**: 理想的な応答特性を持つ1次系
- **PFC**: ASPRモデルとプラントの差分を補償
- **適応則**: MRACベースのパラメータ調整

### ニューラルネットワーク構造

**4層全結合NN**:
- 入力層: 4ユニット（y, yd, ydd, yddd）
- 隠れ層1: 30ユニット（ReLU）
- 隠れ層2: 20ユニット（ReLU）
- 隠れ層3: 10ユニット（ReLU）
- 出力層: 1ユニット（線形）

**訓練手法**:
1. Adam最適化（初期学習: lr=1e-3）
2. SGDM微調整（仕上げ: lr=3e-4, momentum=0.9）
3. Early Stopping, ReduceLROnPlateau

---

## 📚 参考文献

このプロジェクトは以下の研究に基づいています：

- ASPR-based adaptive control theory
- Model Reference Adaptive Control (MRAC)
- Neural network approximation for nonlinear systems

詳細は `Renewed article/explain_0306.pptx` を参照してください。

---

## 👤 使用上の注意

1. **時系列データ**: データはシャッフルせずに時系列順で訓練してください
2. **仮想環境**: 必ず仮想環境を有効化してからPythonスクリプトを実行してください
3. **パス区切り文字**: MATLABでは `\` または `/` どちらも使用可能です
4. **保存先**: 訓練済みモデルは適切なフォルダに配置してください

---

## 📄 ライセンス

本プロジェクトは研究・教育目的で使用してください。

---

## 🔄 更新履歴

- **2025-10-02**: 連続時間システム対応版を作成
- **旧バージョン**: 離散時間システム版は `old/` フォルダに保存

---

**🎓 このプロジェクトについて質問がある場合は、プロジェクト担当者にお問い合わせください。**
