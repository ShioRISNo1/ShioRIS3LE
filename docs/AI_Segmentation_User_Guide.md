# AISegmentation ユーザーガイド

## 📍 AISegmentation機能の起動方法

### メニューバーから起動
1. ShioRIS3を起動
2. メニューバー: **AI** → **Auto Segment** をクリック
3. Auto Segmentation Dialogが表示されます

### ツールバーから起動
1. ShioRIS3のツールバー内の **AI** セクション
2. **Auto Segment** アイコンをクリック

---

## 📥 ONNXモデルファイルの入手方法

### 方法1: 既存のONNXモデルを使用（推奨）

#### A. ONNX Model Zooから取得
公式のONNXモデルリポジトリからダウンロード：

```bash
# ONNX Model Zoo
# https://github.com/onnx/models

# セグメンテーションモデルの例
cd ~/Downloads
wget https://github.com/onnx/models/raw/main/vision/body_analysis/...
```

#### B. MedSAM (Medical Segment Anything Model)
医療画像専用のセグメンテーションモデル：

```bash
# MedSAM ONNX版の探索
# GitHub: https://github.com/bowang-lab/MedSAM
```

#### C. nnU-Net ONNX版
医療画像セグメンテーションで広く使用されているモデル：

```bash
# nnU-Netをトレーニング済みモデルとしてエクスポート
# 詳細: https://github.com/MIC-DKFZ/nnUNet
```

---

### 方法2: PyTorchモデルをONNXに変換

TotalSegmentatorなどのPyTorchモデルをONNX形式に変換します。

#### 手順1: TotalSegmentatorのインストール

```bash
# Python仮想環境の作成
python3 -m venv ~/totalseg_env
source ~/totalseg_env/bin/activate

# TotalSegmentatorのインストール
pip install TotalSegmentator torch torchvision
```

#### 手順2: モデルのダウンロード

```bash
# モデルを自動ダウンロード
TotalSegmentator --download_models

# モデルは ~/.totalsegmentator/nnunet/results に保存されます
```

#### 手順3: PyTorchからONNXへ変換

```python
# convert_to_onnx.py
import torch
import torch.onnx
from totalsegmentator.libs import setup_nnunet

# モデルのロード
model = setup_nnunet()
model.eval()

# ダミー入力の作成（例: 1x1x128x128x128）
dummy_input = torch.randn(1, 1, 128, 128, 128)

# ONNX形式にエクスポート
torch.onnx.export(
    model,
    dummy_input,
    "totalsegmentator.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch', 2: 'depth', 3: 'height', 4: 'width'},
        'output': {0: 'batch', 2: 'depth', 3: 'height', 4: 'width'}
    }
)
```

実行：
```bash
python convert_to_onnx.py
```

---

### 方法3: 事前変換済みモデルの探索

#### Hugging Face Hub
```bash
# Hugging Face CLIのインストール
pip install huggingface-hub

# ONNXモデルの検索
huggingface-cli search --model-type onnx segmentation medical
```

#### Zenodo (研究データリポジトリ)
- URL: https://zenodo.org/
- 検索キーワード: "medical segmentation onnx"
- 学術論文の補足データとして公開されている場合があります

---

## 📂 モデルファイルの配置

### デフォルトのモデルディレクトリ

ShioRIS3は以下のディレクトリにモデルを配置することを推奨します：

```
~/Documents/ShioRIS3/AIModels/
├── onnx/                    # ONNXモデルファイル
│   ├── abdominal_organ.onnx
│   ├── totalsegmentator.onnx
│   └── medsam.onnx
├── temp/                    # 一時ファイル
└── samples/                 # サンプルデータ
```

### モデルディレクトリの初期化

ShioRIS3の初回起動時に自動的に作成されますが、手動で作成することも可能：

```bash
mkdir -p ~/Documents/ShioRIS3/AIModels/{onnx,temp,samples}
```

---

## 🎯 Auto Segmentation Dialogの使用方法

### 1. モデルのロード

1. **Model File** セクション
2. **Browse** ボタンをクリック
3. ONNXモデルファイル (`.onnx`) を選択
4. モデルがロードされたことを確認

### 2. セグメンテーションの実行

1. DICOMボリュームがロードされていることを確認
2. **Start Segmentation** ボタンをクリック
3. 進捗バーで処理状況を確認
4. 完了まで待機

### 3. 結果の確認

1. **Organ Labels** ツリーで各臓器の表示/非表示を切り替え
2. **Preview** セクションでスライスを確認
3. **Statistics** で各臓器のボクセル数・体積を確認

### 4. 結果の調整

- **Threshold**: 閾値調整
- **Smoothing**: 平滑化処理
- **Fill Holes**: 穴埋め処理

### 5. 結果の保存

1. **Export** ボタンをクリック
2. 保存形式を選択：
   - セグメンテーション結果 (`.seg`)
   - RT Structure Set (`.dcm`)
3. 保存先を指定

---

## 🎛️ 精度と計算時間のバランス（品質モード）

精度向上のために推論を複数回実行する **高品質モード** がデフォルトで有効になりました。計算時間は長くなりますが、TTA（Test-Time Augmentation）による投票で精度を高めます。

- **デフォルト: 高品質 (high)**
  - 4方向のフリップ推論を含むTTAを実行し、結果を多数決で統合。
  - 推論時間は単一実行より長くなるが、臓器の抜けや誤検出を減らせます。
- **高速化したい場合**
  - 環境変数 `SHIORIS_AI_QUALITY_MODE=standard` を設定すると従来の単一推論（最速）に戻ります。
- **さらに最高精度が必要な場合**
  - `SHIORIS_AI_QUALITY_MODE=ultra` を設定すると、8方向フリップTTAとスライディングウィンドウを組み合わせた「長考」推論を行います（処理時間は数倍以上に伸び、メモリ消費も増加）。

> 例: LinuxやmacOSのターミナルでアプリ起動前に以下を実行
> ```bash
> export SHIORIS_AI_QUALITY_MODE=ultra
> ./ShioRIS3
> ```

---

## 🔧 トラブルシューティング

### モデルがロードできない

**エラー**: "Failed to load ONNX model"

**原因と解決策**:
1. **ファイル形式の確認**
   - 拡張子が `.onnx` であることを確認
   - ファイルが破損していないか確認

2. **ONNX Runtimeのバージョン**
   ```bash
   # ONNX Runtime情報の確認
   # ShioRIS3のログで確認可能
   ```

3. **モデルの互換性**
   - モデルのopsetバージョンを確認
   - ONNX Runtime 1.11+ を推奨

### セグメンテーションが失敗する

**エラー**: "Segmentation returned empty result"

**原因と解決策**:
1. **入力データの確認**
   - DICOMボリュームが正しくロードされているか
   - ボリュームの次元が正しいか (3D)

2. **モデルの入力仕様**
   - モデルが期待する入力サイズを確認
   - HU値の範囲が適切か (-1024 to 3071)

3. **GPU/CPU設定**
   ```bash
   # GPU情報の確認
   nvidia-smi

   # CUDA利用可能か確認
   # Auto Segmentation Dialog起動時のログで確認
   ```

### パフォーマンスが遅い

**対処方法**:

1. **GPU使用の確認**
   - NVIDIA GPUが検出されているか確認
   - CUDA対応版のONNX Runtimeがインストールされているか

2. **モデルサイズの最適化**
   - より軽量なモデルを使用
   - 入力解像度を下げる

3. **スレッド数の調整**
   - OnnxSegmenterのスレッド設定を変更
   - デフォルト: 4スレッド

---

## 🌐 推奨ONNXモデルソース

### 医療画像セグメンテーション

1. **MONAI Model Zoo**
   - URL: https://github.com/Project-MONAI/model-zoo
   - 医療画像専用のモデルコレクション

2. **Grand Challenge**
   - URL: https://grand-challenge.org/
   - 医療画像処理コンペティションの優勝モデル

3. **Papers with Code**
   - URL: https://paperswithcode.com/
   - 検索: "medical image segmentation onnx"

### 一般的なセグメンテーション

1. **ONNX Model Zoo**
   - URL: https://github.com/onnx/models
   - 公式のモデルコレクション

2. **Hugging Face**
   - URL: https://huggingface.co/models?pipeline_tag=image-segmentation&library=onnx
   - ONNX形式のセグメンテーションモデル

---

## 📊 モデル仕様の確認

### ONNX Visualizerの使用

```bash
# Netronのインストール（ブラウザベースのビューワー）
pip install netron

# モデルの可視化
netron model.onnx
```

### Python スクリプトで確認

```python
import onnx

model = onnx.load("model.onnx")

# 入力情報
for input in model.graph.input:
    print(f"Input: {input.name}")
    print(f"  Shape: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")

# 出力情報
for output in model.graph.output:
    print(f"Output: {output.name}")
    print(f"  Shape: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")
```

---

## ⚙️ 環境設定

### ONNX Runtime GPU版のインストール（Linux）

```bash
# CUDA 11.x の場合
pip install onnxruntime-gpu==1.17.0

# CUDA 12.x の場合
pip install onnxruntime-gpu==1.17.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

### GPU情報の確認

```bash
# NVIDIA GPU情報
nvidia-smi

# CUDA バージョン
nvcc --version

# ShioRIS3内での確認
# メニュー: AI > Auto Segment
# ログに GPU情報が表示されます
```

---

## 📚 参考リンク

- **ONNX公式**: https://onnx.ai/
- **ONNX Runtime**: https://onnxruntime.ai/
- **TotalSegmentator**: https://github.com/wasserth/TotalSegmentator
- **MONAI**: https://monai.io/
- **PyTorch to ONNX**: https://pytorch.org/docs/stable/onnx.html

---

## 📧 サポート

問題が解決しない場合は、以下の情報を添えてお問い合わせください：

1. ShioRIS3のバージョン
2. 使用しているONNXモデル名とサイズ
3. エラーメッセージの全文
4. ログファイル（必要に応じて）

---

**更新日**: 2025-10-25
**バージョン**: 1.0
