# 3D CTセグメンテーション用ONNXモデル ダウンロードガイド

## 推奨モデル一覧

### 1. TotalSegmentator (推奨⭐)

**概要:**
- 全身CT用の包括的セグメンテーションモデル
- 104種類の臓器・構造を自動認識
- 腹部臓器（肝臓、腎臓、脾臓、膵臓など）を含む

**入力形式:**
- 3Dボリューム（NIFTI形式が標準だが、変換可能）
- 任意サイズのCTボリューム対応

**ダウンロード方法:**

```bash
# Pythonとpipが必要
pip install TotalSegmentator

# モデルを事前ダウンロード
totalseg_download_weights -t total

# ONNXモデルのエクスポート（Python経由）
```

**ONNX変換スクリプト:**

```python
# export_totalseg_to_onnx.py
import torch
import onnx
from totalsegmentator.nnunet import get_inference_model

# モデルをロード
model = get_inference_model("total")
model.eval()

# ダミー入力（例：128x256x256のCTボリューム）
dummy_input = torch.randn(1, 1, 128, 256, 256)

# ONNXにエクスポート
torch.onnx.export(
    model,
    dummy_input,
    "totalsegmentator.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch', 2: 'depth', 3: 'height', 4: 'width'},
        'output': {0: 'batch', 2: 'depth', 3: 'height', 4: 'width'}
    }
)
print("✓ ONNXモデル出力: totalsegmentator.onnx")
```

**公式サイト:**
- GitHub: https://github.com/wasserth/TotalSegmentator
- 論文: https://arxiv.org/abs/2208.05868

---

### 2. MONAI Model Zoo - 腹部臓器セグメンテーション

**概要:**
- MONAI（Medical Open Network for AI）の公式モデル集
- 腹部多臓器セグメンテーション専用モデルあり
- 肝臓、脾臓、腎臓（左右）、胃、膵臓など

**モデル:**
- **Swin UNETR** - 腹部臓器セグメンテーション
- **SegResNet** - 軽量版

**ダウンロード方法:**

```bash
# MONAIインストール
pip install monai

# Swin UNETR腹部臓器モデルをダウンロード
```

**ONNX変換スクリプト:**

```python
# export_monai_to_onnx.py
import torch
import monai
from monai.networks.nets import SwinUNETR

# Swin UNETRモデル（腹部13臓器用）
model = SwinUNETR(
    img_size=(96, 96, 96),  # 入力サイズ
    in_channels=1,          # CTは1チャンネル
    out_channels=14,        # 背景+13臓器
    feature_size=48,
    use_checkpoint=True,
)

# 事前学習済み重みをロード（MONAIからダウンロード）
# モデルの重みパス: https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
checkpoint_path = "swin_unetr_pretrained.pt"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
model.eval()

# ダミー入力
dummy_input = torch.randn(1, 1, 96, 96, 96)

# ONNX出力
torch.onnx.export(
    model,
    dummy_input,
    "monai_abdomen_segmentation.onnx",
    export_params=True,
    opset_version=14,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {2: 'depth', 3: 'height', 4: 'width'},
        'output': {2: 'depth', 3: 'height', 4: 'width'}
    }
)
print("✓ ONNX出力完了")
```

**公式リソース:**
- MONAI Model Zoo: https://monai.io/model-zoo.html
- GitHub: https://github.com/Project-MONAI/tutorials
- 事前学習モデル: https://github.com/Project-MONAI/MONAI-extra-test-data/releases

---

### 3. nnU-Net (医療画像セグメンテーションの標準)

**概要:**
- 自己調整型セグメンテーションフレームワーク
- 様々な医療画像タスクで最高精度
- カスタムデータセットでの学習・推論に対応

**ダウンロード:**

```bash
# インストール
pip install nnunetv2

# 事前学習済みモデルのダウンロード（例：Dataset 017 - Abdominal Organs）
# https://zenodo.org/record/7498126 からダウンロード
```

**ONNX変換:**

```python
# export_nnunet_to_onnx.py
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# モデルをロード
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    device=torch.device('cuda', 0)
)

# モデルパラメータをロード
predictor.initialize_from_trained_model_folder(
    model_training_output_dir,
    use_folds=(0,),
    checkpoint_name='checkpoint_final.pth'
)

# 推論ネットワークを取得
network = predictor.network

# ONNX変換
dummy_input = torch.randn(1, 1, 128, 128, 128).cuda()
torch.onnx.export(
    network,
    dummy_input,
    "nnunet_abdomen.onnx",
    opset_version=14,
    input_names=['input'],
    output_names=['output']
)
```

**公式サイト:**
- GitHub: https://github.com/MIC-DKFZ/nnUNet
- 論文: https://www.nature.com/articles/s41592-020-01008-z

---

### 4. 簡単な方法：Hugging Face Model Hubから直接ダウンロード

**利用可能なモデル:**

```bash
# Hugging Face CLIをインストール
pip install huggingface_hub

# モデルを検索・ダウンロード
huggingface-cli download <model-name>
```

**おすすめモデル（ONNX形式）:**

1. **医療画像セグメンテーション用モデル**
   - 検索: https://huggingface.co/models?other=medical+segmentation+onnx

2. **例：腹部CTセグメンテーション**
   ```bash
   # 具体的なモデル名（利用可能な場合）
   huggingface-cli download USERNAME/abdomen-ct-segmentation-onnx
   ```

---

## クイックスタート：軽量テスト用モデル

すぐに試したい場合、以下の軽量モデルを作成してテストできます：

### シンプルな3D U-Netモデル（ONNX）

```python
# create_simple_3d_unet.py
import torch
import torch.nn as nn

class Simple3DUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()
        # エンコーダー
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)

        # ボトルネック
        self.bottleneck = self.conv_block(128, 256)

        # デコーダー
        self.dec3 = self.conv_block(256 + 128, 128)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec1 = self.conv_block(64 + 32, 32)

        # 出力
        self.out = nn.Conv3d(32, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # エンコード
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool3d(e1, 2))
        e3 = self.enc3(nn.functional.max_pool3d(e2, 2))

        # ボトルネック
        b = self.bottleneck(nn.functional.max_pool3d(e3, 2))

        # デコード
        d3 = self.dec3(torch.cat([nn.functional.interpolate(b, scale_factor=2), e3], dim=1))
        d2 = self.dec2(torch.cat([nn.functional.interpolate(d3, scale_factor=2), e2], dim=1))
        d1 = self.dec1(torch.cat([nn.functional.interpolate(d2, scale_factor=2), e1], dim=1))

        return self.out(d1)

# モデル作成
model = Simple3DUNet(in_channels=1, out_channels=4)  # 背景+3臓器
model.eval()

# ONNX出力
dummy_input = torch.randn(1, 1, 128, 256, 256)
torch.onnx.export(
    model,
    dummy_input,
    "simple_3d_unet.onnx",
    export_params=True,
    opset_version=14,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {2: 'depth', 3: 'height', 4: 'width'},
        'output': {2: 'depth', 3: 'height', 4: 'width'}
    }
)
print("✓ Simple 3D U-Net ONNX model created: simple_3d_unet.onnx")
print("  Input: [1, 1, D, H, W]")
print("  Output: [1, 4, D, H, W]")
```

**注意:** このモデルはランダム重みなので、実際のセグメンテーションには使えません。構造テスト用です。

---

## 推奨ワークフロー

### ステップ1: TotalSegmentatorを使う（最も簡単）

```bash
# インストール
pip install TotalSegmentator

# CTボリュームをセグメント（自動でモデルダウンロード）
TotalSegmentator -i input_ct.nii.gz -o output_segmentation/

# 初回実行時に自動的にモデルがダウンロードされます
# モデル保存場所: ~/.totalsegmentator/nnunet/results/
```

### ステップ2: ONNX形式に変換

上記のPythonスクリプトを使用してONNX形式に変換します。

### ステップ3: ShioRIS3で使用

変換したONNXモデルをShioRIS3で読み込んで使用します。

---

## モデル仕様の確認方法

ダウンロードしたONNXモデルの詳細を確認：

```python
# check_onnx_model.py
import onnx

# モデルをロード
model = onnx.load("your_model.onnx")

# 入力・出力の形状を表示
print("=== Model Inputs ===")
for input in model.graph.input:
    print(f"Name: {input.name}")
    print(f"Shape: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in input.type.tensor_type.shape.dim]}")
    print()

print("=== Model Outputs ===")
for output in model.graph.output:
    print(f"Name: {output.name}")
    print(f"Shape: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in output.type.tensor_type.shape.dim]}")
```

または、NetronでGUIで確認：

```bash
pip install netron
netron your_model.onnx
```

ブラウザで `http://localhost:8080` を開くとモデル構造が可視化されます。

---

## ShioRIS3との互換性

ShioRIS3で使用するONNXモデルの要件：

### 推奨入力形式:
```
Input: [batch, channels, depth, height, width]
例: [1, 1, 128, 256, 256]
```

### 推奨出力形式:
```
Output: [batch, num_classes, depth, height, width]
例: [1, 4, 128, 256, 256]  # 背景 + 3臓器
```

### サポートされるクラス:
- Class 0: Background（背景）
- Class 1: Liver（肝臓）
- Class 2: Right Kidney（右腎）
- Class 3: Left Kidney/Spleen（左腎/脾臓）

---

## トラブルシューティング

### Q: PyTorchがインストールできない

**A:** Condaを使用：
```bash
conda create -n medical-ai python=3.10
conda activate medical-ai
conda install pytorch torchvision torchaudio pytorch-cuda=12.0 -c pytorch -c nvidia
pip install monai onnx
```

### Q: メモリ不足エラー

**A:** モデルの入力サイズを小さく：
```python
# 大きすぎる: (1, 1, 512, 512, 512)
# 推奨: (1, 1, 96, 96, 96) または (1, 1, 128, 128, 128)
```

### Q: ONNX変換エラー

**A:** Opsetバージョンを変更：
```python
# opset_version=14 がエラーの場合
torch.onnx.export(..., opset_version=11, ...)
```

---

## 参考リンク

- **TotalSegmentator**: https://github.com/wasserth/TotalSegmentator
- **MONAI**: https://monai.io/
- **nnU-Net**: https://github.com/MIC-DKFZ/nnUNet
- **Hugging Face Medical Models**: https://huggingface.co/models?other=medical
- **3D Slicer Segmentation**: https://www.slicer.org/
- **Medical Segmentation Decathlon**: http://medicaldecathlon.com/

---

## 実際の使用例

完全な実装例をShioRIS3プロジェクトに追加予定：
- `examples/download_segmentation_model.py` - モデルダウンロードスクリプト
- `examples/export_to_onnx.py` - ONNX変換スクリプト
- `examples/test_onnx_model.py` - モデルテストスクリプト

ご不明な点があればお気軽にお問い合わせください！
