# GPU メモリ不足エラーの解決方法

## 問題

```
Failed to allocate memory for requested buffer of size 10291934720
```

Swin UNETRモデルが約10GB以上のGPUメモリを要求していますが、GPUのVRAMが不足しています。

---

## 解決方法

### 方法1: 軽量な3D U-Netモデルを使用（推奨⭐）

Swin UNETRの代わりに、より軽量なモデルを作成します：

```bash
cd ~/ShioRIS3

# 軽量モデル生成スクリプトを作成
cat > scripts/model_tools/create_lightweight_model.py << 'SCRIPT'
#!/usr/bin/env python3
"""
軽量3D U-Netモデル生成スクリプト
GPUメモリが限られている環境向け
"""
import torch
import torch.nn as nn
import onnx

class Lightweight3DUNet(nn.Module):
    """メモリ効率の良い軽量3D U-Net"""
    def __init__(self, in_channels=1, out_channels=4, base_features=16):
        super().__init__()

        # エンコーダー（ダウンサンプリング）
        self.enc1 = self.conv_block(in_channels, base_features)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = self.conv_block(base_features, base_features * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = self.conv_block(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool3d(2)

        # ボトルネック
        self.bottleneck = self.conv_block(base_features * 4, base_features * 8)

        # デコーダー（アップサンプリング）
        self.upconv3 = nn.ConvTranspose3d(base_features * 8, base_features * 4, 2, stride=2)
        self.dec3 = self.conv_block(base_features * 8, base_features * 4)

        self.upconv2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, 2, stride=2)
        self.dec2 = self.conv_block(base_features * 4, base_features * 2)

        self.upconv1 = nn.ConvTranspose3d(base_features * 2, base_features, 2, stride=2)
        self.dec1 = self.conv_block(base_features * 2, base_features)

        # 出力層
        self.out = nn.Conv3d(base_features, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        """3D畳み込みブロック"""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # エンコーダー
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # ボトルネック
        b = self.bottleneck(self.pool3(e3))

        # デコーダー（スキップ接続付き）
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)

def create_and_export():
    print("=" * 70)
    print("Lightweight 3D U-Net Model Generator")
    print("For GPU memory-constrained environments")
    print("=" * 70)
    print()

    # モデル作成（基本特徴量を16に制限してメモリ使用量を削減）
    model = Lightweight3DUNet(
        in_channels=1,      # CT（グレースケール）
        out_channels=4,     # 背景 + 3臓器
        base_features=16    # 軽量化のため16（デフォルトは32-64）
    )
    model.eval()

    print("✓ Model created")
    print(f"  Architecture: Lightweight 3D U-Net")
    print(f"  Base features: 16 (memory-efficient)")
    print(f"  Parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print()

    # ダミー入力（小さめのサイズ）
    dummy_input = torch.randn(1, 1, 64, 64, 64)

    output_path = "lightweight_3d_unet.onnx"

    print(f"Exporting to ONNX: {output_path}")
    print(f"  Input shape: {list(dummy_input.shape)}")

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
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

        print("✓ ONNX export successful!")

        # 検証
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ Model verification passed!")

        import os
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"\nModel file size: {size_mb:.2f} MB")

        print("\n" + "=" * 70)
        print("✓ Lightweight model created successfully!")
        print("=" * 70)
        print(f"\nGenerated: {output_path}")
        print("\nModel specifications:")
        print("  Input:  [batch, 1, D, H, W] - CT volume (0~1 normalized)")
        print("  Output: [batch, 4, D, H, W] - Segmentation probabilities")
        print("  Classes: 0=Background, 1=Liver, 2=R.Kidney, 3=L.Kidney/Spleen")
        print("\nMemory requirements:")
        print("  GPU VRAM: ~2-4GB (depending on input size)")
        print("  Recommended input: 64x64x64 to 96x96x96")
        print("\nNext steps:")
        print("  1. Load in ShioRIS3")
        print("  2. AI Segmentation → Select this model")
        print("  3. Process will automatically resize volume to fit")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = create_and_export()
    sys.exit(0 if success else 1)
SCRIPT

chmod +x scripts/model_tools/create_lightweight_model.py

# 実行
python3 scripts/model_tools/create_lightweight_model.py
```

このモデルは：
- **GPU VRAM要件**: 2-4GB（Swin UNETRの10GB以上と比較）
- **精度**: 事前学習なしだが、構造は医療画像セグメンテーションに適している
- **速度**: Swin UNETRより高速

---

### 方法2: GPUメモリ容量を確認して入力サイズを調整

```bash
# GPU情報確認
nvidia-smi

# メモリ容量を確認（例：8GB, 12GB, 24GBなど）
```

**GPUメモリ別の推奨設定:**

| GPU VRAM | 推奨入力サイズ | 使用可能モデル |
|----------|---------------|---------------|
| 4-6GB    | 64x64x64      | Lightweight U-Net |
| 8GB      | 96x96x96      | Lightweight U-Net |
| 12GB     | 128x128x128   | Standard U-Net |
| 16GB+    | 128x128x128   | Swin UNETR (小サイズ) |
| 24GB+    | Full size     | Swin UNETR (フルサイズ) |

---

### 方法3: CPU推論に切り替え

GPUが使えない場合、CPUで推論を実行：

ShioRIS3を再ビルド（CUDA無効化）：
```bash
cd ~/ShioRIS3/build
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_ONNXRUNTIME=ON -DONNXRUNTIME_USE_CUDA=OFF ..
make -j$(nproc)
```

または、環境変数で一時的に無効化：
```bash
export CUDA_VISIBLE_DEVICES=""
./ShioRIS3
```

**注意**: CPU推論は遅いですが、メモリ不足エラーは発生しません。

---

### 方法4: タイル処理（今後の実装予定）

大きなボリュームを小さなタイルに分割して処理し、結果を統合する方法です。
現在のShioRIS3コードにはまだ実装されていませんが、今後追加予定です。

---

## 推奨アクション

**すぐに試せる解決策（優先順）:**

1. **軽量モデルを使用**（上記スクリプトを実行）
   ```bash
   python3 scripts/model_tools/create_lightweight_model.py
   ```

2. **ShioRIS3で軽量モデルをロード**
   - 生成された `lightweight_3d_unet.onnx` を使用
   - GPU VRAM 2-4GBで動作

3. **動作確認**
   - 小さめのCTボリューム（64-96スライス程度）でテスト

---

## 技術的詳細

### なぜSwin UNETRはメモリを多く使うのか？

1. **Transformerアーキテクチャ**: 自己注意機構が大量のメモリを使用
2. **大きな特徴マップ**: 中間層で大きな3Dテンソルを保持
3. **深いネットワーク**: 多層構造で累積的にメモリ使用

### 軽量U-Netの利点

1. **CNNベース**: Transformerより効率的
2. **浅いネットワーク**: メモリフットプリントが小さい
3. **実績ある構造**: 医療画像セグメンテーションで広く使用

---

## トラブルシューティング

### Q: 軽量モデルでも Out of Memory エラー

**対策:**
```python
# create_lightweight_model.pyの base_features を 16 → 8 に変更
model = Lightweight3DUNet(
    in_channels=1,
    out_channels=4,
    base_features=8  # さらに軽量化
)
```

### Q: 精度が低い

**対策:**
1. 事前学習済みモデルを使用（別途学習が必要）
2. データ拡張を使用してファインチューニング
3. より大きなGPUを使用

### Q: CPUモードが遅い

**対策:**
- 入力サイズを小さくする（64x64x64など）
- マルチスレッド処理を有効化
- より高性能なCPUを使用

---

まずは**軽量モデルスクリプト**を実行してみてください！
