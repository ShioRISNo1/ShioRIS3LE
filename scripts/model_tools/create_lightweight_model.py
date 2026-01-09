#!/usr/bin/env python3
"""
軽量3D U-Netモデル生成スクリプト
GPUメモリが限られている環境向け（2-4GB VRAM）
"""
import torch
import torch.nn as nn
import onnx
import os
import sys

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

        d1 = self.upconv1(d2)  # 修正: d3 → d2
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)

def create_and_export(output_path="lightweight_3d_unet.onnx", base_features=16):
    print("=" * 70)
    print("Lightweight 3D U-Net Model Generator")
    print("For GPU memory-constrained environments")
    print("=" * 70)
    print()

    # モデル作成
    model = Lightweight3DUNet(
        in_channels=1,
        out_channels=4,
        base_features=base_features
    )
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print("✓ Model created")
    print(f"  Architecture: Lightweight 3D U-Net")
    print(f"  Base features: {base_features}")
    print(f"  Total parameters: {num_params / 1e6:.2f}M")
    print()

    # メモリ推定
    if base_features == 8:
        vram = "1-2GB"
        input_size = "48x48x48 to 64x64x64"
    elif base_features == 16:
        vram = "2-4GB"
        input_size = "64x64x64 to 96x96x96"
    elif base_features == 32:
        vram = "4-8GB"
        input_size = "96x96x96 to 128x128x128"
    else:
        vram = "Unknown"
        input_size = "Variable"

    print(f"Estimated GPU VRAM requirement: {vram}")
    print(f"Recommended input size: {input_size}")
    print()

    # ダミー入力
    if base_features <= 16:
        dummy_input = torch.randn(1, 1, 64, 64, 64)
    else:
        dummy_input = torch.randn(1, 1, 96, 96, 96)

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
        print("\nVerifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ Model verification passed!")

        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"\nModel file size: {size_mb:.2f} MB")

        print("\n" + "=" * 70)
        print("✓ Lightweight model created successfully!")
        print("=" * 70)
        print(f"\nGenerated: {output_path}")
        print("\nModel specifications:")
        print("  Input:  [batch, 1, D, H, W] - CT volume (0~1 normalized)")
        print("  Output: [batch, 4, D, H, W] - Segmentation probabilities")
        print("  Classes:")
        print("    0 = Background")
        print("    1 = Liver")
        print("    2 = Right Kidney")
        print("    3 = Left Kidney/Spleen")
        print("\nMemory requirements:")
        print(f"  GPU VRAM: ~{vram}")
        print(f"  Recommended input: {input_size}")
        print("\nNext steps:")
        print("  1. Open ShioRIS3")
        print("  2. Load CT volume (DICOM)")
        print("  3. AI Segmentation → Select this model")
        print("  4. Click 'Start' - volume will auto-resize to fit")
        print()

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate lightweight 3D U-Net for medical image segmentation"
    )
    parser.add_argument(
        "-o", "--output",
        default="lightweight_3d_unet.onnx",
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--base-features",
        type=int,
        default=16,
        choices=[8, 16, 32],
        help="Base feature size (8=ultra-light, 16=light, 32=standard)"
    )

    args = parser.parse_args()

    print(f"Base features: {args.base_features}")
    if args.base_features == 8:
        print("⚡ Ultra-lightweight mode: Minimal GPU memory (~1-2GB)")
    elif args.base_features == 16:
        print("⚡ Lightweight mode: Low GPU memory (~2-4GB)")
    else:
        print("⚙️  Standard mode: Moderate GPU memory (~4-8GB)")
    print()

    success = create_and_export(args.output, args.base_features)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
