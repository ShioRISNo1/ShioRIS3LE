#!/usr/bin/env python3
"""
MONAI 腹部臓器セグメンテーションモデル ONNX Export Script

MONAI Model ZooのSwin UNETRモデルをONNX形式に変換します。
"""

import torch
import onnx
import sys
import os
import requests
import gc

try:
    import monai
    from monai.networks.nets import SwinUNETR
    print(f"✓ MONAI version: {monai.__version__}")
except ImportError:
    print("❌ Error: MONAI not installed")
    print("Please run: pip install monai")
    sys.exit(1)

def download_pretrained_weights(output_path="swin_unetr_pretrained.pt"):
    """
    MONAI Swin UNETR事前学習済み重みをダウンロード
    """
    # MONAI公式の事前学習済みモデルURL
    # Note: 実際のURLは変更される可能性があります
    model_urls = {
        "swin_unetr_btcv": "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt"
    }

    if os.path.exists(output_path):
        print(f"✓ Pretrained weights already exist: {output_path}")
        return output_path

    print("Downloading pretrained weights...")
    print("This may take several minutes...")

    try:
        url = model_urls["swin_unetr_btcv"]
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='')

        print(f"\n✓ Downloaded: {output_path}")
        return output_path

    except Exception as e:
        print(f"\n❌ Error downloading weights: {e}")
        print("\nPlease download manually from:")
        print("https://github.com/Project-MONAI/MONAI-extra-test-data/releases")
        sys.exit(1)

def create_swin_unetr_model(pretrained=True):
    """
    Swin UNETRモデルを作成

    Args:
        pretrained: 事前学習済み重みを使用するか
    """
    print("\nCreating Swin UNETR model...")

    # MONAI 1.5.x での新しいAPI
    # BTCV腹部多臓器データセット用の設定
    try:
        # MONAI 1.5.x以降の新しいAPI
        model = SwinUNETR(
            spatial_dims=3,          # 3D画像
            in_channels=1,           # CTは1チャンネル
            out_channels=14,         # 背景 + 13臓器
            feature_size=48,
            use_checkpoint=False,    # ONNX exportのため無効化
        )
        print("✓ Using MONAI 1.5.x+ API")
    except TypeError:
        # MONAI 1.3.x以前の古いAPI（フォールバック）
        model = SwinUNETR(
            img_size=(96, 96, 96),   # 入力画像サイズ
            in_channels=1,           # CTは1チャンネル
            out_channels=14,         # 背景 + 13臓器
            feature_size=48,
            use_checkpoint=False,    # ONNX exportのため無効化
        )
        print("✓ Using legacy MONAI API")

    if pretrained:
        print("Loading pretrained weights...")
        weights_path = download_pretrained_weights()

        try:
            # PyTorch 2.6+: weights_only=False is required for MONAI pretrained weights
            # This is safe because we're loading from official MONAI releases
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

            # チェックポイントの構造を確認
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=False)
            print("✓ Pretrained weights loaded successfully")

        except Exception as e:
            print(f"⚠ Warning: Could not load pretrained weights: {e}")
            print("Continuing with random initialization...")
            print("Note: This will result in poor segmentation quality!")

    model.eval()
    return model

def export_to_onnx(model, output_path="monai_abdomen_segmentation.onnx",
                   device="cpu", input_size=(96, 96, 96)):
    """
    モデルをONNX形式にエクスポート

    Args:
        model: PyTorchモデル
        output_path: 出力ONNXファイルパス
        device: 使用デバイス ('cpu', 'cuda:0', 'cuda:1' など)
        input_size: 入力サイズ (depth, height, width)
    """
    print(f"\nExporting to ONNX: {output_path}")
    print(f"  Device: {device}")
    print(f"  Input size: {input_size}")

    # メモリクリーンアップ
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        # モデルをデバイスに移動
        device_obj = torch.device(device)
        model = model.to(device_obj)
        print(f"✓ Model moved to {device}")

        # ダミー入力を作成
        dummy_input = torch.randn(1, 1, *input_size, device=device_obj)
        print(f"  Input shape: {dummy_input.shape}")

        # メモリ使用量を表示（GPUの場合）
        if device != "cpu" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device_obj) / 1024**3
            reserved = torch.cuda.memory_reserved(device_obj) / 1024**3
            print(f"  GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        # メモリ使用量を削減するため、推論テストをスキップ
        # （大きなモデルではメモリ不足になるため）
        print("  Skipping inference test to save memory...")
        print("  ⚠ If export fails, try using GPU with --device cuda:0")

        with torch.no_grad():
            # メモリクリーンアップ
            gc.collect()

            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=False,  # メモリ削減のためFalseに変更
                input_names=['input'],
                output_names=['output'],
                # NOTE: dynamic_axes removed - Swin UNETR requires fixed size input
                # The input size is determined by the dummy_input shape
                verbose=False
            )

        print(f"✓ ONNX export completed!")

        # メモリクリーンアップ
        del model
        del dummy_input
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # モデル検証
        print("\nVerifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ Model verification passed!")

        # ファイルサイズ表示
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"\nModel file size: {file_size_mb:.2f} MB")

        return True

    except Exception as e:
        print(f"❌ Error during ONNX export: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("MONAI Swin UNETR → ONNX Export for Abdominal Organ Segmentation")
    print("=" * 70)
    print()

    import argparse
    parser = argparse.ArgumentParser(
        description="Export MONAI Swin UNETR model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export using CPU (recommended for large models, avoids VRAM issues)
  %(prog)s --device cpu

  # Export using GPU 0
  %(prog)s --device cuda:0

  # Export with smaller input size to reduce memory usage
  %(prog)s --device cuda:0 --input-size 64 64 64

  # Export to specific file
  %(prog)s -o my_model.onnx --device cpu
        """
    )
    parser.add_argument(
        "-o", "--output",
        default="monai_swin_unetr_abdomen.onnx",
        help="Output ONNX file path (default: monai_swin_unetr_abdomen.onnx)"
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Don't use pretrained weights (random init)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for export: 'cpu', 'cuda:0', 'cuda:1', etc. (default: cpu)"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=3,
        default=[96, 96, 96],
        metavar=('D', 'H', 'W'),
        help="Input size as depth height width (default: 96 96 96). "
             "Smaller sizes use less VRAM but dynamic_axes allows any size at inference."
    )

    args = parser.parse_args()

    # デバイス情報を表示
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    print()

    # デバイス検証
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("❌ Error: CUDA device requested but CUDA is not available")
        print("Falling back to CPU...")
        args.device = "cpu"

    # モデル作成
    model = create_swin_unetr_model(pretrained=not args.no_pretrained)

    # ONNX出力
    success = export_to_onnx(
        model,
        args.output,
        device=args.device,
        input_size=tuple(args.input_size)
    )

    if success:
        print("\n" + "=" * 70)
        print("✓ Export successful!")
        print("=" * 70)
        print(f"\nModel Information:")
        print(f"  File: {args.output}")
        print(f"  Architecture: Swin UNETR")
        print(f"  Task: Abdominal multi-organ segmentation")
        print(f"  Export device: {args.device}")
        print(f"\n  Input:")
        print(f"    Shape: [1, 1, {args.input_size[0]}, {args.input_size[1]}, {args.input_size[2]}] (FIXED SIZE)")
        print(f"    Type: float32 (normalized CT values)")
        print(f"    ⚠ IMPORTANT: Input must be exactly this size!")
        print(f"\n  Output:")
        print(f"    Shape: [1, 14, {args.input_size[0]}, {args.input_size[1]}, {args.input_size[2]}] (FIXED SIZE)")
        print(f"    Type: float32 (probabilities)")
        print(f"\n  Supported organs (BTCV dataset):")
        organs = [
            "0: Background",
            "1: Spleen", "2: Right Kidney", "3: Left Kidney",
            "4: Gallbladder", "5: Esophagus", "6: Liver",
            "7: Stomach", "8: Aorta", "9: Inferior Vena Cava",
            "10: Portal/Splenic Veins", "11: Pancreas",
            "12: Right Adrenal Gland", "13: Left Adrenal Gland"
        ]
        for organ in organs:
            print(f"    {organ}")

        print(f"\nMemory optimization tips:")
        if args.device == "cpu":
            print(f"  ✓ Using CPU avoids VRAM issues")
        else:
            print(f"  • To avoid VRAM issues, use: --device cpu")
            print(f"  • Or use smaller input size: --input-size 64 64 64")

        print(f"\nNext steps:")
        print(f"1. Open ShioRIS3")
        print(f"2. Load CT volume (DICOM or NIFTI)")
        print(f"3. AI Segmentation → Select model: {args.output}")
        print(f"4. Run segmentation")
        print()
    else:
        print("\n❌ Export failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
