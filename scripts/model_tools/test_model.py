#!/usr/bin/env python3
"""
ONNXモデルの動作テスト
pretrained weightsが正しくロードされているか確認
"""

import onnxruntime as ort
import numpy as np
import sys

def test_model(model_path):
    print(f"Testing ONNX model: {model_path}")

    try:
        # モデルロード
        session = ort.InferenceSession(model_path)

        # 入力情報取得
        input_info = session.get_inputs()[0]
        input_shape = input_info.shape
        print(f"  Input shape: {input_shape}")

        # ダミー入力作成（全て0.5の値）
        # これは正規化されたCT値の中間値
        dummy_input = np.full(input_shape, 0.5, dtype=np.float32)
        print(f"  Input: all values = 0.5 (normalized CT)")

        # 推論実行
        outputs = session.run(None, {input_info.name: dummy_input})

        # 出力確認
        output = outputs[0]
        print(f"  Output shape: {output.shape}")

        # チャンネルごとの平均スコア
        print("\n  Channel average scores:")
        channels = output.shape[1]
        for c in range(min(channels, 14)):  # 最大14チャンネル
            channel_data = output[0, c, :, :, :]
            avg_score = np.mean(channel_data)
            max_score = np.max(channel_data)
            print(f"    Channel {c}: avg={avg_score:.6f}, max={max_score:.6f}")

        # 判定
        print("\n  Analysis:")
        channel_avgs = [np.mean(output[0, c, :, :, :]) for c in range(channels)]

        if all(abs(avg - channel_avgs[0]) < 0.01 for avg in channel_avgs):
            print("  ⚠ WARNING: All channels have similar scores!")
            print("  This suggests the model may not be using pretrained weights.")
            print("  A properly trained model should have varying channel scores.")
            return False
        else:
            print("  ✓ Channels have varying scores - model appears trained")
            return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_model.py <model.onnx>")
        sys.exit(1)

    model_path = sys.argv[1]
    result = test_model(model_path)
    sys.exit(0 if result else 1)
