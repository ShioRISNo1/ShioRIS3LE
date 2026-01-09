# 3D Medical Image Segmentation Models - Download & Export Tools

このディレクトリには、ShioRIS3で使用可能な3Dセグメンテーションモデルをダウンロード・変換するためのツールが含まれています。

## クイックスタート

### 方法1: MONAI Swin UNETR（推奨）

最も精度が高く、使いやすいオプションです。

```bash
# 1. MONAIをインストール
pip install monai torch onnx requests

# 2. モデルをダウンロード・変換
python3 export_monai_to_onnx.py -o monai_abdomen.onnx

# 3. ShioRIS3で使用
# - ShioRIS3を起動
# - CTボリュームを読み込み
# - AI Segmentation → monai_abdomen.onnx を選択
```

**サポート臓器（14種類）:**
- Spleen（脾臓）、Liver（肝臓）、Kidneys（腎臓）
- Gallbladder（胆嚢）、Pancreas（膵臓）、Stomach（胃）
- Aorta（大動脈）、IVC（下大静脈）など

---

### 方法2: TotalSegmentator

包括的な全身セグメンテーション（104臓器）

```bash
# 1. セットアップ
bash download_totalsegmentator.sh

# 2. モデル変換
python3 export_totalseg_to_onnx.py -o totalseg.onnx
```

**注意:** TotalSegmentatorは複雑なため、スクリプトはシンプルな3D U-Netモデルを作成します。

---

## ファイル一覧

### スクリプト

| ファイル | 説明 |
|---------|------|
| `download_totalsegmentator.sh` | TotalSegmentatorのインストールスクリプト |
| `export_totalseg_to_onnx.py` | TotalSegmentator → ONNX変換 |
| `export_monai_to_onnx.py` | MONAI Swin UNETR → ONNX変換 ⭐推奨 |

### ドキュメント

- `../3D_SEGMENTATION_MODELS_GUIDE.md` - 詳細なモデルダウンロードガイド

---

## 使用方法

### MONAI Swin UNETRの使用（推奨）

```bash
# 依存関係インストール
pip install monai torch onnx requests

# ONNX変換（事前学習済み重み使用）
python3 export_monai_to_onnx.py

# またはランダム初期化（テスト用）
python3 export_monai_to_onnx.py --no-pretrained

# カスタム出力パス
python3 export_monai_to_onnx.py -o custom_model.onnx
```

**出力:**
- ファイル: `monai_swin_unetr_abdomen.onnx`
- サイズ: ~250 MB（事前学習済み）
- 入力: `[1, 1, 96, 96, 96]` (batch, channels, D, H, W)
- 出力: `[1, 14, 96, 96, 96]` (14クラス)

---

### TotalSegmentatorの使用

```bash
# ステップ1: インストール
bash download_totalsegmentator.sh

# ステップ2: ONNX変換
python3 export_totalseg_to_onnx.py

# カスタム出力
python3 export_totalseg_to_onnx.py -o my_model.onnx
```

**注意:**
TotalSegmentatorの完全な機能を使うには、Pythonから直接呼び出すことを推奨：

```python
from totalsegmentator.python_api import totalsegmentator
totalsegmentator(input_path="ct_volume.nii.gz", output_path="segmentation/")
```

---

## モデル仕様

### 推奨入力形式

```
Input:
  - Shape: [batch, channels, depth, height, width]
  - Example: [1, 1, 96, 96, 96]
  - Type: float32
  - Value range: 0.0 ~ 1.0 (normalized)
```

### 推奨出力形式

```
Output:
  - Shape: [batch, num_classes, depth, height, width]
  - Example: [1, 14, 96, 96, 96]
  - Type: float32
  - Values: Per-class probabilities (0.0 ~ 1.0)
```

---

## トラブルシューティング

### Q: `ImportError: No module named 'monai'`

**解決策:**
```bash
pip install monai
# または特定バージョン
pip install monai==1.3.0
```

### Q: `RuntimeError: CUDA out of memory`

**解決策:**
モデルの入力サイズを小さくしてください：

```python
# 大きすぎる
dummy_input = torch.randn(1, 1, 512, 512, 512)

# 推奨サイズ
dummy_input = torch.randn(1, 1, 96, 96, 96)
# または
dummy_input = torch.randn(1, 1, 128, 128, 128)
```

### Q: ダウンロードが遅い・失敗する

**解決策:**
事前学習済みモデルを手動でダウンロード：

MONAI Swin UNETR:
https://github.com/Project-MONAI/MONAI-extra-test-data/releases

TotalSegmentator:
```bash
totalseg_download_weights -t total
```

### Q: ONNX変換エラー

**解決策1:** Opsetバージョンを変更
```python
torch.onnx.export(..., opset_version=11, ...)  # 14の代わりに11
```

**解決策2:** PyTorchを更新
```bash
pip install --upgrade torch onnx
```

---

## ShioRIS3での使用

1. **モデル準備**
   ```bash
   python3 export_monai_to_onnx.py -o abdomen_model.onnx
   ```

2. **ShioRIS3起動**
   ```bash
   ./ShioRIS3
   ```

3. **CTボリューム読み込み**
   - File → Open DICOM Directory
   - または NIFTI ファイル読み込み

4. **AIセグメンテーション**
   - Tools → AI Segmentation
   - モデル選択: `abdomen_model.onnx`
   - 開始ボタンクリック

5. **結果確認**
   - セグメンテーション結果が3Dビューに表示
   - 臓器ごとの統計情報を確認
   - 必要に応じて調整・エクスポート

---

## 参考リンク

- **MONAI**: https://monai.io/
- **TotalSegmentator**: https://github.com/wasserth/TotalSegmentator
- **ONNX**: https://onnx.ai/
- **ShioRIS3 Documentation**: ../docs/

---

## ライセンスと利用規約

各モデルには独自のライセンスがあります：

- **MONAI Models**: Apache License 2.0
- **TotalSegmentator**: Apache License 2.0（非商用研究目的）

商用利用の場合は、各プロジェクトのライセンスを確認してください。

---

## サポート

問題が発生した場合：

1. `../3D_SEGMENTATION_MODELS_GUIDE.md` を確認
2. GitHubでIssueを作成
3. 詳細なエラーメッセージとログを提供

---

**最終更新:** 2025-10-26
