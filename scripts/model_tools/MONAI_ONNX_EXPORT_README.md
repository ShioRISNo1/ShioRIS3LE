# MONAI Swin UNETR ONNX Export - VRAM 不足対策

## 問題の説明

Swin UNETR は非常に大きなモデル（約250M parameters）のため、ONNX エクスポート時に VRAM 不足エラーが発生する可能性があります。

## 解決策

### 方法1: CPU でエクスポート（推奨）

**最も確実な方法**です。VRAM を全く使用しないため、メモリ不足エラーが発生しません。

```bash
python scripts/model_tools/export_monai_to_onnx.py --device cpu
```

**メリット:**
- VRAM を使用しないため、確実にエクスポートできる
- 複数のプロセスを同時に実行できる

**デメリット:**
- GPU よりも時間がかかる（ただし、エクスポートは一度だけなので許容範囲）

### 方法2: 単一 GPU でエクスポート（入力サイズを小さくする）

GPU を使用する場合、入力サイズを小さくすることで VRAM 使用量を削減できます。

```bash
# GPU 0 を使用、入力サイズを 64x64x64 に縮小
python scripts/model_tools/export_monai_to_onnx.py --device cuda:0 --input-size 64 64 64
```

**重要:**
- `dynamic_axes` を使用しているため、エクスポート時の入力サイズは参照用です
- 実際の推論時には任意のサイズを使用できます

### 方法3: 特定の GPU を指定

```bash
# GPU 1 を使用
python scripts/model_tools/export_monai_to_onnx.py --device cuda:1

# GPU 0 を使用
python scripts/model_tools/export_monai_to_onnx.py --device cuda:0
```

## 重要な注意事項

### ONNX エクスポート時の複数 GPU 使用について

**ONNX エクスポート時に複数 GPU を直接使用することはできません。**

理由:
- `torch.onnx.export` は単一デバイス上のモデルを期待します
- `DataParallel` でラップされたモデルは ONNX にエクスポートできません
- ONNX はシリアライズされた単一のグラフ構造を必要とします

### 推論時の複数 GPU 使用について

エクスポートされた ONNX モデルを使用する際に複数 GPU を使用したい場合は、別のアプローチが必要です：

1. **ONNX Runtime の場合**: 複数 GPU は複雑
2. **PyTorch で直接推論**: `DataParallel` や `DistributedDataParallel` を使用可能

## 使用例

### 基本的な使用方法

```bash
# CPU でエクスポート（推奨）
python scripts/model_tools/export_monai_to_onnx.py --device cpu

# 出力ファイル名を指定
python scripts/model_tools/export_monai_to_onnx.py --device cpu -o my_model.onnx

# GPU 0 で小さい入力サイズを使用
python scripts/model_tools/export_monai_to_onnx.py --device cuda:0 --input-size 64 64 64
```

### メモリ使用量の確認

スクリプトは自動的に以下の情報を表示します：
- 利用可能な GPU とそのメモリサイズ
- エクスポート時の GPU メモリ使用量
- モデルファイルサイズ

## トラブルシューティング

### VRAM 不足エラーが出る場合

1. **CPU でエクスポートする**（推奨）
   ```bash
   python scripts/model_tools/export_monai_to_onnx.py --device cpu
   ```

2. **入力サイズを小さくする**
   ```bash
   python scripts/model_tools/export_monai_to_onnx.py --device cuda:0 --input-size 48 48 48
   ```

3. **他のプロセスを終了して VRAM を確保する**
   ```bash
   # VRAM 使用状況を確認
   nvidia-smi
   ```

### エクスポートされたモデルの使用時に VRAM 不足が出る場合

これは別の問題です。推論時のメモリ管理が必要です：

- バッチサイズを小さくする
- 画像を小さなパッチに分割して処理する
- スライディングウィンドウ推論を使用する

## 技術的な詳細

### 改善点

1. **デバイス選択**: CPU または特定の GPU を指定可能
2. **入力サイズのカスタマイズ**: メモリ使用量を調整可能
3. **メモリクリーンアップ**: ガベージコレクションと CUDA キャッシュのクリア
4. **詳細な情報表示**: GPU メモリ使用量、デバイス情報など
5. **エラーハンドリング**: トレースバックを表示して問題の特定を容易に

### Dynamic Axes について

スクリプトは `dynamic_axes` を使用しているため：
- エクスポート時の入力サイズは参照用
- 実際の推論時には任意のサイズ（D, H, W）を使用可能
- バッチサイズも動的に変更可能

## まとめ

**推奨される方法:**

3090 x2 (48GB VRAM) の環境であっても、**CPU でのエクスポートを推奨**します：

```bash
python scripts/model_tools/export_monai_to_onnx.py --device cpu -o monai_swin_unetr_abdomen.onnx
```

これにより：
- ✓ VRAM 不足エラーを完全に回避
- ✓ 安定したエクスポート処理
- ✓ 他のプロセスに影響を与えない

エクスポートは一度だけなので、CPU を使用しても問題ありません。
