# ONNX推論時のVRAM最適化ガイド

## 問題の説明

MONAI Swin UNETRなどの大規模なONNXモデルを使用して推論を行う際、VRAM（GPU メモリ）を大量に消費する問題が発生していました。

3090 x2 (48GB VRAM) の環境でも VRAM 不足エラーが発生する場合があります。

## 重要な注意事項

### モデルの固定サイズ制約

**Swin UNETR モデルは、エクスポート時に指定した固定サイズの入力を期待します。**

`export_monai_to_onnx.py` でモデルをエクスポートした際の `--input-size` と、推論時の入力サイズが一致していないと、以下のエラーが発生します：

```
Non-zero status code returned while running Reshape node.
The input tensor cannot be reshaped to the requested shape.
```

**デフォルトのエクスポートサイズ:** 96×96×96

**モデルの出力:** 14チャンネル（BTCVデータセット標準）
- 0: Background
- 1: Spleen, 2: Right Kidney, 3: Left Kidney
- 4: Gallbladder, 5: Esophagus, 6: Liver, 7: Stomach
- 8: Aorta, 9: Inferior Vena Cava
- 10: Portal/Splenic Veins, 11: Pancreas
- 12: Right Adrenal Gland, 13: Left Adrenal Gland

**注意:** モデルの固定サイズ制約により、入力ボリュームは強制的にエクスポートされたサイズにリサンプリングされます。これにより、元のアスペクト比が変わる可能性があります。

### メモリ消費の特性

**Swin UNETR のような大規模Transformerモデルは、入力サイズに対して非線形的に大量のメモリを消費します。**

中間層（特にアテンション層）で、入力テンソルの10-20倍のメモリを使用する可能性があります。

例：
- 入力: 96×96×96 (約3.5MB)
- ピークメモリ: 10-15GB（モデルの重み + 中間アクティベーション含む）

## VRAM不足の解決方法

### 推奨: より小さいサイズでモデルを再エクスポート

デフォルトの96×96×96サイズは、24GB VRAMでも不足する可能性があります。

**64×64×64でモデルを再エクスポート:**

```bash
cd /home/user/ShioRIS3
python scripts/model_tools/export_monai_to_onnx.py \
  --input-size 64 64 64 \
  --device cpu \
  -o monai_swin_unetr_64x64x64.onnx
```

推定VRAM使用量: 4-8 GB

**48×48×48でさらに小さくエクスポート:**

```bash
python scripts/model_tools/export_monai_to_onnx.py \
  --input-size 48 48 48 \
  --device cpu \
  -o monai_swin_unetr_48x48x48.onnx
```

推定VRAM使用量: 2-4 GB

**エクスポート後、ShioRIS3で使用する際に環境変数を設定:**

```bash
# 64x64x64でエクスポートした場合
export SHIORIS_MODEL_INPUT_DEPTH=64
export SHIORIS_MODEL_INPUT_HEIGHT=64
export SHIORIS_MODEL_INPUT_WIDTH=64
export SHIORIS_GPU_DEVICE=0
export SHIORIS_GPU_MEM_LIMIT_GB=8
./ShioRIS3

# 48x48x48でエクスポートした場合
export SHIORIS_MODEL_INPUT_DEPTH=48
export SHIORIS_MODEL_INPUT_HEIGHT=48
export SHIORIS_MODEL_INPUT_WIDTH=48
export SHIORIS_GPU_DEVICE=0
export SHIORIS_GPU_MEM_LIMIT_GB=4
./ShioRIS3
```

## 実装された最適化

### 1. GPU デバイスの選択

**環境変数: `SHIORIS_GPU_DEVICE`**

デフォルトでは GPU 0 を使用しますが、環境変数で別のGPUを指定できます。

```bash
# GPU 0 を使用（デフォルト）
./ShioRIS3

# GPU 1 を使用
export SHIORIS_GPU_DEVICE=1
./ShioRIS3

# GPU を切り替えて複数の推論を並列実行
export SHIORIS_GPU_DEVICE=0
./ShioRIS3 &

export SHIORIS_GPU_DEVICE=1
./ShioRIS3 &
```

### 2. GPU メモリ制限

**環境変数: `SHIORIS_GPU_MEM_LIMIT_GB`**

デフォルト: 8GB

ONNX Runtime の GPU メモリ使用量の上限を設定します。

```bash
# 8GB制限（デフォルト）
./ShioRIS3

# 12GB制限
export SHIORIS_GPU_MEM_LIMIT_GB=12
./ShioRIS3

# 4GB制限（メモリを節約）
export SHIORIS_GPU_MEM_LIMIT_GB=4
./ShioRIS3
```

### 3. 入力サイズの制限

**環境変数: `SHIORIS_MAX_DEPTH`, `SHIORIS_MAX_HEIGHT`, `SHIORIS_MAX_WIDTH`**

デフォルト値（大規模Transformerモデル対応）:
- `SHIORIS_MAX_DEPTH=48` (Swin UNETR対応に最適化)
- `SHIORIS_MAX_HEIGHT=224` (Swin UNETR対応に最適化)
- `SHIORIS_MAX_WIDTH=224` (Swin UNETR対応に最適化)

これらの値を調整することで、VRAM 使用量を制御できます。

**重要:** Swin UNETR のような Transformer モデルは、これらの値を大きくすると指数関数的にメモリを消費します。

```bash
# VRAM 使用量を削減（低解像度）
export SHIORIS_MAX_DEPTH=64
export SHIORIS_MAX_HEIGHT=256
export SHIORIS_MAX_WIDTH=256
./ShioRIS3

# より高解像度で推論（より多くのVRAM必要）
export SHIORIS_MAX_DEPTH=128
export SHIORIS_MAX_HEIGHT=512
export SHIORIS_MAX_WIDTH=512
./ShioRIS3
```

**メモリ使用量の目安（Swin UNETR - モデルのエクスポートサイズ別）:**

| エクスポートサイズ | 入力テンソル | 推定ピークVRAM使用量* | 推奨環境 |
|------------------|-------------|---------------------|---------|
| **32×32×32** | ~0.5 MB | ~0.5-1 GB | VRAMが非常に限られている場合 |
| **48×48×48** | ~1.7 MB | ~2-4 GB | ✓ 推奨（低VRAM環境） |
| **64×64×64** | ~4 MB | ~4-8 GB | ✓ 推奨（バランス型） |
| **96×96×96** (デフォルト) | ~13.5 MB | **10-15 GB** | 24GB+ VRAM必要 |
| 128×128×128 | ~32 MB | **>20 GB** | 40GB+ VRAM必要 |

**重要な注意:**
- \* ピークVRAM使用量 = モデルの重み(~1-2GB) + 入力テンソル + 中間アクティベーション(入力の10-20倍)
- **エクスポートサイズ** = `export_monai_to_onnx.py --input-size D H W` で指定したサイズ
- Swin UNETR のような Transformer モデルは、アテンション機構により中間層で非常に大きなメモリを消費します
- 実際のVRAM使用量はモデルの実装、ONNX Runtime のバージョン、最適化レベルに依存します
- **推奨:** 3090 (24GB VRAM) では、64×64×64 または 48×48×48 でエクスポートしてください

### 3. CUDAメモリ管理の最適化

コード内で以下の最適化を実装しました：

- **GPU メモリ制限**: 16GB までに制限（`gpu_mem_limit`）
- **メモリアリーナ戦略**: 必要に応じてメモリを拡張（`arena_extend_strategy`）
- **CPUメモリアリーナ**: 有効化してCPU-GPU転送を効率化

### 4. VRAM使用量のモニタリング

推論の前後でVRAM使用量が自動的にログ出力されます：

```
=== 3D VOLUME SEGMENTATION ===
VRAM usage before inference (check with nvidia-smi)
1024, 22528    # GPU 0: 1GB使用、22.5GB空き
512, 23040     # GPU 1: 512MB使用、23GB空き
...
VRAM usage after inference:
3584, 20480    # GPU 0: 3.5GB使用、20GB空き
512, 23040     # GPU 1: 512MB使用、23GB空き
```

### 5. CUDAキャッシュのクリーンアップ

**環境変数: `SHIORIS_CLEAR_CUDA_CACHE`**

推論後にCUDAキャッシュのクリーンアップをリクエストできます（実験的）：

```bash
export SHIORIS_CLEAR_CUDA_CACHE=1
./ShioRIS3
```

## 推奨設定（Swin UNETR用）

### 設定 A: 64×64×64でエクスポート（推奨 - 3090 24GB VRAM）

**ステップ1: モデルを64×64×64でエクスポート**

```bash
cd /home/user/ShioRIS3
python scripts/model_tools/export_monai_to_onnx.py \
  --input-size 64 64 64 \
  --device cpu \
  -o monai_swin_unetr_64x64x64.onnx
```

**ステップ2: ShioRIS3で使用**

```bash
export SHIORIS_GPU_DEVICE=0
export SHIORIS_GPU_MEM_LIMIT_GB=8
export SHIORIS_MODEL_INPUT_DEPTH=64
export SHIORIS_MODEL_INPUT_HEIGHT=64
export SHIORIS_MODEL_INPUT_WIDTH=64
./ShioRIS3
```

推定VRAM使用量: 4-8 GB

### 設定 B: 48×48×48でエクスポート（低VRAM環境）

**ステップ1: モデルを48×48×48でエクスポート**

```bash
python scripts/model_tools/export_monai_to_onnx.py \
  --input-size 48 48 48 \
  --device cpu \
  -o monai_swin_unetr_48x48x48.onnx
```

**ステップ2: ShioRIS3で使用**

```bash
export SHIORIS_GPU_DEVICE=0
export SHIORIS_GPU_MEM_LIMIT_GB=4
export SHIORIS_MODEL_INPUT_DEPTH=48
export SHIORIS_MODEL_INPUT_HEIGHT=48
export SHIORIS_MODEL_INPUT_WIDTH=48
./ShioRIS3
```

推定VRAM使用量: 2-4 GB

### 設定 C: デフォルト96×96×96を使用（40GB+ VRAM推奨）

デフォルトでエクスポートされたモデルを使用する場合（VRAMに余裕がある場合のみ）：

```bash
export SHIORIS_GPU_DEVICE=0
export SHIORIS_GPU_MEM_LIMIT_GB=16
# SHIORIS_MODEL_INPUT_* はデフォルト値（96x96x96）を使用
./ShioRIS3
```

推定VRAM使用量: 10-15 GB（**A100 40GBまたはA6000 48GB推奨**）

### 設定 D: 2つのGPUで並列実行

```bash
# ターミナル 1: GPU 0 で64x64x64モデル
export SHIORIS_GPU_DEVICE=0
export SHIORIS_GPU_MEM_LIMIT_GB=8
export SHIORIS_MODEL_INPUT_DEPTH=64
export SHIORIS_MODEL_INPUT_HEIGHT=64
export SHIORIS_MODEL_INPUT_WIDTH=64
./ShioRIS3 &

# ターミナル 2: GPU 1 で64x64x64モデル
export SHIORIS_GPU_DEVICE=1
export SHIORIS_GPU_MEM_LIMIT_GB=8
export SHIORIS_MODEL_INPUT_DEPTH=64
export SHIORIS_MODEL_INPUT_HEIGHT=64
export SHIORIS_MODEL_INPUT_WIDTH=64
./ShioRIS3 &
```

各GPUで4-8 GB VRAM使用。2つのGPUを同時に使用して異なるデータを処理できます。

## トラブルシューティング

### VRAM 不足エラーが出る場合

#### エラー例: "Failed to allocate memory for requested buffer of size 10291934720"

このエラーは、モデルの中間層が10GB以上のメモリを要求していることを意味します。

**解決手順:**

1. **入力サイズを大幅に削減（最優先）**
   ```bash
   export SHIORIS_MAX_DEPTH=32
   export SHIORIS_MAX_HEIGHT=128
   export SHIORIS_MAX_WIDTH=128
   export SHIORIS_GPU_MEM_LIMIT_GB=4
   ./ShioRIS3
   ```

2. **それでも失敗する場合、さらに削減**
   ```bash
   export SHIORIS_MAX_DEPTH=24
   export SHIORIS_MAX_HEIGHT=96
   export SHIORIS_MAX_HEIGHT=96
   export SHIORIS_GPU_MEM_LIMIT_GB=4
   ./ShioRIS3
   ```

3. **GPUメモリ使用状況を確認**
   ```bash
   # リアルタイムで監視
   watch -n 1 nvidia-smi

   # または1回だけ確認
   nvidia-smi
   ```

4. **他のGPUプロセスを終了**
   ```bash
   # GPU を使用している他のプロセスを確認
   nvidia-smi

   # 不要なプロセスを終了
   kill <PID>
   ```

5. **別のGPUを試す**
   ```bash
   export SHIORIS_GPU_DEVICE=1
   ```

6. **最終手段: CPU推論にフォールバック**

   GPUでVRAM不足が解決できない場合、CPU推論を使用：
   - より遅くなりますが、メモリ制限が緩和されます
   - ONNX Runtime はCUDAが利用できない場合、自動的にCPUにフォールバックします

### 推論が遅い場合

- より大きい入力サイズを試す（品質と速度のトレードオフ）
- GPU デバイスが正しく選択されているか確認
- ログで "CUDA Execution Provider successfully enabled" が表示されているか確認

### 複数GPUの効果的な使用

単一のONNX推論セッションで複数GPUを直接使用することはできませんが、以下の方法で複数GPUを活用できます：

1. **異なるプロセスで異なるGPUを使用**
   - 推奨：複数のShioRIS3インスタンスを起動し、それぞれ異なるGPUを使用

2. **バッチ処理**
   - 複数のCTボリュームを処理する場合、GPU 0とGPU 1で並列処理

## 技術的な詳細

### メモリ使用量の計算

```
入力テンソルサイズ = Depth × Height × Width × sizeof(float)
                  = D × H × W × 4 bytes

例: 96 × 384 × 384 × 4 = 56,623,104 bytes ≈ 54 MB

しかし、モデルの重み、中間アクティベーション、出力テンソルなども
VRAMを消費するため、実際の使用量は入力サイズの10-20倍になることがあります。
```

### ONNX Runtime のメモリ管理

- `gpu_mem_limit`: GPU メモリの上限を設定（バイト単位）
- `arena_extend_strategy`: メモリアリーナの拡張戦略
  - `kSameAsRequested`: 必要な分だけ拡張（メモリ効率的）
  - `kNextPowerOfTwo`: 2の累乗で拡張（速度優先）

### デバッグ

デバッグビルドでは、より詳細なログが出力されます：

```bash
# デバッグビルド
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

# 実行
./ShioRIS3
```

## まとめ

### 最も重要なポイント

1. **モデルの固定サイズ制約**: Swin UNETR はエクスポート時に指定したサイズの入力を期待します
2. **モデルの再エクスポートが必要**: デフォルトの96×96×96は大きすぎるため、64×64×64または48×48×48で再エクスポートしてください
3. **非線形的なメモリ消費**: Transformerモデルは入力の10-20倍のVRAMを中間層で使用します

### 推奨ワークフロー（3090 x2 環境）

**ステップ1: モデルを適切なサイズで再エクスポート**

```bash
cd /home/user/ShioRIS3

# 推奨: 64x64x64でエクスポート（4-8GB VRAM使用）
python scripts/model_tools/export_monai_to_onnx.py \
  --input-size 64 64 64 \
  --device cpu \
  -o monai_swin_unetr_64x64x64.onnx
```

**ステップ2: ShioRIS3で使用**

```bash
# 単一GPU
export SHIORIS_GPU_DEVICE=0
export SHIORIS_GPU_MEM_LIMIT_GB=8
export SHIORIS_MODEL_INPUT_DEPTH=64
export SHIORIS_MODEL_INPUT_HEIGHT=64
export SHIORIS_MODEL_INPUT_WIDTH=64
./ShioRIS3
```

**ステップ3: 2つのGPUで並列処理（オプション）**

```bash
# ターミナル1: GPU 0
export SHIORIS_GPU_DEVICE=0
export SHIORIS_GPU_MEM_LIMIT_GB=8
export SHIORIS_MODEL_INPUT_DEPTH=64
export SHIORIS_MODEL_INPUT_HEIGHT=64
export SHIORIS_MODEL_INPUT_WIDTH=64
./ShioRIS3 &

# ターミナル2: GPU 1
export SHIORIS_GPU_DEVICE=1
export SHIORIS_GPU_MEM_LIMIT_GB=8
export SHIORIS_MODEL_INPUT_DEPTH=64
export SHIORIS_MODEL_INPUT_HEIGHT=64
export SHIORIS_MODEL_INPUT_WIDTH=64
./ShioRIS3 &
```

### クイックリファレンス

| エクスポートサイズ | 環境変数設定 | 推定VRAM | 推奨環境 |
|------------------|-------------|----------|---------|
| 48×48×48 | `DEPTH=48 HEIGHT=48 WIDTH=48` | 2-4 GB | 任意 |
| 64×64×64 | `DEPTH=64 HEIGHT=64 WIDTH=64` | 4-8 GB | ✓ 3090 24GB |
| 96×96×96 | デフォルト（設定不要） | 10-15 GB | A100 40GB+ |

この最適化により、3090 x2 (48GB VRAM) 環境で Swin UNETR の ONNX 推論を安定して実行できるようになりました。
