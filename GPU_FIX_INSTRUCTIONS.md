# AIセグメンテーションGPU対応修正手順

## 問題の概要

Linux環境でAIセグメンテーション機能を使用する際、以下の問題が発生していました：

1. **CPUモードしか使用できない**: CUDA Execution Providerが正しく検出・有効化されない
2. **Segmentation開始ボタンがクリックできない**: ボリュームがロードされていない、またはモデルがロードされていない

## 修正内容

### 1. CMakeLists.txtの改善

- CUDA検出ロジックを改善し、より多くのパスで`libonnxruntime_providers_cuda.so`を探索
- CUDA Providerライブラリが見つかった場合、`ONNXRUNTIME_USE_CUDA`マクロを確実に定義
- Shared Providerライブラリ(`libonnxruntime_providers_shared.so`)のサポート追加
- 詳細なデバッグメッセージを追加

### 2. C++コードの改善

#### onnx_segmenter.cpp
- CUDA初期化時の詳細なログ出力を追加
- CUDA Providerが利用可能かどうかを明確に表示
- エラー時のトラブルシューティング情報を提供

#### linux_auto_segmenter.cpp
- `getGPUInfo()`を改善し、nvidia-smiを複数のパスで探索
- より詳細なGPU情報（Compute Capability含む）を取得
- デバッグ用の詳細なログ出力を追加

#### auto_segmentation_dialog.cpp
- UIにGPU/CPU実行環境情報を表示
- モデルロード時に実行プロバイダー（GPU/CPU）を表示
- 開始ボタンが無効な理由をツールチップで表示

## ビルド手順

### 前提条件

以下がインストールされていることを確認してください：

1. **CUDA Toolkit** (12.0以降推奨)
   ```bash
   nvcc --version
   ```

2. **NVIDIA Driver**
   ```bash
   nvidia-smi
   ```

3. **ONNX Runtime with CUDA support**
   - 例: `/usr/local/onnxruntime/`にインストールされている場合
   - 必要なファイル:
     - `include/onnxruntime_cxx_api.h`
     - `lib/libonnxruntime.so`
     - `lib/libonnxruntime_providers_cuda.so`
     - `lib/libonnxruntime_providers_shared.so` (推奨)

### ビルド

1. **既存のビルドディレクトリをクリーンアップ**
   ```bash
   cd /path/to/ShioRIS3
   rm -rf build
   mkdir build
   cd build
   ```

2. **CMakeを実行** (ONNX Runtimeのパスを指定)
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release \
         -DENABLE_ONNXRUNTIME=ON \
         -DONNXRUNTIME_ROOT=/usr/local/onnxruntime \
         ..
   ```

   または、環境変数を使用:
   ```bash
   export ONNXRUNTIME_ROOT=/usr/local/onnxruntime
   cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_ONNXRUNTIME=ON ..
   ```

3. **CMake出力を確認**

   成功した場合、以下のようなメッセージが表示されます：
   ```
   🚀 ===== CUDA SUPPORT ENABLED =====
      CUDA Provider: /usr/local/onnxruntime/lib/libonnxruntime_providers_cuda.so
      Shared Provider: /usr/local/onnxruntime/lib/libonnxruntime_providers_shared.so
      Macro ONNXRUNTIME_USE_CUDA will be defined
   ===================================
   ```

   CPUのみの場合:
   ```
   ℹ CUDA support disabled - using CPU execution provider
   ```

4. **ビルド**
   ```bash
   make -j$(nproc)
   ```

5. **ライブラリパスの設定**
   ```bash
   export LD_LIBRARY_PATH=/usr/local/onnxruntime/lib:$LD_LIBRARY_PATH
   ```

6. **実行**
   ```bash
   ./ShioRIS3
   ```

## 動作確認

### 1. アプリケーション起動時のログ確認

端末でアプリケーションを起動し、以下のログを確認してください：

```
=== CUDA EXECUTION PROVIDER INITIALIZATION ===
Build configuration: ONNXRUNTIME_USE_CUDA is defined
Available execution providers:
  - CPUExecutionProvider
  - CUDAExecutionProvider
✓ CUDAExecutionProvider is available
✓ CUDA Execution Provider successfully enabled (GPU device 0)
✓ GPU-optimized session options configured
=== CUDA EP INITIALIZATION COMPLETE ===
CUDA Enabled: YES
```

CPUモードの場合:
```
=== CUDA EXECUTION PROVIDER INITIALIZATION ===
Build configuration: ONNXRUNTIME_USE_CUDA is defined
Available execution providers:
  - CPUExecutionProvider
❌ CUDAExecutionProvider not in available providers list
This may indicate:
  1. CUDA provider library (libonnxruntime_providers_cuda.so) not found
  2. CUDA runtime libraries not properly installed
  3. GPU drivers not properly configured
```

### 2. UI確認

AIセグメンテーションダイアログを開いて確認：

1. **モデルセクション**:
   - GPU使用時: `🚀 実行環境: GPU: [GPU名] (Driver: [バージョン], ...)`
   - CPU使用時: `💻 実行環境: CPU`

2. **モデルロード後**:
   - GPU使用時: `モデル: ロード完了 (GPU)` (青色)
   - CPU使用時: `モデル: ロード完了 (CPU)` (オレンジ色)

3. **実行ログ**:
   ```
   [時刻] モデルファイルをロードしました: model.onnx
   [時刻] 実行プロバイダー: CUDA Execution Provider (GPU)
   ```

4. **開始ボタン**:
   - ボタンが無効の場合、マウスカーソルを合わせると理由が表示されます:
     - "ONNXモデルをロードしてください"
     - "セグメンテーション対象のボリュームがロードされていません。メイン画面でCTボリュームを読み込んでから..."

## トラブルシューティング

### 問題1: CUDAExecutionProviderが利用不可能

**症状**:
```
❌ CUDAExecutionProvider not in available providers list
```

**解決方法**:

1. **CUDA Providerライブラリの確認**:
   ```bash
   ls -lh /usr/local/onnxruntime/lib/libonnxruntime_providers_cuda.so
   ```

   ファイルが存在しない場合、CUDA対応版のONNX Runtimeをインストールしてください。

2. **LD_LIBRARY_PATHの確認**:
   ```bash
   echo $LD_LIBRARY_PATH
   ```

   ONNX Runtimeのlibディレクトリが含まれているか確認します。

3. **リンクされているライブラリの確認**:
   ```bash
   ldd ./ShioRIS3 | grep onnx
   ```

   出力例:
   ```
   libonnxruntime.so.1 => /usr/local/onnxruntime/lib/libonnxruntime.so.1
   libonnxruntime_providers_cuda.so => /usr/local/onnxruntime/lib/libonnxruntime_providers_cuda.so
   libonnxruntime_providers_shared.so => /usr/local/onnxruntime/lib/libonnxruntime_providers_shared.so
   ```

4. **CUDA Runtimeの確認**:
   ```bash
   ldconfig -p | grep libcudart
   ```

   CUDAランタイムが見つからない場合、CUDA Toolkitをインストールしてください。

### 問題2: nvidia-smiが見つからない

**症状**:
```
GPU detection: FAILED (nvidia-smi not available)
```

**解決方法**:

1. **nvidia-smiの場所を確認**:
   ```bash
   which nvidia-smi
   # または
   find /usr -name nvidia-smi 2>/dev/null
   ```

2. **NVIDIAドライバーを再インストール**:
   ```bash
   # Ubuntuの場合
   sudo ubuntu-drivers autoinstall
   # または
   sudo apt install nvidia-driver-535
   ```

### 問題3: 開始ボタンがクリックできない

**解決方法**:

1. **ボタンのツールチップを確認**: マウスカーソルをボタンに合わせて理由を確認

2. **ボリュームがロードされているか確認**:
   - メイン画面でDICOM CTボリュームを読み込む
   - AIセグメンテーションダイアログを開く（メニューから）

3. **モデルがロードされているか確認**:
   - "参照..."ボタンからONNXモデルを選択
   - "モデル: ロード完了"が表示されることを確認

### 問題4: コンパイルエラー

**症状**:
```
error: 'isCudaEnabled' is not a member of 'OnnxSegmenter'
```

**解決方法**:

すべての修正が適用されていることを確認してください：
- `include/ai/onnx_segmenter.h`に`isCudaEnabled()`と`getExecutionProviderInfo()`が追加されている
- `src/ai/onnx_segmenter.cpp`のCUDA初期化コードが更新されている

## サポート

問題が解決しない場合、以下の情報を提供してください：

1. **システム情報**:
   ```bash
   uname -a
   nvidia-smi
   nvcc --version
   ```

2. **ビルドログ**: CMakeとmakeの完全な出力

3. **実行時ログ**: 端末でアプリケーションを起動した際のすべてのログ

4. **ライブラリ情報**:
   ```bash
   ldd ./ShioRIS3 | grep onnx
   ls -lh /usr/local/onnxruntime/lib/
   ```

## 参考情報

- [ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- ShioRIS3 プロジェクト: https://github.com/ShioRISNo1/ShioRIS3
