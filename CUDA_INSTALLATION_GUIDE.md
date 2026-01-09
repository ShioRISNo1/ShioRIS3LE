# CUDA Toolkit インストールガイド

## 現状確認

現在の状態：
- ✅ NVIDIAドライバー: インストール済み（nvidia-smiが動作）
- ❌ CUDA Toolkit: **未インストール**（nvccが見つからない）

CUDA Toolkitは、GPUプログラミングに必要な開発ツール（コンパイラ、ライブラリ、ヘッダーファイル）のセットです。nvidia-smiはドライバーに含まれますが、開発にはCUDA Toolkitが別途必要です。

## インストール方法（Ubuntu/Debian）

### 方法1: apt経由でインストール（推奨）

1. **利用可能なCUDAバージョンを確認**
   ```bash
   apt search cuda-toolkit
   ```

2. **CUDA Toolkit 12.xをインストール**
   ```bash
   sudo apt update
   sudo apt install nvidia-cuda-toolkit
   ```

   または、特定のバージョン（例：CUDA 12.0）：
   ```bash
   sudo apt install cuda-toolkit-12-0
   ```

3. **環境変数を設定**

   `~/.bashrc`または`~/.zshrc`に以下を追加：
   ```bash
   # CUDA環境変数
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

   設定を反映：
   ```bash
   source ~/.bashrc
   ```

4. **インストール確認**
   ```bash
   nvcc --version
   ```

   以下のような出力が表示されればOK：
   ```
   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2023 NVIDIA Corporation
   Built on ...
   Cuda compilation tools, release 12.x, ...
   ```

### 方法2: NVIDIA公式サイトからインストール

最新版や特定バージョンが必要な場合：

1. **ドライバーバージョンの確認**
   ```bash
   nvidia-smi
   ```

   ドライバーバージョンに対応したCUDA Toolkitを選択してください：
   - Driver 525.x以上 → CUDA 12.0以降をサポート
   - Driver 450.x-524.x → CUDA 11.xをサポート

2. **CUDA Toolkitのダウンロード**

   [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

   または、過去のバージョン：
   [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)

3. **例：CUDA 12.0のインストール（Ubuntu 22.04）**

   ```bash
   # ダウンロード（例）
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
   sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2204-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

4. **シンボリックリンクの確認**
   ```bash
   ls -la /usr/local/cuda
   ```

   `/usr/local/cuda`が最新のCUDAバージョンへのシンボリックリンクになっているはずです。

5. **環境変数を設定**（方法1と同じ）

## インストール後の確認

すべてのツールが正しくインストールされているか確認：

```bash
# nvccコンパイラ
nvcc --version

# CUDA サンプルのビルドディレクトリ確認
ls /usr/local/cuda/samples 2>/dev/null || echo "Samples not found (optional)"

# ライブラリの確認
ls -lh /usr/local/cuda/lib64/libcudart.so*
ls -lh /usr/local/cuda/lib64/libcublas.so*
```

## ONNX Runtime CUDA Providerの確認

CUDA Toolkitインストール後、ONNX RuntimeのCUDA Providerライブラリも確認：

```bash
ls -lh /usr/local/onnxruntime/lib/libonnxruntime_providers_cuda.so
```

このファイルが存在しない場合、CUDA対応版のONNX Runtimeを再インストールする必要があります。

### CUDA対応ONNX Runtimeのインストール

1. **既存のONNX Runtimeのバージョン確認**
   ```bash
   ls -la /usr/local/onnxruntime/lib/libonnxruntime.so*
   ```

2. **CUDA対応版をダウンロード**

   [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)

   例：ONNX Runtime 1.16.3 with CUDA 12.x support
   ```bash
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-gpu-1.16.3.tgz
   tar -xzf onnxruntime-linux-x64-gpu-1.16.3.tgz
   sudo rm -rf /usr/local/onnxruntime  # 既存を削除（バックアップ推奨）
   sudo mv onnxruntime-linux-x64-gpu-1.16.3 /usr/local/onnxruntime
   ```

3. **必要なライブラリの確認**
   ```bash
   ls -lh /usr/local/onnxruntime/lib/
   ```

   以下が含まれているはず：
   - `libonnxruntime.so` - メインライブラリ
   - `libonnxruntime_providers_cuda.so` - CUDA Provider
   - `libonnxruntime_providers_shared.so` - Shared Provider
   - （オプション）`libonnxruntime_providers_tensorrt.so` - TensorRT Provider

## ShioRIS3の再ビルド

CUDA Toolkitインストール後、ShioRIS3を再ビルド：

1. **ビルドディレクトリのクリーンアップ**
   ```bash
   cd /path/to/ShioRIS3
   rm -rf build
   mkdir build
   cd build
   ```

2. **CMake実行**
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release \
         -DENABLE_ONNXRUNTIME=ON \
         -DONNXRUNTIME_ROOT=/usr/local/onnxruntime \
         ..
   ```

3. **CUDA検出の確認**

   CMake出力で以下を確認：
   ```
   -- Checking CUDA availability...
   -- ✓ CUDA Toolkit found: Version 12.x
   -- ✓ CUDA provider library found: /usr/local/onnxruntime/lib/libonnxruntime_providers_cuda.so

   🚀 ===== CUDA SUPPORT ENABLED =====
      CUDA Provider: /usr/local/onnxruntime/lib/libonnxruntime_providers_cuda.so
      Shared Provider: /usr/local/onnxruntime/lib/libonnxruntime_providers_shared.so
      Macro ONNXRUNTIME_USE_CUDA will be defined
   ===================================
   ```

4. **ビルド**
   ```bash
   make -j$(nproc)
   ```

5. **動作確認**
   ```bash
   export LD_LIBRARY_PATH=/usr/local/onnxruntime/lib:$LD_LIBRARY_PATH
   ./ShioRIS3
   ```

   アプリ起動時のログで確認：
   ```
   === CUDA EXECUTION PROVIDER INITIALIZATION ===
   Build configuration: ONNXRUNTIME_USE_CUDA is defined
   Available execution providers:
     - CPUExecutionProvider
     - CUDAExecutionProvider
   ✓ CUDAExecutionProvider is available
   ✓ CUDA Execution Provider successfully enabled (GPU device 0)
   === CUDA EP INITIALIZATION COMPLETE ===
   CUDA Enabled: YES
   ```

## トラブルシューティング

### 問題1: nvcc: command not found（インストール後も）

**解決方法**:
```bash
# CUDAのパスを確認
find /usr/local -name nvcc 2>/dev/null

# シンボリックリンクが正しいか確認
ls -la /usr/local/cuda

# 環境変数を再確認
echo $PATH | grep cuda
echo $CUDA_HOME

# .bashrcを再読み込み
source ~/.bashrc
```

### 問題2: CMakeがCUDA Toolkitを見つけられない

**解決方法**:

CMakeにCUDAのパスを明示的に指定：
```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_ONNXRUNTIME=ON \
      -DONNXRUNTIME_ROOT=/usr/local/onnxruntime \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      ..
```

または、環境変数を設定：
```bash
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_ONNXRUNTIME=ON ..
```

### 問題3: ドライバーとCUDA Toolkitの互換性エラー

**症状**:
```
CUDA driver version is insufficient for CUDA runtime version
```

**解決方法**:

NVIDIAドライバーのバージョンを確認し、対応するCUDA Toolkitを選択：

```bash
nvidia-smi
```

互換性表：
- Driver 525.60.13以上 → CUDA 12.0以降
- Driver 515.43.04以上 → CUDA 11.7-11.8
- Driver 510.39.01以上 → CUDA 11.6

必要に応じてドライバーをアップグレード：
```bash
sudo ubuntu-drivers autoinstall
# または
sudo apt install nvidia-driver-535  # 最新の推奨バージョンに置き換え
```

### 問題4: libcudart.so not found

**症状**:
```
error while loading shared libraries: libcudart.so.12: cannot open shared object file
```

**解決方法**:
```bash
# LD_LIBRARY_PATHにCUDAライブラリパスを追加
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# システムワイドで設定（推奨）
sudo bash -c 'echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf'
sudo ldconfig

# 確認
ldconfig -p | grep cuda
```

## 確認コマンド一覧

インストール成功後、以下のコマンドでシステム全体を確認：

```bash
# 1. CUDA Toolkitバージョン
nvcc --version

# 2. NVIDIAドライバー
nvidia-smi

# 3. CUDA環境変数
echo $CUDA_HOME
echo $PATH | grep cuda
echo $LD_LIBRARY_PATH | grep cuda

# 4. ONNX Runtime CUDA Provider
ls -lh /usr/local/onnxruntime/lib/libonnxruntime_providers_cuda.so

# 5. ShioRIS3のリンク確認
cd /path/to/ShioRIS3/build
ldd ./ShioRIS3 | grep -E "(cuda|onnx)"
```

すべてが正常なら、以下のような出力が得られます：
```
# nvcc --version
Cuda compilation tools, release 12.0, ...

# nvidia-smi
NVIDIA-SMI 525.xx.xx   Driver Version: 525.xx.xx   CUDA Version: 12.0

# ldd ./ShioRIS3 | grep onnx
libonnxruntime.so => /usr/local/onnxruntime/lib/libonnxruntime.so
libonnxruntime_providers_cuda.so => /usr/local/onnxruntime/lib/libonnxruntime_providers_cuda.so
libonnxruntime_providers_shared.so => /usr/local/onnxruntime/lib/libonnxruntime_providers_shared.so
```

## 参考リンク

- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [CUDA GPUs - Compute Capability](https://developer.nvidia.com/cuda-gpus)
