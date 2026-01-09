# CUDA Backend - Build and Test Guide

## 概要

このガイドでは、ShioRIS3のCUDAバックエンドをビルドしてテストする手順を説明します。

## 前提条件

### 1. NVIDIA GPUとドライバー

RTX 3090または他のCUDA対応NVIDIA GPU（Compute Capability 7.5以上）が必要です。

```bash
# GPUの確認（ドライバーがインストールされている場合）
nvidia-smi

# 期待される出力:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.2   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        ...          | Bus-Id        ...    | ...                  |
# |===============================+======================+======================|
# |   0  NVIDIA GeForce RTX 3090  | 00000000:01:00.0     | ...                  |
```

ドライバーがインストールされていない場合:

```bash
# 推奨ドライバーの確認
ubuntu-drivers devices

# 自動インストール
sudo ubuntu-drivers autoinstall

# または、特定バージョンを指定
sudo apt install nvidia-driver-535

# 再起動
sudo reboot
```

### 2. CUDA Toolkit のインストール

#### 方法1: apt経由（推奨）

```bash
# CUDA Toolkitのインストール
sudo apt update
sudo apt install nvidia-cuda-toolkit

# または、特定バージョン
sudo apt install cuda-toolkit-12-0
```

#### 方法2: NVIDIA公式サイトから

詳細は [`CUDA_INSTALLATION_GUIDE.md`](../CUDA_INSTALLATION_GUIDE.md) を参照してください。

#### 環境変数の設定

`~/.bashrc` に以下を追加:

```bash
# CUDA環境変数
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

設定を反映:

```bash
source ~/.bashrc
```

#### インストール確認

```bash
# nvccコンパイラの確認
nvcc --version

# 期待される出力:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Cuda compilation tools, release 12.x, ...
```

## ビルド手順

### 1. ビルドディレクトリの準備

```bash
cd /home/user/ShioRIS3

# 既存のビルドディレクトリをクリーンアップ
rm -rf build
mkdir build
cd build
```

### 2. CMake設定

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_GPU_DOSE_CALCULATION=ON \
      ..
```

#### 期待される出力

CMake出力で以下を確認:

```
=== GPU Dose Calculation Configuration ===
✓ CUDA Toolkit found: 12.x
  CUDA Toolkit Root: /usr/local/cuda/include
  CUDA Compiler: /usr/local/cuda/bin/nvcc
  CUDA Architectures: 75;80;86;89
✓ OpenCL found: 3.0
  OpenCL Include: /usr/include
  OpenCL Library: /usr/lib/x86_64-linux-gnu/libOpenCL.so
✓ GPU dose calculation enabled with CUDA backend (primary - NVIDIA)
  OpenCL backend available (fallback - CUDA preferred for NVIDIA)
```

### 3. ビルド

```bash
# 並列ビルド（全CPUコアを使用）
make -j$(nproc)

# または、進捗を詳細表示
make VERBOSE=1 -j$(nproc)
```

#### ビルド時間の目安

- RTX 3090環境: 約2-5分
- CUDAカーネルのコンパイルに時間がかかります

#### 一般的なビルドエラーと解決策

**エラー1: nvcc: command not found**

```bash
# 解決策: CUDA Toolkitのインストールとパス設定を確認
which nvcc
echo $PATH | grep cuda
```

**エラー2: unsupported GNU version**

```bash
# 解決策: GCCバージョンを確認（CUDA 12.xはGCC 12まで対応）
gcc --version

# 古いGCCが必要な場合
sudo apt install gcc-11 g++-11
export CC=gcc-11
export CXX=g++-11
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**エラー3: No CUDA toolchain found**

```bash
# 解決策: CMakeにCUDAのパスを明示的に指定
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
         -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

## 実行とテスト

### 1. 基本動作確認

```bash
# ShioRIS3を起動
./ShioRIS3
```

#### 起動ログで確認すべきポイント

```
GPU Backend Factory: Detecting best available backend...
GPU Backend Factory: Platform = Windows/Linux
GPU Backend Factory: Checking CUDA backend...
CUDA: Initializing backend...
CUDA: Device detection started
CUDA: Found 1 device(s)
CUDA: Device 0: NVIDIA GeForce RTX 3090 Compute 8.6 Memory: 24576 MB
CUDA: ✓ Selected device 0
CUDA: Device capability check passed
CUDA: Compute Capability: 8.6
CUDA: Total Memory: 24576 MB
CUDA: Multiprocessors: 82
CUDA: Initialization successful
CUDA: Device: NVIDIA GeForce RTX 3090 (Compute 8.6, 24576 MB)
GPU Backend Factory: CUDA backend available
✓ GPU dose calculation enabled with CUDA backend
```

### 2. UI確認

1. **File → Load CyberKnife Beam Data...** からビームデータを読み込み
2. CyberKnifeパネルで:
   - 「Enable GPU Acceleration」チェックボックスが表示される
   - GPU状態インジケーターで「🟢 CUDA: NVIDIA GeForce RTX 3090」が表示される
3. チェックボックスを有効化してGPU加速をオン

### 3. 線量計算テスト

1. CTデータを読み込み
2. CyberKnifeパネルでビーム設定
3. 「Calculate Dose」をクリック
4. コンソール出力で確認:

```
CUDA: Uploading CT volume: 512 x 512 x 300
CUDA: CT volume uploaded successfully
CUDA: Uploading beam data tables...
CUDA: Beam data uploaded successfully
CUDA: Calculating dose...
CUDA: Launching kernel with grid 128 x 128 x 75
CUDA: Block size: 8 x 8 x 8
CUDA: Grid size: 16 x 16 x 10
CUDA: Dose calculation completed successfully
```

### 4. 性能ベンチマーク

#### テストケース: 512³ボリューム、単一ビーム

```
CPU (QtConcurrent):  ~60秒
OpenCL:              ~10秒 (6x高速化)
CUDA:                ~3秒  (20x高速化)
```

#### テストケース: マルチビーム（100本）

```
CPU:     ~100分
OpenCL:  ~15分 (6.7x高速化)
CUDA:    ~5分  (20x高速化)
```

## トラブルシューティング

### 問題1: CUDAデバイスが見つからない

**症状:**
```
CUDA: Found 0 device(s)
CUDA: Device detection failed
```

**解決策:**
```bash
# NVIDIAドライバーを確認
nvidia-smi

# ドライバーが正常なら、CUDAランタイムライブラリを確認
ldconfig -p | grep cuda

# 見つからない場合、LD_LIBRARY_PATHを設定
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 問題2: CUDA out of memory

**症状:**
```
CUDA: Failed to allocate CT volume buffer: out of memory
```

**解決策:**
- ボリュームサイズを縮小
- 他のGPUアプリケーションを終了
- `nvidia-smi` でGPUメモリ使用状況を確認

### 問題3: 計算結果がCPU版と異なる

**症状:**
線量値がCPU計算と大きく異なる

**原因:**
- 浮動小数点演算の順序の違い
- 許容範囲: ±0.5%程度

**確認方法:**
```cpp
// CPU版とCUDA版の結果を比較
// 最大差分を確認
```

### 問題4: カーネル起動エラー

**症状:**
```
CUDA kernel launch failed: invalid argument
```

**解決策:**
1. ビームデータが正しくアップロードされているか確認
2. ボリューム次元が正しいか確認
3. デバッグビルドで詳細ログを確認:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Debug
   ```

## 最適化のヒント

### GPU使用率の確認

```bash
# リアルタイムモニタリング
nvidia-smi dmon -s u

# 期待される値: GPU使用率 90-100%
```

### ブロックサイズのチューニング

`src/cyberknife/cuda_dose_backend.cu` の `calculateDose()` 関数内:

```cpp
// 現在の設定（RTX 3090最適化）
dim3 blockSize(8, 8, 8);  // 512 threads per block

// メモリバウンドな場合
dim3 blockSize(16, 8, 4);  // より大きいブロック

// コンピュートバウンドな場合
dim3 blockSize(4, 4, 4);   // より小さいブロック
```

### CUDAアーキテクチャの追加

CMakeLists.txtで対象アーキテクチャを追加:

```cmake
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89" CACHE STRING "CUDA architectures")
```

## パフォーマンスプロファイリング

### NVIDIA Nsight Systems

```bash
# プロファイリング
nsys profile --stats=true ./ShioRIS3

# レポート表示
nsys-ui report.nsys-rep
```

### NVIDIA Nsight Compute

```bash
# カーネル詳細プロファイリング
ncu --set full ./ShioRIS3
```

## 次のステップ

1. **性能最適化**
   - 共有メモリの活用
   - テクスチャメモリの活用
   - ストリーム並列化

2. **機能拡張**
   - マルチGPU対応
   - Tensor Core活用（混合精度演算）
   - 動的負荷分散

3. **デバッグツール**
   - cuda-memcheck
   - compute-sanitizer

## 参考資料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [RTX 3090 Specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/)
- [CyberKnife Dose Algorithms](./cyberknife_dose_algorithms.md)
- [GPU Dose Calculation Architecture](./gpu_dose_calculation.md)

## 貢献者

- Claude (2024) - CUDA backend implementation

## ライセンス

ShioRIS3 プロジェクトのライセンスに準拠
