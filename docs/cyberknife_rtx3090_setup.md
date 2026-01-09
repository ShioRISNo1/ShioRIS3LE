# CyberKnife線量計算 - RTX3090セットアップガイド

## 環境
- OS: Ubuntu 24.04 LTS
- GPU: NVIDIA RTX 3090
- プロジェクト: ShioRIS3

## 前提条件

### 1. NVIDIAドライバーのインストール

```bash
# 推奨ドライバーの確認
ubuntu-drivers devices

# 推奨ドライバーの自動インストール
sudo ubuntu-drivers autoinstall

# または、特定バージョンを指定
sudo apt install nvidia-driver-535

# 再起動
sudo reboot

# 確認
nvidia-smi
```

正常にインストールされていれば、以下のような出力が得られます：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
```

## オプション1: OpenCLバックエンド（推奨・即座に利用可能）

### OpenCLのインストール

```bash
# OpenCL開発ヘッダーとローダーのインストール
sudo apt update
sudo apt install opencl-headers ocl-icd-opencl-dev

# NVIDIA OpenCL ICD（Installable Client Driver）のインストール
sudo apt install nvidia-opencl-icd

# 確認ツールのインストール
sudo apt install clinfo

# OpenCL動作確認
clinfo
```

`clinfo`の出力で以下を確認：
```
Platform Name: NVIDIA CUDA
Device Name: NVIDIA GeForce RTX 3090
Device Type: GPU
```

### ShioRIS3のビルド

```bash
cd /home/user/ShioRIS3

# ビルドディレクトリを作成（既にある場合はクリーンアップ）
rm -rf build
mkdir build
cd build

# CMake設定（GPU線量計算を有効化）
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_GPU_DOSE_CALCULATION=ON \
      ..

# ビルド時の確認事項
# CMake出力で以下が表示されることを確認：
# ✓ OpenCL found: 3.0
# ✓ GPU dose calculation enabled with OpenCL backend

# ビルド
make -j$(nproc)
```

### 動作確認

```bash
# ShioRIS3を起動
./ShioRIS3
```

アプリケーション起動後：
1. **File → Load CyberKnife Beam Data...** からビームデータを読み込み
2. CyberKnifeパネルで「Enable GPU Acceleration」チェックボックスを有効化
3. GPU状態インジケーターで「🟢 OpenCL: NVIDIA GeForce RTX 3090」が表示されることを確認

## オプション2: CUDAバックエンド（最高性能・要実装）

現在、CUDAバックエンドは未実装です。OpenCLより高速な線量計算を実現するには、以下の実装が必要です。

### CUDA Toolkitのインストール

詳細は [`CUDA_INSTALLATION_GUIDE.md`](../CUDA_INSTALLATION_GUIDE.md) を参照してください。

```bash
# CUDA Toolkitのインストール
sudo apt install nvidia-cuda-toolkit

# または、特定バージョン
sudo apt install cuda-toolkit-12-0

# 環境変数の設定（~/.bashrcに追加）
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 確認
nvcc --version
```

### CUDAバックエンドの実装（TODO）

以下のファイルを作成する必要があります：

1. **`include/cyberknife/cuda_dose_backend.h`**
   - `IGPUDoseBackend`を継承したCUDAバックエンドクラス

2. **`src/cyberknife/cuda_dose_backend.cu`**
   - CUDA実装（C++/CUDA）

3. **`src/cyberknife/cuda_kernels.cu`**
   - CUDAカーネル実装
   - `calculateDoseKernel` - 線量計算カーネル
   - `interpolateVolumeKernel` - 三線形補間カーネル
   - `recalculateDoseWithThresholdKernel` - 閾値再計算カーネル

4. **CMakeLists.txtの修正**
   - CUDAToolkit検出
   - CUDA_ARCHITECTUREの設定（RTX3090: sm_86）
   - CUDAソースファイルのコンパイル設定

参考実装：
- OpenCLバックエンド: `src/cyberknife/opencl_dose_backend.cpp`
- OpenCLカーネル: `src/cyberknife/opencl_kernels.cl`

## 性能比較（予測）

| バックエンド | 単一ビーム（512³） | マルチビーム（100本） | 実装状況 |
|-------------|------------------|---------------------|----------|
| CPU（並列）  | 60秒              | 100分                | ✅ 完成   |
| OpenCL      | 5-10秒            | 8-16分               | ✅ 完成   |
| CUDA        | 2-6秒             | 3-10分               | ❌ 未実装 |

## トラブルシューティング

### 問題1: OpenCLデバイスが見つからない

**症状:**
```
clinfo
Number of platforms: 0
```

**解決方法:**
```bash
# NVIDIAドライバーの再インストール
sudo apt purge nvidia-*
sudo ubuntu-drivers autoinstall
sudo reboot

# OpenCL ICDの再インストール
sudo apt install --reinstall nvidia-opencl-icd
```

### 問題2: GPUが有効化できない（ShioRIS3内）

**確認事項:**
1. CMakeビルド時に `ENABLE_GPU_DOSE_CALCULATION=ON` が設定されているか確認
2. ビルドログで「✓ OpenCL found」が表示されているか確認
3. `clinfo`でOpenCLデバイスが検出されているか確認

**デバッグ:**
```bash
# ビルドディレクトリで確認
cd build
cmake .. -LAH | grep -i opencl

# 期待される出力:
# ENABLE_GPU_DOSE_CALCULATION:BOOL=ON
# OpenCL_FOUND:BOOL=TRUE
```

### 問題3: 線量計算が遅い

**原因と対策:**

1. **CPUモードで動作している**
   - GPU有効化チェックボックスが有効か確認
   - GPU初期化エラーログを確認

2. **OpenCLの最適化が不十分**
   - 作業グループサイズの調整が必要
   - CUDAバックエンドの実装を検討

3. **メモリ転送のボトルネック**
   - CTボリュームサイズの確認
   - ビームデータのアップロード頻度の最適化

## 次のステップ

### 短期（OpenCLで開始）
1. OpenCLセットアップを完了
2. 既存のCPU実装と結果を比較・検証
3. 性能ベンチマークの実施

### 中期（CUDA実装）
1. CUDAバックエンドの設計
2. カーネルの実装（`opencl_kernels.cl`をCUDAに移植）
3. 性能最適化（shared memory、coalesced access）

### 長期（高度な最適化）
1. Tensor Core活用（混合精度演算）
2. マルチGPU対応
3. 動的負荷分散（CPU+GPU hybrid）

## 参考資料

- [OpenCL Programming Guide](https://www.khronos.org/registry/OpenCL/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [RTX 3090 Specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/)
  - Compute Capability: 8.6 (sm_86)
  - CUDA Cores: 10496
  - Tensor Cores: 328 (第3世代)
  - メモリ: 24GB GDDR6X

## 関連ドキュメント

- [`docs/gpu_dose_calculation.md`](./gpu_dose_calculation.md) - GPU線量計算アーキテクチャ
- [`docs/cyberknife_dose_algorithms.md`](./cyberknife_dose_algorithms.md) - 線量計算アルゴリズム
- [`CUDA_INSTALLATION_GUIDE.md`](../CUDA_INSTALLATION_GUIDE.md) - CUDA Toolkitインストール詳細
