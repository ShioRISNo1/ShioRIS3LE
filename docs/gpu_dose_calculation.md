# GPU-Accelerated Dose Calculation

## 概要

CyberKnifeの線量計算をGPU（OpenCL, Metal）を用いて高速化する機能の実装です。

## アーキテクチャ

### 設計原則

- **プラグイン型バックエンド**: 複数のGPUバックエンド（OpenCL, Metal, CUDA）を統一インターフェースでサポート
- **CPUフォールバック**: GPU が利用不可能な場合は既存のCPU実装を使用
- **クロスプラットフォーム**: Windows, Linux, macOS で動作

### ファイル構造

```
include/cyberknife/
  ├── gpu_dose_backend.h           # 統一インターフェース定義
  ├── opencl_dose_backend.h        # OpenCL実装ヘッダー
  └── metal_dose_backend.h         # Metal実装ヘッダー (macOS)

src/cyberknife/
  ├── gpu_dose_backend.cpp         # ファクトリー実装
  ├── opencl_dose_backend.cpp      # OpenCL実装
  ├── opencl_kernels.cl            # OpenCL カーネル（C99）
  ├── metal_dose_backend.mm        # Metal実装 (Objective-C++)
  └── metal_kernels.metal          # Metal シェーダー (MSL)
```

### クラス構造

```
IGPUDoseBackend (抽象インターフェース)
    ↑
    ├── OpenCLDoseBackend   (実装済み - Windows/Linux/macOS)
    ├── MetalDoseBackend    (実装済み - macOS)
    └── CUDADoseBackend     (未実装)

GPUDoseBackendFactory (ファクトリー)
    └── 自動検出・生成 (macOS: Metal優先 → OpenCL fallback)
```

## 実装状況

### ✅ 完了

1. **統合インターフェース設計** (`gpu_dose_backend.h`)
   - `IGPUDoseBackend` 抽象クラス
   - `GPUComputeParams` パラメータ構造体
   - `GPUBeamData` ビームデータ構造体
   - `GPUDoseBackendFactory` ファクトリークラス

2. **OpenCLバックエンド基本実装** (`opencl_dose_backend.cpp`)
   - デバイス検出・選択
   - コンテキスト・コマンドキュー作成
   - カーネルコンパイル
   - バッファ管理

3. **OpenCLカーネル** (`opencl_kernels.cl`)
   - 線量計算カーネル（`calculateDoseKernel`）
   - 三線形補間カーネル（`interpolateVolumeKernel`）
   - 2D/3D補間ヘルパー関数

4. **CMake統合**
   - OpenCL検出とリンク
   - コンパイル定義（`USE_OPENCL_BACKEND`）
   - カーネルファイルのコピー

5. **既存コードへの統合**
   - `CyberKnifeDoseCalculator` にGPUサポート追加
   - `initializeGPU()`, `setGPUEnabled()` メソッド

6. **Metalバックエンド実装** (`metal_dose_backend.mm`, `metal_kernels.metal`)
   - デバイス取得 (`MTLCreateSystemDefaultDevice`)
   - シェーダーコンパイル (.metal ファイルから)
   - コマンドバッファ・エンコーダー管理
   - バッファ管理 (CT, Dose, Parameters)
   - 線量計算カーネル実装 (Metal Shading Language)
   - Apple Silicon ネイティブサポート

7. **CMake Metal統合**
   - Metal/Foundation framework検出とリンク
   - コンパイル定義（`USE_METAL_BACKEND`）
   - シェーダーファイルのアプリバンドルへのコピー
   - macOSでMetal優先、OpenCL fallback設定

8. **GPUステータスUI**
   - CyberKnifeパネルにGPU有効化チェックボックス追加
   - GPU状態表示ラベル（デバイス名、computing状態）
   - 絵文字インジケーター（🟢 Active, 🟡 Available, ⚡ Computing, 💻 CPU）

### ⚠️ 既知の制限事項

- OpenCLバックエンドはmacOS（特にApple Silicon）で非推奨
  - Apple Silicon MacではOpenCLがGPUデバイスを公開しない
  - CPUデバイスのみ利用可能
  - **推奨**: macOSではMetalバックエンドを使用
- 詳細は [`docs/gpu_macos_notes.md`](./gpu_macos_notes.md) を参照

### 🚧 未完成（今後の作業）

1. **ビームデータアップロード**
   - BeamDataManager からGPU用テーブルへの変換
   - OF, TMR, OCR テーブルの適切なフォーマット変換
   - GPU メモリへの効率的な転送

2. **線量計算の統合**
   - `calculateVolumeDose()` でGPUパスの実装
   - Dynamic Grid モードのGPU対応
   - マルチビーム計算の並列化

3. **補間処理の実装**
   - `interpolateVolume()` の完成
   - コンピューテッドマスクの管理

4. **エラーハンドリング**
   - より詳細なエラーメッセージ
   - GPU メモリ不足時の対処
   - カーネル実行失敗時のフォールバック

5. **最適化**
   - 作業グループサイズのチューニング
   - メモリアクセスパターンの最適化
   - カーネル融合

6. **テスト**
   - 単体テスト
   - 統合テスト
   - CPU版との結果比較

## ビルド方法

### 前提条件

- OpenCL SDK がインストールされていること
  - **Linux**: `opencl-headers` + ベンダー提供のICD
  - **Windows**: NVIDIA/AMD/Intel SDK
  - **macOS**: 標準で利用可能（ただし非推奨）

### CMakeオプション

```bash
cmake -DENABLE_GPU_DOSE_CALCULATION=ON ..
```

GPU無効化する場合：
```bash
cmake -DENABLE_GPU_DOSE_CALCULATION=OFF ..
```

### バックエンド優先順位

- **macOS**: Metal (優先) → OpenCL (fallback, 非推奨)
- **Windows/Linux**: OpenCL

macOS環境では自動的にMetalが選択されます（利用可能な場合）。

## 使用方法

### C++ API

```cpp
#include "cyberknife/dose_calculator.h"

CyberKnifeDoseCalculator calculator;
calculator.initialize(beamDataPath);

#ifdef ENABLE_GPU_DOSE_CALCULATION
// GPU初期化
if (calculator.initializeGPU()) {
    qDebug() << "GPU Device:" << calculator.getGPUDeviceInfo();
    calculator.setGPUEnabled(true);
} else {
    qDebug() << "GPU not available, using CPU";
}
#endif

// 線量計算（GPUが有効なら自動的にGPUで計算）
calculator.calculateVolumeDose(ctVolume, beam, resultDose);
```

## 性能目標

| 処理 | CPU時間 | GPU目標時間 | 期待高速化率 |
|-----|---------|------------|------------|
| 単一ビーム（512³ボクセル） | 60秒 | 2-6秒 | 10-30倍 |
| マルチビーム（100ビーム） | 100分 | 3-10分 | 10-30倍 |
| Dynamic Grid（3パス） | 30秒 | 1-3秒 | 10-30倍 |

## 今後の拡張

1. **Metal バックエンド最適化** (macOS)
   - ✅ 基本実装完了
   - 🚧 Unified Memory 活用
   - 🚧 Apple Silicon特有の最適化

2. **CUDA バックエンド** (NVIDIA GPU 最適化)
   - Tensor Core 活用
   - より高度な最適化

3. **Vulkan Compute** (次世代クロスプラットフォーム)
   - モダンなAPI
   - 低オーバーヘッド

## 参考資料

- [OpenCL 1.2 Specification](https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf)
- [CyberKnife Dose Algorithm Documentation](./cyberknife_dose_algorithms.md)

## 貢献者

- Claude (2024) - 初期実装

## ライセンス

ShioRIS3 プロジェクトのライセンスに準拠
