# Whisper音声認識モードのビルド手順

## 初回セットアップ

### 1. サブモジュールの初期化

whisper.cppはgitサブモジュールとして追加されているため、初回ビルド時にサブモジュールを初期化する必要があります。

#### Windows (PowerShell)

```powershell
cd C:\Projects\ShioRIS3

# サブモジュールを初期化・更新
git submodule init
git submodule update
```

#### macOS / Linux

```bash
cd /Users/shiomi/Projects/ShioRIS3/ShioRIS3

# サブモジュールを初期化・更新
git submodule update --init --recursive
```

これにより、`external/whisper.cpp`ディレクトリにwhisper.cppのソースコードがクローンされます。

### 2. ビルド

サブモジュールの初期化が完了したら、通常通りビルドできます：

#### Windows

```powershell
# ビルドスクリプトを実行
.\build.ps1
```

#### macOS / Linux

```bash
# ビルドディレクトリを作成（存在しない場合）
mkdir -p build
cd build

# CMakeを実行
cmake ..

# ビルド
cmake --build . -j8

# または make を使用
make -j8
```

## macOS固有の設定

### Metal/CoreML サポート (オプション、推奨)

macOSでは、Metal（GPU加速）とCoreML（Neural Engine加速）が利用可能です。
これらを有効にすると、Whisperの推論速度が大幅に向上します。

#### Metal サポート (自動有効)

CMakeLists.txtはMetal対応を自動的に検出して有効化します。特に設定は不要です。

#### CoreML サポート (オプション)

CoreMLを有効にする場合は、以下の手順が必要です：

```bash
# Python環境の準備（Homebrewのpythonを使用）
brew install python@3.11

# 必要なPythonパッケージをインストール
pip3 install ane_transformers openai-whisper coremltools torch==2.1.0

# CoreMLを有効にしてビルド
cd build
cmake .. -DWHISPER_COREML=1
cmake --build . -j8
```

**注意**: CoreMLモデルの変換は初回のみ約20分かかります。

## トラブルシューティング

### エラー: "does not contain a CMakeLists.txt file"

**原因**: gitサブモジュールが初期化されていない

**解決策**:
```bash
git submodule update --init --recursive
```

### エラー: "Qt6::Multimedia not found"

**原因**: Qt Multimediaがインストールされていない

**解決策** (Homebrew使用時):
```bash
# Qt6を再インストール（Multimediaを含む）
brew reinstall qt@6
```

### エラー: "whisper target not found"

**原因**: whisper.cppのビルドに失敗している

**解決策**:
```bash
# ビルドディレクトリをクリーンアップ
rm -rf build
mkdir build
cd build

# 再度ビルド
cmake ..
cmake --build . -j8
```

## Whisperモデルのダウンロード

ビルドが成功したら、Whisperモデルをダウンロードする必要があります。

### Windows

#### 方法1: PowerShellで直接ダウンロード（推奨）

```powershell
# モデル保存ディレクトリを作成
$modelsDir = "$env:APPDATA\ShioRIS3 Development Team\ShioRIS3\whisper\models"
New-Item -ItemType Directory -Force -Path $modelsDir

# baseモデルをダウンロード (推奨: 142MB)
Start-BitsTransfer -Source "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin" -Destination "$modelsDir\ggml-base.bin"

# または tinyモデル (最速: 75MB)
# Start-BitsTransfer -Source "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin" -Destination "$modelsDir\ggml-tiny.bin"
```

#### 方法2: whisper.cppのダウンロードスクリプトを使用

```powershell
# プロジェクトディレクトリに移動
cd C:\Projects\ShioRIS3

# モデル保存ディレクトリを指定してダウンロード
.\external\whisper.cpp\models\download-ggml-model.cmd base "$env:APPDATA\ShioRIS3 Development Team\ShioRIS3\whisper\models"
```

### macOS

#### 推奨モデル: base (142MB)

```bash
# モデル保存ディレクトリを作成
mkdir -p ~/Library/Application\ Support/ShioRIS3/whisper/models

# baseモデルをダウンロード
cd ~/Library/Application\ Support/ShioRIS3/whisper/models
curl -L -o ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
```

### 利用可能なモデル

- **tiny** (75MB, 最速): `ggml-tiny.bin` - 開発・テスト用、高速だが精度低め
- **base** (142MB, 推奨): `ggml-base.bin` - バランスの取れた選択
- **small** (466MB, 高精度): `ggml-small.bin` - より高精度だが処理速度は遅め
- **medium** (1.5GB): `ggml-medium.bin` - 高精度、大容量
- **large-v3** (2.9GB): `ggml-large-v3.bin` - 最高精度、大容量・低速

macOSの場合:
```bash
# tinyモデル（開発・テスト用）
curl -L -o ggml-tiny.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin

# smallモデル（高精度）
curl -L -o ggml-small.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
```

## ビルド完了後

ビルドが成功すると、以下のファイルが生成されます：

```
build/ShioRIS3.app/                    # macOSアプリケーションバンドル
build/ShioRIS3.app/Contents/MacOS/ShioRIS3  # 実行ファイル
```

### 実行方法

```bash
# GUIから実行
open build/ShioRIS3.app

# またはターミナルから実行
./build/ShioRIS3.app/Contents/MacOS/ShioRIS3
```

## 動作確認

1. ShioRIS3を起動
2. AI Control Panelを開く
3. 音声入力セクションが表示されることを確認
4. マイクアクセス許可のダイアログが表示されたら「許可」をクリック
5. 「🎤 録音」ボタンをクリック
6. 何か話す
7. 「⏹ 停止」ボタンをクリック
8. 文字起こし結果が表示されることを確認

## まとめ

### Windows (PowerShell)

```powershell
# 完全なセットアップ手順（まとめ）
cd C:\Projects\ShioRIS3

# 1. サブモジュールを初期化
git submodule init
git submodule update

# 2. ビルド
.\build.ps1

# 3. モデルをダウンロード
$modelsDir = "$env:APPDATA\ShioRIS3 Development Team\ShioRIS3\whisper\models"
New-Item -ItemType Directory -Force -Path $modelsDir
Start-BitsTransfer -Source "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin" -Destination "$modelsDir\ggml-base.bin"

# 4. 実行
.\build\Release\ShioRIS3.exe
```

### macOS

```bash
# 完全なセットアップ手順（まとめ）
cd /Users/shiomi/Projects/ShioRIS3/ShioRIS3

# 1. サブモジュールを初期化
git submodule update --init --recursive

# 2. ビルド
mkdir -p build
cd build
cmake ..
cmake --build . -j8

# 3. モデルをダウンロード
mkdir -p ~/Library/Application\ Support/ShioRIS3/whisper/models
cd ~/Library/Application\ Support/ShioRIS3/whisper/models
curl -L -o ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin

# 4. 実行
cd /Users/shiomi/Projects/ShioRIS3/ShioRIS3
open build/ShioRIS3.app
```

これで、Whisper音声認識機能を含むShioRIS3のビルドが完了します！
