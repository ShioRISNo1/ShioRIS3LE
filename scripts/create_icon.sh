#!/bin/bash

# PNG画像から macOS用の .icns ファイルを作成するスクリプト

# 入力ファイル
INPUT_PNG="resources/images/ShioRIS3_Logo.png"
# 出力ファイル
OUTPUT_ICNS="resources/images/ShioRIS3.icns"

# 一時ディレクトリの作成
ICONSET_DIR="ShioRIS3.iconset"
mkdir -p "$ICONSET_DIR"

# 各サイズのアイコンを生成
sips -z 16 16     "$INPUT_PNG" --out "${ICONSET_DIR}/icon_16x16.png"
sips -z 32 32     "$INPUT_PNG" --out "${ICONSET_DIR}/icon_16x16@2x.png"
sips -z 32 32     "$INPUT_PNG" --out "${ICONSET_DIR}/icon_32x32.png"
sips -z 64 64     "$INPUT_PNG" --out "${ICONSET_DIR}/icon_32x32@2x.png"
sips -z 128 128   "$INPUT_PNG" --out "${ICONSET_DIR}/icon_128x128.png"
sips -z 256 256   "$INPUT_PNG" --out "${ICONSET_DIR}/icon_128x128@2x.png"
sips -z 256 256   "$INPUT_PNG" --out "${ICONSET_DIR}/icon_256x256.png"
sips -z 512 512   "$INPUT_PNG" --out "${ICONSET_DIR}/icon_256x256@2x.png"
sips -z 512 512   "$INPUT_PNG" --out "${ICONSET_DIR}/icon_512x512.png"
sips -z 1024 1024 "$INPUT_PNG" --out "${ICONSET_DIR}/icon_512x512@2x.png"

# .icns ファイルの作成
iconutil -c icns "$ICONSET_DIR" -o "$OUTPUT_ICNS"

# 一時ディレクトリの削除
rm -rf "$ICONSET_DIR"

echo "✓ アイコンファイルを作成しました: $OUTPUT_ICNS"
