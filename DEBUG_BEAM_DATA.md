# CyberKnifeビームデータ読み込みエラーのデバッグ手順

## エラー: 「ビームデータディレクトリを特定できませんでした」

### 1. ビームデータファイルの確認

指定したディレクトリに以下の**すべてのファイル**が存在することを確認してください：

```bash
cd /path/to/beam_data
ls -la
```

**必須ファイル:**
- `DMTable.dat`
- `TMRtable.dat`
- `OCRtable*.dat` (少なくとも1つ、例: OCRtable5.dat, OCRtable7.dat など)

### 2. ファイル名の大文字小文字を確認 (Linux)

Linuxでは大文字小文字が厳密に区別されます：

❌ **間違い:**
```
dmtable.dat
DMtable.dat
dmTable.dat
```

✅ **正しい:**
```
DMTable.dat
TMRtable.dat
OCRtable5.dat
```

ファイル名を確認：
```bash
ls -1 /path/to/beam_data/ | grep -E "DMTable|TMRtable|OCRtable"
```

ファイル名を修正（必要な場合）：
```bash
mv dmtable.dat DMTable.dat
mv tmrtable.dat TMRtable.dat
mv ocrtable5.dat OCRtable5.dat
```

### 3. ファイルの読み取り権限を確認

```bash
ls -la /path/to/beam_data/*.dat
```

権限がない場合（例: `---------- 1 user user`）は修正：
```bash
chmod 644 /path/to/beam_data/*.dat
```

### 4. デバッグログを有効化

詳細なログを出力して、どのディレクトリが検査されているか確認：

```bash
export QT_LOGGING_RULES="cyberknife.beamdata.locator.debug=true"
./ShioRIS3
```

出力例：
```
cyberknife.beamdata.locator: Candidate "/home/user/beam_data" is missing required beam data files.
cyberknife.beamdata.locator: Beam data directory resolved: "/usr/share/shioris3/beam_data"
```

### 5. 環境変数でパスを明示的に指定

ディレクトリの自動検出がうまくいかない場合、環境変数で明示的に指定：

```bash
export CYBERKNIFE_BEAM_DATA_PATH=/path/to/beam_data
./ShioRIS3
```

### 6. ディレクトリ検索の優先順位

`BeamDataLocator` は以下の順序でディレクトリを検索します：

1. プログラムで明示的に指定されたパス
2. 環境変数 `CYBERKNIFE_BEAM_DATA_PATH`
3. QSettings `cyberknife/beamDataPath`
4. QSettings `paths/beamDataRoot` (レガシー)
5. 実行ファイルと同じディレクトリの `beam_data/`
6. `../share/shioris3/beam_data/` (Linuxパッケージ用)

各候補について：
- ディレクトリが存在するか
- 必須ファイル（DMTable.dat, TMRtable.dat, OCRtable*.dat）がすべて揃っているか

をチェックします。

### 7. 設定ファイルのクリア（問題がある場合）

古い設定が残っている場合、クリアして再試行：

```bash
# QSettings の場所を確認
find ~/.config -name "ShioRIS3.conf" 2>/dev/null
find ~/.local/share -name "ShioRIS3" 2>/dev/null

# 設定を削除（必要に応じて）
rm ~/.config/ShioRIS3/ShioRIS3.conf
```

### 8. テストコマンド

すべてが正しく設定されているか確認するテストスクリプト：

```bash
#!/bin/bash

BEAM_DATA_DIR="/path/to/your/beam_data"

echo "=== CyberKnife Beam Data Directory Check ==="
echo "Directory: $BEAM_DATA_DIR"
echo ""

# ディレクトリの存在確認
if [ ! -d "$BEAM_DATA_DIR" ]; then
    echo "❌ Error: Directory does not exist"
    exit 1
fi
echo "✅ Directory exists"

# DMTable.dat の確認
if [ ! -r "$BEAM_DATA_DIR/DMTable.dat" ]; then
    echo "❌ Error: DMTable.dat not found or not readable"
    exit 1
fi
echo "✅ DMTable.dat found and readable"

# TMRtable.dat の確認
if [ ! -r "$BEAM_DATA_DIR/TMRtable.dat" ]; then
    echo "❌ Error: TMRtable.dat not found or not readable"
    exit 1
fi
echo "✅ TMRtable.dat found and readable"

# OCRtable*.dat の確認
OCR_COUNT=$(find "$BEAM_DATA_DIR" -maxdepth 1 -name "OCRtable*.dat" -readable | wc -l)
if [ "$OCR_COUNT" -eq 0 ]; then
    echo "❌ Error: No readable OCRtable*.dat files found"
    exit 1
fi
echo "✅ Found $OCR_COUNT OCRtable*.dat file(s)"

echo ""
echo "=== All required files are present ==="
echo "Files:"
ls -lh "$BEAM_DATA_DIR"/*.dat
```

保存して実行：
```bash
chmod +x check_beam_data.sh
./check_beam_data.sh
```

## よくある問題と解決策

### 問題1: "Candidate xxx is missing required beam data files."

**原因:** ファイルが存在しないか、ファイル名が間違っている

**解決策:**
```bash
cd /path/to/beam_data
ls -1 | grep -iE "dm|tmr|ocr"  # 大文字小文字を無視して検索
```

ファイル名をリネーム：
```bash
rename 's/^dm/DM/' dm*.dat
rename 's/^tmr/TMR/' tmr*.dat
rename 's/^ocr/OCR/' ocr*.dat
```

### 問題2: ファイルは存在するのにエラーが出る

**原因:** ファイルの読み取り権限がない

**解決策:**
```bash
chmod 644 /path/to/beam_data/*.dat
```

### 問題3: シンボリックリンクが機能しない

**原因:** リンク先が壊れている

**解決策:**
```bash
ls -la /path/to/beam_data/
readlink -f /path/to/beam_data/DMTable.dat  # リンク先の実体を確認
```

## 参考

- ソースコード: `src/cyberknife/beam_data_locator.cpp:94-110`
- ドキュメント: `docs/cyberknife_beam_data.md`
