# CyberKnife ビームデータの登録手順

本ドキュメントでは、新たに追加された `CyberKnifeBeamData` テーブルへ計測データを登録する手順を示します。アプリケーションは SQLite データベースを使用しているため、`sqlite3` コマンドラインツールを利用するのがもっとも手軽です。

## 前提条件

- `resources/sql/create_tables.sql` の更新後にアプリケーションを 1 度起動し、データベースに `CyberKnifeBeamData` テーブルが作成されていること。
- SQLite3 コマンドラインツールが使用できること。

## 1. データベースファイルの場所を確認する

既定では、ユーザーデータディレクトリ配下に `ShioRIS3.db` が生成されます。開発環境で位置が不明な場合は、アプリケーションの起動ログに出力されるデータベースパスを参照してください。

## 2. `sqlite3` でテーブル状態を確認する

```bash
sqlite3 /path/to/ShioRIS3.db
```

```sql
.tables
.schema CyberKnifeBeamData
```

テーブル定義が表示されれば準備完了です。

## 3. 手動でデータを挿入する

DM（Output Factor）・OCR（Off-Center Ratio）・TMR のいずれのデータも、下記のように 1 行ずつ `INSERT` することができます。`data_type` を `DM` / `OCR` / `TMR` のいずれかに切り替え、用途に応じて `collimator_size`・`depth`・`radius`・`factor_value` を設定してください。

```sql
INSERT INTO CyberKnifeBeamData (
    data_type,
    collimator_size,
    depth,
    radius,
    factor_value,
    file_source
) VALUES ('DM', 5.0, 15.0, NULL, 0.987, 'DMTable.dat');
```

- `collimator_size`・`depth`・`radius` は該当しない場合 `NULL` のままで構いません。
- `factor_value` に各種係数（出力係数・軸外比・TMR 値）を設定します。

## 4. ファイルからの一括投入（推奨）

大量の行を登録する場合は、以下の手順でタブ区切りファイル（TSV）をそのまま取り込むことができます。

1. 解析済みデータを以下の形式で保存します。

   ```text
   data_type\tcollimator_size\tdepth\tradius\tfactor_value\tfile_source
   DM\t5.0\t15.0\t\t0.987\tDMTable.dat
   OCR\t7.5\t10.0\t2.0\t0.954\tOCRtable7.dat
   TMR\t60.0\t50.0\t\t0.876\tTMRtable.dat
   ```

2. `.mode tabs` を使って TSV を読み込みます。

   ```sql
   .mode tabs
   .import /path/to/cyberknife_beam_data.tsv CyberKnifeBeamData
   ```

   `NULL` を挿入したい列は空欄のままにしておくと、自動的に `NULL` として登録されます。

## 5. 登録結果の確認

```sql
SELECT data_type, collimator_size, depth, radius, factor_value
FROM CyberKnifeBeamData
ORDER BY data_type, collimator_size, depth, radius;
```

投入した行が表示されれば登録完了です。

---

## 6. アプリケーションからの自動読み込み

### 6.1 メニューからの読み込み

ShioRIS3 のメイン画面から **File → Load CyberKnife Beam Data...** を選択すると、DMTable/OCRtable/TMRtable が格納されたフォルダを指定してビームデータを読み込めます。

1. メニューで **File → Load CyberKnife Beam Data...** を選択します。
2. 計測ファイルが置かれているディレクトリを選択します。
3. 読み込みに成功すると、ステータスバーに解決したフォルダパスが表示され、同じ場所が設定に保存されます。

フォルダ内に必須ファイル（`DMTable.dat`、`TMRtable.dat`、`OCRtable*.dat`）が揃っていない場合は警告ダイアログが表示されます。その場合はファイル構成を確認してから再度実行してください。

### 6.2 C++ コードからの初期化

`src/cyberknife/beam_data_manager.h` で提供される `CyberKnife::BeamDataManager` を利用すると、計測ファイル群（`DMTable.dat`、`OCRtable*.dat`、`TMRtable.dat`）を指定ディレクトリから自動的に解析・読み込みできます。線量計算エンジンでは `CyberKnife::CyberKnifeDoseCalculator::initialize()` がこのマネージャーを内部的に生成して利用します。

`initialize()` に空文字列を渡す（あるいは引数を省略する）と、内部で `BeamDataLocator` が次の優先順位でビームデータフォルダを探索します。

1. 呼び出し側で明示したパス
2. 環境変数 `CYBERKNIFE_BEAM_DATA_PATH`
3. アプリケーション設定 `ShioRIS3/cyberknife/beamDataPath`（`QSettings`）
4. 実行ファイルと同階層にある `beam_data/`
5. `../share/shioris3/beam_data/`（Linux パッケージを想定）

いずれかのディレクトリに `DMTable.dat`・`TMRtable.dat` と 1 個以上の `OCRtable*.dat` が揃っていれば自動的に読み込みが行われ、探索結果は設定ファイルに保存されます。探索に失敗した場合は `false` が返り、詳細はログ（`cyberknife.beamdata.locator`）で確認できます。

```cpp
using namespace CyberKnife;

BeamDataManager manager;
if (!manager.loadBeamData("/path/to/beam_data")) {
    qWarning() << manager.getValidationErrors();
    return;
}

const double of = manager.getOutputFactor(7.5);          // 出力係数
const double ocr = manager.getOCRRatio(10.0, 15.0, 2.0);  // 軸外比
const double tmr = manager.getTMRValue(60.0, 50.0);       // TMR 値
```

- `loadBeamData()` は `.dat` ファイルを順に解析し、内部メモリへ展開します。読み込みに失敗した場合は `false` を返し、`getValidationErrors()` に失敗理由が格納されます。
- 解析済みのデータに対しては補間を含む取得 API（`getOutputFactor` / `getOCRRatio` / `getTMRValue`）をそのまま呼び出せます。
- アプリケーションコードからは `CyberKnifeDoseCalculator` を通じて `initialize()` を呼び出すだけで、必要なビームデータが自動的に読み込まれます。

```cpp
CyberKnifeDoseCalculator calculator;
if (!calculator.initialize()) {
    qWarning() << "Beam data auto-discovery failed.";
    return;
}

// 初回成功時には探索結果が QSettings("ShioRIS3", "ShioRIS3") に保存され、
// 次回以降は設定から同じフォルダが自動的に再利用されます。
```

引き続き、データベースへの登録が必要な場合はセクション 3〜5 の手順を使用してください。
