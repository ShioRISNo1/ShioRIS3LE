# ShioRIS3 Vision Pro Integration

## 概要

ShioRIS3のVision Pro統合により、Apple Vision Proで放射線治療計画データを没入型3D空間で可視化できます。WebXR技術を使用し、ブラウザベースで動作するため、専用アプリのインストールは不要です。

## 機能

### 実装済み機能（Phase 1 & 2）
- ✅ WebベースHTTPS/HTTPサーバー（Qt Network + OpenSSL使用）
- ✅ SSL/TLS暗号化対応（自己署名証明書）
- ✅ REST APIエンドポイント（患者データ、ボリューム、線量、輪郭）
- ✅ Three.js 3Dレンダリング
- ✅ WebXR VRモード対応（HTTPS必須）
- ✅ Vision Pro Safari対応
- ✅ インタラクティブな3D操作（回転、ズーム、パン）
- ✅ **実際のDICOMボリュームデータの3D表示**（スライススタッキング）
- ✅ **RT Structure輪郭の3D表示**（カラー付き3D線画）
- ✅ **輝度ベースの自動透明化**（CT値が低い部分を自動的に透明化、背景・エアー部分が見えなくなる）

### 今後実装予定（Phase 3+）
- 🔲 線量等線量面の3D表示
- 🔲 Window/Levelコントロール（UI調整）
- 🔲 プログレッシブローディング（低解像度→高解像度）
- 🔲 ハンドトラッキングによる高度な操作
- 🔲 音声コマンド統合
- 🔲 SharePlayによる協調作業

---

## セットアップ手順

### 1. ShioRIS3のビルドと実行

#### Linuxの場合
```bash
cd /home/user/ShioRIS3
mkdir -p build && cd build
cmake ..
make -j$(nproc)
./ShioRIS3
```

#### macOSの場合
```bash
cd /path/to/ShioRIS3
mkdir -p build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
open ShioRIS3.app
```

#### Windowsの場合
```powershell
cd C:\path\to\ShioRIS3
mkdir build
cd build
cmake ..
cmake --build . --config Release
.\Release\ShioRIS3.exe
```

### 2. SSL証明書の生成（初回のみ必須）

Vision ProでWebXRを使用するには、HTTPS接続が必要です。初回セットアップ時に、SSL証明書を生成してください。

```bash
cd /home/user/ShioRIS3
./scripts/generate_ssl_cert.sh
```

このスクリプトは自己署名SSL証明書を生成します。証明書は1年間有効です。

### 3. Webサーバーの起動確認

ShioRIS3を起動すると、自動的にHTTPSサーバーがポート8443で起動します。

コンソールに以下のようなメッセージが表示されます：
```
SSL/TLS enabled
Certificate loaded successfully
Web server started on port 8443 (HTTPS)
Access the web interface at: https://localhost:8443
```

**注意**: 証明書が見つからない場合、HTTPモードで起動します（ポート8080）。

### 4. MacのIPアドレスを確認

Vision ProからアクセスするためMacのローカルIPアドレスを確認します。

#### macOS/Linuxの場合
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

または
```bash
ip addr show | grep "inet " | grep -v 127.0.0.1
```

例：`192.168.1.100`

#### Windowsの場合
```powershell
ipconfig
```

### 5. ファイアウォールの設定

ポート8443（HTTPS）への接続を許可します。HTTPを使用する場合はポート8080も許可してください。

#### macOSの場合
システム設定 → ネットワーク → ファイアウォール → オプション → ShioRIS3を許可

#### Windowsの場合
コントロールパネル → システムとセキュリティ → Windows Defender ファイアウォール → 詳細設定 → 受信の規則 → 新しい規則 → ポート8080を許可

---

## Vision Proでの使用方法

### 1. Safariでアクセス

1. Vision ProでSafariブラウザを開く
2. アドレスバーに入力：`https://192.168.1.100:8443`
   - `192.168.1.100`の部分は、実際のMacのIPアドレスに置き換えてください
3. **証明書の警告が表示されます**（初回のみ）
4. 証明書警告の対処方法：
   - 「**Show Details**」をタップ
   - 「**visit this website**」をタップ
   - デバイスのパスコードを入力して証明書を信頼
5. ShioRIS3 Webインターフェースが表示されます

**注意**: HTTPSは必須です。Vision ProのWebXR APIはセキュアコンテキスト（HTTPS）でのみ動作します。

### 2. 3Dビューの操作

#### 通常モード（2Dウィンドウ）
- **ドラッグ**: 3Dモデルを回転
- **ピンチ**: ズームイン/アウト
- **2本指ドラッグ**: パン（移動）

### 3. 空間モード（Vision Proパススルー）に入る

1. Webページ右側の「VR Mode」パネルにある**Enter Spatial Mode (Passthrough)**ボタンをタップ（通常、「Enter VR Mode」ボタンのすぐ下に常時表示されます）
2. ボタンがグレーアウトしている場合は、HTTPS接続とSafariのWebXR設定（設定 → Safari → 詳細 → Feature Flags → WebXR Device API）を確認してください。要件を満たすと自動的にアクティブになります
3. Vision Proでは没入型AR（パススルー）セッションが自動選択され、現実空間の中に3Dビューが浮かび上がります
4. もしパススルーが開始できない場合は自動的にVRモードへフォールバックします。必要に応じて上段の「Enter VR Mode」ボタンで従来の没入型VRも利用できます

#### 空間モードでの操作
- **視線移動**: 見たい方向を見る
- **ハンドジェスチャー**:
  - ピンチで選択
  - ドラッグで回転
  - 2本指でズーム
- **モードを終了**: 「Exit Spatial Mode」または「Exit VR Mode」ボタンをタップ、またはVision Proのクラウンボタンを押す

### XRヘルスインジケーターの読み方

VRパネルの下部に`Secure / Spatial API / Blend`の3行と警告メッセージをまとめたミニダッシュボードを追加しました。各行の意味は以下のとおりです。

- **Secure**: `HTTPS / Secure`と表示されていればパススルー要件の「セキュアコンテキスト」を満たしています。`HTTP / Blocked`となっている場合はポート8443のHTTPSで再アクセスしてください（HTTPではVision Proがカメラ透過レイヤーを開きません）。
- **Spatial API**: `immersive-ar ready`であればSafariが`immersive-ar`セッションを受理しています。`Vision Pro detected - enable WebXR Device API in Safari settings`などの警告が出ている場合は、Vision Proの設定→Safari→詳細→Feature FlagsでWebXR Device APIを有効化し、HTTPSで再読み込みしてください。
- **Blend**: WebXRセッション開始後にVision Proが返す`environmentBlendMode`を表示します。`alpha-blend`以外（`opaque`など）が表示された場合はパススルー層が拒否されています。自動的に警告が表示されるので、HTTPS・Feature Flags・証明書信頼状況を再確認してください。

警告行には直近のエラー理由が表示されます。`Spatial Mode requires HTTPS (Secure Context)`のまま変わらない場合は証明書の信頼やサーバー設定に問題があります。`Vision ProがenvironmentBlendMode="opaque"を返しました`と表示された場合はVision Pro側がパススルーを拒否しているため、Feature Flagsやブラウザ再起動、キャッシュ削除を行ってください。

---

## トラブルシューティング

### Webページに接続できない

**症状**: Vision ProのSafariで「接続できません」と表示される

**解決策**:
1. MacとVision Proが同じWi-Fiネットワークに接続されているか確認
2. MacのIPアドレスが正しいか確認
3. ShioRIS3が起動しているか確認（コンソールに「Web server started」と表示されているか）
4. ファイアウォールがポート8443をブロックしていないか確認
5. 別のデバイス（iPhone、iPad）でアクセスできるかテスト：`https://MacのIP:8443`
6. SSL証明書が正しく生成されているか確認：`ls -l ssl_certs/`

### SSL証明書エラー

**症状**: 「証明書が信頼できません」と表示される

**解決策**:
1. Vision Proで証明書を手動で信頼する（上記の手順4を参照）
2. 証明書が期限切れの場合、再生成：`./scripts/generate_ssl_cert.sh`
3. 証明書にローカルIPが含まれているか確認

### VRボタンが無効（グレーアウト）

**症状**: 「Enter VR Mode」ボタンがクリックできない、または「VR Not Supported」と表示される

**解決策**:
1. **HTTPSで接続していることを確認**（http://ではなくhttps://を使用）
2. Vision ProのvisionOSバージョンを確認（2.0以降が必要）
3. SafariでWebXRが有効か確認：
   - 設定 → Safari → 詳細 → Feature Flags
   - 「WebXR Device API」を有効化
4. ページをリロード
5. デベロッパーコンソールでエラーを確認（Safari設定 → 詳細 → Webインスペクタ）

### 3Dモデルが表示されない

**症状**: 黒い画面または空の画面が表示される

**解決策**:
1. ブラウザのコンソールでエラーを確認
2. Three.jsライブラリが正しくロードされているか確認
3. WebGLがサポートされているか確認（通常Vision Proは対応）
4. ページをリロード

### パフォーマンスが悪い

**症状**: 3Dモデルがカクカクする、フレームレートが低い

**解決策**:
1. Wi-Fi接続が安定しているか確認（5GHz帯推奨）
2. MacとVision Proの距離を近づける
3. 他のアプリを閉じてVision Proのリソースを確保
4. デモモードで確認（実データよりも軽量）

---

## 開発者向け情報

### APIエンドポイント

ShioRIS3 Webサーバーは以下のREST APIを提供します：

#### 患者リスト取得
```
GET /api/patients
```

レスポンス例：
```json
{
  "status": "success",
  "patients": [
    {
      "id": "patient001",
      "name": "Test Patient 1",
      "studyDate": "2025-01-01"
    }
  ],
  "count": 1
}
```

#### ボリュームデータ取得
```
GET /api/volume/{patientId}
```

レスポンス例：
```json
{
  "status": "success",
  "patientId": "patient001",
  "volume": {
    "width": 512,
    "height": 512,
    "depth": 120,
    "spacingX": 0.5,
    "spacingY": 0.5,
    "spacingZ": 2.5
  }
}
```

#### 輪郭データ取得
```
GET /api/structures/{patientId}
```

#### 線量データ取得
```
GET /api/dose/{patientId}
```

### ファイル構成

```
ShioRIS3/
├── include/
│   └── web/
│       └── web_server.h          # Webサーバーヘッダー
├── src/
│   └── web/
│       └── web_server.cpp        # Webサーバー実装
└── web_client/                   # Webクライアント（Vision Pro側）
    ├── index.html                # メインHTML
    ├── css/
    │   └── style.css             # スタイルシート
    └── js/
        ├── main.js               # メインアプリケーション
        ├── api_client.js         # API通信
        ├── volume_renderer.js    # 3Dレンダリング
        └── vr_controller.js      # VRモード制御
```

### カスタマイズ

#### ポート番号の変更

`src/visualization/dicom_viewer.cpp`の1644行付近：
```cpp
bool useSSL = true;
quint16 port = useSSL ? 8443 : 8080;  // ポート番号を変更
m_webServer->start(port, useSSL);
```

#### HTTP/HTTPSの切り替え

HTTPSを無効にしてHTTPのみを使用する場合（非推奨、WebXR動作不可）：
```cpp
bool useSSL = false;  // HTTPSを無効化
quint16 port = 8080;
m_webServer->start(port, useSSL);
```

#### APIエンドポイントの追加

`src/web/web_server.cpp`の`handleApiRequest()`メソッドに新しいエンドポイントを追加できます。

---

## 既知の制限事項

1. **WebXR APIの制限**:
   - 高度なアイトラッキングは利用不可
   - SharePlay（複数人協調）は利用不可（ネイティブアプリのみ）

2. **パフォーマンス**:
   - 大容量のDICOMデータは転送に時間がかかる
   - ネットワーク遅延がある

3. **ブラウザ制限**:
   - SafariのWebXR実装に依存
   - 一部の機能はvisionOS 2.0以降が必要

---

## 今後の開発ロードマップ

### Phase 1: 基本機能強化（優先度: 高）
- [ ] 実際のDICOMボリュームデータのボリュームレンダリング実装
- [ ] RT Structure輪郭の3Dメッシュ生成と表示
- [ ] 線量等線量面の3D表示（カラーマップ付き）
- [ ] データ転送の最適化（圧縮、キャッシング）

### Phase 2: インタラクション強化（優先度: 中）
- [ ] ハンドトラッキングによるROI描画
- [ ] リアルタイム線量計算の統合
- [ ] 音声コマンド統合（既存のWhisper統合を活用）

### Phase 3: 高度な機能（優先度: 中）
- [ ] DVH/DPSD表示の空間UI
- [ ] Brachytherapy線源の空間配置
- [ ] AI自動セグメンテーション結果の3D表示
- [ ] スナップショット・動画キャプチャ機能

### Phase 4: ネイティブアプリ化（優先度: 低）
- [ ] SwiftUI + RealityKitによるネイティブvisionOSアプリ
- [ ] SharePlayによる協調作業機能
- [ ] オフライン動作対応
- [ ] 最適化されたパフォーマンス

---

## サポート

### 質問・バグ報告
GitHub Issues: https://github.com/ShioRISNo1/ShioRIS3/issues

### 貢献
プルリクエストを歓迎します！Vision Pro統合の改善にご協力ください。

---

## ライセンス

ShioRIS3のライセンスに準じます。

---

**最終更新**: 2025年11月13日
**対応visionOS**: 2.0以降
**対応ShioRIS3バージョン**: 1.0.0以降
