# LmStudioClient - 使用ガイド

## 概要

`LmStudioClient` は、LM Studio サーバーとの通信を管理する強化されたクライアントクラスです。以下の高度な機能を提供します：

- ✅ **タイムアウト管理** - リクエストのタイムアウト設定
- ✅ **自動リトライ** - ネットワークエラー時の自動再試行
- ✅ **ストリーミングレスポンス** - リアルタイムのチャット応答
- ✅ **接続監視** - サーバー接続状態の確認
- ✅ **リクエストキャンセル** - 実行中のリクエストの中止

## 基本的な使用方法

### 1. インスタンスの作成と設定

```cpp
#include "ai/lmstudio_client.h"

// クライアントの作成
LmStudioClient *client = new LmStudioClient(this);

// エンドポイントの設定
client->setEndpoint(QUrl("http://localhost:1234/v1/chat/completions"));

// モデルの設定
client->setModel("mistralai/magistral-small-2509");

// オプション: APIキーの設定（必要な場合）
client->setApiKey("your-api-key");

// タイムアウトの設定（デフォルト: 30秒）
client->setTimeout(60000);  // 60秒

// リトライ回数の設定（デフォルト: 3回）
client->setMaxRetries(5);

// リトライ遅延の設定（デフォルト: 1秒）
client->setRetryDelay(2000);  // 2秒
```

### 2. シグナルの接続

```cpp
// リクエスト開始
connect(client, &LmStudioClient::requestStarted, this, [=]() {
    qDebug() << "リクエストを開始しました";
});

// リクエスト完了
connect(client, &LmStudioClient::requestFinished, this, [=](const QString &response) {
    qDebug() << "応答:" << response;
});

// リクエスト失敗
connect(client, &LmStudioClient::requestFailed, this, [=](const QString &error) {
    qWarning() << "エラー:" << error;
});

// リトライ中
connect(client, &LmStudioClient::requestRetrying, this, [=](int attempt, int maxAttempts) {
    qDebug() << QString("リトライ中 %1/%2").arg(attempt).arg(maxAttempts);
});
```

### 3. チャット補完リクエストの送信

```cpp
QString systemPrompt = "あなたは親切なアシスタントです。";
QString userPrompt = "日本の首都はどこですか？";
double temperature = 0.7;

client->sendChatCompletion(systemPrompt, userPrompt, temperature);
```

## 高度な機能

### ストリーミングレスポンス

リアルタイムでレスポンスを受信する場合：

```cpp
// ストリーミングシグナルの接続
connect(client, &LmStudioClient::streamChunkReceived, this, [=](const QString &chunk) {
    qDebug() << "チャンク受信:" << chunk;
    // UIに逐次表示
    textEdit->insertPlainText(chunk);
});

connect(client, &LmStudioClient::streamFinished, this, [=]() {
    qDebug() << "ストリーミング完了";
});

connect(client, &LmStudioClient::streamFailed, this, [=](const QString &error) {
    qWarning() << "ストリーミングエラー:" << error;
});

// ストリーミングリクエストの送信
client->sendChatCompletionStream(systemPrompt, userPrompt, temperature);
```

### 接続確認

サーバーとの接続を確認する場合：

```cpp
connect(client, &LmStudioClient::connectionCheckStarted, this, [=]() {
    qDebug() << "接続確認を開始...";
});

connect(client, &LmStudioClient::connectionCheckSucceeded, this, [=]() {
    qDebug() << "接続成功！";
});

connect(client, &LmStudioClient::connectionCheckFailed, this, [=](const QString &error) {
    qWarning() << "接続失敗:" << error;
});

// 接続確認の実行
client->checkConnection();
```

### リクエストのキャンセル

実行中のリクエストをキャンセルする場合：

```cpp
connect(client, &LmStudioClient::requestCancelled, this, [=]() {
    qDebug() << "リクエストがキャンセルされました";
});

// リクエストのキャンセル
client->cancelCurrentRequest();
```

### 利用可能なモデルの取得

```cpp
connect(client, &LmStudioClient::modelsUpdated, this, [=](const QStringList &models) {
    qDebug() << "利用可能なモデル:" << models;
    for (const QString &model : models) {
        comboBox->addItem(model);
    }
});

connect(client, &LmStudioClient::modelsFetchFailed, this, [=](const QString &error) {
    qWarning() << "モデル取得エラー:" << error;
});

// モデル一覧の取得
client->fetchAvailableModels();
```

## 完全な使用例

```cpp
#include "ai/lmstudio_client.h"
#include <QApplication>
#include <QTextEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>

class ChatWindow : public QWidget {
    Q_OBJECT

public:
    ChatWindow(QWidget *parent = nullptr) : QWidget(parent) {
        // UI設定
        textEdit = new QTextEdit(this);
        sendButton = new QPushButton("送信", this);

        QVBoxLayout *layout = new QVBoxLayout(this);
        layout->addWidget(textEdit);
        layout->addWidget(sendButton);

        // LmStudioClient設定
        client = new LmStudioClient(this);
        client->setEndpoint(QUrl("http://localhost:1234/v1/chat/completions"));
        client->setModel("mistralai/magistral-small-2509");
        client->setTimeout(60000);  // 60秒
        client->setMaxRetries(3);

        // シグナル接続
        connect(sendButton, &QPushButton::clicked, this, &ChatWindow::sendMessage);

        connect(client, &LmStudioClient::requestStarted, this, [=]() {
            sendButton->setEnabled(false);
            textEdit->append("--- リクエスト送信中... ---");
        });

        connect(client, &LmStudioClient::requestFinished, this, [=](const QString &response) {
            sendButton->setEnabled(true);
            textEdit->append("AI: " + response);
        });

        connect(client, &LmStudioClient::requestFailed, this, [=](const QString &error) {
            sendButton->setEnabled(true);
            textEdit->append("エラー: " + error);
        });

        connect(client, &LmStudioClient::requestRetrying, this, [=](int attempt, int max) {
            textEdit->append(QString("--- リトライ中 %1/%2 ---").arg(attempt).arg(max));
        });
    }

private slots:
    void sendMessage() {
        QString userInput = textEdit->toPlainText().split('\n').last();
        QString systemPrompt = "あなたは親切なアシスタントです。";

        client->sendChatCompletion(systemPrompt, userInput, 0.7);
    }

private:
    LmStudioClient *client;
    QTextEdit *textEdit;
    QPushButton *sendButton;
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    ChatWindow window;
    window.resize(600, 400);
    window.show();

    return app.exec();
}

#include "main.moc"
```

## エラーハンドリング

### リトライ可能なエラー

以下のエラーは自動的にリトライされます：

- `ConnectionRefusedError` - 接続拒否
- `RemoteHostClosedError` - リモートホスト切断
- `HostNotFoundError` - ホストが見つからない
- `TimeoutError` - タイムアウト
- `TemporaryNetworkFailureError` - 一時的なネットワーク障害
- `NetworkSessionFailedError` - ネットワークセッション失敗
- `UnknownNetworkError` - 不明なネットワークエラー
- `UnknownServerError` - 不明なサーバーエラー
- `ServiceUnavailableError` - サービス利用不可

### エクスポネンシャルバックオフ

リトライ時の遅延は指数関数的に増加します：

- 1回目のリトライ: `baseDelay` (デフォルト 1秒)
- 2回目のリトライ: `baseDelay * 2` (2秒)
- 3回目のリトライ: `baseDelay * 4` (4秒)
- 最大遅延: 30秒

## パフォーマンス設定

### 推奨設定

**短いリクエスト（質問応答など）:**
```cpp
client->setTimeout(30000);   // 30秒
client->setMaxRetries(3);
client->setRetryDelay(1000); // 1秒
```

**長いリクエスト（コード生成など）:**
```cpp
client->setTimeout(120000);  // 2分
client->setMaxRetries(2);
client->setRetryDelay(2000); // 2秒
```

**ストリーミング:**
```cpp
client->setTimeout(300000);  // 5分（長時間のストリーム用）
client->setMaxRetries(1);    // ストリームはリトライ少なめ
client->setRetryDelay(500);  // 500ms
```

## トラブルシューティング

### 接続できない場合

1. LM Studio が起動していることを確認
2. 正しいポート番号を確認（デフォルト: 1234）
3. ファイアウォール設定を確認

```cpp
// 接続テスト
client->checkConnection();
```

### タイムアウトが頻発する場合

```cpp
// タイムアウト時間を延長
client->setTimeout(120000);  // 2分に延長

// またはストリーミングを使用
client->sendChatCompletionStream(systemPrompt, userPrompt);
```

### リトライが多すぎる場合

```cpp
// リトライ回数を制限
client->setMaxRetries(1);

// または完全に無効化
client->setMaxRetries(0);
```

## API リファレンス

### 設定メソッド

| メソッド | 説明 | デフォルト値 |
|---------|------|-------------|
| `setEndpoint(QUrl)` | APIエンドポイントURL | `http://localhost:1234/v1/chat/completions` |
| `setModel(QString)` | 使用するモデル名 | `mistralai/magistral-small-2509` |
| `setApiKey(QString)` | API認証キー | なし |
| `setTimeout(int)` | タイムアウト時間（ミリ秒） | 30000 (30秒) |
| `setMaxRetries(int)` | 最大リトライ回数 | 3 |
| `setRetryDelay(int)` | リトライ基本遅延（ミリ秒） | 1000 (1秒) |

### リクエストメソッド

| メソッド | 説明 |
|---------|------|
| `sendChatCompletion(...)` | チャット補完リクエスト送信 |
| `sendChatCompletionStream(...)` | ストリーミングリクエスト送信 |
| `fetchAvailableModels()` | 利用可能なモデル一覧取得 |
| `cancelCurrentRequest()` | 実行中のリクエストキャンセル |
| `checkConnection()` | 接続状態確認 |

### シグナル

**リクエストライフサイクル:**
- `requestStarted()` - リクエスト開始
- `requestFinished(QString)` - リクエスト完了
- `requestFailed(QString)` - リクエスト失敗
- `requestCancelled()` - リクエストキャンセル
- `requestRetrying(int, int)` - リトライ中

**ストリーミング:**
- `streamChunkReceived(QString)` - チャンク受信
- `streamFinished()` - ストリーム完了
- `streamFailed(QString)` - ストリーム失敗

**モデル管理:**
- `modelsUpdated(QStringList)` - モデル一覧更新
- `modelsFetchFailed(QString)` - モデル取得失敗

**接続状態:**
- `connectionCheckStarted()` - 接続確認開始
- `connectionCheckSucceeded()` - 接続成功
- `connectionCheckFailed(QString)` - 接続失敗

## 変更履歴

### v2.0 (2025-10)
- ✅ タイムアウト管理機能追加
- ✅ 自動リトライメカニズム実装
- ✅ ストリーミングレスポンス対応
- ✅ 接続監視機能追加
- ✅ リクエストキャンセル機能追加
- ✅ エクスポネンシャルバックオフ実装
- ✅ エラーハンドリング改善

### v1.0 (2024)
- 基本的なチャット補完機能
- モデル一覧取得機能

## ライセンス

このコードは ShioRIS3 プロジェクトの一部です。
