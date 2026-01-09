# Whisper音声認識モード実装ガイド

このドキュメントは、DicomViewerにWhisper音声認識機能を統合するための実装コードを提供します。

## 実装が必要な箇所

### 1. インクルードの追加 (dicom_viewer.cpp 冒頭)

```cpp
#include "ai/whisper_client.h"
#include "ai/audio_recorder.h"
#include <QProgressBar>
```

### 2. コンストラクタでの初期化 (DicomViewer::DicomViewer)

LmStudioClientの初期化の後に追加:

```cpp
  // Whisper音声認識クライアントの初期化
  m_whisperClient = new WhisperClient(this);
  connect(m_whisperClient, &WhisperClient::transcriptionReady,
          this, &DicomViewer::onWhisperTranscriptionReady);
  connect(m_whisperClient, &WhisperClient::error,
          this, &DicomViewer::onWhisperError);
  connect(m_whisperClient, &WhisperClient::modelLoaded,
          this, &DicomViewer::onWhisperModelLoaded);

  // オーディオレコーダーの初期化
  m_audioRecorder = new AudioRecorder(this);
  connect(m_audioRecorder, &AudioRecorder::recordingStarted,
          this, &DicomViewer::onAudioRecordingStarted);
  connect(m_audioRecorder, &AudioRecorder::recordingStopped,
          this, &DicomViewer::onAudioRecordingStopped);
  connect(m_audioRecorder, &AudioRecorder::audioLevelChanged,
          this, &DicomViewer::onAudioLevelChanged);
  connect(m_audioRecorder, &AudioRecorder::error,
          this, &DicomViewer::onWhisperError);
```

### 3. AI Control PanelのUI初期化 (行2490付近、promptLabelの前)

```cpp
  // 音声入力セクション
  QGroupBox *voiceInputGroup = new QGroupBox(tr("音声入力 (Whisper)"), m_aiControlPanel);
  QVBoxLayout *voiceLayout = new QVBoxLayout(voiceInputGroup);

  // 録音ボタンと状態表示
  QHBoxLayout *voiceControlLayout = new QHBoxLayout();
  m_aiVoiceRecordButton = new QPushButton(tr("🎤 録音"), m_aiControlPanel);
  m_aiVoiceRecordButton->setCheckable(false);
  m_aiVoiceStatusLabel = new QLabel(tr("準備完了"), m_aiControlPanel);
  voiceControlLayout->addWidget(m_aiVoiceRecordButton);
  voiceControlLayout->addWidget(m_aiVoiceStatusLabel, 1);
  voiceLayout->addLayout(voiceControlLayout);

  // プログレスバー
  m_aiVoiceProgressBar = new QProgressBar(m_aiControlPanel);
  m_aiVoiceProgressBar->setRange(0, 100);
  m_aiVoiceProgressBar->setValue(0);
  m_aiVoiceProgressBar->setTextVisible(false);
  m_aiVoiceProgressBar->setMaximumHeight(6);
  voiceLayout->addWidget(m_aiVoiceProgressBar);

  // 設定行
  QHBoxLayout *voiceSettingsLayout = new QHBoxLayout();

  // モデル選択
  QLabel *modelLabel = new QLabel(tr("モデル:"), m_aiControlPanel);
  m_aiWhisperModelCombo = new QComboBox(m_aiControlPanel);
  m_aiWhisperModelCombo->addItem("tiny (75MB)");
  m_aiWhisperModelCombo->addItem("base (142MB)");
  m_aiWhisperModelCombo->addItem("small (466MB)");
  m_aiWhisperModelCombo->setCurrentIndex(1); // デフォルトはbase

  // 言語選択
  QLabel *langLabel = new QLabel(tr("言語:"), m_aiControlPanel);
  m_aiLanguageCombo = new QComboBox(m_aiControlPanel);
  m_aiLanguageCombo->addItem(tr("自動検出"), "auto");
  m_aiLanguageCombo->addItem(tr("日本語"), "ja");
  m_aiLanguageCombo->addItem(tr("English"), "en");
  m_aiLanguageCombo->setCurrentIndex(1); // デフォルトは日本語

  voiceSettingsLayout->addWidget(modelLabel);
  voiceSettingsLayout->addWidget(m_aiWhisperModelCombo, 1);
  voiceSettingsLayout->addWidget(langLabel);
  voiceSettingsLayout->addWidget(m_aiLanguageCombo, 1);
  voiceLayout->addLayout(voiceSettingsLayout);

  // 自動送信チェックボックス
  m_aiVoiceAutoSendCheck = new QCheckBox(tr("文字起こし後に自動送信"), m_aiControlPanel);
  m_aiVoiceAutoSendCheck->setChecked(false);
  voiceLayout->addWidget(m_aiVoiceAutoSendCheck);

  // 文字起こしプレビュー
  QLabel *previewLabel = new QLabel(tr("文字起こしプレビュー:"), m_aiControlPanel);
  m_aiTranscriptionPreview = new QLabel(m_aiControlPanel);
  m_aiTranscriptionPreview->setWordWrap(true);
  m_aiTranscriptionPreview->setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; border-radius: 3px; }");
  m_aiTranscriptionPreview->setMinimumHeight(40);
  m_aiTranscriptionPreview->setText(tr("(文字起こし結果がここに表示されます)"));
  voiceLayout->addWidget(previewLabel);
  voiceLayout->addWidget(m_aiTranscriptionPreview);

  aiControlLayout->addWidget(voiceInputGroup);

  // 録音ボタンのシグナル接続
  connect(m_aiVoiceRecordButton, &QPushButton::clicked, this,
          &DicomViewer::onAiVoiceRecordButtonClicked);
  connect(m_aiWhisperModelCombo, &QComboBox::currentIndexChanged, this,
          [this]() { saveAiSettings(); });
  connect(m_aiLanguageCombo, &QComboBox::currentIndexChanged, this,
          [this]() { saveAiSettings(); });
  connect(m_aiVoiceAutoSendCheck, &QCheckBox::toggled, this,
          [this]() { saveAiSettings(); });

  // (既存のpromptLabel以降のコードが続く)
  QLabel *promptLabel = new QLabel(tr("ユーザープロンプト"), m_aiControlPanel);
  // ...
```

### 4. スロット実装 (dicom_viewer.cppの末尾またはAI関連スロットのセクション)

```cpp
void DicomViewer::onAiVoiceRecordButtonClicked() {
  if (!m_isRecording) {
    // 録音開始
    if (m_audioRecorder->startRecording()) {
      m_isRecording = true;
      m_aiVoiceRecordButton->setText(tr("⏹ 停止"));
      m_aiVoiceRecordButton->setStyleSheet("QPushButton { background-color: #ff4444; color: white; }");
    } else {
      QMessageBox::warning(this, tr("録音エラー"),
                          tr("録音を開始できませんでした。マイクが接続されているか確認してください。"));
    }
  } else {
    // 録音停止
    m_audioRecorder->stopRecording();
    // onAudioRecordingStoppedで処理が続く
  }
}

void DicomViewer::onAudioRecordingStarted() {
  m_aiVoiceStatusLabel->setText(tr("録音中..."));
  m_aiVoiceProgressBar->setRange(0, 0); // インジケーターモード
  appendAiLog(tr("🎤 録音を開始しました"));
}

void DicomViewer::onAudioRecordingStopped(const QByteArray &audioData) {
  m_isRecording = false;
  m_aiVoiceRecordButton->setText(tr("🎤 録音"));
  m_aiVoiceRecordButton->setStyleSheet("");
  m_aiVoiceProgressBar->setRange(0, 100);
  m_aiVoiceProgressBar->setValue(0);

  if (audioData.isEmpty()) {
    m_aiVoiceStatusLabel->setText(tr("録音データなし"));
    appendAiLog(tr("⚠ 録音データが空です"));
    return;
  }

  m_aiVoiceStatusLabel->setText(tr("文字起こし中..."));
  appendAiLog(tr("🔄 文字起こしを開始しています..."));

  // Whisperモデルが未ロードの場合はロード
  if (!m_whisperClient->isModelLoaded()) {
    WhisperClient::ModelSize modelSize = WhisperClient::ModelSize::Base;
    int modelIndex = m_aiWhisperModelCombo->currentIndex();

    switch (modelIndex) {
      case 0: modelSize = WhisperClient::ModelSize::Tiny; break;
      case 1: modelSize = WhisperClient::ModelSize::Base; break;
      case 2: modelSize = WhisperClient::ModelSize::Small; break;
    }

    appendAiLog(tr("📦 Whisperモデル (%1) をロード中...").arg(m_aiWhisperModelCombo->currentText()));

    bool loaded = m_whisperClient->loadDefaultModel(modelSize);
    if (!loaded) {
      QString modelPath = WhisperClient::getDefaultModelPath(modelSize);
      QMessageBox::warning(this, tr("モデル読み込みエラー"),
                          tr("Whisperモデルが見つかりません。\n\n"
                             "以下のパスにモデルをダウンロードしてください:\n%1\n\n"
                             "ダウンロード元:\nhttps://huggingface.co/ggerganov/whisper.cpp/tree/main")
                          .arg(modelPath));
      m_aiVoiceStatusLabel->setText(tr("モデルエラー"));
      return;
    }
  }

  // 言語設定
  QString langCode = m_aiLanguageCombo->currentData().toString();
  if (langCode == "ja") {
    m_whisperClient->setLanguage(WhisperClient::Language::Japanese);
  } else if (langCode == "en") {
    m_whisperClient->setLanguage(WhisperClient::Language::English);
  } else {
    m_whisperClient->setLanguage(WhisperClient::Language::Auto);
  }

  // 非同期で文字起こし実行
  QtConcurrent::run([this, audioData]() {
    QString transcription = m_whisperClient->transcribeFromWav(audioData);
    // transcriptionReadyシグナルは既に発行されている
  });
}

void DicomViewer::onAudioLevelChanged(float level) {
  // 音声レベルをプログレスバーに表示
  m_aiVoiceProgressBar->setValue(static_cast<int>(level * 100));
}

void DicomViewer::onWhisperTranscriptionReady(const QString &text) {
  m_aiVoiceStatusLabel->setText(tr("完了"));
  m_aiVoiceProgressBar->setValue(0);

  if (text.isEmpty()) {
    m_aiTranscriptionPreview->setText(tr("(音声が検出されませんでした)"));
    appendAiLog(tr("⚠ 文字起こし結果が空です"));
    return;
  }

  // プレビューに表示
  m_aiTranscriptionPreview->setText(text);

  // プロンプト入力欄に挿入
  m_aiPromptEdit->setPlainText(text);

  appendAiLog(tr("✅ 文字起こし完了: %1").arg(text));

  // 自動送信が有効なら送信
  if (m_aiVoiceAutoSendCheck->isChecked()) {
    appendAiLog(tr("🚀 自動送信を実行します"));
    QTimer::singleShot(500, this, &DicomViewer::onAiSendPrompt);
  }
}

void DicomViewer::onWhisperError(const QString &errorMsg) {
  m_aiVoiceStatusLabel->setText(tr("エラー"));
  m_aiVoiceProgressBar->setValue(0);
  appendAiLog(tr("❌ Whisperエラー: %1").arg(errorMsg));
  QMessageBox::critical(this, tr("音声認識エラー"), errorMsg);
}

void DicomViewer::onWhisperModelLoaded(bool success) {
  if (success) {
    appendAiLog(tr("✅ Whisperモデルのロードに成功しました"));
  } else {
    appendAiLog(tr("❌ Whisperモデルのロードに失敗しました"));
  }
}
```

### 5. 設定の保存/読み込み (既存のloadAiSettings/saveAiSettingsに追加)

**saveAiSettings()に追加:**

```cpp
  // Whisper設定
  settings.setValue("ai/whisperModel", m_aiWhisperModelCombo->currentIndex());
  settings.setValue("ai/whisperLanguage", m_aiLanguageCombo->currentIndex());
  settings.setValue("ai/voiceAutoSend", m_aiVoiceAutoSendCheck->isChecked());
```

**loadAiSettings()に追加:**

```cpp
  // Whisper設定
  if (settings.contains("ai/whisperModel")) {
    m_aiWhisperModelCombo->setCurrentIndex(settings.value("ai/whisperModel").toInt());
  }
  if (settings.contains("ai/whisperLanguage")) {
    m_aiLanguageCombo->setCurrentIndex(settings.value("ai/whisperLanguage").toInt());
  }
  if (settings.contains("ai/voiceAutoSend")) {
    m_aiVoiceAutoSendCheck->setChecked(settings.value("ai/voiceAutoSend").toBool());
  }
```

## モデルのダウンロード

Whisperモデルは以下からダウンロード可能です:

- **ダウンロード元**: https://huggingface.co/ggerganov/whisper.cpp/tree/main
- **保存先** (プラットフォーム依存):
  - **Linux**: `~/.local/share/ShioRIS3/whisper/models/`
  - **macOS**: `~/Library/Application Support/ShioRIS3/whisper/models/`
  - **Windows**: `C:\Users\<username>\AppData\Local\ShioRIS3\whisper\models\`

**推奨モデル:**
- 開発・テスト: `ggml-tiny.bin` (75MB)
- 実用: `ggml-base.bin` (142MB) または `ggml-small.bin` (466MB)

## 使用方法

1. ShioRIS3を起動
2. AI Control Panelを開く
3. 音声入力セクションでモデルと言語を選択
4. 「🎤 録音」ボタンをクリックして録音開始
5. 話し終わったら「⏹ 停止」ボタンをクリック
6. 文字起こし結果がプロンプト入力欄に自動挿入される
7. 必要に応じて編集し、「送信」ボタンでAIに送信

## トラブルシューティング

### マイクが認識されない
- システムのマイク権限を確認
- 別のマイクデバイスを試す

### モデルが見つからない
- エラーメッセージに表示されたパスにモデルファイルをダウンロード
- ファイル名が正確に一致しているか確認

### 文字起こしが空になる
- 録音時間が短すぎないか確認
- マイクの音量が適切か確認
- 別のモデルを試す (base → small)

## まとめ

この実装により、ShioRIS3のAI操作モードにWhisper音声認識機能が統合されます。
ユーザーは音声でコマンドを入力し、AIがそれを解釈してShioRIS3の操作を自動化できます。
