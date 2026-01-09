#include "visualization/translator_window.h"
#include "ai/lmstudio_client.h"
#include "ai/whisper_client.h"
#include "ai/audio_recorder.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QSplitter>
#include <QDateTime>
#include <QDebug>
#include <QCloseEvent>
#include <QMessageBox>
#include <QCoreApplication>
#include <QRegularExpression>
#include <QTextCursor>
#include <QDir>
#include <QFile>
#include <QStandardPaths>

#ifdef Q_OS_MACOS
#include "platform/macos_audio_permissions.h"
#endif

TranslatorWindow::TranslatorWindow(QWidget *parent)
    : QWidget(parent, Qt::Window)
{
    setWindowTitle(tr("Real-time Translator"));
    setAttribute(Qt::WA_DeleteOnClose, false);

    // デバッグ音声保存ディレクトリを設定
    m_debugAudioDir = QDir::homePath() + "/ShioRIS3_debug_audio";
    QDir dir;
    if (!dir.exists(m_debugAudioDir)) {
        if (dir.mkpath(m_debugAudioDir)) {
            qDebug() << "Created debug audio directory:" << m_debugAudioDir;
        } else {
            qWarning() << "Failed to create debug audio directory:" << m_debugAudioDir;
            m_saveDebugAudio = false;
        }
    }

    setupUI();
    createClients();  // Creates clients and connects signals internally

    resize(900, 700);
}

namespace {

QString normalizeWord(const QString &word)
{
    QString normalized = word.toLower();
    normalized.remove(QRegularExpression("^[\\W_]+|[\\W_]+$"));
    return normalized;
}

QStringList splitIntoWords(const QString &text)
{
    return text.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
}

}  // namespace

TranslatorWindow::~TranslatorWindow()
{
    // Clean up clients if we own them
    if (m_ownClients) {
        delete m_lmStudioClient;
        delete m_whisperClient;
        delete m_audioRecorder;
    }
}

void TranslatorWindow::setupUI()
{
    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(10, 10, 10, 10);
    mainLayout->setSpacing(10);

    // Status label
    m_statusLabel = new QLabel(tr("Status: Ready"), this);
    m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #2c3e50; padding: 5px; }");
    mainLayout->addWidget(m_statusLabel);

    // Model selection
    auto *modelLayout = new QHBoxLayout();
    m_modelLabel = new QLabel(tr("LLM Model:"), this);
    m_modelLabel->setStyleSheet("QLabel { font-weight: bold; }");
    modelLayout->addWidget(m_modelLabel);

    m_modelComboBox = new QComboBox(this);
    m_modelComboBox->setEditable(false);
    m_modelComboBox->addItem(tr("mistralai/magistral-small-2509"));  // Default
    m_modelComboBox->setMinimumWidth(300);
    m_modelComboBox->setStyleSheet(
        "QComboBox { "
        "  padding: 5px; "
        "  border: 2px solid #3498db; "
        "  border-radius: 3px; "
        "  background-color: #1e1e1e; "
        "  color: #ffffff; "
        "}"
        "QComboBox:hover { "
        "  border: 2px solid #2980b9; "
        "  background-color: #2a2a2a; "
        "}"
        "QComboBox::drop-down { "
        "  border: none; "
        "  background-color: #3498db; "
        "}"
        "QComboBox::down-arrow { "
        "  image: none; "
        "  border-left: 5px solid transparent; "
        "  border-right: 5px solid transparent; "
        "  border-top: 5px solid white; "
        "  width: 0; "
        "  height: 0; "
        "}"
        "QComboBox QAbstractItemView { "
        "  background-color: #1e1e1e; "
        "  color: #ffffff; "
        "  selection-background-color: #3498db; "
        "  selection-color: #ffffff; "
        "  border: 2px solid #3498db; "
        "}"
    );
    modelLayout->addWidget(m_modelComboBox);
    modelLayout->addStretch();

    mainLayout->addLayout(modelLayout);

    // Create splitter for resizable areas
    auto *splitter = new QSplitter(Qt::Vertical, this);

    // Source text group (English)
    auto *sourceGroup = new QGroupBox(tr("English (Source)"), this);
    auto *sourceLayout = new QVBoxLayout(sourceGroup);

    m_sourceLabel = new QLabel(tr("Recognized speech will appear here"), this);
    m_sourceLabel->setStyleSheet("QLabel { color: #95a5a6; font-style: italic; }");
    sourceLayout->addWidget(m_sourceLabel);

    m_sourceTextEdit = new QTextEdit(this);
    m_sourceTextEdit->setPlaceholderText(tr("English speech will be transcribed here..."));
    m_sourceTextEdit->setReadOnly(true);
    m_sourceTextEdit->setMinimumHeight(100);
    m_sourceTextEdit->setStyleSheet(
        "QTextEdit { "
        "  background-color: #1e1e1e; "
        "  color: #ffffff; "
        "  border: 2px solid #3498db; "
        "  border-radius: 5px; "
        "  padding: 10px; "
        "  font-size: 14px; "
        "}"
    );
    sourceLayout->addWidget(m_sourceTextEdit);

    splitter->addWidget(sourceGroup);

    // Target text group (Japanese)
    auto *targetGroup = new QGroupBox(tr("Japanese (Translation)"), this);
    auto *targetLayout = new QVBoxLayout(targetGroup);

    m_targetLabel = new QLabel(tr("Translation will appear here"), this);
    m_targetLabel->setStyleSheet("QLabel { color: #95a5a6; font-style: italic; }");
    targetLayout->addWidget(m_targetLabel);

    m_targetTextEdit = new QTextEdit(this);
    m_targetTextEdit->setPlaceholderText(tr("Japanese translation will appear here..."));
    m_targetTextEdit->setReadOnly(true);
    m_targetTextEdit->setMinimumHeight(100);
    m_targetTextEdit->setStyleSheet(
        "QTextEdit { "
        "  background-color: #1e1e1e; "
        "  color: #00ff00; "
        "  border: 2px solid #27ae60; "
        "  border-radius: 5px; "
        "  padding: 10px; "
        "  font-size: 14px; "
        "}"
    );
    targetLayout->addWidget(m_targetTextEdit);

    splitter->addWidget(targetGroup);

    // Log group
    auto *logGroup = new QGroupBox(tr("Activity Log"), this);
    auto *logLayout = new QVBoxLayout(logGroup);

    m_logTextEdit = new QTextEdit(this);
    m_logTextEdit->setReadOnly(true);
    m_logTextEdit->setMaximumHeight(120);
    m_logTextEdit->setStyleSheet(
        "QTextEdit { "
        "  background-color: #0a0a0a; "
        "  color: #00ff00; "
        "  border: 1px solid #555555; "
        "  border-radius: 3px; "
        "  padding: 5px; "
        "  font-size: 11px; "
        "  font-family: monospace; "
        "}"
    );
    logLayout->addWidget(m_logTextEdit);

    splitter->addWidget(logGroup);

    // Set splitter sizes (proportions)
    splitter->setStretchFactor(0, 3);  // Source text
    splitter->setStretchFactor(1, 3);  // Target text
    splitter->setStretchFactor(2, 2);  // Log

    mainLayout->addWidget(splitter);

    // Control buttons
    auto *buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(10);

    m_recordButton = new QPushButton(tr("🎤 Start Recording"), this);
    m_recordButton->setEnabled(false);  // Disabled until Whisper model is loaded
    m_recordButton->setMinimumHeight(40);
    m_recordButton->setStyleSheet(
        "QPushButton { "
        "  background-color: #3498db; "
        "  color: white; "
        "  border: none; "
        "  border-radius: 5px; "
        "  padding: 10px; "
        "  font-size: 14px; "
        "  font-weight: bold; "
        "}"
        "QPushButton:hover { "
        "  background-color: #2980b9; "
        "}"
        "QPushButton:pressed { "
        "  background-color: #21618c; "
        "}"
        "QPushButton:disabled { "
        "  background-color: #95a5a6; "
        "}"
    );
    buttonLayout->addWidget(m_recordButton);

    m_translateButton = new QPushButton(tr("🔄 Translate"), this);
    m_translateButton->setEnabled(false);
    m_translateButton->setMinimumHeight(40);
    m_translateButton->setStyleSheet(
        "QPushButton { "
        "  background-color: #27ae60; "
        "  color: white; "
        "  border: none; "
        "  border-radius: 5px; "
        "  padding: 10px; "
        "  font-size: 14px; "
        "  font-weight: bold; "
        "}"
        "QPushButton:hover { "
        "  background-color: #229954; "
        "}"
        "QPushButton:pressed { "
        "  background-color: #1e8449; "
        "}"
        "QPushButton:disabled { "
        "  background-color: #95a5a6; "
        "}"
    );
    buttonLayout->addWidget(m_translateButton);

    m_clearButton = new QPushButton(tr("🗑️ Clear"), this);
    m_clearButton->setMinimumHeight(40);
    m_clearButton->setStyleSheet(
        "QPushButton { "
        "  background-color: #e74c3c; "
        "  color: white; "
        "  border: none; "
        "  border-radius: 5px; "
        "  padding: 10px; "
        "  font-size: 14px; "
        "  font-weight: bold; "
        "}"
        "QPushButton:hover { "
        "  background-color: #c0392b; "
        "}"
        "QPushButton:pressed { "
        "  background-color: #a93226; "
        "}"
    );
    buttonLayout->addWidget(m_clearButton);

    // Debug recording checkbox
    buttonLayout->addSpacing(20);
    m_debugRecordingCheckbox = new QCheckBox(tr("💾 Save Audio & Text"), this);
    m_debugRecordingCheckbox->setChecked(m_saveDebugAudio);
    m_debugRecordingCheckbox->setStyleSheet(
        "QCheckBox { "
        "  color: #ecf0f1; "
        "  font-size: 13px; "
        "  font-weight: bold; "
        "}"
        "QCheckBox::indicator { "
        "  width: 20px; "
        "  height: 20px; "
        "}"
        "QCheckBox::indicator:unchecked { "
        "  background-color: #34495e; "
        "  border: 2px solid #7f8c8d; "
        "  border-radius: 3px; "
        "}"
        "QCheckBox::indicator:checked { "
        "  background-color: #27ae60; "
        "  border: 2px solid #27ae60; "
        "  border-radius: 3px; "
        "}"
    );
    buttonLayout->addWidget(m_debugRecordingCheckbox);

    mainLayout->addLayout(buttonLayout);

    // Connect button signals
    connect(m_recordButton, &QPushButton::clicked,
            this, &TranslatorWindow::onRecordButtonClicked);
    connect(m_translateButton, &QPushButton::clicked,
            this, &TranslatorWindow::onTranslateButtonClicked);
    connect(m_clearButton, &QPushButton::clicked,
            this, &TranslatorWindow::onClearButtonClicked);
    connect(m_debugRecordingCheckbox, &QCheckBox::stateChanged,
            this, &TranslatorWindow::onDebugRecordingCheckboxChanged);

    // Connect model selection
    connect(m_modelComboBox, &QComboBox::currentTextChanged,
            this, &TranslatorWindow::onModelChanged);
}

void TranslatorWindow::createClients()
{
    // Create AI clients if not set externally
    if (!m_whisperClient) {
        m_whisperClient = new WhisperClient(this);
        m_ownClients = true;
        appendLog(tr("Initializing Whisper client..."));
        qDebug() << "TranslatorWindow: WhisperClient created";

        // Connect Whisper signals BEFORE loading model (Qt::UniqueConnection prevents duplicates)
        connect(m_whisperClient, &WhisperClient::transcriptionReady,
                this, &TranslatorWindow::onTranscriptionReady, Qt::UniqueConnection);
        connect(m_whisperClient, &WhisperClient::error,
                this, &TranslatorWindow::onWhisperError, Qt::UniqueConnection);
        connect(m_whisperClient, &WhisperClient::modelLoaded,
                this, &TranslatorWindow::onWhisperModelLoaded, Qt::UniqueConnection);

        // Set language to English for recognition
        m_whisperClient->setLanguage(WhisperClient::Language::English);
        appendLog(tr("Set Whisper language to English"));

        // Try to load any available Whisper model
        appendLog(tr("Attempting to load Whisper model..."));
        qDebug() << "TranslatorWindow: Calling loadAnyAvailableModel()";
        bool loadStarted = m_whisperClient->loadAnyAvailableModel();
        qDebug() << "TranslatorWindow: loadAnyAvailableModel() returned:" << loadStarted;

        if (!loadStarted) {
            appendLog(tr("WARNING: Whisper model load did not start - model files may be missing"));
        }
    }

    if (!m_lmStudioClient) {
        m_lmStudioClient = new LmStudioClient(this);
        m_ownClients = true;
        appendLog(tr("Initializing LM Studio client..."));

        // Connect LM Studio signals (Qt::UniqueConnection prevents duplicates)
        connect(m_lmStudioClient, &LmStudioClient::requestFinished,
                this, &TranslatorWindow::onTranslationComplete, Qt::UniqueConnection);
        connect(m_lmStudioClient, &LmStudioClient::streamChunkReceived,
                this, &TranslatorWindow::onTranslationChunk, Qt::UniqueConnection);
        connect(m_lmStudioClient, &LmStudioClient::requestFailed,
                this, &TranslatorWindow::onLmStudioError, Qt::UniqueConnection);
        connect(m_lmStudioClient, &LmStudioClient::modelsUpdated,
                this, &TranslatorWindow::onModelsListReceived, Qt::UniqueConnection);

        // Set default model
        m_lmStudioClient->setModel("mistralai/magistral-small-2509");

        // Fetch available models from LM Studio
        appendLog(tr("Fetching available LLM models..."));
        m_lmStudioClient->fetchAvailableModels();
    }

    if (!m_audioRecorder) {
        m_audioRecorder = new AudioRecorder(this);
        m_ownClients = true;
        appendLog(tr("Initializing audio recorder..."));

        // リアルタイム処理を有効化（精度向上のため長めのチャンクを使用）
        m_audioRecorder->setRealtimeProcessing(true);
        m_audioRecorder->setChunkInterval(5000);   // 5秒ごと（より多くの文脈を含む）
        m_audioRecorder->setOverlapDuration(2000); // 2秒のオーバーラップ（文の途中で切れにくくなる）
        appendLog(tr("Realtime processing enabled (5s chunks, 2s overlap)"));

        // Connect audio recorder signals
        // NOTE: Cannot use Qt::UniqueConnection with lambda expressions
        connect(m_audioRecorder, &AudioRecorder::recordingStarted,
                this, &TranslatorWindow::onRecordingStarted);
        connect(m_audioRecorder, &AudioRecorder::recordingStopped,
                this, &TranslatorWindow::onRecordingStopped);
        connect(m_audioRecorder, &AudioRecorder::audioChunkReady,
                this, &TranslatorWindow::onAudioChunkReady);
        connect(m_audioRecorder, &AudioRecorder::error,
                this, [this](const QString &errorMsg) {
            appendLog(tr("AudioRecorder ERROR: %1").arg(errorMsg));
            m_statusLabel->setText(tr("Status: Recording error"));
            m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #e74c3c; padding: 5px; }");
        });
    }
}

void TranslatorWindow::connectSignals()
{
    // This method is kept for setXXXClient() methods
    // Qt::UniqueConnection prevents duplicate connections

    // Whisper signals
    if (m_whisperClient) {
        connect(m_whisperClient, &WhisperClient::transcriptionReady,
                this, &TranslatorWindow::onTranscriptionReady, Qt::UniqueConnection);
        connect(m_whisperClient, &WhisperClient::error,
                this, &TranslatorWindow::onWhisperError, Qt::UniqueConnection);
        connect(m_whisperClient, &WhisperClient::modelLoaded,
                this, &TranslatorWindow::onWhisperModelLoaded, Qt::UniqueConnection);
    }

    // LM Studio signals
    if (m_lmStudioClient) {
        connect(m_lmStudioClient, &LmStudioClient::requestFinished,
                this, &TranslatorWindow::onTranslationComplete, Qt::UniqueConnection);
        connect(m_lmStudioClient, &LmStudioClient::streamChunkReceived,
                this, &TranslatorWindow::onTranslationChunk, Qt::UniqueConnection);
        connect(m_lmStudioClient, &LmStudioClient::requestFailed,
                this, &TranslatorWindow::onLmStudioError, Qt::UniqueConnection);
        connect(m_lmStudioClient, &LmStudioClient::modelsUpdated,
                this, &TranslatorWindow::onModelsListReceived, Qt::UniqueConnection);
    }

    // Audio recorder signals
    if (m_audioRecorder) {
        connect(m_audioRecorder, &AudioRecorder::recordingStarted,
                this, &TranslatorWindow::onRecordingStarted);
        connect(m_audioRecorder, &AudioRecorder::recordingStopped,
                this, &TranslatorWindow::onRecordingStopped);
        connect(m_audioRecorder, &AudioRecorder::audioChunkReady,
                this, &TranslatorWindow::onAudioChunkReady);
        connect(m_audioRecorder, &AudioRecorder::error,
                this, [this](const QString &errorMsg) {
            appendLog(tr("AudioRecorder ERROR: %1").arg(errorMsg));
            m_statusLabel->setText(tr("Status: Recording error"));
            m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #e74c3c; padding: 5px; }");
        });
    }
}

void TranslatorWindow::setLmStudioClient(LmStudioClient *client)
{
    if (m_lmStudioClient && m_ownClients) {
        delete m_lmStudioClient;
    }
    m_lmStudioClient = client;
    m_ownClients = false;
    connectSignals();
}

void TranslatorWindow::setWhisperClient(WhisperClient *client)
{
    if (m_whisperClient && m_ownClients) {
        delete m_whisperClient;
    }
    m_whisperClient = client;
    m_ownClients = false;
    connectSignals();
}

void TranslatorWindow::setAudioRecorder(AudioRecorder *recorder)
{
    if (m_audioRecorder && m_ownClients) {
        delete m_audioRecorder;
    }
    m_audioRecorder = recorder;
    m_ownClients = false;
    connectSignals();
}

void TranslatorWindow::onRecordButtonClicked()
{
    if (!m_audioRecorder) {
        appendLog(tr("ERROR: Audio recorder not available"));
        return;
    }

    if (m_isRecording) {
        // Stop recording
        m_audioRecorder->stopRecording();
    } else {
        // Check microphone permissions before starting
        checkMicrophonePermissions();

        // Start recording
        if (m_audioRecorder->startRecording()) {
            appendLog(tr("Recording started..."));
        } else {
            appendLog(tr("ERROR: Failed to start recording"));
            QMessageBox::warning(this, tr("Recording Error"),
                               tr("Failed to start audio recording. Please check your microphone."));
        }
    }
}

void TranslatorWindow::onTranslateButtonClicked()
{
    QString sourceText = m_sourceTextEdit->toPlainText().trimmed();

    if (sourceText.isEmpty()) {
        appendLog(tr("WARNING: No text to translate"));
        QMessageBox::information(this, tr("No Text"),
                                tr("Please record some speech first."));
        return;
    }

    translateText(sourceText);
}

void TranslatorWindow::onClearButtonClicked()
{
    m_sourceTextEdit->clear();
    m_targetTextEdit->clear();
    m_currentTranslation.clear();
    m_sourceLabel->setText(tr("Recognized speech will appear here"));
    m_targetLabel->setText(tr("Translation will appear here"));
    appendLog(tr("Cleared all text"));
}

void TranslatorWindow::onRecordingStarted()
{
    m_isRecording = true;
    m_lastTranscription.clear();  // リアルタイム処理用の状態をリセット
    m_audioChunkCounter = 0;      // チャンクカウンターをリセット
    updateRecordButtonState(true);
    m_statusLabel->setText(tr("Status: Recording..."));
    m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #e74c3c; padding: 5px; }");
}

void TranslatorWindow::onRecordingStopped(const QByteArray &audioData)
{
    m_isRecording = false;
    updateRecordButtonState(false);
    m_statusLabel->setText(tr("Status: Processing..."));
    m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #f39c12; padding: 5px; }");

    appendLog(tr("Recording stopped"));
    appendLog(tr("Audio data size: %1 bytes").arg(audioData.size()));

    // デバッグ用に音声を保存
    QString audioFile = saveAudioToFile(audioData, "recording");

    // 音声が保存された場合、現在のテキストも保存
    if (!audioFile.isEmpty()) {
        QString transcription = m_sourceTextEdit->toPlainText();
        QString translation = m_targetTextEdit->toPlainText();
        saveTranscriptionText(audioFile, transcription, translation);
    }

    if (audioData.size() <= 44) {
        QString errorMsg = tr("ERROR: No audio data recorded (only WAV header)");
        appendLog(errorMsg);
        appendLog(tr("Possible causes:"));
        appendLog(tr("  - Microphone access permission denied"));
        appendLog(tr("  - No audio input device available"));
        appendLog(tr("  - Audio device not configured correctly"));
        appendLog(tr("Check console output for detailed error messages"));
        m_statusLabel->setText(tr("Status: Recording failed - no audio data"));
        m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #e74c3c; padding: 5px; }");
        return;
    }

    if (!m_whisperClient) {
        appendLog(tr("ERROR: Whisper client not available"));
        m_statusLabel->setText(tr("Status: Error"));
        m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #e74c3c; padding: 5px; }");
        return;
    }

    appendLog(tr("Starting final transcription..."));
    m_whisperClient->transcribeFromWav(audioData);
}

void TranslatorWindow::onAudioChunkReady(const QByteArray &audioChunk)
{
    appendLog(tr("Audio chunk received: %1 bytes").arg(audioChunk.size()));

    // チャンクカウンターのみ更新（音声は保存しない）
    m_audioChunkCounter++;

    if (audioChunk.size() <= 44) {
        appendLog(tr("WARNING: Audio chunk too small, skipping"));
        return;
    }

    if (!m_whisperClient) {
        appendLog(tr("ERROR: Whisper client not available"));
        return;
    }

    m_statusLabel->setText(tr("Status: Processing chunk..."));
    m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #f39c12; padding: 5px; }");

    appendLog(tr("Transcribing audio chunk..."));
    m_whisperClient->transcribeFromWav(audioChunk);
}

void TranslatorWindow::onWhisperModelLoaded(bool success)
{
    qDebug() << "TranslatorWindow::onWhisperModelLoaded() called with success =" << success;

    if (success) {
        m_recordButton->setEnabled(true);
        appendLog(tr("✓ Whisper model loaded successfully"));
        m_statusLabel->setText(tr("Status: Ready"));
        m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #27ae60; padding: 5px; }");
    } else {
        m_recordButton->setEnabled(false);
        appendLog(tr("✗ ERROR: Failed to load Whisper model"));
        m_statusLabel->setText(tr("Status: Whisper model not available"));
        m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #e74c3c; padding: 5px; }");

        // Note: Don't show message box immediately as it's intrusive
        appendLog(tr("Check that Whisper model files are in the correct location"));
        appendLog(tr("Expected paths: ~/.cache/whisper/ or ./models/"));
    }
}

void TranslatorWindow::onTranscriptionReady(const QString &text)
{
    appendLog(tr("Transcription: %1").arg(text));

    // Check for blank audio or special markers
    QString trimmedText = text.trimmed();
    if (trimmedText.isEmpty() ||
        trimmedText == "[BLANK_AUDIO]" ||
        trimmedText.startsWith("[") && trimmedText.endsWith("]")) {
        appendLog(tr("⚠ No speech detected or blank audio"));
        m_statusLabel->setText(tr("Status: No speech detected"));
        m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #f39c12; padding: 5px; }");
        return;  // Don't translate blank audio or special markers
    }

    QString currentWords = text.trimmed();
    QStringList currentWordList = splitIntoWords(currentWords);

    // リアルタイム処理時の重複除去と補正適用
    QString newText = currentWords;
    QString sourceText = m_sourceTextEdit->toPlainText();
    if (m_isRecording && !currentWordList.isEmpty()) {
        // 直近のテキストとの重複だけでなく、既存テキスト末尾への上書きも考慮
        if (!m_lastTranscription.isEmpty() && m_lastTranscription.trimmed() == currentWords) {
            appendLog(tr("Skipping duplicate transcription (identical to previous)"));
            m_statusLabel->setText(tr("Status: Ready"));
            m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #27ae60; padding: 5px; }");
            return;
        }

        QStringList sourceWordList = splitIntoWords(sourceText);

        // 既存テキスト末尾に現在のチャンクの先頭がどこまで一致するかを探索
        int searchStart = qMax(0, sourceWordList.size() - 50);  // 過去50語程度を対象（精度向上）
        int bestMatchStart = -1;
        int bestMatchLength = 0;

        for (int start = searchStart; start < sourceWordList.size(); ++start) {
            int matchLen = 0;
            while (start + matchLen < sourceWordList.size() &&
                   matchLen < currentWordList.size() &&
                   normalizeWord(sourceWordList[start + matchLen]) ==
                       normalizeWord(currentWordList[matchLen])) {
                ++matchLen;
            }

            if (matchLen > bestMatchLength) {
                bestMatchLength = matchLen;
                bestMatchStart = start;
            }
        }

        // 2語以上一致した場合は、その位置から末尾を差し替えて誤認識を修正
        if (bestMatchLength >= 2 && bestMatchStart >= 0) {
            QStringList updatedWordList = sourceWordList.mid(0, bestMatchStart);
            updatedWordList.append(currentWordList);
            sourceText = updatedWordList.join(" ");

            QString overlapRemovedText = currentWordList.mid(bestMatchLength).join(" ");
            newText = overlapRemovedText.isEmpty() ? QString() : overlapRemovedText;

            appendLog(tr("Replaced %1 words with corrected transcription (matched %2 words)" )
                          .arg(sourceWordList.size() - bestMatchStart)
                          .arg(bestMatchLength));
        } else if (!m_lastTranscription.isEmpty()) {
            // 従来通り前チャンクとのオーバーラップを除去
            QStringList lastWordList = splitIntoWords(m_lastTranscription.trimmed());

            int maxOverlapWords = qMin(qMin(lastWordList.size() * 4 / 5, 5), currentWordList.size());
            int overlapCount = 0;

            for (int i = 1; i <= maxOverlapWords; ++i) {
                QStringList lastTail = lastWordList.mid(lastWordList.size() - i);
                QStringList currentHead = currentWordList.mid(0, i);

                if (lastTail == currentHead) {
                    overlapCount = i;
                }
            }

            if (overlapCount > 0) {
                QStringList newWordList = currentWordList.mid(overlapCount);
                newText = newWordList.join(" ");
                appendLog(tr("Removed %1 overlapping words").arg(overlapCount));
            }
        }
    }

    // 文字起こし結果を保存（次回の重複除去用）
    m_lastTranscription = text;

    // Whisperの文脈継続用に前回の結果をinitial_promptとして設定
    // これにより次のチャンクでより一貫した文字起こしが期待できる
    if (m_whisperClient && !text.trimmed().isEmpty()) {
        // 最後の50語程度を保持（長すぎると逆効果）
        QStringList words = text.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        if (words.size() > 50) {
            words = words.mid(words.size() - 50);
        }
        m_whisperClient->setInitialPrompt(words.join(" "));
    }

    // Update source text
    if (!sourceText.isEmpty() || !newText.trimmed().isEmpty()) {
        QString currentText = sourceText;
        if (!newText.trimmed().isEmpty()) {
            if (!currentText.isEmpty() && !currentText.endsWith('\n')) {
                currentText += " ";
            }
            currentText += newText.trimmed();
        }
        m_sourceTextEdit->setPlainText(currentText);

        // 自動スクロール：テキストの最後に移動
        QTextCursor cursor = m_sourceTextEdit->textCursor();
        cursor.movePosition(QTextCursor::End);
        m_sourceTextEdit->setTextCursor(cursor);
        m_sourceTextEdit->ensureCursorVisible();

        // Update label
        m_sourceLabel->setText(tr("Last recognized: %1").arg(
            QDateTime::currentDateTime().toString("hh:mm:ss")));

        // Enable translate button
        m_translateButton->setEnabled(true);

        // Automatically start translation (with new text only)
        translateText(newText.trimmed());
    }
}

void TranslatorWindow::onTranslationComplete(const QString &text)
{
    appendLog(tr("Translation complete"));

    m_statusLabel->setText(tr("Status: Ready"));
    m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #27ae60; padding: 5px; }");

    // Update target label
    m_targetLabel->setText(tr("Translated at: %1").arg(
        QDateTime::currentDateTime().toString("hh:mm:ss")));
}

void TranslatorWindow::onTranslationChunk(const QString &chunk)
{
    // Append streaming translation chunks
    m_targetTextEdit->insertPlainText(chunk);
    m_currentTranslation += chunk;

    // 自動スクロール：テキストの最後に移動
    QTextCursor cursor = m_targetTextEdit->textCursor();
    cursor.movePosition(QTextCursor::End);
    m_targetTextEdit->setTextCursor(cursor);
    m_targetTextEdit->ensureCursorVisible();
}

void TranslatorWindow::onWhisperError(const QString &errorMsg)
{
    appendLog(tr("Whisper ERROR: %1").arg(errorMsg));
    m_statusLabel->setText(tr("Status: Error"));
    m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #e74c3c; padding: 5px; }");
}

void TranslatorWindow::onLmStudioError(const QString &errorMsg)
{
    appendLog(tr("Translation ERROR: %1").arg(errorMsg));
    m_statusLabel->setText(tr("Status: Error"));
    m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #e74c3c; padding: 5px; }");

    QMessageBox::warning(this, tr("Translation Error"),
                        tr("Failed to translate text: %1").arg(errorMsg));
}

void TranslatorWindow::translateText(const QString &sourceText)
{
    if (!m_lmStudioClient) {
        appendLog(tr("ERROR: LM Studio client not available"));
        m_statusLabel->setText(tr("Status: Error - No LM Studio client"));
        m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #e74c3c; padding: 5px; }");
        return;
    }

    QString trimmedText = sourceText.trimmed();
    if (trimmedText.isEmpty()) {
        appendLog(tr("WARNING: Empty text, skipping translation"));
        return;
    }

    // Skip translation for special markers or blank audio
    if (trimmedText == "[BLANK_AUDIO]" ||
        (trimmedText.startsWith("[") && trimmedText.endsWith("]"))) {
        appendLog(tr("WARNING: Skipping translation for special marker: %1").arg(trimmedText));
        return;
    }

    appendLog(tr("Starting translation..."));
    m_statusLabel->setText(tr("Status: Translating..."));
    m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #3498db; padding: 5px; }");

    // リアルタイム処理中は翻訳結果をクリアせず、追加していく
    if (!m_isRecording) {
        // 録音停止後または手動翻訳時のみクリア
        m_currentTranslation.clear();
        m_targetTextEdit->clear();
    } else {
        // リアルタイム処理中は、前の翻訳との間にスペースを追加
        if (!m_targetTextEdit->toPlainText().isEmpty()) {
            m_targetTextEdit->insertPlainText(" ");
        }
    }

    // Prepare enhanced system prompt for medical/technical translation
    QString systemPrompt = tr(
        "You are a professional medical/radiology English-to-Japanese translator. "
        "This is real-time speech-to-text translation, so the input may contain recognition errors.\n\n"

        "STRICT RULES:\n"
        "1. Output ONLY Japanese translation - NO explanations, notes, or English text\n"
        "2. Translate medical/technical terms accurately using standard Japanese terminology\n"
        "3. For proper nouns (names, places), use katakana: John → ジョン\n"
        "4. Maintain professional but natural tone - avoid overly formal keigo\n"
        "5. If input has speech recognition errors, infer the correct meaning and translate that\n"
        "6. For unclear abbreviations, use the most common medical interpretation\n"
        "7. Preserve numbers, measurements, and units accurately\n"
        "8. Keep technical acronyms recognizable: CT, MRI, PET, etc.\n\n"

        "COMMON MEDICAL TERMS:\n"
        "- CT scan → CTスキャン\n"
        "- MRI → MRI検査\n"
        "- radiation therapy → 放射線治療\n"
        "- radiotherapy → 放射線療法\n"
        "- dose → 線量\n"
        "- tumor → 腫瘍\n"
        "- patient → 患者\n"
        "- treatment plan → 治療計画\n"
        "- beam → ビーム\n"
        "- target → 標的\n"
        "- imaging → 画像撮影\n"
        "- diagnosis → 診断\n\n"

        "EXAMPLE:\n"
        "Input: 'The patient needs a CT scan to evaluate the tumor size before radiation therapy.'\n"
        "Output: '患者は放射線治療の前に腫瘍のサイズを評価するためにCTスキャンが必要です。'\n\n"

        "Now translate the following:"
    );

    QString userPrompt = sourceText;

    // Use streaming for real-time translation display
    // Temperature 0.05 for highly deterministic, consistent translations
    m_lmStudioClient->sendChatCompletionStream(systemPrompt, userPrompt, 0.05);
}

void TranslatorWindow::appendLog(const QString &message)
{
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");
    m_logTextEdit->append(QString("[%1] %2").arg(timestamp, message));
}

void TranslatorWindow::updateRecordButtonState(bool isRecording)
{
    if (isRecording) {
        m_recordButton->setText(tr("⏹️ Stop Recording"));
        m_recordButton->setStyleSheet(
            "QPushButton { "
            "  background-color: #e74c3c; "
            "  color: white; "
            "  border: none; "
            "  border-radius: 5px; "
            "  padding: 10px; "
            "  font-size: 14px; "
            "  font-weight: bold; "
            "}"
            "QPushButton:hover { "
            "  background-color: #c0392b; "
            "}"
            "QPushButton:pressed { "
            "  background-color: #a93226; "
            "}"
        );
    } else {
        m_recordButton->setText(tr("🎤 Start Recording"));
        m_recordButton->setStyleSheet(
            "QPushButton { "
            "  background-color: #3498db; "
            "  color: white; "
            "  border: none; "
            "  border-radius: 5px; "
            "  padding: 10px; "
            "  font-size: 14px; "
            "  font-weight: bold; "
            "}"
            "QPushButton:hover { "
            "  background-color: #2980b9; "
            "}"
            "QPushButton:pressed { "
            "  background-color: #21618c; "
            "}"
        );
    }
}

void TranslatorWindow::checkMicrophonePermissions()
{
    qDebug() << "TranslatorWindow::checkMicrophonePermissions() called";

#ifdef Q_OS_MACOS
    qDebug() << "Q_OS_MACOS is defined - checking macOS permissions";
    appendLog(tr("Checking microphone permissions on macOS..."));

    // Check microphone authorization status
    using MacOSAudioPermissions::PermissionStatus;
    PermissionStatus status = MacOSAudioPermissions::checkMicrophonePermission();
    qDebug() << "Permission status:" << static_cast<int>(status);

    switch (status) {
        case PermissionStatus::Authorized:
            appendLog(tr("✓ Microphone access: GRANTED"));
            qDebug() << "Microphone permission: GRANTED";
            break;

        case PermissionStatus::Denied:
            appendLog(tr("✗ Microphone access: DENIED"));
            appendLog(tr("  Please enable microphone access in System Settings > Privacy & Security > Microphone"));
            qWarning() << "CRITICAL: Microphone permission DENIED by system";
            QMessageBox::critical(this, tr("Microphone Access Denied"),
                tr("Microphone access is DENIED.\n\n"
                   "Recording will NOT work until you grant permission.\n\n"
                   "To enable:\n"
                   "1. Open System Settings (or System Preferences)\n"
                   "2. Go to Privacy & Security > Microphone\n"
                   "3. Enable access for this application\n"
                   "4. RESTART the application\n\n"
                   "The application name may appear as 'ShioRIS3' or the executable name."));
            break;

        case PermissionStatus::Restricted:
            appendLog(tr("✗ Microphone access: RESTRICTED (parental controls or MDM)"));
            qWarning() << "Microphone permission: RESTRICTED";
            QMessageBox::warning(this, tr("Microphone Access Restricted"),
                tr("Microphone access is restricted by system policies."));
            break;

        case PermissionStatus::NotDetermined:
            appendLog(tr("⚠ Microphone access: NOT DETERMINED (requesting permission...)"));
            qDebug() << "Microphone permission: NOT DETERMINED - requesting now";

            // Request permission
            MacOSAudioPermissions::requestMicrophonePermission([this](bool granted) {
                if (granted) {
                    appendLog(tr("✓ Microphone permission GRANTED by user"));
                    qDebug() << "User GRANTED microphone permission";
                    QMessageBox::information(this, tr("Permission Granted"),
                        tr("Microphone permission granted!\n\nPlease try recording again."));
                } else {
                    appendLog(tr("✗ Microphone permission DENIED by user"));
                    qWarning() << "User DENIED microphone permission";
                    QMessageBox::warning(this, tr("Microphone Access Required"),
                        tr("Microphone access is required for speech recognition.\n"
                           "Please try recording again and grant permission when prompted."));
                }
            });
            break;
    }
#else
    qDebug() << "Q_OS_MACOS NOT defined - not on macOS or build issue";
    appendLog(tr("Platform: Not macOS, skipping permission check"));
    appendLog(tr("⚠ If you are on macOS and seeing this, there may be a build configuration issue"));
    qDebug() << "checkMicrophonePermissions: Not on macOS, permissions check not needed";
#endif
}

void TranslatorWindow::closeEvent(QCloseEvent *event)
{
    // Stop recording if active
    if (m_isRecording && m_audioRecorder) {
        m_audioRecorder->stopRecording();
    }

    event->accept();
}

void TranslatorWindow::onModelChanged(const QString &model)
{
    if (!m_lmStudioClient) {
        return;
    }

    appendLog(tr("Changing LLM model to: %1").arg(model));
    m_lmStudioClient->setModel(model);
}

void TranslatorWindow::onModelsListReceived(const QStringList &models)
{
    if (!m_modelComboBox) {
        return;
    }

    if (models.isEmpty()) {
        appendLog(tr("No models available from LM Studio"));
        return;
    }

    appendLog(tr("Received %1 models from LM Studio").arg(models.size()));

    // Save current selection
    QString currentModel = m_modelComboBox->currentText();

    // Clear and repopulate
    m_modelComboBox->clear();
    m_modelComboBox->addItems(models);

    // Restore selection if it exists in the new list
    int index = m_modelComboBox->findText(currentModel);
    if (index >= 0) {
        m_modelComboBox->setCurrentIndex(index);
    } else if (!models.isEmpty()) {
        // Select first model if current selection not found
        m_modelComboBox->setCurrentIndex(0);
    }
}

QString TranslatorWindow::saveAudioToFile(const QByteArray &audioData, const QString &prefix)
{
    if (!m_saveDebugAudio || audioData.isEmpty()) {
        return QString();
    }

    // タイムスタンプ付きのファイル名を生成
    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss_zzz");
    QString filename = QString("%1/%2_%3.wav").arg(m_debugAudioDir, prefix, timestamp);

    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly)) {
        qWarning() << "Failed to open file for writing:" << filename;
        appendLog(tr("⚠ Failed to save debug audio: %1").arg(filename));
        return QString();
    }

    qint64 bytesWritten = file.write(audioData);
    file.close();

    if (bytesWritten != audioData.size()) {
        qWarning() << "Failed to write all audio data:" << bytesWritten << "of" << audioData.size();
        appendLog(tr("⚠ Failed to write complete audio data"));
        return QString();
    }

    qDebug() << "Saved debug audio:" << filename << "(" << audioData.size() << "bytes)";
    appendLog(tr("✓ Saved debug audio: %1 (%2 bytes)").arg(filename).arg(audioData.size()));

    return filename;
}

void TranslatorWindow::saveTranscriptionText(const QString &audioFilename, const QString &transcription, const QString &translation)
{
    if (!m_saveDebugAudio || audioFilename.isEmpty()) {
        return;
    }

    // 音声ファイル名から拡張子を除いて、.txtを追加
    QString textFilename = audioFilename;
    textFilename.replace(".wav", ".txt");

    QFile file(textFilename);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Failed to open text file for writing:" << textFilename;
        appendLog(tr("⚠ Failed to save transcription text: %1").arg(textFilename));
        return;
    }

    QTextStream out(&file);
    // Qt6では自動的にUTF-8エンコーディングが使用される

    out << "========================================\n";
    out << "Audio Transcription and Translation\n";
    out << "========================================\n";
    out << "Time: " << QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss") << "\n";
    out << "Audio File: " << QFileInfo(audioFilename).fileName() << "\n";
    out << "========================================\n\n";

    out << "Original (English):\n";
    out << "----------------------------------------\n";
    out << transcription << "\n\n";

    out << "Translation (Japanese):\n";
    out << "----------------------------------------\n";
    out << translation << "\n";

    file.close();

    qDebug() << "Saved transcription text:" << textFilename;
    appendLog(tr("✓ Saved transcription text: %1").arg(textFilename));
}

void TranslatorWindow::onDebugRecordingCheckboxChanged(int state)
{
    m_saveDebugAudio = (state == Qt::Checked);
    qDebug() << "Debug recording" << (m_saveDebugAudio ? "enabled" : "disabled");
    appendLog(m_saveDebugAudio
        ? tr("✓ Audio and text recording enabled")
        : tr("Audio and text recording disabled"));
}
