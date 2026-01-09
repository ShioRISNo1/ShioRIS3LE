#pragma once

#include <QWidget>
#include <QPushButton>
#include <QTextEdit>
#include <QComboBox>
#include <QCheckBox>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>

// Forward declarations
class LmStudioClient;
class WhisperClient;
class AudioRecorder;

/**
 * @brief TranslatorWindow provides real-time translation functionality
 *
 * This window captures English audio using Whisper speech recognition
 * and translates it to Japanese using LLM Studio.
 */
class TranslatorWindow : public QWidget {
    Q_OBJECT

public:
    explicit TranslatorWindow(QWidget *parent = nullptr);
    ~TranslatorWindow() override;

    // Set external clients (if shared from MainWindow or DicomViewer)
    void setLmStudioClient(LmStudioClient *client);
    void setWhisperClient(WhisperClient *client);
    void setAudioRecorder(AudioRecorder *recorder);

public slots:
    void onTranscriptionReady(const QString &text);
    void onTranslationComplete(const QString &text);
    void onTranslationChunk(const QString &chunk);
    void onWhisperError(const QString &errorMsg);
    void onLmStudioError(const QString &errorMsg);

signals:
    void translationRequested(const QString &sourceText,
                             const QString &sourceLang,
                             const QString &targetLang);

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void setupUI();
    void onRecordButtonClicked();
    void onTranslateButtonClicked();
    void onClearButtonClicked();
    void onRecordingStarted();
    void onRecordingStopped(const QByteArray &audioData);
    void onAudioChunkReady(const QByteArray &audioChunk);
    void onWhisperModelLoaded(bool success);
    void onModelChanged(const QString &model);
    void onModelsListReceived(const QStringList &models);
    void onDebugRecordingCheckboxChanged(int state);

private:
    void createClients();
    void connectSignals();
    void translateText(const QString &sourceText);
    void appendLog(const QString &message);
    void updateRecordButtonState(bool isRecording);
    void checkMicrophonePermissions();

    // Debug functionality
    QString saveAudioToFile(const QByteArray &audioData, const QString &prefix);
    void saveTranscriptionText(const QString &audioFilename, const QString &transcription, const QString &translation);

    // UI Components
    QTextEdit *m_sourceTextEdit{nullptr};      // English audio text display
    QTextEdit *m_targetTextEdit{nullptr};      // Japanese translation display
    QTextEdit *m_logTextEdit{nullptr};         // Log display

    QPushButton *m_recordButton{nullptr};      // Start/Stop recording
    QPushButton *m_translateButton{nullptr};   // Manual translation trigger
    QPushButton *m_clearButton{nullptr};       // Clear all text
    QCheckBox *m_debugRecordingCheckbox{nullptr};  // Debug recording checkbox

    QLabel *m_sourceLabel{nullptr};
    QLabel *m_targetLabel{nullptr};
    QLabel *m_statusLabel{nullptr};
    QLabel *m_modelLabel{nullptr};

    QComboBox *m_modelComboBox{nullptr};       // LLM model selection

    // AI Clients
    LmStudioClient *m_lmStudioClient{nullptr};
    WhisperClient *m_whisperClient{nullptr};
    AudioRecorder *m_audioRecorder{nullptr};

    // State
    bool m_isRecording{false};
    bool m_ownClients{false};  // Whether we own the client objects
    QString m_currentTranslation;

    // Realtime processing state
    QString m_lastTranscription;  // 前回の文字起こし結果（重複除去用）

    // Debug settings
    bool m_saveDebugAudio{false};  // 録音された音声をデバッグ用に保存するかどうか
    QString m_debugAudioDir;       // デバッグ音声ファイルの保存先ディレクトリ
    int m_audioChunkCounter{0};    // チャンクのカウンター
    QString m_lastSavedAudioFile;  // 最後に保存した音声ファイル名
    QString m_currentChunkTranscription;  // 現在のチャンクの文字起こし
    QString m_currentChunkTranslation;    // 現在のチャンクの翻訳
};
