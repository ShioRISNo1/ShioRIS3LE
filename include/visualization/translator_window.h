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

/**
 * @brief TranslatorWindow provides real-time translation functionality
 *
 * This window translates text to Japanese using LLM Studio.
 */
class TranslatorWindow : public QWidget {
    Q_OBJECT

public:
    explicit TranslatorWindow(QWidget *parent = nullptr);
    ~TranslatorWindow() override;

    // Set external clients (if shared from MainWindow or DicomViewer)
    void setLmStudioClient(LmStudioClient *client);

public slots:
    void onTranslationComplete(const QString &text);
    void onTranslationChunk(const QString &chunk);
    void onLmStudioError(const QString &errorMsg);

signals:
    void translationRequested(const QString &sourceText,
                             const QString &sourceLang,
                             const QString &targetLang);

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void setupUI();
    void onTranslateButtonClicked();
    void onClearButtonClicked();
    void onModelChanged(const QString &model);
    void onModelsListReceived(const QStringList &models);

private:
    void createClients();
    void connectSignals();
    void translateText(const QString &sourceText);
    void appendLog(const QString &message);

    // UI Components
    QTextEdit *m_sourceTextEdit{nullptr};      // Source text display
    QTextEdit *m_targetTextEdit{nullptr};      // Japanese translation display
    QTextEdit *m_logTextEdit{nullptr};         // Log display

    QPushButton *m_translateButton{nullptr};   // Manual translation trigger
    QPushButton *m_clearButton{nullptr};       // Clear all text

    QLabel *m_sourceLabel{nullptr};
    QLabel *m_targetLabel{nullptr};
    QLabel *m_statusLabel{nullptr};
    QLabel *m_modelLabel{nullptr};

    QComboBox *m_modelComboBox{nullptr};       // LLM model selection

    // AI Clients
    LmStudioClient *m_lmStudioClient{nullptr};

    // State
    bool m_ownClients{false};  // Whether we own the client objects
    QString m_currentTranslation;
};
