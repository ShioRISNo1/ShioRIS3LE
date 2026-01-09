#include "visualization/translator_window.h"
#include "ai/lmstudio_client.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QSplitter>
#include <QDateTime>
#include <QDebug>
#include <QCloseEvent>
#include <QMessageBox>
#include <QCoreApplication>
#include <QTextCursor>
#include <QDir>
#include <QFile>
#include <QStandardPaths>

TranslatorWindow::TranslatorWindow(QWidget *parent)
    : QWidget(parent, Qt::Window)
{
    setWindowTitle(tr("Real-time Translator"));
    setAttribute(Qt::WA_DeleteOnClose, false);

    setupUI();
    createClients();  // Creates clients and connects signals internally

    resize(900, 700);
}

TranslatorWindow::~TranslatorWindow()
{
    // Clean up clients if we own them
    if (m_ownClients) {
        delete m_lmStudioClient;
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

    mainLayout->addLayout(buttonLayout);

    // Connect button signals
    connect(m_translateButton, &QPushButton::clicked,
            this, &TranslatorWindow::onTranslateButtonClicked);
    connect(m_clearButton, &QPushButton::clicked,
            this, &TranslatorWindow::onClearButtonClicked);

    // Connect model selection
    connect(m_modelComboBox, &QComboBox::currentTextChanged,
            this, &TranslatorWindow::onModelChanged);
}

void TranslatorWindow::createClients()
{
    // Create AI clients if not set externally
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
}

void TranslatorWindow::connectSignals()
{
    // This method is kept for setXXXClient() methods
    // Qt::UniqueConnection prevents duplicate connections

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

void TranslatorWindow::onTranslateButtonClicked()
{
    QString sourceText = m_sourceTextEdit->toPlainText().trimmed();

    if (sourceText.isEmpty()) {
        appendLog(tr("WARNING: No text to translate"));
        QMessageBox::information(this, tr("No Text"),
                                tr("Please enter some text first."));
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

    appendLog(tr("Starting translation..."));
    m_statusLabel->setText(tr("Status: Translating..."));
    m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #3498db; padding: 5px; }");

    // Clear previous translation
    m_currentTranslation.clear();
    m_targetTextEdit->clear();

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

void TranslatorWindow::closeEvent(QCloseEvent *event)
{
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
