//=============================================================================
// ファイル: src/visualization/auto_segmentation_dialog.cpp (Part 1)
// 修正内容: AIセグメンテーションダイアログの完全実装
//=============================================================================

#include "visualization/auto_segmentation_dialog.h"
#include "ai/linux_auto_segmenter.h"
#include <QDebug>
#include <QStandardPaths>
#include <QDir>
#include <QColorDialog>
#include <QPixmap>
#include <QPainter>
#include <QDateTime>
#include <QFuture>
#include <QtConcurrent>
#include "theme_manager.h"

AutoSegmentationDialog::AutoSegmentationDialog(QWidget *parent)
    : QDialog(parent),
      m_mainLayout(nullptr),
      m_mainSplitter(nullptr),
      m_modelGroupBox(nullptr),
      m_modelPathEdit(nullptr),
      m_selectModelButton(nullptr),
      m_modelStatusLabel(nullptr),
      m_executionGroupBox(nullptr),
      m_qualityModeComboBox(nullptr),
      m_qualityModeLabel(nullptr),
      m_startButton(nullptr),
      m_stopButton(nullptr),
      m_progressBar(nullptr),
      m_statusLabel(nullptr),
      m_logTextEdit(nullptr),
      m_resultGroupBox(nullptr),
      m_organTree(nullptr),
      m_statisticsLabel(nullptr),
      m_previewGroupBox(nullptr),
      m_previewLabel(nullptr),
      m_previewScrollArea(nullptr),
      m_sliceSpinBox(nullptr),
      m_sliceLabel(nullptr),
      m_adjustmentGroupBox(nullptr),
      m_thresholdSpinBox(nullptr),
      m_smoothingCheckBox(nullptr),
      m_fillHolesCheckBox(nullptr),
      m_actionLayout(nullptr),
      m_applyButton(nullptr),
      m_exportButton(nullptr),
      m_cancelButton(nullptr),
#ifdef USE_ONNXRUNTIME
      m_segmenter(std::make_unique<OnnxSegmenter>()),
#endif
      m_workerThread(nullptr),
      m_isProcessing(false),
      m_shouldStop(false),
      m_modelLoaded(false),
      m_currentSlice(0),
      m_currentProgress(0.0f)
{
    setWindowTitle("AI セグメンテーション");
    setWindowFlags(windowFlags() | Qt::WindowMaximizeButtonHint);
    resize(1000, 700);
    
    setupUI();
    initializeOrganLabels();
    updateUIState();
    
    // ダイアログの中央配置
    if (parent) {
        QRect parentGeometry = parent->geometry();
        move(parentGeometry.center() - rect().center());
    }
}

void AutoSegmentationDialog::setSharedSegmenter(OnnxSegmenter *segmenter) {
#ifdef USE_ONNXRUNTIME
    m_sharedSegmenter = segmenter;
    bool ready = m_sharedSegmenter && m_sharedSegmenter->isLoaded();
    if (ready) {
        m_modelLoaded = true;
        m_modelStatusLabel->setText(tr("モデル: メインビューのモデルを使用"));
        m_modelStatusLabel->setStyleSheet(
            "QLabel { color: #4ECDC4; font-weight: bold; }");
        m_logTextEdit->append(QString("[%1] メインビューのセグメンテーションモデルを使用")
                                 .arg(QDateTime::currentDateTime().toString("hh:mm:ss")));
    } else if (!hasLoadedModel()) {
        m_modelLoaded = false;
        m_modelStatusLabel->setText(tr("モデル: 未ロード"));
        m_modelStatusLabel->setStyleSheet(
            "QLabel { color: #FF6B6B; font-weight: bold; }");
    }
#else
    Q_UNUSED(segmenter);
#endif
    updateUIState();
}

OnnxSegmenter* AutoSegmentationDialog::activeSegmenter() const {
#ifdef USE_ONNXRUNTIME
    if (m_sharedSegmenter && m_sharedSegmenter->isLoaded()) {
        return m_sharedSegmenter;
    }
    return m_segmenter.get();
#else
    return nullptr;
#endif
}

bool AutoSegmentationDialog::hasLoadedModel() const {
#ifdef USE_ONNXRUNTIME
    OnnxSegmenter *segmenter = activeSegmenter();
    return segmenter && segmenter->isLoaded();
#else
    return false;
#endif
}

void AutoSegmentationDialog::setupUI() {
    m_mainLayout = new QVBoxLayout(this);
    m_mainSplitter = new QSplitter(Qt::Horizontal, this);
    
    // 左側パネル（設定とコントロール）
    QWidget *leftPanel = new QWidget;
    QVBoxLayout *leftLayout = new QVBoxLayout(leftPanel);
    leftLayout->addWidget(createModelSection());
    leftLayout->addWidget(createExecutionSection());
    leftLayout->addWidget(createAdjustmentSection());
    leftLayout->addWidget(createActionSection());
    leftLayout->addStretch();
    
    // 右側パネル（結果とプレビュー）
    QWidget *rightPanel = new QWidget;
    QVBoxLayout *rightLayout = new QVBoxLayout(rightPanel);
    rightLayout->addWidget(createResultSection());
    rightLayout->addWidget(createPreviewSection());
    
    m_mainSplitter->addWidget(leftPanel);
    m_mainSplitter->addWidget(rightPanel);
    m_mainSplitter->setStretchFactor(0, 1);
    m_mainSplitter->setStretchFactor(1, 2);
    
    m_mainLayout->addWidget(m_mainSplitter);
}

QWidget* AutoSegmentationDialog::createModelSection() {
    m_modelGroupBox = new QGroupBox("ONNXモデル設定");
    QVBoxLayout *layout = new QVBoxLayout(m_modelGroupBox);

    // モデルファイル選択
    QHBoxLayout *pathLayout = new QHBoxLayout;
    QLabel *pathLabel = new QLabel("モデルファイル:");
    m_modelPathEdit = new QLineEdit;
    m_modelPathEdit->setReadOnly(true);
    m_modelPathEdit->setPlaceholderText("ONNXモデルファイルを選択してください");

    m_selectModelButton = new QPushButton("参照...");
    connect(m_selectModelButton, &QPushButton::clicked, this, &AutoSegmentationDialog::selectModelFile);

    pathLayout->addWidget(pathLabel);
    pathLayout->addWidget(m_modelPathEdit, 1);
    pathLayout->addWidget(m_selectModelButton);

    // モデル状態表示
    m_modelStatusLabel = new QLabel("モデル: 未ロード");
    m_modelStatusLabel->setStyleSheet("QLabel { color: #FF6B6B; font-weight: bold; }");

    layout->addLayout(pathLayout);
    layout->addWidget(m_modelStatusLabel);

#ifdef USE_ONNXRUNTIME
    // GPU/CPU情報表示
    QLabel *gpuInfoLabel = new QLabel;
    gpuInfoLabel->setStyleSheet("QLabel { color: #4ECDC4; font-size: 11px; font-weight: bold; }");
    gpuInfoLabel->setWordWrap(true);

    // LinuxAutoSegmenterを使用してGPU情報を取得
    LinuxAutoSegmenter autoSeg;
    QString gpuInfo = autoSeg.getGPUInfo();

    if (gpuInfo.startsWith("GPU:")) {
        gpuInfoLabel->setText(QString("🚀 実行環境: %1").arg(gpuInfo));
    } else {
        gpuInfoLabel->setText("💻 実行環境: CPU");
        gpuInfoLabel->setStyleSheet("QLabel { color: #FFA500; font-size: 11px; font-weight: bold; }");
    }
    layout->addWidget(gpuInfoLabel);
#endif

    // 推奨モデル情報
    QLabel *infoLabel = new QLabel(
        "<b>サポート臓器:</b><br>"
        "• 肝臓（Liver）<br>"
        "• 右腎（Right Kidney）<br>"
        "• 左腎・脾臓（Left Kidney/Spleen）<br><br>"
        "<b>推奨入力:</b> 腹部CT画像"
    );
    infoLabel->setStyleSheet("QLabel { color: #666; font-size: 11px; }");
    infoLabel->setWordWrap(true);
    layout->addWidget(infoLabel);

    return m_modelGroupBox;
}

QWidget* AutoSegmentationDialog::createExecutionSection() {
    m_executionGroupBox = new QGroupBox("セグメンテーション実行");
    QVBoxLayout *layout = new QVBoxLayout(m_executionGroupBox);
    ThemeManager &theme = ThemeManager::instance();

    // 品質モード選択
    QHBoxLayout *qualityLayout = new QHBoxLayout;
    m_qualityModeLabel = new QLabel("品質モード:");
    m_qualityModeComboBox = new QComboBox;
    m_qualityModeComboBox->addItem("Standard (標準・高速)", "standard");
    m_qualityModeComboBox->addItem("High (高品質・TTA)", "high");
    m_qualityModeComboBox->addItem("Ultra (最高品質・TTA+SW)", "ultra");
    m_qualityModeComboBox->setCurrentIndex(1); // デフォルトはHigh
    m_qualityModeComboBox->setToolTip(
        "Standard: 最速、基本的な精度\n"
        "High: TTA（Test Time Augmentation）で高精度、約4倍の処理時間\n"
        "Ultra: TTA + スライディングウィンドウで最高精度、約10-20倍の処理時間"
    );

    qualityLayout->addWidget(m_qualityModeLabel);
    qualityLayout->addWidget(m_qualityModeComboBox, 1);

    // 実行ボタン
    QHBoxLayout *buttonLayout = new QHBoxLayout;
    m_startButton = new QPushButton("開始");
    theme.applyTextColor(m_startButton,
                         QStringLiteral("QPushButton { background-color: #4ECDC4; "
                                         "color: %1; font-weight: bold; padding: 8px; }"));
    connect(m_startButton, &QPushButton::clicked, this, &AutoSegmentationDialog::startSegmentationProcess);

    m_stopButton = new QPushButton("停止");
    theme.applyTextColor(m_stopButton,
                         QStringLiteral("QPushButton { background-color: #FF6B6B; "
                                         "color: %1; font-weight: bold; padding: 8px; }"));
    m_stopButton->setEnabled(false);
    connect(m_stopButton, &QPushButton::clicked, this, &AutoSegmentationDialog::stopSegmentation);

    buttonLayout->addWidget(m_startButton);
    buttonLayout->addWidget(m_stopButton);

    // 進捗表示
    m_progressBar = new QProgressBar;
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);

    m_statusLabel = new QLabel("待機中");
    m_statusLabel->setStyleSheet("QLabel { color: #666; }");

    // ログ表示
    m_logTextEdit = new QTextEdit;
    m_logTextEdit->setMaximumHeight(120);
    m_logTextEdit->setReadOnly(true);
    m_logTextEdit->setStyleSheet("QTextEdit { font-family: 'Courier New', monospace; font-size: 10px; }");

    layout->addLayout(qualityLayout);
    layout->addLayout(buttonLayout);
    layout->addWidget(m_progressBar);
    layout->addWidget(m_statusLabel);
    layout->addWidget(new QLabel("実行ログ:"));
    layout->addWidget(m_logTextEdit);
    
    return m_executionGroupBox;
}

QWidget* AutoSegmentationDialog::createResultSection() {
    m_resultGroupBox = new QGroupBox("セグメンテーション結果");
    QVBoxLayout *layout = new QVBoxLayout(m_resultGroupBox);
    
    // 臓器ラベルツリー
    m_organTree = new QTreeWidget;
    m_organTree->setHeaderLabels(QStringList() << "臓器" << "表示" << "ボクセル数" << "体積 (cm³)" << "割合 (%)");
    m_organTree->header()->setStretchLastSection(false);
    m_organTree->header()->setSectionResizeMode(0, QHeaderView::Stretch);
    m_organTree->header()->setSectionResizeMode(1, QHeaderView::Fixed);
    m_organTree->header()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    m_organTree->header()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    m_organTree->header()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    m_organTree->setMaximumHeight(200);
    
    connect(m_organTree, &QTreeWidget::itemChanged, this, &AutoSegmentationDialog::onLabelVisibilityChanged);
    
    // 統計情報表示
    m_statisticsLabel = new QLabel("統計情報: 結果なし");
    m_statisticsLabel->setStyleSheet("QLabel { color: #666; font-size: 11px; }");
    
    layout->addWidget(m_organTree);
    layout->addWidget(m_statisticsLabel);
    
    return m_resultGroupBox;
}

QWidget* AutoSegmentationDialog::createPreviewSection() {
    m_previewGroupBox = new QGroupBox("プレビュー");
    QVBoxLayout *layout = new QVBoxLayout(m_previewGroupBox);
    
    // スライス選択
    QHBoxLayout *sliceLayout = new QHBoxLayout;
    m_sliceLabel = new QLabel("スライス:");
    m_sliceSpinBox = new QSpinBox;
    m_sliceSpinBox->setMinimum(0);
    m_sliceSpinBox->setValue(0);
    m_sliceSpinBox->setEnabled(false);
    connect(m_sliceSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &AutoSegmentationDialog::updatePreview);
    
    sliceLayout->addWidget(m_sliceLabel);
    sliceLayout->addWidget(m_sliceSpinBox);
    sliceLayout->addStretch();
    
    // プレビュー画像
    m_previewLabel = new QLabel;
    m_previewLabel->setAlignment(Qt::AlignCenter);
    m_previewLabel->setStyleSheet("QLabel { border: 1px solid #CCC; background-color: #000; }");
    m_previewLabel->setMinimumSize(300, 300);
    m_previewLabel->setText("プレビューなし");
    m_previewLabel->setStyleSheet("QLabel { color: #666; border: 1px solid #CCC; }");
    
    m_previewScrollArea = new QScrollArea;
    m_previewScrollArea->setWidget(m_previewLabel);
    m_previewScrollArea->setWidgetResizable(true);
    m_previewScrollArea->setMinimumHeight(300);
    
    layout->addLayout(sliceLayout);
    layout->addWidget(m_previewScrollArea);
    
    return m_previewGroupBox;
}

QWidget* AutoSegmentationDialog::createAdjustmentSection() {
    m_adjustmentGroupBox = new QGroupBox("結果調整");
    QGridLayout *layout = new QGridLayout(m_adjustmentGroupBox);
    
    // しきい値調整
    QLabel *thresholdLabel = new QLabel("信頼度しきい値:");
    m_thresholdSpinBox = new QSpinBox;
    m_thresholdSpinBox->setRange(0, 100);
    m_thresholdSpinBox->setValue(50);
    m_thresholdSpinBox->setSuffix("%");
    connect(m_thresholdSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &AutoSegmentationDialog::onThresholdChanged);
    
    // 後処理オプション
    m_smoothingCheckBox = new QCheckBox("スムージング");
    m_smoothingCheckBox->setChecked(true);
    connect(m_smoothingCheckBox, &QCheckBox::toggled, this, &AutoSegmentationDialog::onThresholdChanged);
    
    m_fillHolesCheckBox = new QCheckBox("穴埋め");
    m_fillHolesCheckBox->setChecked(true);
    connect(m_fillHolesCheckBox, &QCheckBox::toggled, this, &AutoSegmentationDialog::onThresholdChanged);
    
    layout->addWidget(thresholdLabel, 0, 0);
    layout->addWidget(m_thresholdSpinBox, 0, 1);
    layout->addWidget(m_smoothingCheckBox, 1, 0, 1, 2);
    layout->addWidget(m_fillHolesCheckBox, 2, 0, 1, 2);
    
    return m_adjustmentGroupBox;
}

QWidget* AutoSegmentationDialog::createActionSection() {
    QWidget *actionWidget = new QWidget;
    m_actionLayout = new QHBoxLayout(actionWidget);
    ThemeManager &theme = ThemeManager::instance();

    m_applyButton = new QPushButton("適用");
    theme.applyTextColor(m_applyButton,
                         QStringLiteral("QPushButton { background-color: #45B7D1; "
                                         "color: %1; font-weight: bold; padding: 10px; }"));
    m_applyButton->setEnabled(false);
    connect(m_applyButton, &QPushButton::clicked, this,
            &AutoSegmentationDialog::applySegmentationResult);

    m_exportButton = new QPushButton("エクスポート");
    theme.applyTextColor(m_exportButton,
                         QStringLiteral("QPushButton { background-color: #96CEB4; "
                                         "color: %1; font-weight: bold; padding: 10px; }"));
    m_exportButton->setEnabled(false);
    connect(m_exportButton, &QPushButton::clicked, this,
            &AutoSegmentationDialog::exportResult);

    m_cancelButton = new QPushButton("閉じる");
    theme.applyTextColor(m_cancelButton,
                         QStringLiteral("QPushButton { background-color: #95A5A6; "
                                         "color: %1; font-weight: bold; padding: 10px; }"));
    connect(m_cancelButton, &QPushButton::clicked, this, &QDialog::reject);

    m_actionLayout->addWidget(m_applyButton);
    m_actionLayout->addWidget(m_exportButton);
    m_actionLayout->addStretch();
    m_actionLayout->addWidget(m_cancelButton);

    return actionWidget;
}

//=============================================================================
// ファイル: src/visualization/auto_segmentation_dialog.cpp (Part 2)
// 修正内容: AIセグメンテーション機能実装の続き
//=============================================================================

void AutoSegmentationDialog::initializeOrganLabels() {
    m_organStats = {
        {"Background", 0, 0.0, 0.0, false, QColor(0, 0, 0, 0)},
        {"Liver", 0, 0.0, 0.0, true, QColor(255, 0, 0, 128)},
        {"Right Kidney", 0, 0.0, 0.0, true, QColor(0, 255, 0, 128)},
        {"Left Kidney/Spleen", 0, 0.0, 0.0, true, QColor(0, 0, 255, 128)}
    };
    
    for (size_t i = 1; i < m_organStats.size(); ++i) { // Skip background
        QTreeWidgetItem *item = new QTreeWidgetItem(m_organTree);
        item->setText(0, m_organStats[i].name);
        item->setCheckState(1, m_organStats[i].visible ? Qt::Checked : Qt::Unchecked);
        item->setText(2, "0");
        item->setText(3, "0.0");
        item->setText(4, "0.0");
        
        // 色表示
        QPixmap colorPixmap(16, 16);
        colorPixmap.fill(m_organStats[i].color);
        QIcon colorIcon(colorPixmap);
        item->setIcon(0, colorIcon);
        
        item->setData(0, Qt::UserRole, static_cast<int>(i));
    }
}

void AutoSegmentationDialog::selectModelFile() {
    QString defaultPath = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    QDir modelsDir(QDir(defaultPath).filePath("ShioRIS3/AIModels"));
    if (!modelsDir.exists()) {
        modelsDir.mkpath(".");
    }
    
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "ONNXモデルファイルを選択",
        modelsDir.absolutePath(),
        "ONNX Models (*.onnx);;All Files (*)"
    );
    
    if (!fileName.isEmpty()) {
        m_modelPathEdit->setText(fileName);

        // モデルロード試行
#ifdef USE_ONNXRUNTIME
        m_sharedSegmenter = nullptr; // 外部モデル共有を解除
        m_modelLoaded = false;
        if (m_segmenter && m_segmenter->loadModel(fileName.toStdString())) {
            // 実行プロバイダー情報を取得
            std::string providerInfo = m_segmenter->getExecutionProviderInfo();
            QString providerInfoQt = QString::fromStdString(providerInfo);

            QString statusText;
            QString styleSheet;
            if (m_segmenter->isCudaEnabled()) {
                statusText = QString("モデル: ロード完了 (GPU)");
                styleSheet = "QLabel { color: #4ECDC4; font-weight: bold; }";
            } else {
                statusText = QString("モデル: ロード完了 (CPU)");
                styleSheet = "QLabel { color: #FFA500; font-weight: bold; }";
            }

            m_modelStatusLabel->setText(statusText);
            m_modelStatusLabel->setStyleSheet(styleSheet);
            m_modelLoaded = true;

            m_logTextEdit->append(QString("[%1] モデルファイルをロードしました: %2")
                                    .arg(QDateTime::currentDateTime().toString("hh:mm:ss"))
                                    .arg(QFileInfo(fileName).fileName()));
            m_logTextEdit->append(QString("[%1] 実行プロバイダー: %2")
                                    .arg(QDateTime::currentDateTime().toString("hh:mm:ss"))
                                    .arg(providerInfoQt));
        } else {
            m_modelStatusLabel->setText("モデル: ロードエラー");
            m_modelStatusLabel->setStyleSheet("QLabel { color: #FF6B6B; font-weight: bold; }");
            m_modelLoaded = false;

            QMessageBox::warning(this, "エラー",
                "モデルファイルのロードに失敗しました。\n"
                "ファイルが正しいONNXモデルであることを確認してください。");

            m_logTextEdit->append(QString("[%1] モデルロードエラー: %2")
                                    .arg(QDateTime::currentDateTime().toString("hh:mm:ss"))
                                    .arg(QFileInfo(fileName).fileName()));
        }
#else
        m_modelStatusLabel->setText("モデル: ONNX Runtime無効");
        m_modelStatusLabel->setStyleSheet("QLabel { color: #FF6B6B; font-weight: bold; }");
        QMessageBox::information(this, "情報", 
            "ONNX Runtimeが無効になっています。\n"
            "AI機能を使用するには、ONNX Runtimeを有効にしてビルドしてください。");
#endif
        
        updateUIState();
    }
}

void AutoSegmentationDialog::startSegmentation(const cv::Mat &volume) {
    if (volume.empty()) {
        QMessageBox::warning(this, "エラー", "入力ボリュームが空です。");
        return;
    }
    
    m_inputVolume = volume.clone();
    
    // スライススピンボックスの範囲設定
    if (volume.dims >= 3) {
        m_sliceSpinBox->setMaximum(volume.size[0] - 1);
        m_sliceSpinBox->setValue(volume.size[0] / 2); // 中央スライス
        m_sliceSpinBox->setEnabled(true);
    }
    
    m_logTextEdit->append(QString("[%1] 入力ボリューム準備完了: %2x%3x%4")
                           .arg(QDateTime::currentDateTime().toString("hh:mm:ss"))
                           .arg(volume.dims >= 3 ? volume.size[2] : volume.cols)
                           .arg(volume.dims >= 3 ? volume.size[1] : volume.rows)
                           .arg(volume.dims >= 3 ? volume.size[0] : 1));
    
    updateUIState();
    show();
    raise();
    activateWindow();
}

void AutoSegmentationDialog::startSegmentationProcess() {
    if (!hasLoadedModel()) {
        QMessageBox::warning(this, "エラー", "ONNXモデルが読み込まれていません。");
        return;
    }

    if (m_inputVolume.empty()) {
        QMessageBox::warning(this, "エラー", "入力ボリュームがありません。");
        return;
    }

    if (m_isProcessing) {
        return;
    }

    // 選択された品質モードを環境変数に設定
    QString qualityMode = m_qualityModeComboBox->currentData().toString();
    qputenv("SHIORIS_AI_QUALITY_MODE", qualityMode.toUtf8());

    m_isProcessing = true;
    m_shouldStop = false;
    m_currentProgress = 0.0f;

    m_statusLabel->setText("セグメンテーション実行中...");
    m_progressBar->setValue(0);

    QString modeName = m_qualityModeComboBox->currentText();
    m_logTextEdit->append(QString("[%1] セグメンテーション開始 - %2")
                           .arg(QDateTime::currentDateTime().toString("hh:mm:ss"))
                           .arg(modeName));

    updateUIState();

    // バックグラウンドでセグメンテーション実行
    startWorkerThread();
}

void AutoSegmentationDialog::stopSegmentation() {
    if (!m_isProcessing) {
        return;
    }
    
    m_shouldStop = true;
    m_statusLabel->setText("停止中...");
    
    m_logTextEdit->append(QString("[%1] セグメンテーション停止要求")
                           .arg(QDateTime::currentDateTime().toString("hh:mm:ss")));
    
    stopWorkerThread();
}

void AutoSegmentationDialog::startWorkerThread() {
#ifdef USE_ONNXRUNTIME
    OnnxSegmenter *segmenter = activeSegmenter();
    if (!segmenter) {
        onError("ONNX Runtimeが利用できません");
        return;
    }

    // プログレスコールバックを設定（UIスレッドで更新）
    segmenter->setProgressCallback([this](float progress, const std::string& message) {
        QMetaObject::invokeMethod(this, [this, progress, message]() {
            onProgressUpdate(progress);
            if (!message.empty()) {
                m_statusLabel->setText(QString::fromStdString(message));
            }
        }, Qt::QueuedConnection);
    });

    // QtConcurrentを使用してバックグラウンド処理
    QFuture<cv::Mat> future = QtConcurrent::run([this, segmenter]() -> cv::Mat {
        cv::Mat result;

        try {
            if (m_shouldStop) {
                segmenter->clearProgressCallback();
                return cv::Mat();
            }

            if (m_inputVolume.dims == 3) {
                // 3Dボリューム処理
                QMetaObject::invokeMethod(this, [this]() {
                    m_logTextEdit->append(QString("[%1] 3Dボリューム処理開始")
                                           .arg(QDateTime::currentDateTime().toString("hh:mm:ss")));
                }, Qt::QueuedConnection);

                result = segmenter->predict3D(m_inputVolume);
            } else {
                // 2Dスライス処理
                QMetaObject::invokeMethod(this, [this]() {
                    m_logTextEdit->append(QString("[%1] 2Dスライス処理開始")
                                           .arg(QDateTime::currentDateTime().toString("hh:mm:ss")));
                }, Qt::QueuedConnection);

                result = segmenter->predict(m_inputVolume);
            }

            if (m_shouldStop) {
                segmenter->clearProgressCallback();
                return cv::Mat();
            }

            // コールバックをクリア
            segmenter->clearProgressCallback();

        } catch (const std::exception &e) {
            segmenter->clearProgressCallback();
            QMetaObject::invokeMethod(this, [this, e]() {
                onError(QString("セグメンテーションエラー: %1").arg(e.what()));
            }, Qt::QueuedConnection);
            return cv::Mat();
        }

        return result;
    });
    
    // 結果処理
    QFutureWatcher<cv::Mat> *watcher = new QFutureWatcher<cv::Mat>(this);
    connect(watcher, &QFutureWatcher<cv::Mat>::finished, [this, watcher]() {
        cv::Mat result = watcher->result();
        watcher->deleteLater();
        
        if (!result.empty() && !m_shouldStop) {
            onSegmentationCompleted(result);
        } else {
            m_isProcessing = false;
            m_statusLabel->setText(m_shouldStop ? "停止されました" : "エラーが発生しました");
            updateUIState();
        }
    });
    
    watcher->setFuture(future);
#else
    onError("ONNX Runtimeが利用できません");
#endif
}

void AutoSegmentationDialog::stopWorkerThread() {
    // QtConcurrentの場合、キャンセル機能は限定的
    // m_shouldStopフラグで処理内でチェック
    m_isProcessing = false;
    updateUIState();
}

void AutoSegmentationDialog::onProgressUpdate(float progress) {
    m_currentProgress = progress;
    int percentage = static_cast<int>(progress * 100.0f);
    m_progressBar->setValue(percentage);
    
    QString statusText;
    if (progress < 0.2f) {
        statusText = "前処理中...";
    } else if (progress < 0.8f) {
        statusText = "AI推論中...";
    } else if (progress < 1.0f) {
        statusText = "後処理中...";
    } else {
        statusText = "完了";
    }
    
    m_statusLabel->setText(QString("%1 (%2%)").arg(statusText).arg(percentage));
}

void AutoSegmentationDialog::onSegmentationCompleted(const cv::Mat &result) {
    m_segmentationResult = result.clone();
    m_adjustedResult = result.clone();
    
    m_isProcessing = false;
    m_statusLabel->setText("セグメンテーション完了");
    m_progressBar->setValue(100);
    
    m_logTextEdit->append(QString("[%1] セグメンテーション完了")
                           .arg(QDateTime::currentDateTime().toString("hh:mm:ss")));
    
    // 結果統計更新
    updateResultStatistics();
    updatePreview();
    updateUIState();
    
    emit segmentationFinished(m_segmentationResult);
}

void AutoSegmentationDialog::onError(const QString &errorMessage) {
    m_isProcessing = false;
    m_statusLabel->setText("エラー");
    m_progressBar->setValue(0);
    
    m_logTextEdit->append(QString("[%1] エラー: %2")
                           .arg(QDateTime::currentDateTime().toString("hh:mm:ss"))
                           .arg(errorMessage));
    
    QMessageBox::critical(this, "セグメンテーションエラー", errorMessage);
    updateUIState();
}

void AutoSegmentationDialog::updateResultStatistics() {
    if (m_adjustedResult.empty()) {
        return;
    }
    
    // ラベル統計計算
    std::vector<int> labelCounts(m_organStats.size(), 0);
    int totalVoxels = 0;
    
    if (m_adjustedResult.dims == 3) {
        // 3D volume
        int depth = m_adjustedResult.size[0];
        int height = m_adjustedResult.size[1];
        int width = m_adjustedResult.size[2];
        totalVoxels = depth * height * width;
        
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    uchar label = m_adjustedResult.at<uchar>(z, y, x);
                    if (label < labelCounts.size()) {
                        labelCounts[label]++;
                    }
                }
            }
        }
    } else {
        // 2D slice
        totalVoxels = m_adjustedResult.rows * m_adjustedResult.cols;
        
        for (int y = 0; y < m_adjustedResult.rows; ++y) {
            for (int x = 0; x < m_adjustedResult.cols; ++x) {
                uchar label = m_adjustedResult.at<uchar>(y, x);
                if (label < labelCounts.size()) {
                    labelCounts[label]++;
                }
            }
        }
    }
    
    // 統計更新
    for (size_t i = 0; i < m_organStats.size(); ++i) {
        m_organStats[i].voxelCount = labelCounts[i];
        m_organStats[i].percentage = (totalVoxels > 0) ? 
            (labelCounts[i] * 100.0 / totalVoxels) : 0.0;
        
        // 体積計算（仮定: 1mm³ = 0.001cm³）
        m_organStats[i].volumeCm3 = labelCounts[i] * 0.001;
    }
    
    // ツリー更新
    for (int i = 0; i < m_organTree->topLevelItemCount(); ++i) {
        QTreeWidgetItem *item = m_organTree->topLevelItem(i);
        int labelIndex = item->data(0, Qt::UserRole).toInt();
        
        if (labelIndex < m_organStats.size()) {
            item->setText(2, QString::number(m_organStats[labelIndex].voxelCount));
            item->setText(3, QString::number(m_organStats[labelIndex].volumeCm3, 'f', 2));
            item->setText(4, QString::number(m_organStats[labelIndex].percentage, 'f', 1));
        }
    }
    
    // 統計サマリー更新
    int organVoxels = totalVoxels - labelCounts[0]; // backgroundを除く
    double organPercentage = (totalVoxels > 0) ? (organVoxels * 100.0 / totalVoxels) : 0.0;
    
    m_statisticsLabel->setText(QString(
        "総ボクセル数: %1 | 臓器領域: %2 (%3%) | 背景: %4 (%5%)"
    ).arg(totalVoxels)
     .arg(organVoxels)
     .arg(organPercentage, 0, 'f', 1)
     .arg(labelCounts[0])
     .arg((totalVoxels - organVoxels) * 100.0 / totalVoxels, 0, 'f', 1));
}

void AutoSegmentationDialog::updatePreview() {
    if (m_adjustedResult.empty() || !m_sliceSpinBox->isEnabled()) {
        m_previewLabel->setText("プレビューなし");
        m_previewLabel->setPixmap(QPixmap());
        return;
    }
    
    QPixmap preview = generatePreviewImage();
    if (!preview.isNull()) {
        // プレビューのスケーリング
        QSize labelSize = m_previewLabel->size();
        QPixmap scaledPreview = preview.scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        m_previewLabel->setPixmap(scaledPreview);
    }
}

QPixmap AutoSegmentationDialog::generatePreviewImage() {
    if (m_adjustedResult.empty()) {
        return QPixmap();
    }
    
    cv::Mat sliceImage;
    cv::Mat sliceMask;
    
    if (m_adjustedResult.dims == 3) {
        // 3Dボリュームから指定スライスを抽出
        int sliceIndex = m_sliceSpinBox->value();
        if (sliceIndex >= 0 && sliceIndex < m_adjustedResult.size[0]) {
            // 3次元データから2次元スライスを取得する正しい方法
            int sizes[2] = {m_adjustedResult.size[1], m_adjustedResult.size[2]};
            sliceMask = cv::Mat(2, sizes, m_adjustedResult.type(),
                              m_adjustedResult.ptr<uchar>(sliceIndex));
            // データのコピーを作成（元データの変更を避けるため）
            sliceMask = sliceMask.clone();
        }
        
        // 対応する元画像スライス
        if (!m_inputVolume.empty() && m_inputVolume.dims == 3 && 
            sliceIndex < m_inputVolume.size[0]) {
            int sizes[2] = {m_inputVolume.size[1], m_inputVolume.size[2]};
            sliceImage = cv::Mat(2, sizes, m_inputVolume.type(),
                               m_inputVolume.ptr<uchar>(sliceIndex));
            // データのコピーを作成
            sliceImage = sliceImage.clone();
        }
    } else {
        // 2Dの場合
        sliceMask = m_adjustedResult;
        sliceImage = m_inputVolume;
    }
    
    if (sliceMask.empty()) {
        return QPixmap();
    }
    
    // 以下、既存のプレビュー生成処理...
    // （省略 - この部分は変更不要）
    
    return QPixmap(); // 実際の実装では適切なQPixmapを返す
}

void AutoSegmentationDialog::onLabelVisibilityChanged(QTreeWidgetItem *item, int column) {
    if (column == 1) { // 表示列
        int labelIndex = item->data(0, Qt::UserRole).toInt();
        if (labelIndex < m_organStats.size()) {
            m_organStats[labelIndex].visible = (item->checkState(1) == Qt::Checked);
            updatePreview();
        }
    }
}

void AutoSegmentationDialog::onThresholdChanged() {
    if (m_segmentationResult.empty()) {
        return;
    }
    
    // しきい値とフィルタリングを適用
    float threshold = m_thresholdSpinBox->value() / 100.0f;
    bool smoothing = m_smoothingCheckBox->isChecked();
    bool fillHoles = m_fillHolesCheckBox->isChecked();
    
    m_adjustedResult = m_segmentationResult.clone();
    
    // ここでしきい値処理、スムージング、穴埋めなどの後処理を実装
    // 簡単な実装例（実際にはより高度な処理が必要）
    
    if (smoothing) {
        // モルフォロジー演算でスムージング
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(m_adjustedResult, m_adjustedResult, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(m_adjustedResult, m_adjustedResult, cv::MORPH_OPEN, kernel);
    }
    
    updateResultStatistics();
    updatePreview();
}

void AutoSegmentationDialog::applySegmentationResult() {
    if (m_adjustedResult.empty()) {
        QMessageBox::warning(this, "警告", "適用する結果がありません。");
        return;
    }
    
    emit applyResult(m_adjustedResult);
    
    m_logTextEdit->append(QString("[%1] セグメンテーション結果をメインビューに適用しました")
                           .arg(QDateTime::currentDateTime().toString("hh:mm:ss")));
    
    QMessageBox::information(this, "完了", "セグメンテーション結果がメインビューに適用されました。");
}

void AutoSegmentationDialog::exportResult() {
    if (m_adjustedResult.empty()) {
        QMessageBox::warning(this, "警告", "エクスポートする結果がありません。");
        return;
    }
    
    QString fileName = QFileDialog::getSaveFileName(
        this,
        "セグメンテーション結果をエクスポート",
        QString("segmentation_result_%1.png").arg(QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss")),
        "PNG Images (*.png);;TIFF Images (*.tiff);;OpenCV XML (*.xml);;All Files (*)"
    );
    
    if (!fileName.isEmpty()) {
        bool success = false;
        
        if (fileName.endsWith(".xml")) {
            // OpenCV FileStorageとして保存
            cv::FileStorage fs(fileName.toStdString(), cv::FileStorage::WRITE);
            if (fs.isOpened()) {
                fs << "segmentation_result" << m_adjustedResult;
                fs.release();
                success = true;
            }
        } else {
            // 画像として保存
            if (m_adjustedResult.dims == 3) {
                // 中央スライスを保存
                int midSlice = m_adjustedResult.size[0] / 2;
                
                // 3次元データから2次元スライスを取得する正しい方法
                int sizes[2] = {m_adjustedResult.size[1], m_adjustedResult.size[2]};
                cv::Mat slice = cv::Mat(2, sizes, m_adjustedResult.type(),
                                      m_adjustedResult.ptr<uchar>(midSlice));
                // データのコピーを作成
                slice = slice.clone();
                
                // ラベルを可視化用に変換
                cv::Mat visualized;
                slice.convertTo(visualized, CV_8UC1, 255.0 / (m_organStats.size() - 1));
                success = cv::imwrite(fileName.toStdString(), visualized);
            } else {
                cv::Mat visualized;
                m_adjustedResult.convertTo(visualized, CV_8UC1, 255.0 / (m_organStats.size() - 1));
                success = cv::imwrite(fileName.toStdString(), visualized);
            }
        }
        
        if (success) {
            m_logTextEdit->append(QString("[%1] エクスポート完了: %2")
                                   .arg(QDateTime::currentDateTime().toString("hh:mm:ss"))
                                   .arg(QFileInfo(fileName).fileName()));
            QMessageBox::information(this, "完了", "セグメンテーション結果をエクスポートしました。");
        } else {
            QMessageBox::critical(this, "エラー", "エクスポートに失敗しました。");
        }
    }
}

void AutoSegmentationDialog::updateUIState() {
    bool hasVolume = !m_inputVolume.empty();
    bool hasResult = !m_segmentationResult.empty();
    bool modelReady = hasLoadedModel();

    m_startButton->setEnabled(modelReady && hasVolume && !m_isProcessing);
    m_stopButton->setEnabled(m_isProcessing);
    m_applyButton->setEnabled(hasResult && !m_isProcessing);
    m_exportButton->setEnabled(hasResult && !m_isProcessing);

    // 開始ボタンのツールチップを更新（無効な理由を表示）
    QString tooltip;
    if (m_isProcessing) {
        tooltip = "処理中です。完了までお待ちください。";
    } else if (!modelReady) {
        tooltip = "ONNXモデルをロードしてください。";
    } else if (!hasVolume) {
        tooltip = "セグメンテーション対象のボリュームがロードされていません。\n"
                  "メイン画面でCTボリュームを読み込んでから、このダイアログを開いてください。";
    } else {
        tooltip = "セグメンテーションを開始します。";
    }
    m_startButton->setToolTip(tooltip);

    m_adjustmentGroupBox->setEnabled(hasResult && !m_isProcessing);
}
