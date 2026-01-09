#include "mainwindow.h"
#include "visualization/volume_viewer_window.h"
#include "web/web_server.h"
#include <QActionGroup>
#include <QApplication>
#include <QColorDialog>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QProgressDialog>
#include <QSettings>
#include <QStringList>
#include <QToolBar>
#include <opencv2/core.hpp>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), m_dicomViewer(nullptr), m_volumeWindow(nullptr) {
  setupUI();
  setupMenus();
  setupStatusBar();
  initializeCyberKnifeDoseCalculator();

  // ウィンドウ設定
  setMinimumSize(800, 600);
  resize(1200, 800);
  updateWindowTitle();

  // 中央にウィンドウを配置
  setWindowState(Qt::WindowMaximized);

  // Load saved data folder preference if available
  {
    QSettings settings("ShioRIS3", "ShioRIS3");
    const QString savedRoot = settings.value("dataRoot").toString();
    if (!savedRoot.isEmpty()) {
      m_dbManager.setDataRoot(savedRoot.toStdString());
    }
  }

  // Initialize database at selected data root and start scanning/sync
  if (m_dbManager.open()) {
    m_scanner.fullScanAndRepair();
    m_metadata.writeGlobalIndex();
    m_syncManager = new DatabaseSyncManager(m_dbManager, m_scanner, this);
    connect(m_syncManager, &DatabaseSyncManager::syncEvent, this,
            [this](const QString &) { m_metadata.writeGlobalIndex(); });
    m_syncManager->start();
  }

  // Show DataWindow on startup (regardless of DB open success)
  showDataWindow();
}

MainWindow::~MainWindow() = default;

void MainWindow::setupUI() {
  // 中央ウィジェットとしてDICOMビューアーを設定
  m_dicomViewer = new DicomViewer(this);
  setCentralWidget(m_dicomViewer);
  m_dicomViewer->setCyberKnifeDoseCalculator(&m_cyberKnifeDoseCalculator);
  m_dicomViewer->setDatabaseManager(&m_dbManager);

  // シグナル・スロット接続
  connect(m_dicomViewer, &DicomViewer::imageLoaded, this,
          &MainWindow::onImageLoaded);
  connect(m_dicomViewer, &DicomViewer::windowLevelChanged, this,
          &MainWindow::onWindowLevelChanged);
  connect(m_dicomViewer, &DicomViewer::doseLoadProgress, this,
          &MainWindow::onDoseLoadProgress);
  connect(m_dicomViewer, &DicomViewer::structureLoadProgress, this,
          &MainWindow::onStructureLoadProgress);
}

void MainWindow::setupMenus() {
  m_menuBar = menuBar();

  // File メニュー
  m_fileMenu = m_menuBar->addMenu("&File");

  m_openAction = new QAction("&Open DICOM File...", this);
  m_openAction->setShortcut(QKeySequence::Open);
  m_openAction->setStatusTip("Open a DICOM file");
  connect(m_openAction, &QAction::triggered, this, &MainWindow::openDicomFile);

  m_openFolderAction = new QAction("Open DICOM &Folder...", this);
  m_openFolderAction->setStatusTip("Open a folder containing DICOM files");
  connect(m_openFolderAction, &QAction::triggered, this,
          &MainWindow::openDicomFolder);

  m_openVolumeAction = new QAction("Open DICOM &Volume...", this);
  m_openVolumeAction->setStatusTip("Open a folder as volume");
  connect(m_openVolumeAction, &QAction::triggered, this,
          &MainWindow::openDicomVolume);

  m_openRTDoseAction = new QAction("Open &RT Dose...", this);
  m_openRTDoseAction->setStatusTip("Open a DICOM RT Dose file");
  connect(m_openRTDoseAction, &QAction::triggered, this,
          &MainWindow::openRTDoseFile);

  m_openRTStructAction = new QAction("Open RT &Struct...", this);
  m_openRTStructAction->setStatusTip("Open a DICOM RT Structure Set");
  connect(m_openRTStructAction, &QAction::triggered, this,
          &MainWindow::openRTStructFile);

  m_loadCyberKnifeBeamDataAction =
      new QAction(tr("Load &CyberKnife Beam Data..."), this);
  m_loadCyberKnifeBeamDataAction->setStatusTip(
      tr("Load CyberKnife beam data from a directory"));
  connect(m_loadCyberKnifeBeamDataAction, &QAction::triggered, this,
          &MainWindow::loadCyberKnifeBeamData);

  m_exitAction = new QAction("E&xit", this);
  m_exitAction->setShortcut(QKeySequence::Quit);
  m_exitAction->setStatusTip("Exit the application");
  connect(m_exitAction, &QAction::triggered, this,
          &MainWindow::exitApplication);

  m_fileMenu->addAction(m_openAction);
  m_fileMenu->addAction(m_openFolderAction);
  m_fileMenu->addAction(m_openVolumeAction);
  m_fileMenu->addAction(m_openRTDoseAction);
  m_fileMenu->addAction(m_openRTStructAction);
  m_fileMenu->addSeparator();
  m_fileMenu->addAction(m_loadCyberKnifeBeamDataAction);

  m_cyberKnifeExportMenu = m_fileMenu->addMenu(tr("Export &CyberKnife Data"));
  m_exportCyberKnifeCsvAction = m_cyberKnifeExportMenu->addAction(
      tr("Export &Beam Data Bundle (CSV)..."));
  connect(m_exportCyberKnifeCsvAction, &QAction::triggered, this,
          &MainWindow::exportCyberKnifeCsvBundle);
  m_exportCyberKnifeCsvAction->setStatusTip(
      tr("Export DM, OCR, and TMR tables to timestamped CSV files"));

  m_cyberKnifeMenu = m_menuBar->addMenu(tr("&CyberKnife"));
  m_cyberKnifeMenu->addAction(m_loadCyberKnifeBeamDataAction);
  m_cyberKnifeMenu->addSeparator();
  m_cyberKnifeMenu->addAction(m_exportCyberKnifeCsvAction);

  m_fileMenu->addSeparator();
  m_fileMenu->addAction(m_exitAction);

  // View メニュー
  QMenu *viewMenu = m_menuBar->addMenu("&View");

  QActionGroup *viewGroup = new QActionGroup(this);

  m_singleViewAction = new QAction("&Single", this);
  m_singleViewAction->setCheckable(true);
  m_singleViewAction->setChecked(true);
  viewGroup->addAction(m_singleViewAction);
  connect(m_singleViewAction, &QAction::triggered, this,
          &MainWindow::setSingleView);

  m_dualViewAction = new QAction("&Double", this);
  m_dualViewAction->setCheckable(true);
  viewGroup->addAction(m_dualViewAction);
  connect(m_dualViewAction, &QAction::triggered, this,
          &MainWindow::setDualView);

  m_quadViewAction = new QAction("&Quad", this);
  m_quadViewAction->setCheckable(true);
  viewGroup->addAction(m_quadViewAction);
  connect(m_quadViewAction, &QAction::triggered, this,
          &MainWindow::setQuadView);

  m_fiveViewAction = new QAction("&Five", this);
  m_fiveViewAction->setCheckable(true);
  viewGroup->addAction(m_fiveViewAction);
  connect(m_fiveViewAction, &QAction::triggered, this,
          &MainWindow::setFiveView);

  viewMenu->addAction(m_singleViewAction);
  viewMenu->addAction(m_dualViewAction);
  viewMenu->addAction(m_quadViewAction);
  viewMenu->addAction(m_fiveViewAction);

  m_openFusionDialogAction = new QAction(tr("Image &Fusion..."), this);
  connect(m_openFusionDialogAction, &QAction::triggered, this,
          &MainWindow::openFusionDialog);
  viewMenu->addSeparator();
  viewMenu->addAction(m_openFusionDialogAction);

  // Data Window
  m_openDataWindowAction = new QAction("&Data Window", this);
  connect(m_openDataWindowAction, &QAction::triggered, this,
          &MainWindow::showDataWindow);
  viewMenu->addSeparator();
  viewMenu->addAction(m_openDataWindowAction);

  // Appearance メニュー
  ThemeManager &themeManager = ThemeManager::instance();
  m_appearanceMenu = m_menuBar->addMenu(tr("&Appearance"));
  m_textColorMenu = m_appearanceMenu->addMenu(tr("Text &Color"));

  QActionGroup *textColorGroup = new QActionGroup(this);
  textColorGroup->setExclusive(true);

  auto createThemeAction = [&](const QString &text,
                               ThemeManager::TextTheme theme) {
    QAction *action = new QAction(text, this);
    action->setCheckable(true);
    textColorGroup->addAction(action);
    m_textColorMenu->addAction(action);
    connect(action, &QAction::triggered, this,
            [theme]() { ThemeManager::instance().setTextTheme(theme); });
    return action;
  };

  m_textColorWhiteAction =
      createThemeAction(tr("&White"), ThemeManager::TextTheme::DefaultWhite);
  m_textColorGreenAction =
      createThemeAction(tr("&Green"), ThemeManager::TextTheme::Green);
  m_textColorDarkRedAction =
      createThemeAction(tr("&Dark Red"), ThemeManager::TextTheme::DarkRed);

  // Add separator before custom color option
  m_textColorMenu->addSeparator();

  // Custom color action
  m_textColorCustomAction = new QAction(tr("&Custom..."), this);
  m_textColorCustomAction->setCheckable(true);
  textColorGroup->addAction(m_textColorCustomAction);
  m_textColorMenu->addAction(m_textColorCustomAction);
  connect(m_textColorCustomAction, &QAction::triggered, this,
          &MainWindow::selectCustomTextColor);

  updateTextThemeSelection(themeManager.currentTheme());
  connect(&themeManager, &ThemeManager::themeChanged, this,
          [this](ThemeManager::TextTheme theme) {
            updateTextThemeSelection(theme);
          });

  // Database メニュー
  m_databaseMenu = m_menuBar->addMenu("&Database");

  m_openDatabaseAction = new QAction("&Database", this);
  m_openDatabaseAction->setStatusTip(tr("Open the database window"));
  connect(m_openDatabaseAction, &QAction::triggered, this,
          &MainWindow::showDataWindow);
  m_databaseMenu->addAction(m_openDatabaseAction);

  // Database ツールバー
  QToolBar *databaseToolBar = addToolBar("Database");
  databaseToolBar->addAction(m_openDatabaseAction);

  // Display ツールバー
  QToolBar *displayToolBar = addToolBar("Display");
  displayToolBar->addAction(m_singleViewAction);
  displayToolBar->addAction(m_dualViewAction);
  displayToolBar->addAction(m_quadViewAction);
  displayToolBar->addAction(m_fiveViewAction);

  // AI メニュー
  m_aiMenu = m_menuBar->addMenu("&AI");

  m_autoSegmentationAction = new QAction("&Auto Segment", this);
  m_autoSegmentationAction->setStatusTip(
      tr("Open the auto segmentation dialog"));
  connect(m_autoSegmentationAction, &QAction::triggered, this,
          &MainWindow::startAutoSegmentation);
  m_aiMenu->addAction(m_autoSegmentationAction);

  m_openTranslatorAction = new QAction("&Translator", this);
  m_openTranslatorAction->setStatusTip(
      tr("Open the real-time translator window"));
  connect(m_openTranslatorAction, &QAction::triggered, this,
          &MainWindow::showTranslatorWindow);
  m_aiMenu->addAction(m_openTranslatorAction);

  // AI ツールバー
  QToolBar *aiToolBar = addToolBar("AI");
  aiToolBar->addAction(m_autoSegmentationAction);

  // Help メニュー
  m_helpMenu = m_menuBar->addMenu("&Help");

  m_aboutAction = new QAction("&About ShioRIS3", this);
  m_aboutAction->setStatusTip("About this application");
  connect(m_aboutAction, &QAction::triggered, this,
          &MainWindow::aboutApplication);

  m_helpMenu->addAction(m_aboutAction);

  updateCyberKnifeExportActions();
}

void MainWindow::showDataWindow() {
  if (!m_dataWindow) {
    // Create as top-level (no parent) so it doesn't hide under main window
    m_dataWindow = new DataWindow(m_dbManager, m_scanner, m_metadata, nullptr);
    connect(m_dataWindow, &DataWindow::openStudyRequested, this,
            [this](const QStringList &imageDirs, const QStringList &modalities,
                   const QStringList &rtssPaths,
                   const QStringList &rtdosePaths,
                   const QStringList &rtplanPaths) {
              bool studyLoaded = false;
              if (!imageDirs.isEmpty()) {
                const QString path = imageDirs.first();
                if (!path.isEmpty()) {
                  bool autoRtss = rtssPaths.isEmpty();
                  bool autoDose = rtdosePaths.isEmpty();
                  studyLoaded =
                      m_dicomViewer->loadDicomDirectory(path, true, autoRtss,
                                                        autoDose, imageDirs,
                                                        modalities, 0);

                  if (studyLoaded) {
                    const int total =
                        rtdosePaths.size() + rtplanPaths.size() +
                        (rtssPaths.isEmpty() ? 0 : 1);
                    bool pendingPrimaryDose = true;
                    auto loadDose = [&](const QString &dosePath) {
                      bool treatAsPrimary = pendingPrimaryDose;
                      bool ok = m_dicomViewer->loadRTDoseFile(dosePath,
                                                              treatAsPrimary);
                      if (ok && treatAsPrimary)
                        pendingPrimaryDose = false;
                      return ok;
                    };
                    if (total > 0) {
                      QProgressDialog progress(tr("Loading RT files..."),
                                               QString(), 0, total, this);
                      progress.setWindowModality(Qt::ApplicationModal);
                      progress.setMinimumDuration(0);
                      progress.setValue(0);
                      progress.show();
                      QApplication::processEvents();

                      int step = 0;
                      for (const QString &d : rtdosePaths) {
                        loadDose(d);
                        progress.setValue(++step);
                        QApplication::processEvents();
                      }
                      for (const QString &p : rtplanPaths) {
                        m_dicomViewer->loadBrachyPlanFile(p);
                        progress.setValue(++step);
                        QApplication::processEvents();
                      }
                      if (!rtssPaths.isEmpty()) {
                        m_dicomViewer->loadRTStructFile(rtssPaths.first());
                        progress.setValue(++step);
                        QApplication::processEvents();
                      }
                      progress.close();
                    } else {
                      for (const QString &d : rtdosePaths)
                        loadDose(d);
                      for (const QString &p : rtplanPaths)
                        m_dicomViewer->loadBrachyPlanFile(p);
                      if (!rtssPaths.isEmpty())
                        m_dicomViewer->loadRTStructFile(rtssPaths.first());
                    }
                    m_dicomViewer->update();
                    updateWindowTitle(path);
                  }
                }
              }
              if (studyLoaded && m_dataWindow)
                m_dataWindow->close();
            });
    connect(m_dataWindow, &DataWindow::openDicomFileRequested, this,
            [this](const QString &file) {
              if (file.isEmpty())
                return;
              if (m_dicomViewer->loadDicomFile(file)) {
                updateWindowTitle(file);
                if (m_dataWindow)
                  m_dataWindow->close();
              }
            });
    connect(m_dataWindow, &DataWindow::openDicomFolderRequested, this,
            [this](const QString &dir) {
              if (dir.isEmpty())
                return;
              if (m_dicomViewer->loadDicomDirectory(dir)) {
                updateWindowTitle(dir);
                if (m_dataWindow)
                  m_dataWindow->close();
              }
            });
    connect(m_dataWindow, &DataWindow::changeDataRootRequested, this,
            [this](const QString &newRoot) {
              if (newRoot.isEmpty())
                return;
              if (m_syncManager)
                m_syncManager->stop();
              m_dbManager.setDataRoot(newRoot.toStdString());
              if (!m_dbManager.open()) {
                QMessageBox::warning(this, "Error",
                                     "Failed to open selected data folder");
                return;
              }
              // Persist selection for next launch
              QSettings settings("ShioRIS3", "ShioRIS3");
              settings.setValue("dataRoot", newRoot);
              m_scanner.fullScanAndRepair();
              m_metadata.writeGlobalIndex();
              if (m_syncManager)
                m_syncManager->start();
              m_dataWindow->refresh();
            });
    m_dataWindow->setWindowTitle("ShioRIS3 Data");
    m_dataWindow->resize(800, 600);
  }
  // Ensure it shows on top when opened
  m_dataWindow->setWindowFlag(Qt::WindowStaysOnTopHint, true);
  // Set width smaller than Main Window and clamp within screen
  QRect mainGeom = this->geometry();
  QRect target = mainGeom;
  // Prefer the main window's screen if available
  QScreen *scr = this->screen() ? this->screen()
                                : (m_dataWindow->screen()
                                       ? m_dataWindow->screen()
                                       : QGuiApplication::primaryScreen());
  if (scr) {
    QRect avail = scr->availableGeometry();
    const int margin = 20; // keep a small margin from edges
    int wFromMain = static_cast<int>(mainGeom.width() * 0.8); // 80% of main
    int wFromScreen = (avail.width() * 2) / 3; // <= 2/3 of screen
    int w = qMin(wFromMain, wFromScreen);
    w = qMin(w, avail.width() - margin);
    w = qMax(400, w); // reasonable minimum
    int h = qMin(mainGeom.height(), avail.height() - margin);
    h = qMax(400, h);
    int x = mainGeom.x() + (mainGeom.width() - w) / 2;
    int y = qBound(avail.y(), mainGeom.y(), avail.bottom() - h + 1);
    // Ensure fully visible
    if (x < avail.left())
      x = avail.left();
    if (x + w > avail.right() + 1)
      x = avail.right() - w + 1;
    target = QRect(x, y, w, h);
    // Also cap maximum size so user cannot resize beyond screen
    m_dataWindow->setMaximumSize(avail.size());
  }
  m_dataWindow->setMinimumSize(QSize(400, 400));
  m_dataWindow->setGeometry(target);
  // Enforce geometry after show in case sizeHint tries to expand
  m_dataWindow->show();
  m_dataWindow->setGeometry(target);
  m_dataWindow->raise();
  m_dataWindow->activateWindow();
}

void MainWindow::showTranslatorWindow() {
  if (!m_translatorWindow) {
    // Create as top-level window (no parent)
    m_translatorWindow = new TranslatorWindow(nullptr);
    m_translatorWindow->setWindowTitle("ShioRIS3 Translator");
    m_translatorWindow->resize(900, 700);
  }

  m_translatorWindow->show();
  m_translatorWindow->raise();
  m_translatorWindow->activateWindow();
}

/*
void MainWindow::startAutoSegmentation() {
    if (!m_dicomViewer) {
        QMessageBox::warning(this, "警告", "DICOMビューアーが初期化されていません。");
        return;
    }
    
    // 現在ロードされているボリュームの確認
    bool hasVolume = m_dicomViewer->isVolumeLoaded();
    if (!hasVolume) {
        QMessageBox::information(this, "AI セグメンテーション", 
            "セグメンテーションを実行するDICOM画像をロードしてください。\n\n"
            "推奨:\n"
            "• 腹部CT画像（複数スライス）\n"
            "• HU値範囲: -1024〜3071\n"
            "• 対象臓器: 肝臓、右腎、左腎・脾臓");
        return;
    }
    
    // ボリューム情報の取得と表示
    QString volumeInfo = getVolumeInfoString();
    
    // AI機能の利用可能性チェック
#ifndef USE_ONNXRUNTIME
    QMessageBox::warning(this, "AI機能未対応", 
        "ONNX Runtimeが有効になっていません。\n\n"
        "AI機能を使用するには、ENABLE_ONNXRUNTIME=ON でビルドしてください。");
    return;
#endif
    
    // 確認ダイアログ
    int result = QMessageBox::question(this, "AI セグメンテーション", 
        QString("以下のボリュームでAIセグメンテーションを実行しますか？\n\n%1\n\n"
                "対象臓器: 肝臓、右腎、左腎・脾臓\n"
                "処理時間: 数秒〜数分（ボリュームサイズによる）")
                .arg(volumeInfo),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
    
    if (result == QMessageBox::Yes) {
        // DicomViewerの既存AI機能を呼び出し
        showAIFeaturesDialog();
    }
}
  */
void MainWindow::startAutoSegmentation() {
#ifndef USE_ONNXRUNTIME
  QMessageBox::warning(this, tr("AI機能未対応"),
                       tr("ONNX Runtime が有効になっていません。\n\n"
                          "AI セグメンテーション機能を使用するには、"
                          "ENABLE_ONNXRUNTIME=ON でビルドしてください。"));
  return;
#else
  if (!m_dicomViewer) {
    QMessageBox::warning(this, tr("警告"),
                         tr("DICOM ビューアーが初期化されていません。"));
    return;
  }

  cv::Mat volume = m_dicomViewer->getSegmentationVolumeRaw();
  if (volume.empty()) {
    QMessageBox::information(this, tr("AI セグメンテーション"),
                             tr("セグメンテーションを実行する DICOM 画像が"
                                "読み込まれていません。\n\n"
                                "推奨:\n"
                                "• 腹部 CT 画像（複数スライス）\n"
                                "• HU 値範囲: -1024〜3071"));
    return;
  }

  if (!m_autoSegDialog) {
    m_autoSegDialog = new AutoSegmentationDialog(this);
    connect(m_autoSegDialog, &AutoSegmentationDialog::applyResult,
            this, &MainWindow::applySegmentationResult);
    connect(m_autoSegDialog, &AutoSegmentationDialog::segmentationFinished,
            this, &MainWindow::onSegmentationFinished);
  }

  // RTSS Structure bounding boxをOnnxSegmenterに適用
  m_dicomViewer->applyBoundingBoxToSegmenter();

  m_autoSegDialog->setSharedSegmenter(m_dicomViewer->getSegmentationModel());
  m_autoSegDialog->startSegmentation(volume);
#endif
}

void MainWindow::openFusionDialog() {
  if (!m_fusionDialog) {
    m_fusionDialog = new FusionDialog(m_dbManager, m_dicomViewer, this);
  } else {
    m_fusionDialog->setPrimaryFromViewer(m_dicomViewer);
  }
  m_fusionDialog->show();
  m_fusionDialog->raise();
  m_fusionDialog->activateWindow();
}

QString MainWindow::getVolumeInfoString() {
    if (!m_dicomViewer) {
        return "DICOMビューアー未初期化";
    }
    
    if (m_dicomViewer->isVolumeLoaded()) {
        return "3Dボリューム: 複数スライス読み込み済み\n"
               "データ型: DICOM CT画像";
    } else {
        return "2D画像: 単一スライス\n"
               "注意: 3Dボリュームの読み込みを推奨";
    }
}

void MainWindow::showAIFeaturesDialog() {
    // AI機能の説明と案内ダイアログ
    QDialog *aiDialog = new QDialog(this);
    aiDialog->setWindowTitle("AI セグメンテーション機能");
    aiDialog->resize(500, 400);
    aiDialog->setAttribute(Qt::WA_DeleteOnClose);

    QVBoxLayout *layout = new QVBoxLayout(aiDialog);
    ThemeManager &theme = ThemeManager::instance();
    
    // 説明テキスト
    QLabel *descLabel = new QLabel(
        "<h3>AI セグメンテーション機能</h3>"
        "<p>ShioRIS3のAI機能は、腹部CT画像から臓器を自動的に抽出します。</p>"
        "<p><b>対象臓器:</b></p>"
        "<ul>"
        "<li>🔴 肝臓 (Liver)</li>"
        "<li>🟢 右腎 (Right Kidney)</li>"
        "<li>🔵 左腎・脾臓 (Left Kidney/Spleen)</li>"
        "</ul>"
        "<p><b>使用方法:</b></p>"
        "<ol>"
        "<li>右側のタブから「AI」パネルを選択</li>"
        "<li>「Load ONNX Model」でモデルファイルを読み込み</li>"
        "<li>「Run Segmentation」でセグメンテーション実行</li>"
        "<li>「Show Segmentation」で結果の表示切り替え</li>"
        "</ol>"
    );
    descLabel->setWordWrap(true);
    descLabel->setStyleSheet("QLabel { padding: 10px; }");
    
    layout->addWidget(descLabel);
    
    // ボタン
    QHBoxLayout *buttonLayout = new QHBoxLayout;
    
    QPushButton *openAIPanelButton = new QPushButton("AIパネルを開く");
    theme.applyTextColor(openAIPanelButton,
                         QStringLiteral("QPushButton { background-color: #4ECDC4; "
                                         "color: %1; font-weight: bold; padding: 8px; }"));
    connect(openAIPanelButton, &QPushButton::clicked, [this, aiDialog]() {
        // DicomViewerの右側タブでAIパネルを選択
        if (m_dicomViewer) {
            QMessageBox::information(this, "AI 機能案内", 
    "右側の「AI」タブから ONNX モデルをロードして\n"
    "セグメンテーションを実行してください。");
        }
        aiDialog->accept();
    });
    
    QPushButton *helpButton = new QPushButton("使用説明書");
    theme.applyTextColor(helpButton,
                         QStringLiteral("QPushButton { background-color: #96CEB4; "
                                         "color: %1; font-weight: bold; padding: 8px; }"));
    connect(helpButton, &QPushButton::clicked, [this]() {
        showAIUsageGuide();
    });
    
    QPushButton *closeButton = new QPushButton("閉じる");
    theme.applyTextColor(closeButton,
                         QStringLiteral("QPushButton { background-color: #95A5A6; "
                                         "color: %1; font-weight: bold; padding: 8px; }"));
    connect(closeButton, &QPushButton::clicked, aiDialog, &QDialog::reject);
    
    buttonLayout->addWidget(openAIPanelButton);
    buttonLayout->addWidget(helpButton);
    buttonLayout->addStretch();
    buttonLayout->addWidget(closeButton);
    
    layout->addLayout(buttonLayout);
    
    // ステータス更新
    m_statusLabel->setText("AI セグメンテーション機能の説明を表示中");
    
    aiDialog->show();
}

void MainWindow::showAIUsageGuide() {
    // 使用説明書ダイアログ
    QDialog *guideDialog = new QDialog(this);
    guideDialog->setWindowTitle("AI機能 使用説明書");
    guideDialog->resize(600, 500);
    guideDialog->setAttribute(Qt::WA_DeleteOnClose);
    
    QVBoxLayout *layout = new QVBoxLayout(guideDialog);
    
    QTextEdit *guideText = new QTextEdit;
    guideText->setReadOnly(true);
    guideText->setHtml(
        "<h2>ShioRIS3 AI Segmentation 使用説明書</h2>"
        
        "<h3>1. 準備</h3>"
        "<ul>"
        "<li><b>ONNXモデル</b>: 腹部多臓器セグメンテーション用のONNXファイルを準備</li>"
        "<li><b>CT画像</b>: 腹部CT DICOMファイル（複数スライス推奨）</li>"
        "<li><b>推奨HU値範囲</b>: -1024 〜 3071</li>"
        "</ul>"
        
        "<h3>2. 基本操作</h3>"
        "<ol>"
        "<li><b>DICOM読み込み</b>: File → Open でCT画像を読み込み</li>"
        "<li><b>AIパネル表示</b>: 右側タブの「AI」を選択</li>"
        "<li><b>モデルロード</b>: \"Load ONNX Model\" をクリック</li>"
        "<li><b>セグメンテーション実行</b>: \"Run Segmentation\" をクリック</li>"
        "<li><b>結果確認</b>: \"Show Segmentation\" で表示切り替え</li>"
        "</ol>"
        
        "<h3>3. 結果の見方</h3>"
        "<ul>"
        "<li><b>赤色</b>: 肝臓 (Liver)</li>"
        "<li><b>緑色</b>: 右腎 (Right Kidney)</li>"
        "<li><b>青色</b>: 左腎・脾臓 (Left Kidney/Spleen)</li>"
        "</ul>"
        
        "<h3>4. トラブルシューティング</h3>"
        "<ul>"
        "<li><b>\"ONNX Runtime not found\"</b>: ENABLE_ONNXRUNTIME=ON でリビルド</li>"
        "<li><b>モデルロードエラー</b>: ONNXファイルの形式を確認</li>"
        "<li><b>セグメンテーション失敗</b>: 腹部CT画像であることを確認</li>"
        "<li><b>結果が表示されない</b>: \"Show Segmentation\" がONになっているか確認</li>"
        "</ul>"
        
        "<h3>5. パフォーマンス</h3>"
        "<ul>"
        "<li><b>処理時間</b>: 小ボリューム 5-10秒、大ボリューム 30-60秒</li>"
        "<li><b>メモリ使用量</b>: ボリュームサイズに比例（4MB〜64MB）</li>"
        "<li><b>推奨システム</b>: 8GB RAM以上、SSD推奨</li>"
        "</ul>"
        
        "<h3>6. 推奨ONNXモデル</h3>"
        "<ul>"
        "<li>TotalSegmentator (腹部臓器)</li>"
        "<li>nnU-Net Abdominal Organs</li>"
        "<li>MONAI Abdominal Segmentation</li>"
        "</ul>"
        
        "<p><i>詳細情報: <a href='https://docs.shioris3.org/ai'>docs.shioris3.org/ai</a></i></p>"
    );
    
    layout->addWidget(guideText);
    
    QPushButton *closeButton = new QPushButton("閉じる");
    connect(closeButton, &QPushButton::clicked, guideDialog, &QDialog::accept);
    
    QHBoxLayout *buttonLayout = new QHBoxLayout;
    buttonLayout->addStretch();
    buttonLayout->addWidget(closeButton);
    layout->addLayout(buttonLayout);
    
    guideDialog->show();
}

QString MainWindow::getVolumeInfoString(const cv::Mat &volume) {
    if (volume.empty()) {
        return "空のボリューム";
    }
    
    QString info;
    
    if (volume.dims == 3) {
        int depth = volume.size[0];
        int height = volume.size[1];
        int width = volume.size[2];
        
        info = QString("3Dボリューム: %1×%2×%3 (%4スライス)")
               .arg(width).arg(height).arg(depth).arg(depth);
    } else if (volume.dims == 2) {
        info = QString("2D画像: %1×%2")
               .arg(volume.cols).arg(volume.rows);
    } else {
        info = QString("%1次元データ").arg(volume.dims);
    }
    
    // データ型情報
    QString typeStr;
    switch (volume.type()) {
        case CV_8UC1: typeStr = "8-bit グレースケール"; break;
        case CV_16UC1: typeStr = "16-bit 符号なし"; break;
        case CV_16SC1: typeStr = "16-bit 符号付き"; break;
        case CV_32FC1: typeStr = "32-bit 浮動小数点"; break;
        default: typeStr = QString("Type %1").arg(volume.type()); break;
    }
    
    info += QString("\nデータ型: %1").arg(typeStr);
    
    // メモリ使用量
    size_t totalElements = 1;
    for (int i = 0; i < volume.dims; ++i) {
        totalElements *= volume.size[i];
    }
    
    size_t bytesPerElement = volume.elemSize();
    size_t totalBytes = totalElements * bytesPerElement;
    
    QString memoryStr;
    if (totalBytes > 1024 * 1024) {
        memoryStr = QString("%1 MB").arg(totalBytes / (1024.0 * 1024.0), 0, 'f', 1);
    } else if (totalBytes > 1024) {
        memoryStr = QString("%1 KB").arg(totalBytes / 1024.0, 0, 'f', 1);
    } else {
        memoryStr = QString("%1 bytes").arg(totalBytes);
    }
    
    info += QString("\nメモリ使用量: %1").arg(memoryStr);
    
    return info;
}

void MainWindow::applySegmentationResult(const cv::Mat &segmentationResult) {
    if (segmentationResult.empty()) {
        QMessageBox::warning(this, "警告", "セグメンテーション結果が空です。");
        return;
    }
    
    if (!m_dicomViewer) {
        QMessageBox::warning(this, "警告", "DICOMビューアーが利用できません。");
        return;
    }
    
    try {
        QString summary;
        if (!m_dicomViewer->applySegmentationVolume(segmentationResult, &summary)) {
            QMessageBox::warning(this, tr("警告"),
                                 tr("セグメンテーション結果の適用に失敗しました。"));
            return;
        }

        m_statusLabel->setText(tr("AI セグメンテーション結果を適用しました"));

        QString messageBody =
            tr("セグメンテーション結果を適用しました。\n右側のAIタブで表示を切り替えられます。");
        if (!summary.isEmpty()) {
            messageBody.append("\n\n").append(summary);
        }

        QMessageBox::information(this, tr("完了"), messageBody);

        qDebug() << "Segmentation result applied successfully";

    } catch (const std::exception &e) {
        QMessageBox::critical(this, tr("エラー"),
            QString(tr("セグメンテーション適用中にエラーが発生しました:\n%1"))
            .arg(e.what()));

        qCritical() << "Error applying segmentation result:" << e.what();
    }
}

void MainWindow::onSegmentationFinished(const cv::Mat &result) {
    if (result.empty()) {
        m_statusLabel->setText("セグメンテーション完了（結果なし）");
        return;
    }
    
    m_statusLabel->setText("セグメンテーション完了");
    
    QMessageBox::information(this, "完了", 
        "AIセグメンテーションが完了しました。\n"
        "右側のAIタブで結果を確認してください。");
}

QString MainWindow::getSegmentationStatsString(const cv::Mat &segmentation) {
    if (segmentation.empty()) {
        return "空の結果";
    }
    
    // ラベル統計の計算
    std::vector<int> labelCounts(4, 0); // 4クラス想定
    int totalVoxels = 0;
    
    if (segmentation.dims == 3) {
        int depth = segmentation.size[0];
        int height = segmentation.size[1];
        int width = segmentation.size[2];
        totalVoxels = depth * height * width;
        
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    uchar label = segmentation.at<uchar>(z, y, x);
                    if (label < labelCounts.size()) {
                        labelCounts[label]++;
                    }
                }
            }
        }
    } else {
        totalVoxels = segmentation.rows * segmentation.cols;
        
        for (int y = 0; y < segmentation.rows; ++y) {
            for (int x = 0; x < segmentation.cols; ++x) {
                uchar label = segmentation.at<uchar>(y, x);
                if (label < labelCounts.size()) {
                    labelCounts[label]++;
                }
            }
        }
    }
    
    // 臓器領域の合計（背景を除く）
    int organVoxels = totalVoxels - labelCounts[0];
    double organPercentage = (totalVoxels > 0) ? (organVoxels * 100.0 / totalVoxels) : 0.0;
    
    return QString("臓器領域 %1% (%2/%3 ボクセル)")
           .arg(organPercentage, 0, 'f', 1)
           .arg(organVoxels)
           .arg(totalVoxels);
}
/*
void MainWindow::showSegmentationStatistics(const cv::Mat &segmentation) {
    if (segmentation.empty()) {
        return;
    }
    
    // 詳細統計をダイアログで表示
    QDialog *statsDialog = new QDialog(this);
    statsDialog->setWindowTitle("セグメンテーション統計");
    statsDialog->setModal(false);
    statsDialog->resize(400, 300);
    
    QVBoxLayout *layout = new QVBoxLayout(statsDialog);
    
    // 統計テーブル作成
    QTableWidget *table = new QTableWidget;
    table->setColumnCount(3);
    table->setHorizontalHeaderLabels(QStringList() << "臓器" << "ボクセル数" << "割合 (%)");
    
    // ラベル統計計算
    std::vector<int> labelCounts(4, 0);
    int totalVoxels = 0;
    
    if (segmentation.dims == 3) {
        int depth = segmentation.size[0];
        int height = segmentation.size[1];
        int width = segmentation.size[2];
        totalVoxels = depth * height * width;
        
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    uchar label = segmentation.at<uchar>(z, y, x);
                    if (label < labelCounts.size()) {
                        labelCounts[label]++;
                    }
                }
            }
        }
    } else {
        totalVoxels = segmentation.rows * segmentation.cols;
        
        for (int y = 0; y < segmentation.rows; ++y) {
            for (int x = 0; x < segmentation.cols; ++x) {
                uchar label = segmentation.at<uchar>(y, x);
                if (label < labelCounts.size()) {
                    labelCounts[label]++;
                }
            }
        }
    }
    
    // テーブルに結果を設定
    QStringList organNames = {"Background", "Liver", "Right Kidney", "Left Kidney/Spleen"};
    table->setRowCount(organNames.size());
    
    for (int i = 0; i < organNames.size(); ++i) {
        table->setItem(i, 0, new QTableWidgetItem(organNames[i]));
        table->setItem(i, 1, new QTableWidgetItem(QString::number(labelCounts[i])));
        
        double percentage = (totalVoxels > 0) ? (labelCounts[i] * 100.0 / totalVoxels) : 0.0;
        table->setItem(i, 2, new QTableWidgetItem(QString::number(percentage, 'f', 2)));
        
        // 背景以外に色付け
        if (i > 0 && labelCounts[i] > 0) {
            QColor colors[] = {QColor(), QColor(255, 200, 200), QColor(200, 255, 200), QColor(200, 200, 255)};
            for (int col = 0; col < 3; ++col) {
                table->item(i, col)->setBackground(colors[i]);
            }
        }
    }
    
    table->resizeColumnsToContents();
    layout->addWidget(table);
    
    // 閉じるボタン
    QPushButton *closeButton = new QPushButton("閉じる");
    connect(closeButton, &QPushButton::clicked, statsDialog, &QDialog::accept);
    
    QHBoxLayout *buttonLayout = new QHBoxLayout;
    buttonLayout->addStretch();
    buttonLayout->addWidget(closeButton);
    layout->addLayout(buttonLayout);
    
    statsDialog->show();
}
*/
void MainWindow::setupStatusBar() {
  m_statusBar = statusBar();

  // ステータスラベル
  m_statusLabel = new QLabel("Ready");
  m_statusBar->addWidget(m_statusLabel);

  m_cyberKnifeStatusLabel = new QLabel(tr("CyberKnife: 未ロード"));
  m_statusBar->addWidget(m_cyberKnifeStatusLabel);

  // Window/Level表示
  m_windowLevelLabel = new QLabel("W/L: -/-");
  m_statusBar->addPermanentWidget(m_windowLevelLabel);

  // プログレスバー（将来の拡張用）
  m_progressBar = new QProgressBar();
  m_progressBar->setVisible(false);
  m_statusBar->addPermanentWidget(m_progressBar);
}

void MainWindow::initializeCyberKnifeDoseCalculator() {
  if (m_cyberKnifeDoseCalculator.isReady())
    return;

  if (m_cyberKnifeDoseCalculator.initialize(QString())) {
    if (m_statusBar) {
      QSettings settings("ShioRIS3", "ShioRIS3");
      const QString resolvedPath =
          settings.value("cyberknife/beamDataPath").toString();
      if (!resolvedPath.isEmpty()) {
        const QString displayPath =
            QDir::toNativeSeparators(resolvedPath);
        m_statusBar->showMessage(
            tr("CyberKnife beam data loaded: %1").arg(displayPath), 5000);
      }
    }
  }

  if (m_dicomViewer)
    m_dicomViewer->refreshCyberKnifeCalculatorState();

  updateCyberKnifeExportActions();
}

void MainWindow::loadCyberKnifeBeamData() {
  QSettings settings("ShioRIS3", "ShioRIS3");
  const QString lastPath = settings.value("cyberknife/beamDataPath").toString();
  const QString initialDir = lastPath.isEmpty() ? QDir::homePath() : lastPath;

  const QString selectedDir = QFileDialog::getExistingDirectory(
      this, tr("Select CyberKnife Beam Data Directory"), initialDir,
      QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

  if (selectedDir.isEmpty()) {
    return;
  }

  m_statusLabel->setText(tr("Loading CyberKnife beam data..."));
  m_statusBar->showMessage(tr("Loading CyberKnife beam data..."));
  if (m_cyberKnifeStatusLabel) {
    m_cyberKnifeStatusLabel->setText(tr("CyberKnife: 読み込み中..."));
  }

  QApplication::setOverrideCursor(Qt::WaitCursor);
  const bool loaded = m_cyberKnifeDoseCalculator.initialize(selectedDir);
  QApplication::restoreOverrideCursor();

  if (loaded) {
    QSettings updatedSettings("ShioRIS3", "ShioRIS3");
    const QString resolvedPath =
        updatedSettings.value("cyberknife/beamDataPath").toString();
    const QString displayPath =
        QDir::toNativeSeparators(resolvedPath.isEmpty() ? selectedDir
                                                        : resolvedPath);
    const QString exportHint =
        tr("CSV出力は「CyberKnife」メニュー、または"
           "「ファイル > Export CyberKnife Data」から実行できます。");

    m_statusLabel->setText(tr("CyberKnife beam data ready"));
    m_statusBar->showMessage(
        tr("CyberKnife beam data loaded: %1").arg(displayPath), 5000);
    QMessageBox::information(
        this, tr("CyberKnife Beam Data"),
        tr("CyberKnife beam data was loaded successfully from:\n%1\n\n%2")
            .arg(displayPath, exportHint));
  } else {
    m_statusLabel->setText(tr("Failed to load CyberKnife beam data"));
    m_statusBar->showMessage(tr("Failed to load CyberKnife beam data"), 5000);
    const QStringList errors = m_cyberKnifeDoseCalculator.lastErrors();
    const QString errorDetails = errors.join(QStringLiteral("\n"));
    QString message = tr("Failed to load CyberKnife beam data from:\n%1\n\nVerify that the "
                        "directory contains DMTable.dat, TMRtable.dat, and "
                        "OCRtable*.dat files.")
                           .arg(QDir::toNativeSeparators(selectedDir));
    if (!errorDetails.isEmpty()) {
      message.append(QStringLiteral("\n\n"));
      message.append(errorDetails);
    }
    QMessageBox::warning(this, tr("CyberKnife Beam Data"), message);
  }
  updateCyberKnifeExportActions();
  if (!loaded && m_cyberKnifeStatusLabel) {
    m_cyberKnifeStatusLabel->setText(tr("CyberKnife: 読み込み失敗"));
  }
  if (m_dicomViewer)
    m_dicomViewer->refreshCyberKnifeCalculatorState();
}

void MainWindow::exportCyberKnifeCsvBundle() {
  const CyberKnife::BeamDataManager *manager =
      m_cyberKnifeDoseCalculator.beamDataManager();
  if (!manager || !m_cyberKnifeDoseCalculator.isReady()) {
    QMessageBox::warning(this, tr("Export CyberKnife Data"),
                         tr("CyberKnifeのビームデータが読み込まれていません。"));
    return;
  }

  QSettings settings("ShioRIS3", "ShioRIS3");
  const QString defaultDir = settings
                                  .value("cyberknife/lastExportDir",
                                         settings.value("cyberknife/beamDataPath",
                                                        QDir::homePath()))
                                  .toString();

  const QString targetDir = QFileDialog::getExistingDirectory(
      this, tr("Export CyberKnife Beam Data (CSV)"), defaultDir,
      QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
  if (targetDir.isEmpty()) {
    return;
  }

  settings.setValue("cyberknife/lastExportDir", targetDir);

  QStringList exportedFiles;
  const bool ok = manager->exportAllDataToCsv(targetDir, &exportedFiles);

  QStringList displayPaths;
  displayPaths.reserve(exportedFiles.size());
  for (const QString &path : exportedFiles) {
    displayPaths.append(QDir::toNativeSeparators(path));
  }

  if (ok) {
    const QString detail = displayPaths.isEmpty()
                               ? QDir::toNativeSeparators(targetDir)
                               : displayPaths.join(QStringLiteral("\n"));
    QMessageBox::information(
        this, tr("Export CyberKnife Data"),
        tr("以下のCSVファイルを出力しました:\n%1").arg(detail));
  } else {
    QString message = tr("CyberKnife CSVの出力に失敗しました。");
    if (!displayPaths.isEmpty()) {
      message.append(QStringLiteral("\n\n"));
      message.append(tr("成功したファイル:\n%1")
                         .arg(displayPaths.join(QStringLiteral("\n"))));
    }
    QMessageBox::warning(this, tr("Export CyberKnife Data"), message);
  }
}

void MainWindow::updateCyberKnifeExportActions() {
  const bool ready = m_cyberKnifeDoseCalculator.isReady();
  if (m_cyberKnifeExportMenu) {
    m_cyberKnifeExportMenu->setEnabled(ready);
  }
  if (m_exportCyberKnifeCsvAction) {
    m_exportCyberKnifeCsvAction->setEnabled(ready);
  }
  if (m_cyberKnifeStatusLabel) {
    m_cyberKnifeStatusLabel->setText(
        ready ? tr("CyberKnife: CSV出力可") : tr("CyberKnife: 未ロード"));
  }
}

void MainWindow::updateTextThemeSelection(ThemeManager::TextTheme theme) {
  if (m_textColorWhiteAction)
    m_textColorWhiteAction->setChecked(
        theme == ThemeManager::TextTheme::DefaultWhite);
  if (m_textColorGreenAction)
    m_textColorGreenAction->setChecked(
        theme == ThemeManager::TextTheme::Green);
  if (m_textColorDarkRedAction)
    m_textColorDarkRedAction->setChecked(
        theme == ThemeManager::TextTheme::DarkRed);
  if (m_textColorCustomAction)
    m_textColorCustomAction->setChecked(
        theme == ThemeManager::TextTheme::Custom);
}

void MainWindow::openDicomFile() {
  QString filename = QFileDialog::getOpenFileName(
      this, "Open DICOM File", QDir::homePath(),
      "DICOM Files (*.dcm *.DCM *.dicom *.DICOM);;All Files (*.*)");

  if (!filename.isEmpty()) {
    m_statusLabel->setText("Loading DICOM file...");
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0); // インデターミネート

    QApplication::processEvents(); // UIを更新

    bool loaded = m_dicomViewer->loadDicomFile(filename);
    if (loaded) {
      m_currentFilename = filename;
      updateWindowTitle(filename);
      m_statusLabel->setText("DICOM file loaded successfully");
    } else {
      QMessageBox::warning(
          this, "Error",
          QString("Failed to load DICOM file:\n%1").arg(filename));
      m_statusLabel->setText("Failed to load DICOM file");
    }

    m_progressBar->setVisible(false);
  }
}

void MainWindow::openDicomFolder() {
  QString dir = QFileDialog::getExistingDirectory(this, "Open DICOM Folder",
                                                  QDir::homePath());

  if (!dir.isEmpty()) {
    m_statusLabel->setText("Loading DICOM folder...");
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0);

    QApplication::processEvents();

    // **修正**: デバッグ出力を条件付きにする
#ifdef QT_DEBUG
    static bool debugFolderLoading = qEnvironmentVariableIsSet("SHIORIS3_DEBUG_FOLDER");
    if (debugFolderLoading) {
        qDebug() << "=== CT Folder Loading Debug ===";
        qDebug() << "Loading folder:" << dir;

        // フォルダ内のDICOMファイルをチェック
        QDir dirObj(dir);
        QStringList filters{"*.dcm", "*.DCM", "*.dicom", "*.DICOM"};
        QStringList files = dirObj.entryList(filters, QDir::Files, QDir::Name);

        if (!files.isEmpty()) {
          QString firstFile = dirObj.absoluteFilePath(files.first());
          qDebug() << "First CT file:" << firstFile;

          // 最初のファイルの詳細座標情報を確認
          DicomReader debugReader;
          if (debugReader.loadDicomFile(firstFile)) {
            double x, y, z;
            if (debugReader.getImagePositionPatient(x, y, z)) {
              qDebug() << QString("First CT file position: (%1, %2, %3)")
                              .arg(x, 0, 'f', 1)
                              .arg(y, 0, 'f', 1)
                              .arg(z, 0, 'f', 1);
            } else {
              qDebug() << "First CT file: No position information found";
            }

            double r1, r2, r3, c1, c2, c3;
            if (debugReader.getImageOrientationPatient(r1, r2, r3, c1, c2, c3)) {
              qDebug() << QString("First CT file orientation: [%1,%2,%3,%4,%5,%6]")
                              .arg(r1, 0, 'f', 3)
                              .arg(r2, 0, 'f', 3)
                              .arg(r3, 0, 'f', 3)
                              .arg(c1, 0, 'f', 3)
                              .arg(c2, 0, 'f', 3)
                              .arg(c3, 0, 'f', 3);
            } else {
              qDebug() << "First CT file: No orientation information found";
            }

            double row, col;
            if (debugReader.getPixelSpacing(row, col)) {
              qDebug() << QString("First CT file spacing: [%1, %2]")
                              .arg(row, 0, 'f', 3)
                              .arg(col, 0, 'f', 3);
            } else {
              qDebug() << "First CT file: No spacing information found";
            }
          }
        }
    }
#endif

    bool loaded = m_dicomViewer->loadDicomDirectory(dir);

    // **修正**: ボリューム状態確認のデバッグ出力も条件付きにする
#ifdef QT_DEBUG
    if (debugFolderLoading && loaded && m_dicomViewer->isVolumeLoaded()) {
        // DicomViewer::loadDicomDirectory()後のCTボリューム状態を確認
        // （詳細なボリューム情報のデバッグ出力）
    }
#endif

    if (loaded) {
      // フォルダパスから患者名を抽出してタイトルに表示
      QDir dirInfo(dir);
      QString folderName = dirInfo.dirName();
      updateWindowTitle(QString("Folder: %1").arg(folderName));
      m_statusLabel->setText("DICOM folder loaded successfully");
    } else {
      QMessageBox::warning(
          this, "Error",
          QString("Failed to load DICOM folder:\n%1").arg(dir));
      m_statusLabel->setText("Failed to load DICOM folder");
    }

    m_progressBar->setVisible(false);
  }
}

void MainWindow::openDicomVolume() {
  QString dir = QFileDialog::getExistingDirectory(this, "Open DICOM Volume",
                                                  QDir::homePath());

  if (!dir.isEmpty()) {
    if (!m_volumeWindow)
      m_volumeWindow = new VolumeViewerWindow(this);
    m_volumeWindow->show();
    m_volumeWindow->raise();
    m_volumeWindow->loadVolume(dir);
  }
}

void MainWindow::openRTDoseFile() {
  QString filename = QFileDialog::getOpenFileName(
      this, "Open RT Dose File", QDir::homePath(),
      "DICOM Files (*.dcm *.DCM *.dicom *.DICOM);;All Files (*.*)");

  if (!filename.isEmpty()) {
    m_statusLabel->setText("Loading RT Dose file...");
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0);
    m_progressBar->setValue(0);
    QApplication::processEvents();
    bool ok = m_dicomViewer->loadRTDoseFile(filename);
    if (!ok) {
      QMessageBox::warning(
          this, "Error",
          QString("Failed to load RT Dose file:\n%1").arg(filename));
      m_statusLabel->setText("Failed to load RT Dose file");
    } else {
      m_statusLabel->setText("RT Dose loaded");
    }
    m_progressBar->setVisible(false);
  }
}

void MainWindow::openRTStructFile() {
  QString filename = QFileDialog::getOpenFileName(
      this, "Open RT Struct File", QDir::homePath(),
      "DICOM Files (*.dcm *.DCM *.dicom *.DICOM);;All Files (*.*)");

  if (!filename.isEmpty()) {
    m_statusLabel->setText("Loading RT Struct...");
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0);
    QApplication::processEvents();

    bool ok = m_dicomViewer->loadRTStructFile(filename);

    m_progressBar->setVisible(false);
    if (!ok) {
      QMessageBox::warning(
          this, "Error",
          QString("Failed to load RT Struct file:\n%1").arg(filename));
    } else {
      m_statusLabel->setText("RT Struct loaded");
    }
  }
}

void MainWindow::exitApplication() { QApplication::quit(); }

void MainWindow::aboutApplication() {
  QMessageBox::about(this, "About ShioRIS3",
                     "<h2>ShioRIS3</h2>"
                     "<p>Radiation Treatment Planning System</p>"
                     "<p>Version 1.0.0</p>"
                     "<p>Built with:</p>"
                     "<ul>"
                     "<li>Qt 6.9</li>"
                     "<li>DCMTK 3.6.9</li>"
                     "<li>OpenCV 4.11</li>"
                     "<li>OpenGL 4.1</li>"
                     "</ul>"
                     "<p>© 2025 ShioRIS3 Development Team</p>");
}

void MainWindow::onImageLoaded(const QString &filename) {
  QFileInfo fileInfo(filename);
  m_statusLabel->setText(QString("Loaded: %1").arg(fileInfo.fileName()));
}

void MainWindow::onWindowLevelChanged(double window, double level) {
  m_windowLevelLabel->setText(QString("W/L: %1/%2")
                                  .arg(static_cast<int>(window))
                                  .arg(static_cast<int>(level)));

  // Synchronize Window/Level with WebServer for VR mode
  if (m_dicomViewer) {
    WebServer* webServer = m_dicomViewer->getWebServer();
    if (webServer) {
      webServer->setWindowLevel(window, level);
    }
  }
}

void MainWindow::onDoseLoadProgress(int current, int total) {
  if (total > 0) {
    m_progressBar->setRange(0, total);
    m_progressBar->setValue(current);
    QApplication::processEvents();
  }
}

void MainWindow::onStructureLoadProgress(int current, int total) {
  if (total > 0) {
    m_progressBar->setRange(0, total);
    m_progressBar->setValue(current);
    QApplication::processEvents();
  }
}

void MainWindow::updateWindowTitle(const QString &filename) {
  QString title = "ShioRIS3 - DICOM Viewer";

  if (!filename.isEmpty()) {
    QFileInfo fileInfo(filename);
    title += QString(" - %1").arg(fileInfo.fileName());
  }

  setWindowTitle(title);
}

void MainWindow::setSingleView() {
  if (m_dicomViewer) {
    m_dicomViewer->setViewMode(DicomViewer::ViewMode::Single);
  }
}

void MainWindow::setDualView() {
  if (m_dicomViewer) {
    m_dicomViewer->setViewMode(DicomViewer::ViewMode::Dual);
  }
}

void MainWindow::setQuadView() {
  if (m_dicomViewer) {
    m_dicomViewer->setViewMode(DicomViewer::ViewMode::Quad);
  }
}

void MainWindow::setFiveView() {
  if (m_dicomViewer) {
    m_dicomViewer->setViewMode(DicomViewer::ViewMode::Five);
  }
}

void MainWindow::selectCustomTextColor() {
  ThemeManager &themeManager = ThemeManager::instance();
  QColor currentColor = themeManager.textColor();

  // Open color picker dialog
  QColor color = QColorDialog::getColor(
      currentColor, this, tr("Select Text Color"),
      QColorDialog::ShowAlphaChannel | QColorDialog::DontUseNativeDialog);

  if (color.isValid()) {
    themeManager.setCustomTextColor(color);
  } else {
    // User cancelled - revert to previous theme selection
    updateTextThemeSelection(themeManager.currentTheme());
  }
}
