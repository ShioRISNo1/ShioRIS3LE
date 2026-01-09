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

  // Help メニュー
  m_helpMenu = m_menuBar->addMenu("&Help");

  m_aboutAction = new QAction("&About ShioRIS3", this);
  m_aboutAction->setStatusTip("About this application");
  connect(m_aboutAction, &QAction::triggered, this,
          &MainWindow::aboutApplication);

  m_helpMenu->addAction(m_aboutAction);
}

void MainWindow::showDataWindow() {
  if (!m_dataWindow) {
    // Create as top-level (no parent) so it doesn't hide under main window
    m_dataWindow = new DataWindow(m_dbManager, m_scanner, m_metadata, nullptr);
    connect(m_dataWindow, &DataWindow::openStudyRequested, this,
            [this](const QStringList &imageDirs, const QStringList &modalities,
                   const QStringList &rtssPaths,
                   const QStringList &rtdosePaths) {
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
                        rtdosePaths.size() +
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
                      if (!rtssPaths.isEmpty()) {
                        m_dicomViewer->loadRTStructFile(rtssPaths.first());
                        progress.setValue(++step);
                        QApplication::processEvents();
                      }
                      progress.close();
                    } else {
                      for (const QString &d : rtdosePaths)
                        loadDose(d);
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

void MainWindow::setupStatusBar() {
  m_statusBar = statusBar();

  // ステータスラベル
  m_statusLabel = new QLabel("Ready");
  m_statusBar->addWidget(m_statusLabel);

  // Window/Level表示
  m_windowLevelLabel = new QLabel("W/L: -/-");
  m_statusBar->addPermanentWidget(m_windowLevelLabel);

  // プログレスバー（将来の拡張用）
  m_progressBar = new QProgressBar();
  m_progressBar->setVisible(false);
  m_statusBar->addPermanentWidget(m_progressBar);
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
