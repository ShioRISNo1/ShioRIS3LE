#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMenuBar>
#include <QStatusBar>
#include <QAction>
#include <QFileDialog>
#include <QMessageBox>
#include <QProgressBar>
#include <QLabel>
#include <QToolBar>

#include <QTableWidget>
#include <QTableWidgetItem>
#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QHeaderView>
#include <QTextEdit>

#include "visualization/dicom_viewer.h"
#include "visualization/volume_viewer_window.h"
#include "database/database_manager.h"
#include "database/smart_scanner.h"
#include "database/database_sync_manager.h"
#include "data/metadata_generator.h"
#include "visualization/data_window.h"
#include "visualization/auto_segmentation_dialog.h"
#include "visualization/fusion_dialog.h"
#include "visualization/translator_window.h"
#include "cyberknife/dose_calculator.h"
#include "theme_manager.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void openDicomFile();
    void openDicomFolder();
    void openDicomVolume();
    void openRTDoseFile();
    void openRTStructFile();
    void loadCyberKnifeBeamData();
    void exportCyberKnifeCsvBundle();
    void exitApplication();
    void aboutApplication();
    void onImageLoaded(const QString& filename);
    void onWindowLevelChanged(double window, double level);
    void onDoseLoadProgress(int current, int total);
    void onStructureLoadProgress(int current, int total);
    void setSingleView();
    void setDualView();
    void setQuadView();
    void setFiveView();
    void startAutoSegmentation();
    void openFusionDialog();
    void selectCustomTextColor();

//    void startAutoSegmentation();
    void applySegmentationResult(const cv::Mat &segmentationResult);
    void onSegmentationFinished(const cv::Mat &result);

  //  void startAutoSegmentation();

private:
    void setupUI();
    void setupMenus();
    void setupStatusBar();
    void initializeCyberKnifeDoseCalculator();
    void updateWindowTitle(const QString& filename = QString());
    void showDataWindow();
    void showTranslatorWindow();

    QString getVolumeInfoString(const cv::Mat &volume);
    QString getSegmentationStatsString(const cv::Mat &segmentation);
    void showSegmentationStatistics(const cv::Mat &segmentation);

    void showAIFeaturesDialog();
    void showAIUsageGuide();

    QString getVolumeInfoString();
    void updateCyberKnifeExportActions();
    void updateTextThemeSelection(ThemeManager::TextTheme theme);

    // UI components
    DicomViewer* m_dicomViewer;
    
    // Menus
    QMenuBar* m_menuBar;
    QMenu* m_fileMenu;
    QMenu* m_cyberKnifeExportMenu{nullptr};
    QMenu* m_cyberKnifeMenu{nullptr};
    QMenu* m_aiMenu;
    QMenu* m_appearanceMenu{nullptr};
    QMenu* m_textColorMenu{nullptr};
    QMenu* m_databaseMenu;
    QMenu* m_helpMenu;
    
    // Actions
    QAction* m_openAction;
    QAction* m_openFolderAction;
    QAction* m_openVolumeAction;
    QAction* m_openRTDoseAction;
    QAction* m_openRTStructAction;
    QAction* m_loadCyberKnifeBeamDataAction;
    QAction* m_exportCyberKnifeCsvAction{nullptr};
    QAction* m_exitAction;
    QAction* m_aboutAction;
    QAction* m_openDataWindowAction;
    QAction* m_singleViewAction;
    QAction* m_dualViewAction;
    QAction* m_quadViewAction;
    QAction* m_fiveViewAction;
    QAction* m_autoSegmentationAction;
    QAction* m_openDatabaseAction;
    QAction* m_openFusionDialogAction;
    QAction* m_textColorWhiteAction{nullptr};
    QAction* m_textColorGreenAction{nullptr};
    QAction* m_textColorDarkRedAction{nullptr};
    QAction* m_textColorCustomAction{nullptr};
    QAction* m_openTranslatorAction{nullptr};
    
    // Status bar
    QStatusBar* m_statusBar;
    QLabel* m_statusLabel;
    QLabel* m_windowLevelLabel;
    QLabel* m_zoomLabel;
    QProgressBar* m_progressBar;
    QLabel* m_cyberKnifeStatusLabel{nullptr};

    VolumeViewerWindow* m_volumeWindow;
    DataWindow* m_dataWindow{nullptr};
    AutoSegmentationDialog* m_autoSegDialog{nullptr};
    FusionDialog* m_fusionDialog{nullptr};
    TranslatorWindow* m_translatorWindow{nullptr};

    QString m_currentFilename;

    // Data/DB managers
    DatabaseManager m_dbManager{DatabaseManager::defaultDataRoot()};
    SmartScanner m_scanner{m_dbManager};
    DatabaseSyncManager* m_syncManager{nullptr};
    MetadataGenerator m_metadata{m_dbManager};

    CyberKnife::CyberKnifeDoseCalculator m_cyberKnifeDoseCalculator;
};

#endif // MAINWINDOW_H
