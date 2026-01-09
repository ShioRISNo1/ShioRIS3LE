#ifndef DICOM_VIEWER_H
#define DICOM_VIEWER_H

#include "visualization/collapsible_group_box.h"
#include "visualization/opengl_3d_widget.h"
#include "visualization/opengl_image_widget.h"
#include "brachy/brachy_dose_calculator.h"
#include "brachy/dwell_time_optimizer.h"
#include <QCheckBox>
#include <QComboBox>
#include <QColor>
#include <QCursor>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QEvent>
#include <QFormLayout>
#include <QFutureWatcher>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHash>
#include <QHeaderView>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMap>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter> // カラーバー描画用に追加
#include <QJsonArray>
#include <QJsonObject>
#include <QScopedValueRollback>
#include <QPlainTextEdit>
#include <QPointF>
#include <QPushButton>
#include <QScrollArea>
#include <QSet>
#include <QSlider>
#include <QSpinBox>
#include <QStackedLayout>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QToolButton>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVBoxLayout>
#include <QVector>
#include <QVector3D>
#include <QWheelEvent>
#include <QWidget>
#include <array>
#include <memory>
#include <optional>
#include <vector>

class QProgressBar;

// Forward declaration to avoid heavy header include
namespace cv {
class VideoCapture;
}

#include "dicom/brachy_plan.h"
#include "dicom/dicom_reader.h"
#include "dicom/dicom_volume.h"
#include "dicom/dose_resampled_volume.h"
#include "dicom/dpsd_calculator.h"
#include "dicom/dvh_calculator.h"
#include "dicom/rtdose_volume.h"
#include "dicom/rtstruct.h"
#include "visualization/dvh_window.h"

class DoseItemWidget;
class DoseProfileWindow;
class MultiRowTabWidget;
class GammaAnalysisWindow;
class DatabaseManager;

// 線量変換モード
enum class DoseCalcMode {
  Physical, // 物理線量
  BED,
  EqD2
};

// カラーバー描画用のカスタムウィジェット
class DoseColorBar : public QWidget {
  Q_OBJECT

public:
  explicit DoseColorBar(QWidget *parent = nullptr);

  void setDoseRange(double minDose, double maxDose);
  void setDisplayMode(DoseResampledVolume::DoseDisplayMode mode);
  void setReferenceDose(double referenceDose);
  void setVisible(bool visible) override; // override を追加
  int preferredWidth() const;

protected:
  void paintEvent(QPaintEvent *event) override;
  QSize sizeHint() const override;

private:
  double m_minDose{0.0};
  double m_maxDose{1.0};
  double m_referenceDose{1.0};
  DoseResampledVolume::DoseDisplayMode m_displayMode{
      DoseResampledVolume::DoseDisplayMode::Colorful};
  bool m_isVisible{false};

  // カラーマッピング関数
  QRgb mapDoseToColor(float doseRatio) const;
  QRgb mapDoseToIsodose(float doseRatio) const;
  QRgb mapDoseToHot(float doseRatio) const;
};

struct DicomStudyInfo {
  QString patientID;
  QString patientName;
  QString modality;
  QString studyDescription;
  QString studyDate;
  QString frameOfReferenceUID;
  QString seriesDirectory;
  bool isValid() const { return !seriesDirectory.isEmpty(); }
};

class DicomViewer : public QWidget {
  Q_OBJECT

public:
  static constexpr int VIEW_COUNT = 5;

  explicit DicomViewer(QWidget *parent = nullptr, bool showControls = true);
  ~DicomViewer();

  bool loadDicomFile(const QString &filename);
  bool loadDicomDirectory(const QString &directory, bool loadCt = true,
                          bool loadRtss = true, bool loadRtdose = true,
                          const QStringList &imageSeries = QStringList(),
                          const QStringList &modalities = QStringList(),
                          int activeSeriesIndex = 0);
  bool loadRTDoseFile(const QString &filename, bool activate = true);
  bool loadRTStructFile(const QString &filename);
  bool loadBrachyPlanFile(const QString &filename);
  void setDatabaseManager(DatabaseManager *dbManager);
  void setWindowLevel(double window, double level);
  void showNextImage();
  void showPreviousImage();
  void clearImage();
  // Reset all state related to a study (CT/Dose/RTSTRUCT/DVH)
  void resetStudyState();
  enum class ViewMode { Single, Dual, Quad, Five };
  void setViewMode(ViewMode mode);
  ViewMode viewMode() const { return m_viewMode; }
  void setFourViewMode(bool enabled); // compatibility
  const DicomVolume &getVolume() const { return m_volume; }
  bool isVolumeLoaded() const { return m_volume.depth() > 0; }
  const RTStructureSet &getRTStruct() const { return m_rtstruct; }
  bool isRTStructLoaded() const { return m_rtstructLoaded; }
  const QVector<DoseIsosurface>& getDoseIsosurfaces() const { return m_doseIsosurfaces; }
  class WebServer* getWebServer() const { return m_webServer; }
  DicomStudyInfo currentStudyInfo() const;
  void setSyncScale(bool enabled) {
    m_syncScale = enabled;
    updateSyncedScale();
  }
  bool isSyncScale() const { return m_syncScale; }

  cv::Mat getCurrentVolume() const;

  void setFusionPreviewImage(const QImage &image, double spacingX,
                             double spacingY);
  void clearFusionPreviewImage();
  bool showExternalImageSeries(const QString &directory,
                               const QString &modality,
                               const DicomVolume &volume, double window,
                               double level);

signals:
  void imageLoaded(const QString &filename);
  void windowLevelChanged(double window, double level);
  void doseLoadProgress(int current, int total);
  void structureLoadProgress(int current, int total);

private slots:
  void onWindowChanged(int value);
  void onLevelChanged(int value);
  void onZoomIn();
  void onZoomOut();
  void onResetZoom();
  void onFitToWindow();
  void onWindowLevelButtonClicked();
  void onWindowLevelTimeout();
  void onPanTimeout();
  void onZoomTimeout();
  void onImageSliderChanged(int value);
  void onPanModeToggled(bool checked);
  void onZoomModeToggled(bool checked);
  void onDoseDisplayModeChanged(); // 新規追加
  void onDoseRangeEditingFinished();
  void onDoseOpacityChanged(int value);
  void onDoseCalculateClicked();
  void onDoseIsosurfaceClicked();
  void onRandomStudyClicked();
  void onGammaAnalysisClicked();
  void onDoseListContextMenu(const QPoint &pos);
  void onDoseSaveRequested(DoseItemWidget *widget);
  void onStructureVisibilityChanged(QListWidgetItem *item);
  void onShowAllStructures();
  void onHideAllStructures();
  void onStructureLineWidthChanged(int value);
  void onStructurePointsToggled(bool checked);
  void onShowDVH();
  void onDvhCalculationRequested(const QString &roiName);
  void onDvhVisibilityChanged(int roiIndex, bool visible);
  void onReadBrachyPlan();
  void onLoadBrachyData();
  void onCalculateBrachyDose();
  void onAddDoseEvaluationPoint();
  void onClearDoseEvaluationPoints();
  void onOptimizeDwellTimes();
  void onGenerateRandomSources();
  void onGenerateTestSource();
  bool loadBrachySourceData(const QString &filename);
  void autoLoadBrachySourceData();
  void updateReferencePointsDisplay(
      const QVector<Brachy::ReferencePointError> &errors = QVector<Brachy::ReferencePointError>());
  void showReferencePointErrorDialog(const QVector<Brachy::ReferencePointError> &errors,
                                      double normalizationFactor);
  void onShowRefPointsChanged(int state);
  void onDvhCalcMaxChanged(double calcMaxGy);
  void onProfileLineSelection(int viewIndex);
  void onProfileLineSaveRequested(int viewIndex, int slotIndex);
  void onProfileLineLoadRequested(int viewIndex, int slotIndex);
  void onImageDoubleClicked(int viewIndex);
  void onSlicePositionToggled(bool checked);
  void onExportButtonClicked(int viewIndex);
  void onGenerateQr();
  void onDecodeQrFromImage();
  void onSaveQrImage();
  void onClearQr();
  void onStartQrCamera();
  void onStopQrCamera();
  void onQrCameraTick();

protected:
  void wheelEvent(QWheelEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void keyPressEvent(QKeyEvent *event) override;
  void enterEvent(QEnterEvent *event) override;
  void leaveEvent(QEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;
  bool eventFilter(QObject *obj, QEvent *event) override;

private:
  void setupUI();
  void setBrachyStatusStyle(const QString &borderColor);
  void updateImage();
  void updateImage(int viewIndex, bool updateStructure = true);
  void update3DView(int viewIndex);
  void updateImageInfo();
  void loadSlice(int viewIndex, int sliceIndex);
  void loadVolumeSlice(int viewIndex, int sliceIndex);
  int sliceCountForOrientation(DicomVolume::Orientation ori) const;
  void showOrientationMenu(int viewIndex, const QPoint &pos);
  void showJumpToMenu(int viewIndex, const QPoint &pos);
  void setZoomFactor(double factor);
  void startWindowLevelDrag();
  void stopWindowLevelDrag();
  void updateWindowLevelFromMouse(const QPoint &currentPos);
  void updateSliderPosition();
  void updateViewLayout();
  void updateSliceLabels();
  void updateInfoOverlays();
  QString patientInfoText() const;
  void updateCoordLabel(int viewIndex, const QPoint &widgetPos);
  QVector3D patientCoordinateAt(int viewIndex, const QPoint &widgetPos) const;
  QPointF planeCoordinateFromPatient(int viewIndex,
                                     const QVector3D &patient) const;
  float sampleResampledDose(const QVector3D &voxel) const;
  void regenerateStructureSurfaceCache();
  void invalidateStructureSurfaceCache();
  std::optional<double> sampleCtValue(const QVector3D &voxel) const;
  std::optional<double> sampleDoseValue(const QVector3D &voxel) const;
  void computeDoseProfile();
  void loadProfileLinePresets();
  void writeProfileLineSlot(int viewIndex, int slotIndex);
  void updateProfileSlotButtons(int viewIndex);
  int viewIndexFromGlobalPos(const QPoint &globalPos) const;
  void updateDoseAlignment();
  void updateColorBars(); // 新規追加
  void updateDoseShiftLabels();
  void resetDoseRange();

  // Colormap helpers for dose display
  QColor getDoseColor(double doseGy, double maxDoseGy) const;
  QRgb mapDoseToColorHSV(float doseRatio) const;
  QRgb mapDoseToIsodose(float doseRatio) const;
  QRgb mapDoseToHot(float doseRatio) const;

  void syncZoomToAllViews(double zoomFactor);
  void copyDoseAt(int index);
  void deleteDoseAt(int index);
  void updateOrientationButtonTexts();
  void showOrientationMenuForView(int viewIndex);
  void updateImageSeriesButtons();
  void showImageSeriesMenu(int viewIndex);
  bool switchToImageSeries(int index);
  void updateSyncedScale();
  void setViewToImage(int viewIndex);
  void setViewTo3D(int viewIndex);
  void setViewToDVH(int viewIndex);
  void setViewToProfile(int viewIndex);
  bool switchViewContentFromString(int viewIndex, const QString &mode);
  int visibleViewCount() const;
  bool isViewIndexVisible(int index) const;
  int clampToVisibleViewIndex(int index) const;
  int findVisibleDvhView() const;
  int activeOrDefaultViewIndex(int fallback = 0) const;
  void loadAiSettings();
  void saveAiSettings() const;
  void loadAiPromptDiagnostics();
  void saveAiPromptDiagnostics() const;
  void appendAiLog(const QString &message);
  void handleAiCommands(const QJsonArray &commands);
  bool shouldAutoExecuteCommands(const QJsonArray &commands,
                                 QStringList *blockingReasons = nullptr) const;
  bool executeAiCommand(const QJsonObject &command);
  bool ensureDvhDataReady(int roiIndex, DVHCalculator::DVHData **outData = nullptr);
  int findRoiIndex(const QString &roiName) const;
  QString formatDvhMetrics(const DVHCalculator::DVHData &data,
                           const QStringList &metrics) const;
  double doseAtVolumePercent(const DVHCalculator::DVHData &data,
                             double volumePercent) const;
  double volumeAtDoseGy(const DVHCalculator::DVHData &data,
                        double doseGy) const;
  void refreshDvhWidgets();
  bool runDpsdAnalysis(const QString &roiName, const QString &sampleRoiName,
                       double startMm, double endMm,
                       DPSDCalculator::Mode mode);
  QString sanitizeAiJsonText(const QString &rawText) const;
  bool isAiPromptOutdated(const QString &prompt) const;
  struct AiCommandStep {
    enum class Status { Pending, Running, Success, Failed, RolledBack };
    QJsonObject command;
    Status status{Status::Pending};
    QString message;
  };
  struct AiExecutionSnapshot {
    bool valid{false};
    bool hasDataset{false};
    QString datasetPath;
    bool datasetIsDirectory{false};
    ViewMode viewMode{ViewMode::Single};
    QStringList viewContents;
    int activeViewIndex{0};
    double zoomFactor{1.0};
    bool syncScale{true};
    bool rtstructLoaded{false};
    QString rtstructPath;
    QVector<int> structureCheckStates;
  };
  struct RoiListItemSnapshot {
    QString text;
    Qt::CheckState checkState{Qt::Unchecked};
    QIcon icon;
  };
  class ScopedRoiUiReadGuard;
  friend class ScopedRoiUiReadGuard;

  struct RoiUiSnapshot {
    bool hasStructureList{false};
    QVector<RoiListItemSnapshot> items;
    int currentRow{-1};
  };
  void clearAiCommandList();
  void populateAiCommandList();
  void updateAiCommandStatus(int index, AiCommandStep::Status status,
                             const QString &message = QString());
  QString aiCommandDescription(const QJsonObject &command) const;
  QString aiCommandStatusText(AiCommandStep::Status status) const;
  QColor aiCommandStatusColor(AiCommandStep::Status status) const;
  void startAiCommandExecution();
  AiExecutionSnapshot captureAiViewerSnapshot() const;
  void restoreAiViewerSnapshot(const AiExecutionSnapshot &snapshot);
  RoiUiSnapshot captureRoiUiSnapshot() const;
  void restoreRoiUiSnapshot(const RoiUiSnapshot &snapshot);
  QString viewContentToString(int viewIndex) const;
  bool applyViewContentFromString(int viewIndex, const QString &content);
  void rollbackToSnapshot(const AiExecutionSnapshot &snapshot);
  void loadAiMacros();
  void persistAiMacros() const;
  void refreshAiMacroCombo();
  QJsonObject aiExecutionSnapshotToJson(
      const AiExecutionSnapshot &snapshot) const;
  QString buildAiViewerContextJson() const;
  QString buildAiViewerContextSection() const;
  QString buildAiFeedbackSection() const;
  QString resolveAiSystemPromptTemplate(const QString &templateText) const;
  QString defaultAiSystemPromptTemplate() const;
  QStringList aiPromptFeedbackMessages() const;
  void recordAiPromptFailure(const QString &category,
                             const QString &detail = QString());
  void collectMacroPlaceholders(const QJsonValue &value,
                                QSet<QString> *placeholders) const;
  QJsonValue applyMacroParameters(const QJsonValue &value,
                                  const QHash<QString, QString> &parameters) const;
  QJsonArray instantiateMacro(const QString &macroName,
                              const QJsonArray &commands,
                              const QStringList &placeholders);
  QStringList normalizePlaceholderList(const QSet<QString> &placeholders) const;
  QString macroDescriptionForSave(const QString &name) const;
  void fetchAiModelList(bool force = false);
  ViewMode m_previousViewMode{ViewMode::Single};
  int m_fullScreenViewIndex{0};
  bool m_hasSavedView0{false};
  QImage m_savedImage0;
  int m_savedIndex0{0};
  DicomVolume::Orientation m_savedOrientation0{DicomVolume::Orientation::Axial};
  QPointF m_savedPanOffset0;
  bool m_savedIsDVH0{false};
  bool m_savedIs3D0{false};
  int m_savedSliderMin0{0};
  int m_savedSliderMax0{0};
  int m_savedSliderValue0{0};
  bool m_savedSliderEnabled0{false};
  std::vector<DVHCalculator::DVHData> m_savedDVHData0;
  double m_savedPrescriptionDose0{0.0};
  bool m_savedXAxisPercent0{false};
  bool m_savedYAxisCc0{false};
  int m_savedCurrentRoiIndex0{-1};

  int getCurrentSliceIndex() const;

  cv::Mat getSingleSliceAsMatrix() const;
  cv::Mat buildVolumeFromFiles() const;
  // UI components
  QVBoxLayout *m_mainLayout;
  QHBoxLayout *m_controlLayout;
  QScrollArea *m_scrollArea;
  QWidget *m_gridWidget;
  QWidget *m_imageContainer;
  QGridLayout *m_imageLayout;
  QWidget *m_viewContainers[VIEW_COUNT];
  QScrollArea *m_viewScrollAreas[VIEW_COUNT];
  OpenGLImageWidget *m_imageWidgets[VIEW_COUNT];
  QSlider *m_sliceSliders[VIEW_COUNT];
  QLabel *m_sliceIndexLabels[VIEW_COUNT];
  QLabel *m_infoOverlays[VIEW_COUNT];
  QLabel *m_coordLabels[VIEW_COUNT];
  QLabel *m_cursorDoseLabels[VIEW_COUNT];
  QVector3D m_lastPatientCoordinates[VIEW_COUNT]; // Last displayed patient coordinates
  QLabel *m_doseShiftLabels[VIEW_COUNT]{nullptr};
  DoseColorBar *m_colorBars[VIEW_COUNT]; // 新規追加
  bool m_colorBarPersistentVisibility[VIEW_COUNT];
  MultiRowTabWidget *m_rightTabWidget{nullptr};
  QWidget *m_doseManagerPanel{nullptr};
  QListWidget *m_doseListWidget{nullptr};
  QPushButton *m_randomStudyButton{nullptr};
  class WebServer *m_webServer{nullptr};
  QWidget *m_qrPanel{nullptr};
  QPlainTextEdit *m_qrTextEdit{nullptr};
  QCheckBox *m_qrEscapeCheck{nullptr};
  QCheckBox *m_qrUtf8EciCheck{nullptr};
  QPushButton *m_qrGenerateButton{nullptr};
  QPushButton *m_qrDecodeImageButton{nullptr};
  QPushButton *m_qrSaveButton{nullptr};
  QPushButton *m_qrClearButton{nullptr};
  QLabel *m_qrImageLabel{nullptr};
  QPushButton *m_qrStartCamButton{nullptr};
  QPushButton *m_qrStopCamButton{nullptr};
  QTimer *m_qrCamTimer{nullptr};
  bool m_qrCamRunning{false};
  std::unique_ptr<cv::VideoCapture> m_qrCapture;
  QImage m_lastQrImage;
  QCheckBox *m_qrHighAccuracyCheck{nullptr};
  QElapsedTimer m_qrTimer;
  int m_qrDecodeIntervalMs{200};
  QWidget *m_brachyPanel{nullptr};
  QPushButton *m_brachyReadButton{nullptr};
  QListWidget *m_brachyListWidget{nullptr};
  QPushButton *m_brachyLoadDataButton{nullptr};
  QPushButton *m_brachyCalcDoseButton{nullptr};
  QPushButton *m_brachyRandomSourceButton{nullptr};
  QPushButton *m_brachyTestSourceButton{nullptr};
  QProgressBar *m_brachyProgressBar{nullptr};
  QLabel *m_brachyDataStatus{nullptr};
  QString m_brachyStatusBorderColor{QStringLiteral("#666666")};
  QDoubleSpinBox *m_brachyVoxelSizeSpinBox{nullptr};

  // Dose optimization UI
  QPushButton *m_brachyAddEvalPointButton{nullptr};
  QPushButton *m_brachyClearEvalPointsButton{nullptr};
  QPushButton *m_brachyOptimizeButton{nullptr};
  QListWidget *m_brachyEvalPointsList{nullptr};
  QSpinBox *m_brachyOptimizationIterations{nullptr};
  QDoubleSpinBox *m_brachyOptimizationTolerance{nullptr};

  // Reference Points UI
  QListWidget *m_brachyRefPointsList{nullptr};
  QCheckBox *m_brachyShowRefPointsCheck{nullptr};


  // Window/Level controls
  CollapsibleGroupBox *m_windowLevelGroup;
  QSlider *m_windowSlider;
  QSlider *m_levelSlider;
  QSpinBox *m_windowSpinBox;
  QSpinBox *m_levelSpinBox;
  QPushButton *m_windowLevelButton;
  QCheckBox *m_slicePositionCheck{nullptr};

  // Zoom controls
  CollapsibleGroupBox *m_zoomGroup;
  // Zoom in/out buttons removed
  QPushButton *m_resetZoomButton;
  QPushButton *m_fitToWindowButton;

  // Dose display controls (新規追加)
  CollapsibleGroupBox *m_doseGroup;
  QComboBox *m_doseColorMapCombo{nullptr};

  // Dose range controls (統合)
  QDoubleSpinBox *m_doseMinSpinBox{nullptr};
  QDoubleSpinBox *m_doseMaxSpinBox{nullptr};
  QSlider *m_doseOpacitySlider{nullptr};

  // Image info
  CollapsibleGroupBox *m_infoGroup;
  QPlainTextEdit *m_infoTextBox;
  QToolButton *m_privacyButton;
  QDoubleSpinBox *m_doseRefSpinBox{nullptr};
  QPushButton *m_doseRefMaxButton{nullptr};
  double m_doseReference{0.0};
  DoseCalcMode m_doseCalcMode{DoseCalcMode::Physical};
  double m_doseAlphaBeta{10.0};
  QComboBox *m_doseModeCombo{nullptr};
  QDoubleSpinBox *m_doseAlphaBetaSpin{nullptr};
  QPushButton *m_doseCalcButton{nullptr};
  QPushButton *m_doseIsosurfaceButton{nullptr};
  QPushButton *m_gammaAnalysisButton{nullptr};
  GammaAnalysisWindow *m_gammaAnalysisWindow{nullptr};
  QVector<DoseIsosurface> m_doseIsosurfaces;
  double transformDose(double rawDose, double dataFr, double displayFr) const;
  QVector3D computeAlignmentShift(const RTDoseVolume &dose) const;
  QPushButton *m_panButton;
  QPushButton *m_zoomButton;
  CollapsibleGroupBox *m_structureGroup{nullptr};
  QListWidget *m_structureList{nullptr};
  QPushButton *m_structureAllButton{nullptr};
  QPushButton *m_structureNoneButton{nullptr};
  QPushButton *m_dvhButton{nullptr};
  QCheckBox *m_showPointsCheck{nullptr};
  QSpinBox *m_structureLineWidthSpin{nullptr};
  int m_structureLineWidth{1};
  bool m_showStructurePoints{false};
  bool m_showSlicePosition{false};

  bool m_privacyMode{false};

  // Data
  std::unique_ptr<DicomReader> m_dicomReader;
  double m_zoomFactor;
  QImage m_originalImages[VIEW_COUNT];
  QStringList m_dicomFiles;
  QString m_ctFilename;
  QString m_rtDoseFilename;
  int m_currentIndices[VIEW_COUNT];
  int m_activeViewIndex{0};

  DicomVolume m_volume;
  RTDoseVolume m_doseVolume;
  DoseResampledVolume m_resampledDose;
  struct BrachyDoseResult {
    bool success{false};
    RTDoseVolume dose;
    QString errorMessage;
    double normalizationFactor{1.0};
    QVector<Brachy::ReferencePointError> referencePointErrors;
  };
  struct DoseItem {
    DoseResampledVolume volume;
    DoseItemWidget *widget{nullptr};
    RTDoseVolume dose; // original RT Dose for re-resampling with shifts
    bool isSaved{false}; // whether this dose has been saved to file
    QString savedFilePath; // file path if saved
  };
  std::vector<DoseItem> m_doseItems;
  DatabaseManager *m_databaseManager{nullptr};
  bool m_doseLoaded{false};
  bool m_doseVisible{true};
  RTStructureSet m_rtstruct;
  bool m_rtstructLoaded{false};
  QString m_lastRtStructPath;
  BrachyPlan m_brachyPlan;
  bool m_brachyLoaded{false};
  std::unique_ptr<Brachy::BrachyDoseCalculator> m_brachyDoseCalc;
  std::unique_ptr<Brachy::DwellTimeOptimizer> m_brachyOptimizer;
  QStackedLayout *m_viewStacks[VIEW_COUNT]{nullptr};
  QWidget *m_imagePanels[VIEW_COUNT]{nullptr};
  DVHWindow *m_dvhWidgets[VIEW_COUNT]{nullptr};
  OpenGL3DWidget *m_3dWidgets[VIEW_COUNT]{nullptr};
  bool m_isDVHView[VIEW_COUNT]{false};
  bool m_is3DView[VIEW_COUNT]{false};
  DoseProfileWindow *m_profileWidgets[VIEW_COUNT]{nullptr};
  bool m_isProfileView[VIEW_COUNT]{false};
  bool m_selectingProfileLine{false};
  bool m_profileLineHasStart{false};
  int m_profileSelectingView{-1};
  int m_profileRequester{-1};
  QVector3D m_profileStartPatient;
  QVector3D m_profileEndPatient;
  StructureLine m_profileLine;
  bool m_profileLineVisible{false};
  int m_profileLineView{-1};
  bool m_dragProfileStart{false};
  bool m_dragProfileEnd{false};
  struct SavedProfileLine {
    bool valid{false};
    QVector3D startPatient;
    QVector3D endPatient;
  };
  std::array<std::array<SavedProfileLine, 3>, VIEW_COUNT> m_savedProfileLines{};
  DVHCalculator m_dvhCalculator;
  std::vector<DVHCalculator::DVHData> m_dvhData;
  QHash<int, QFutureWatcher<DVHCalculator::DVHData> *> m_dvhWatchers;
  QVector3D m_doseShift{0.0, 0.0, 0.0};
  // currently applied dose display range
  double m_doseMinRange{0.0};
  double m_doseMaxRange{0.0};
  double m_doseOpacity{0.8};
  DicomVolume::Orientation m_viewOrientations[VIEW_COUNT];
  QImage m_orientationImages[3];
  int m_orientationIndices[3]{0, 0, 0};
  DoseResampledVolume::DoseDisplayMode m_doseDisplayMode{
      DoseResampledVolume::DoseDisplayMode::Colorful}; // 新規追加

  // Window/Level drag functionality
  bool m_windowLevelDragActive;
  QPoint m_dragStartPos;
  double m_dragStartWindow;
  double m_dragStartLevel;
  QTimer *m_windowLevelTimer;
  QTimer *m_panTimer;
  QTimer *m_zoomTimer;

  // Pan functionality
  bool m_panDragActive{false};
  bool m_panMode{false};
  QPoint m_panStartPos;
  QPointF m_panOffsets[VIEW_COUNT];

  // Zoom functionality
  bool m_zoomDragActive{false};
  bool m_zoomMode{false};
  QPoint m_zoomStartPos;
  double m_zoomStartFactor{1.5};
  bool m_rotationDragActive{false};
  QPoint m_rotationStartPos;

  ViewMode m_viewMode;
  bool m_fourViewMode;
  bool m_showControls;

  // Constants
  static constexpr double MIN_ZOOM = 0.1;
  static constexpr double MAX_ZOOM = 10.0;
  static constexpr double MAX_ZOOM_3D = 25.0;
  static constexpr double ZOOM_STEP = 1.2;
  static constexpr double WINDOW_LEVEL_SENSITIVITY = 2.0;
  static constexpr int WINDOW_LEVEL_TIMEOUT = 2500;

  QPushButton *m_orientationButtons[VIEW_COUNT]{
      nullptr}; // 各ビューの方向変更ボタン
  QPushButton *m_imageToggleButtons[VIEW_COUNT]{nullptr}; // 3D画像表示切り替えボタン
  QPushButton *m_lineToggleButtons[VIEW_COUNT]{nullptr}; // 3D Structure Line表示切り替えボタン
  QPushButton *m_exportButtons[VIEW_COUNT]{nullptr}; // 3D VisionPro USDZ export buttons
  QPushButton *m_imageSeriesButtons[VIEW_COUNT]{nullptr};
  QPushButton *m_viewWindowLevelButtons[VIEW_COUNT]{nullptr};
  QPushButton *m_viewPanButtons[VIEW_COUNT]{nullptr};
  QPushButton *m_viewZoomButtons[VIEW_COUNT]{nullptr};
  bool m_show3DImages[VIEW_COUNT]{true}; // 3D view CT image visibility
  bool m_show3DLines[VIEW_COUNT]{true};  // 3D view slice line visibility
  QVector<StructureSurface> m_cachedStructureSurfaces;
  bool m_structureSurfacesDirty{true};
  QStringList m_imageSeriesDirs;
  QStringList m_imageSeriesModalities;
  int m_activeImageSeriesIndex{0};
  int m_primaryImageSeriesIndex{0};
  bool m_aiSuppressSourceTracking{false};
  bool m_aiHasDicomSource{false};
  bool m_aiCurrentDicomSourceIsDirectory{false};
  QString m_aiCurrentDicomSourcePath;
  struct ImageSeriesCacheEntry {
    bool prepared{false};
    DicomVolume volume;
  };
  std::vector<ImageSeriesCacheEntry> m_seriesVolumeCache;

  void updateInteractionButtonVisibility(int viewIndex);
  void updateOverlayInteractionStates();
  void onViewWindowLevelToggled(int viewIndex, bool checked);
  void onViewPanToggled(int viewIndex, bool checked);
  void onViewZoomToggled(int viewIndex, bool checked);
  void onImageToggleClicked(int viewIndex);
  void onLineToggleClicked(int viewIndex);
  QVector<double> m_seriesWindowValues;
  QVector<double> m_seriesLevelValues;
  QVector<bool> m_seriesWindowLevelInitialized;

  bool m_fusionViewActive{false};
  QImage m_fusionViewImage;
  double m_fusionSpacingX{1.0};
  double m_fusionSpacingY{1.0};
  ViewMode m_viewModeBeforeFusion{ViewMode::Single};
  bool m_restoreViewModeAfterFusion{false};

  bool m_syncScale{true};
  // Always sample RTDose in native patient coordinates (ignore patientShift)
  // Checkbox removed per request; native is enforced.
  QCheckBox *m_nativeDoseCheck{nullptr};
  QCheckBox *m_doseGuideCheck{nullptr};
  bool m_showDoseGuide{false};

  // Generic visualization grid overlay (mm-based)
  QCheckBox *m_gridCheck{nullptr};
  bool m_showGrid{false};

  void initializeSeriesWindowLevel(int index, const QString &directory);
  bool ensureImageSeriesVolume(int index);
};

#endif // DICOM_VIEWER_H
