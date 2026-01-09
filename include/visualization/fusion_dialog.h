#ifndef FUSION_DIALOG_H
#define FUSION_DIALOG_H

#include <QDialog>
#include <QImage>
#include <QVector3D>
#include <QWidget>
#include <QString>
#include <QStringList>
#include <QFutureWatcher>
#include <array>
#include <atomic>
#include <vector>

class QComboBox;
class QPushButton;
class QLabel;
class QSlider;
class QGridLayout;
class QDoubleSpinBox;
class QFormLayout;
class QGroupBox;
class QProgressBar;
class QTimer;
class DatabaseManager;
class DicomViewer;

#include "dicom/dicom_volume.h"
#include <QQuaternion>

struct FusionStudyRecord {
  int id{0};
  QString patientKey;
  QString patientName;
  QString modality;
  QString studyName;
  QString studyDate;
  QString path;
  QString normalizedPath;
  QString frameUid;
  QString normalizedFrameUid;
};

struct FusionTransferJobResult {
  enum class Error { None, ResampleFailed, VolumeCreationFailed };

  Error error{Error::None};
  DicomVolume volume;
};

class FusionSliceView : public QWidget {
  Q_OBJECT

public:
  explicit FusionSliceView(QWidget *parent = nullptr);

  void setBaseImage(const QImage &image);
  void setOverlayImage(const QImage &image);
  void setMessage(const QString &text);
  void setOrientationLabel(const QString &text);
  void setOrientation(DicomVolume::Orientation orientation);
  void setSliceIndex(int index);
  void setVolumeDimensions(int width, int height, int depth);
  void setROI(const QVector3D &minEdges, const QVector3D &maxEdges);
  void setROIEnabled(bool enabled);
  void setPixelSpacing(double horizontal, double vertical);
  void setZoomFactor(double factor);

signals:
  void roiChanging(const QVector3D &minEdges, const QVector3D &maxEdges);
  void roiChanged(const QVector3D &minEdges, const QVector3D &maxEdges);

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void leaveEvent(QEvent *event) override;
  bool hasHeightForWidth() const override;
  int heightForWidth(int w) const override;
  QSize sizeHint() const override;
  QSize minimumSizeHint() const override;

private:
  enum class DragMode {
    None,
    Move,
    ResizeLeft,
    ResizeRight,
    ResizeTop,
    ResizeBottom,
    ResizeTopLeft,
    ResizeTopRight,
    ResizeBottomLeft,
    ResizeBottomRight
  };

  struct AxisMapping {
    int axisH{0};
    int axisV{1};
    int axisPerp{2};
    bool invertH{false};
    bool invertV{false};
  };

  AxisMapping mappingForOrientation() const;
  QRectF computeTargetRect() const;
  QRectF roiRectInWidget(const QRectF &imageRect) const;
  double axisLength(int axis) const;
  bool mapPointToAxes(const QPointF &pt, double &outH, double &outV) const;
  double horizontalDeltaForPixels(double pixels) const;
  double verticalDeltaForPixels(double pixels) const;
  DragMode hitTestHandles(const QPointF &pos, const QRectF &roiRect) const;
  void updateHoverCursor(const QPointF &pos);

  QImage m_baseImage;
  QImage m_overlayImage;
  QString m_message;
  QString m_orientationLabel;
  DicomVolume::Orientation m_orientation{DicomVolume::Orientation::Axial};
  int m_sliceIndex{0};
  int m_volumeWidth{0};
  int m_volumeHeight{0};
  int m_volumeDepth{0};
  QVector3D m_roiMinEdges{0.0f, 0.0f, 0.0f};
  QVector3D m_roiMaxEdges{0.0f, 0.0f, 0.0f};
  bool m_roiEnabled{false};
  double m_horizontalSpacing{1.0};
  double m_verticalSpacing{1.0};
  double m_zoomFactor{1.0};
  DragMode m_dragMode{DragMode::None};
  QPointF m_dragStartPos;
  QVector3D m_dragStartMinEdges{0.0f, 0.0f, 0.0f};
  QVector3D m_dragStartMaxEdges{0.0f, 0.0f, 0.0f};
};

class FusionDialog : public QDialog {
  Q_OBJECT

public:
  explicit FusionDialog(DatabaseManager &db, DicomViewer *viewer,
                        QWidget *parent = nullptr);
  ~FusionDialog();
  void setPrimaryFromViewer(DicomViewer *viewer);

private slots:
  void onPrimaryStudyChanged(int index);
  void onSecondaryStudyChanged(int index);
  void onSliceSliderChanged(int orientationIndex, int value);
  void onPrimaryWindowChanged(int value);
  void onPrimaryLevelChanged(int value);
  void onSecondaryWindowChanged(int value);
  void onSecondaryLevelChanged(int value);
  void onOpacityChanged(int value);
  void onPrimaryDisplayModeChanged(int index);
  void onSecondaryDisplayModeChanged(int index);
  void onRefreshStudies();
  void onSendSecondaryToViewer();
  void onResetTransform();
  void onTranslationChanged(int axis, double value);
  void onRotationChanged(int axis, double value);
  void onAutoAlign();
  void onZoomSliderChanged(int orientationIndex, int value);
  void onSliceViewROIChanging(int viewIndex, const QVector3D &minEdges,
                              const QVector3D &maxEdges);
  void onSliceViewROIChanged(int viewIndex, const QVector3D &minEdges,
                             const QVector3D &maxEdges);
  void onTransferComputationFinished();

private:
  enum class PrimaryDisplayMode { Grayscale = 0, Inverted, BoneHighlight };
  enum class SecondaryDisplayMode { RedOverlay = 0, CyanOverlay, EdgeHighlight };

  using TransferJobResult = FusionTransferJobResult;

  struct SamplePoint {
    QVector3D voxel;
    QVector3D patient;
    double primaryValue{0.0};
    QVector3D primaryGradient{0.0f, 0.0f, 0.0f};
    double gradientMagnitude{0.0};
  };

  void setupUi();
  void setupTransformControls(QGroupBox *group);
  void loadStudyList();
  void populateStudyCombos();
  void updatePrimaryInfoDisplay();
  void selectSecondaryForPrimaryPatient();
  QString normalizedPatientKey(const QString &value) const;
  QString normalizedStudyPath(const QString &path) const;
  QString normalizedFrameUid(const QString &uid) const;
  QString studyRecordKey(const FusionStudyRecord &rec) const;
  bool studyDirectoryHasDicomFiles(const QString &path) const;
  bool studyRecordLooksValid(const FusionStudyRecord &rec) const;
  bool recordMatchesPrimaryPatient(const FusionStudyRecord &rec,
                                   const QString &primaryIdKey,
                                   const QString &primaryNameKey,
                                   const QString &primaryKeyKey) const;
  bool recordIsPrimaryStudy(const FusionStudyRecord &rec,
                            const QString &primaryPathKey,
                            const QString &primaryFrameUid) const;
  bool loadVolumeFromStudy(int studyIndex, DicomVolume &volume,
                           bool &loadedFlag, int &currentStudyIndex,
                           QLabel *infoLabel);
  void updateSliceRanges();
  void updateCenterShift();
  void updateImages();
  void updateStatusLabels();
  void updateTransferButtonState();
  void startTransferProgress(int totalSteps);
  void stopTransferProgress();
  void updateTransferProgressBar();
  void initializeDefaultROI();
  void disableROI();
  void propagateROIToViews();
  bool sanitizeROIEdges(QVector3D &minEdges, QVector3D &maxEdges) const;
  void applyROIEdges(const QVector3D &minEdges, const QVector3D &maxEdges,
                     bool finalChange);
  bool roiEnabled() const;
  QImage createBaseImage(const cv::Mat &slice) const;
  QImage createOverlayImage(const cv::Mat &slice) const;
  cv::Mat extractPrimarySlice(int sliceIndex,
                              DicomVolume::Orientation orientation) const;
  cv::Mat resampleSecondarySlice(int sliceIndex,
                                 DicomVolume::Orientation orientation) const;
  cv::Mat resampleSecondaryVolumeToPrimary() const;
  double sampleVolumeValue(const cv::Mat &volume, double x, double y,
                           double z) const;
  QVector3D sampleVolumeGradient(const cv::Mat &volume, double x, double y,
                                 double z) const;
  std::vector<SamplePoint> generatePrimarySamples(int maxSamples) const;
  double evaluateAlignmentCost(const QVector3D &manualTranslation,
                               const QQuaternion &rotation,
                               const std::vector<SamplePoint> &samples) const;
  QVector3D computeVolumeCenter(const DicomVolume &volume) const;
  QVector3D transformToSecondaryPatient(const QVector3D &primaryPoint) const;
  int orientationIndex(DicomVolume::Orientation orientation) const;
  DicomVolume::Orientation orientationFromIndex(int idx) const;
  QString orientationDisplayName(DicomVolume::Orientation orientation) const;
  std::array<double, 2>
  viewPixelSpacing(DicomVolume::Orientation orientation) const;
  bool loadFusionVolume(const FusionStudyRecord &rec, DicomVolume &volume,
                        double *outWindow, double *outLevel,
                        QStringList *infoLines);
  bool saveFusionResult(const DicomVolume &volume, QString &outDirectory,
                        QString &outModality);
  bool writeFusionMetadata(const QString &directory, const QString &modality,
                           const QString &volumeFileName,
                           const QString &seriesUid) const;
  bool recordFusionStudyInDatabase(const QString &patientKey,
                                   const QString &patientName,
                                   const QString &patientInfoPath,
                                   const QString &modality,
                                   const QString &directory,
                                   const QString &metaFileName,
                                   const QString &volumeFileName,
                                   const QString &seriesUid,
                                   int &outStudyId);
  void updatePrimaryPatientKey();

  DatabaseManager &m_db;
  DicomViewer *m_viewer{nullptr};
  DicomVolume m_primaryVolume;
  DicomVolume m_secondaryVolume;
  bool m_primaryLoaded{false};
  bool m_secondaryLoaded{false};
  QString m_primaryPatientKey;
  QString m_primaryPatientId;
  QString m_primaryPatientName;
  QString m_primaryStudyPath;
  QString m_primaryStudyDescription;
  QString m_primaryStudyDate;
  QString m_primaryModality;
  QString m_primaryFrameUid;
  QString m_secondaryStudyPath;
  QString m_secondaryModality;
  QVector3D m_centerShift{0.0, 0.0, 0.0};
  QVector3D m_manualTranslation{0.0, 0.0, 0.0};
  QVector3D m_manualRotation{0.0, 0.0, 0.0};
  QQuaternion m_rotationQuat;

  std::vector<FusionStudyRecord> m_studyRecords;
  int m_currentPrimaryStudy{-1};
  int m_currentSecondaryStudy{-1};

  QComboBox *m_primaryStudyCombo{nullptr};
  QComboBox *m_secondaryStudyCombo{nullptr};
  QLabel *m_primaryInfoLabel{nullptr};
  QLabel *m_secondaryInfoLabel{nullptr};
  QLabel *m_primaryUidLabel{nullptr};
  QLabel *m_secondaryUidLabel{nullptr};
  QPushButton *m_refreshButton{nullptr};
  QPushButton *m_autoAlignButton{nullptr};
  QPushButton *m_transferSecondaryButton{nullptr};
  QProgressBar *m_transferProgressBar{nullptr};
  QTimer *m_transferProgressTimer{nullptr};

  std::array<FusionSliceView *, 3> m_sliceViews{};
  std::array<QSlider *, 3> m_sliceSliders{};
  std::array<QLabel *, 3> m_sliceValueLabels{};
  std::array<QSlider *, 3> m_zoomSliders{};
  std::array<QLabel *, 3> m_zoomValueLabels{};
  std::array<double, 3> m_zoomFactors{{1.0, 1.0, 1.0}};

  QSlider *m_primaryWindowSlider{nullptr};
  QLabel *m_primaryWindowValueLabel{nullptr};
  QSlider *m_primaryLevelSlider{nullptr};
  QLabel *m_primaryLevelValueLabel{nullptr};
  QSlider *m_secondaryWindowSlider{nullptr};
  QLabel *m_secondaryWindowValueLabel{nullptr};
  QSlider *m_secondaryLevelSlider{nullptr};
  QLabel *m_secondaryLevelValueLabel{nullptr};
  QSlider *m_opacitySlider{nullptr};
  QLabel *m_opacityValueLabel{nullptr};
  QComboBox *m_primaryDisplayModeCombo{nullptr};
  QComboBox *m_secondaryDisplayModeCombo{nullptr};
  std::array<QDoubleSpinBox *, 3> m_translationSpins{};
  std::array<QDoubleSpinBox *, 3> m_rotationSpins{};

  double m_primaryWindow{400.0};
  double m_primaryLevel{40.0};
  double m_secondaryWindow{400.0};
  double m_secondaryLevel{40.0};
  double m_overlayOpacity{0.5};
  PrimaryDisplayMode m_primaryDisplayMode{PrimaryDisplayMode::Grayscale};
  SecondaryDisplayMode m_secondaryDisplayMode{SecondaryDisplayMode::RedOverlay};
  QVector3D m_roiMinEdges{0.0f, 0.0f, 0.0f};
  QVector3D m_roiMaxEdges{0.0f, 0.0f, 0.0f};
  bool m_roiActive{false};
  QFutureWatcher<TransferJobResult> m_transferWatcher;
  bool m_transferInProgress{false};
  QString m_pendingTransferPath;
  QString m_pendingTransferModality;
  double m_pendingTransferWindow{400.0};
  double m_pendingTransferLevel{40.0};
  QString m_transferButtonDefaultText;
  std::atomic<int> m_transferProgressCounter{0};
  int m_transferProgressTotal{0};
};

#endif // FUSION_DIALOG_H
