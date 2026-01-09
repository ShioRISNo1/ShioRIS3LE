#ifndef DOSE_PROFILE_WINDOW_H
#define DOSE_PROFILE_WINDOW_H

#include "dicom/dicom_volume.h"
#include "dicom/dose_resampled_volume.h"
#include "dicom/dpsd_calculator.h"
#include "dicom/rtstruct.h"
#include <QColor>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFutureWatcher>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QCheckBox>
#include <QLabel>
#include <QResizeEvent>
#include <QShowEvent>
#include <QPair>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QPoint>
#include <QEvent>
#include <QStringList>
#include <QTimer>
#include <QVBoxLayout>
#include <QVector>
#include <QWidget>
#include <qcustomplot.h>
#include <optional>

// 線量プロファイル表示用ウィンドウ
class DoseProfileWindow : public QWidget {
  Q_OBJECT
  Q_DISABLE_COPY_MOVE(DoseProfileWindow)
public:
  explicit DoseProfileWindow(QWidget *parent = nullptr);

  // ROIセグメント情報
  struct Segment {
    double startMm;
    double endMm;
    double maxDoseGy;
    double minDoseGy;
    QColor color;
    QString name;
  };

  struct SamplePoint {
    double positionMm{0.0};
    std::optional<double> ctHu;
    std::optional<double> doseGy;
  };

  // 線量プロファイルデータを設定
  void setProfile(const QVector<double> &positions,
                  const QVector<double> &doses,
                  const QVector<Segment> &segments);
  // テキスト情報を設定
  void setStats(double lengthMm, double minDoseGy, double maxDoseGy,
                const QVector<Segment> &segments,
                const QVector<SamplePoint> &samplePoints = {});

  // DPS-D解析用にROI名とDICOMデータを設定
  void setROINames(const QStringList &names);
  void setDicomData(const DicomVolume *ct, const DoseResampledVolume *dose,
                    const RTStructureSet *structures);
  void setLineSlotAvailable(int slotIndex, bool hasLine);

signals:
  // 線分選択要求
  void requestLineSelection();
  void saveLineRequested(int slotIndex);
  void loadLineRequested(int slotIndex);

private slots:
  void onCalculateDPSD();
  void performDPSDCalculation(int roiIndex, double startMm, double endMm,
                              DPSDCalculator::Mode mode,
                              int sampleRoiIndex = -1);
  void onExportDPSD();

private:
  void setDPSDData(const DPSDCalculator::Result &data,
                   const DPSDCalculator::Result &sampleData);
  void updatePlotForProfile();
  void updatePlotForDPSD();
  void updateOverlayPositions();
  void resizeEvent(QResizeEvent *event) override;
  void showEvent(QShowEvent *event) override;
  bool eventFilter(QObject *obj, QEvent *event) override;
  void updateCursorDisplay(const QPoint &pos);
  void updateSlotButtonState(int slotIndex);
  void refreshSlotButtonStates();
  QCustomPlot *m_plot{nullptr};
  QPlainTextEdit *m_infoBox{nullptr};
  QPushButton *m_pickButton{nullptr};
  QPushButton *m_saveButton{nullptr};
  QVector<QPushButton *> m_slotButtons;
  QVector<bool> m_slotHasData;
  bool m_pendingSave{false};
  QVector<QCPItemLine *> m_roiLines;
  QVector<QCPItemText *> m_roiLabels;
  QCPItemLine *m_zeroAxis{nullptr};
  QLabel *m_dpsdTitleLabel{nullptr};
  QLabel *m_filterRoiLabel{nullptr};
  QCPItemLine *m_cursorLine{nullptr};
  QLabel *m_cursorInfoLabel{nullptr};

  // カーソル表示切替用チェックボックス
  QCheckBox *m_showCursorCheck{nullptr};

  // DPS-D解析用ウィジェット
  QGroupBox *m_dpsdGroup{nullptr};
  QComboBox *m_roiCombo{nullptr};
  QComboBox *m_sampleCombo{nullptr};
  QDoubleSpinBox *m_startSpin{nullptr};
  QDoubleSpinBox *m_endSpin{nullptr};
  QComboBox *m_modeCombo{nullptr};
  QPushButton *m_calcButton{nullptr};
  QPushButton *m_exportButton{nullptr};
  QRadioButton *m_profileRadio{nullptr};
  QRadioButton *m_dpsdRadio{nullptr};
  QProgressBar *m_progress{nullptr};
  QFutureWatcher<QPair<DPSDCalculator::Result, DPSDCalculator::Result>>
      *m_calcWatcher{nullptr};
  int m_pendingROIIndex{-1};
  int m_pendingSampleIndex{-1};
  QTimer *m_progressTimer{nullptr};

  // プロットデータの保持
  QVector<double> m_profilePositions;
  QVector<double> m_profileDoses;
  QVector<Segment> m_profileSegments;
  DPSDCalculator::Result m_dpsdResult;
  DPSDCalculator::Result m_dpsdSampleResult;
  int m_dpsdSampleIndex{-1};
  bool m_hasDPSDResult{false};

  // DICOMデータ参照
  const DicomVolume *m_ct{nullptr};
  const DoseResampledVolume *m_dose{nullptr};
  const RTStructureSet *m_structures{nullptr};
};

#endif // DOSE_PROFILE_WINDOW_H
