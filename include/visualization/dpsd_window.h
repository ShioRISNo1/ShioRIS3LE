#ifndef DPSD_WINDOW_H
#define DPSD_WINDOW_H

#include <QFutureWatcher>
#include <QtConcurrent>
#include "dicom/dicom_volume.h"
#include "dicom/dose_resampled_volume.h"
#include "dicom/rtstruct.h"

#include "dicom/dpsd_calculator.h"
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QCheckBox>
#include <QLabel>
#include <QPlainTextEdit>
#include <QPoint>
#include <QProgressBar>
#include <QPushButton>
#include <QResizeEvent>
#include <QShowEvent>
#include <QStringList>
#include <QTimer>
#include <QVector>
#include <QWidget>
#include <qcustomplot.h>

class DPSDWindow : public QWidget {
  Q_OBJECT
  Q_DISABLE_COPY_MOVE(DPSDWindow)
public:
  explicit DPSDWindow(QWidget *parent = nullptr);

  void setROINames(const QStringList &names);
  void setDPSDData(int roiIndex, const DPSDCalculator::Result &data,
                   const DPSDCalculator::Result &roiData);
  void setCurrentROI(int index);
  void setBusy(bool busy);
  void setDicomData(const DicomVolume *ct, const DoseResampledVolume *dose,
                    const RTStructureSet *structures);

signals:
  void requestCalculation(int roiIndex, int roiFilterIndex, double startMm,
                          double endMm, DPSDCalculator::Mode mode);

private slots:
  void onCalculate();
  void onExport();

private:
  bool eventFilter(QObject *obj, QEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;
  void showEvent(QShowEvent *event) override;
  void updateOverlayPositions();
  void updateCursorDisplay(const QPoint &pos);

  // DICOMデータ参照
  const DicomVolume *m_ct{nullptr};
  const DoseResampledVolume *m_dose{nullptr};
  const RTStructureSet *m_structures{nullptr};
  
  // 非同期計算用
  QFutureWatcher<QPair<DPSDCalculator::Result, DPSDCalculator::Result>> *m_calcWatcher{nullptr};

  
  QCustomPlot *m_plot{nullptr};
  QComboBox *m_roiCombo{nullptr};
  QComboBox *m_modeCombo{nullptr};
  QComboBox *m_roiFilterCombo{nullptr};
  QDoubleSpinBox *m_startSpin{nullptr};
  QDoubleSpinBox *m_endSpin{nullptr};
  QPushButton *m_calcButton{nullptr};
  QPushButton *m_exportButton{nullptr};
  QCheckBox *m_showDataCheck{nullptr};
  QPlainTextEdit *m_infoBox{nullptr};
  QProgressBar *m_progress{nullptr};
  QTimer *m_progressTimer{nullptr};
  QCPItemLine *m_zeroAxis{nullptr};
  QLabel *m_titleLabel{nullptr};
  QLabel *m_roiLabel{nullptr};
  QCPItemLine *m_cursorLine{nullptr};
  QLabel *m_cursorInfoLabel{nullptr};
  DPSDCalculator::Result m_currentResult;
  DPSDCalculator::Result m_currentRoiResult;
};

#endif // DPSD_WINDOW_H
