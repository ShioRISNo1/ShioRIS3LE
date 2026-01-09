#ifndef DVH_WINDOW_H
#define DVH_WINDOW_H

#include "dicom/dvh_calculator.h"
#include <QButtonGroup>
#include <QEvent>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QDoubleSpinBox>
#include <QMouseEvent>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QString>
#include <QStringList>
#include <QVBoxLayout>
#include <QWidget>
#include <qcustomplot.h>

class DVHWindow : public QWidget {
  Q_OBJECT
public:
  explicit DVHWindow(QWidget *parent = nullptr);
  ~DVHWindow();

  void setROINames(const QStringList &roiNames);
  void setDVHData(const std::vector<DVHCalculator::DVHData> &dvhData);
  void updateVisibility(int roiIndex, bool visible);
  // ROI名でチェック状態を変更（シグナルは発火しない）
  void setROIChecked(const QString &roiName, bool checked);
  const std::vector<DVHCalculator::DVHData> &dvhData() const;
  void setPatientInfo(const QString &text);
  double prescriptionDose() const { return m_prescriptionDose; }
  bool isXAxisPercent() const {
    return m_xPercentButton && m_xPercentButton->isChecked();
  }
  bool isYAxisCc() const { return m_yCcButton && m_yCcButton->isChecked(); }
  void setAxisUnits(bool xPercent, bool yCc);
  int currentRoiIndex() const;
  void setCurrentRoiIndex(int index);
  // CalcMax controls
  double calcMaxGy() const { return m_calcMaxGy; }
  bool isCalcMaxAuto() const { return !m_calcMaxUserSet || m_calcMaxGy <= 0.0; }
  void setCalcMaxAuto();
  void setCalcMaxGyNoRecalc(double gy);

public slots:
  void setPrescriptionDose(double doseGy);
  void setCalculationProgress(int processed, int total);

signals:
  void recalculateRequested(const QString &roiName);
  void calcMaxChanged(double gy);
  void doubleClicked();
  void visibilityChanged(int roiIndex, bool visible);

private slots:
  void onExportCSV();
  void onExportPNG();
  void onExportPDF();
  void onSelectAll();
  void onSelectNone();
  void onRoiItemChanged(QListWidgetItem *item);
  void onCurrentRoiChanged(int row);
  void onPlotMouseMove(QMouseEvent *event);
  void onAxisUnitChanged();
  void onCalcMaxChanged(double value);
  void onCalcMaxTo100();

protected:
  bool eventFilter(QObject *obj, QEvent *event) override;
  void closeEvent(QCloseEvent *event) override;

private:
  QCustomPlot *m_plot{nullptr};
  QListWidget *m_roiList{nullptr};
  QPlainTextEdit *m_detailBox{nullptr};
  QPushButton *m_exportCsvButton{nullptr};
  QPushButton *m_exportPngButton{nullptr};
  QPushButton *m_exportPdfButton{nullptr};
  QPushButton *m_allButton{nullptr};
  QPushButton *m_noneButton{nullptr};
  QLabel *m_roiNameLabel{nullptr};
  QLabel *m_cursorInfoLabel{nullptr};
  QLabel *m_patientInfoLabel{nullptr};
  QCPItemLine *m_cursorVLine{nullptr};
  QCPItemLine *m_cursorHLine{nullptr};
  std::vector<DVHCalculator::DVHData> m_data;
  QRadioButton *m_xPercentButton{nullptr};
  QRadioButton *m_xGyButton{nullptr};
  QRadioButton *m_yCcButton{nullptr};
  QRadioButton *m_yPercentButton{nullptr};
  QDoubleSpinBox *m_calcMaxSpin{nullptr};
  QPushButton *m_calcMax100Button{nullptr};
  QProgressBar *m_progressBar{nullptr};
  QButtonGroup *m_xUnitGroup{nullptr};
  QButtonGroup *m_yUnitGroup{nullptr};
  double m_globalMaxDose{0.0};
  double m_globalMaxVolumeCc{0.0};
  double m_prescriptionDose{0.0};
  double m_calcMaxGy{0.0};
  bool m_calcMaxUserSet{false};

  bool m_isClosing{false};

  void updateRoiInfo(int index);
  void updateOverlayPosition();
  void updatePlotUnits();
  void cleanup();
  double niceTickStep(double range) const;
};

#endif // DVH_WINDOW_H
