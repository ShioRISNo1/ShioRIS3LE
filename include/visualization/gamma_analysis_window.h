#ifndef GAMMA_ANALYSIS_WINDOW_H
#define GAMMA_ANALYSIS_WINDOW_H

#include <QWidget>
#include <QFutureWatcher>
#include <QVector3D>
#include <QString>
#include "dicom/rtdose_volume.h"
#include <vector>

class QComboBox;
class QDoubleSpinBox;
class HistogramWidget;
class QSpinBox;
class QPushButton;
class QLabel;

class GammaAnalysisWindow : public QWidget {
  Q_OBJECT

public:
  struct DoseEntry {
    QString name;
    RTDoseVolume dose;
    double dataFractions{1.0};
    double displayFractions{1.0};
    double factor{1.0};
    QVector3D shift;
  };

  struct GammaResult {
    double passRate{0.0};
    double meanGamma{0.0};
    double maxGamma{0.0};
    int totalPoints{0};
    int passedPoints{0};
    int invalidPoints{0};
    std::vector<double> gammaValues;
  };

  explicit GammaAnalysisWindow(QWidget *parent = nullptr);
  void setDoseEntries(const std::vector<DoseEntry> &entries, int calcMode,
                      double alphaBeta);

private slots:
  void onAnalyze();
  void onAnalysisFinished();

private:
  struct AnalysisSettings {
    double doseDiffPercent{3.0};
    double distanceMm{3.0};
    double thresholdPercent{10.0};
    int sampleStep{1};
    int calcMode{0};
    double alphaBeta{10.0};
  };

  std::vector<DoseEntry> m_entries;

  QComboBox *m_refCombo{nullptr};
  QComboBox *m_evalCombo{nullptr};
  QDoubleSpinBox *m_doseDiffSpin{nullptr};
  QDoubleSpinBox *m_distanceSpin{nullptr};
  QDoubleSpinBox *m_thresholdSpin{nullptr};
  QSpinBox *m_sampleStepSpin{nullptr};
  QPushButton *m_analyzeButton{nullptr};
  QLabel *m_statusLabel{nullptr};
  QLabel *m_passRateLabel{nullptr};
  QLabel *m_meanGammaLabel{nullptr};
  QLabel *m_maxGammaLabel{nullptr};
  QLabel *m_pointsLabel{nullptr};
  QLabel *m_invalidLabel{nullptr};
  HistogramWidget *m_histogramView{nullptr};

  QFutureWatcher<GammaResult> *m_watcher{nullptr};
  int m_calcMode{0};
  double m_alphaBeta{10.0};

  AnalysisSettings buildSettings() const;
  void updateResultLabels(const GammaResult &result);
  void updateHistogram(const GammaResult &result);
  void setUiRunning(bool running);

  static double transformDose(double rawDose, double dataFractions,
                              double displayFractions, double factor,
                              int calcMode, double alphaBeta);
  static GammaResult computeGamma(const DoseEntry &refEntry,
                                  const DoseEntry &evalEntry,
                                  const AnalysisSettings &settings);
};

#endif
