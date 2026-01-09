#ifndef RANDOM_STUDY_DIALOG_H
#define RANDOM_STUDY_DIALOG_H

#include <QDialog>
#include <QtGlobal>
#include <QStringList>
#include <QCloseEvent>

class QLabel;
class QPushButton;
class QVBoxLayout;
class QComboBox;
class QCustomPlot;
class QSpinBox;
class QDoubleSpinBox;
class QGroupBox;
class QProgressBar;
class QCheckBox;
class QSplitter;
class QCPItemLine;
class QCPItemText;
class QPlainTextEdit;

// Simple dialog placeholder for future DVH and related displays
class RandomStudyDialog : public QDialog {
  Q_OBJECT

public:
  explicit RandomStudyDialog(QWidget *parent = nullptr);
  ~RandomStudyDialog();

  void setROINames(const QStringList &roiNames);
  // Plotting helpers
  void clearPlot();
  void addDVHCurve(const QVector<double> &xGy, const QVector<double> &yPct,
                   const QColor &color, bool replot = true, int width = 1);
  void setDoseAxisMax(double maxGy);

  // Parameter getters
  int selectedROIIndex() const;
  int fractionCount() const;
  int iterationCount() const;
  // View/Histogram settings
  enum class ViewMode { DVH = 0, Histogram = 1 };
  enum class HistType { DxCc = 0, DxPct = 1, VdCc = 2, MinDose = 3, MaxDose = 4, MeanDose = 5 };
  ViewMode viewMode() const;
  HistType histogramType() const;
  double histogramParam() const; // cc for DxCc, % for DxPct, Gy for VdCc; ignored for Min/Max
  int histogramBins() const;
  // Axis percent mode for dose-like X
  bool isXAxisPercent() const;
  double xPercentRefGy() const;
  // Systematic error: mean and sigma (mm) per axis
  void systematicError(double &mx, double &my, double &mz,
                       double &sx, double &sy, double &sz) const;
  // Random error: sigma (mm) per axis (zero mean)
  void randomError(double &sx, double &sy, double &sz) const;

  // Seed control
  bool isSeedFixed() const;
  quint64 seedValue() const;

  // Running state
  void setRunning(bool running);

  // Histogram plotting helper: values are per-iteration metrics
  void plotHistogram(const QVector<double> &values, int bins,
                     const QString &xLabel, const QString &yLabel);
  // Switch plot axes to DVH defaults
  void prepareDVHAxes();
  // Overlay markers for histogram stats (mean and percentiles)
  void setHistogramMarkers(double mean,
                           const QVector<QPair<double, QString>> &marks);

  // Cache API (kept until ROI changes or Start is clicked; cleared on close)
  // Store all per-iteration histograms for instant metric recompute (Dx%, etc.)
  void clearCache();
  void setCachedHistograms(const QVector<QVector<float>> &hists,
                           double binSize,
                           double totalVolume);
  bool hasCache() const { return !m_cachedHists.isEmpty(); }
  // Recompute and draw current histogram from cache using UI params
  void plotHistogramFromCache();
  // Rebuild DVH curves from cache (up to ~100 curves)
  void plotDVHsFromCache(int maxCurves = 100);
  // Baseline (no-error) histogram setter
  void setBaselineHistogram(const QVector<float> &hist, double binSize, double totalVolume) {
    m_baselineHist = hist;
    m_baselineBinSize = binSize;
    m_baselineTotalVolume = totalVolume;
  }
  bool hasBaseline() const { return !m_baselineHist.isEmpty() && m_baselineBinSize > 0.0 && m_baselineTotalVolume > 0.0; }

signals:
  void startCalculationRequested();
  void cancelCalculationRequested();

public slots:
  void setCalculationProgress(int processed, int total);

private:
  void initUi();
  void setupHalfScreenSize();
  void adjustSplitter();
  void closeEvent(QCloseEvent *event) override;

  // Root
  QVBoxLayout *m_layout{nullptr};

  // Left: DVH plot area
  QCustomPlot *m_plot{nullptr};

  // Right controls
  QComboBox *m_roiCombo{nullptr};
  QSpinBox *m_fractionSpin{nullptr};
  QSpinBox *m_iterationsSpin{nullptr};
  // View & Histogram controls
  QComboBox *m_viewModeCombo{nullptr};
  QComboBox *m_histTypeCombo{nullptr};
  QDoubleSpinBox *m_histParamSpin{nullptr};
  QSpinBox *m_histBinsSpin{nullptr};
  QLabel *m_histParamLabel{nullptr};
  QCheckBox *m_fixSeedCheck{nullptr};
  QSpinBox *m_seedSpin{nullptr};
  // X-axis percent controls
  QCheckBox *m_xPercentCheck{nullptr};
  QDoubleSpinBox *m_xPercentRef{nullptr};
  // Systematic error (± value per axis)
  QDoubleSpinBox *m_sysX{nullptr};
  QDoubleSpinBox *m_sysY{nullptr};
  QDoubleSpinBox *m_sysZ{nullptr};
  // ± boxes (stddev) for each axis
  QDoubleSpinBox *m_sysXpm{nullptr};
  QDoubleSpinBox *m_sysYpm{nullptr};
  QDoubleSpinBox *m_sysZpm{nullptr};
  // Random error (value per axis)
  QDoubleSpinBox *m_randX{nullptr};
  QDoubleSpinBox *m_randY{nullptr};
  QDoubleSpinBox *m_randZ{nullptr};
  QPushButton *m_startButton{nullptr};
  QPushButton *m_cancelButton{nullptr};
  QProgressBar *m_progress{nullptr};
  QSplitter *m_splitter{nullptr};
  QPlainTextEdit *m_infoText{nullptr};

  // Histogram statistic markers
  QCPItemLine *m_meanLine{nullptr};
  QCPItemText *m_meanLabel{nullptr};
  QVector<QCPItemLine*> m_pctLines;   // P5,P10,P25,P75,P90,P95
  QVector<QCPItemText*> m_pctLabels;
  // Baseline (no-error) marker
  QCPItemLine *m_baselineLine{nullptr};
  QCPItemText *m_baselineLabel{nullptr};

  // Cached per-iteration histograms and metadata
  QVector<QVector<float>> m_cachedHists; // size: iterations x bins (non-cumulative volume per bin)
  double m_cachedBinSize{0.0};
  double m_cachedTotalVolume{0.0};
  // Baseline histogram and metadata
  QVector<float> m_baselineHist; // same bins/binSize as cached (non-cumulative)
  double m_baselineBinSize{0.0};
  double m_baselineTotalVolume{0.0};
};

#endif // RANDOM_STUDY_DIALOG_H
