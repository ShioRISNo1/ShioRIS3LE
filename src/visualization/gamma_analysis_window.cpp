#include "visualization/gamma_analysis_window.h"

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFutureWatcher>
#include <QGridLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPainter>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QtConcurrent>
#include <algorithm>
#include <cmath>
#include <limits>

#include "dicom/rtdose_volume.h"

namespace {
constexpr double kEpsilonDose = 1e-6;
}

class HistogramWidget : public QWidget {
public:
  explicit HistogramWidget(QWidget *parent = nullptr) : QWidget(parent) {
    setMinimumHeight(160);
  }

  void setData(const std::vector<double> &values, double maxGamma) {
    m_values = values;
    m_maxGamma = std::max(1.0, maxGamma);
    update();
  }

protected:
  void paintEvent(QPaintEvent *event) override {
    Q_UNUSED(event);
    QPainter painter(this);
    painter.fillRect(rect(), palette().window());

    QRect plotRect = rect().adjusted(10, 10, -10, -30);
    painter.setPen(palette().text().color());
    painter.drawRect(plotRect);

    if (m_values.empty()) {
      painter.drawText(plotRect, Qt::AlignCenter,
                       tr("No valid gamma points"));
      return;
    }

    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.save();
    QColor gridColor = palette().text().color();
    gridColor.setAlpha(40);
    painter.setPen(QPen(gridColor, 1, Qt::DashLine));
    const int gridLines = 5;
    for (int i = 1; i < gridLines; ++i) {
      int x = plotRect.left() + i * plotRect.width() / gridLines;
      painter.drawLine(QPoint(x, plotRect.top()),
                       QPoint(x, plotRect.bottom()));
      int y = plotRect.bottom() - i * plotRect.height() / gridLines;
      painter.drawLine(QPoint(plotRect.left(), y),
                       QPoint(plotRect.right(), y));
    }
    painter.restore();

    std::vector<int> counts(m_bins, 0);
    for (double gamma : m_values) {
      int idx =
          static_cast<int>(std::floor(gamma / m_maxGamma * m_bins));
      idx = std::clamp(idx, 0, m_bins - 1);
      counts[idx]++;
    }
    int maxCount = *std::max_element(counts.begin(), counts.end());
    if (maxCount <= 0)
      return;

    double binWidth = static_cast<double>(plotRect.width()) / m_bins;
    QColor barColor(80, 160, 255);
    painter.setBrush(barColor);
    painter.setPen(Qt::NoPen);
    for (int i = 0; i < m_bins; ++i) {
      double ratio = static_cast<double>(counts[i]) / maxCount;
      int barHeight = static_cast<int>(ratio * plotRect.height());
      QRectF bar(plotRect.left() + i * binWidth,
                 plotRect.bottom() - barHeight, binWidth - 2, barHeight);
      painter.drawRect(bar);
    }

    QColor gammaLineColor(255, 120, 120);
    gammaLineColor.setAlpha(200);
    painter.setPen(QPen(gammaLineColor, 2));
    double gammaOneX =
        plotRect.left() + (1.0 / m_maxGamma) * plotRect.width();
    if (gammaOneX >= plotRect.left() && gammaOneX <= plotRect.right()) {
      painter.drawLine(QPointF(gammaOneX, plotRect.top()),
                       QPointF(gammaOneX, plotRect.bottom()));
      painter.drawText(QPointF(gammaOneX + 4, plotRect.top() + 12), tr("1.0"));
    }

    painter.setPen(palette().text().color());
    painter.drawText(plotRect.left(), plotRect.bottom() + 20,
                     tr("0"));
    painter.drawText(plotRect.right() - 30, plotRect.bottom() + 20,
                     QString::number(m_maxGamma, 'f', 2));
  }

private:
  std::vector<double> m_values;
  double m_maxGamma{1.0};
  int m_bins{20};
};

GammaAnalysisWindow::GammaAnalysisWindow(QWidget *parent) : QWidget(parent) {
  setWindowTitle(tr("Gamma Analysis"));
  setWindowFlag(Qt::Window, true);
  setMinimumWidth(420);

  auto *layout = new QVBoxLayout(this);

  auto *selectionLayout = new QGridLayout();
  selectionLayout->addWidget(new QLabel(tr("Reference Dose")), 0, 0);
  m_refCombo = new QComboBox(this);
  selectionLayout->addWidget(m_refCombo, 0, 1);
  selectionLayout->addWidget(new QLabel(tr("Evaluation Dose")), 1, 0);
  m_evalCombo = new QComboBox(this);
  selectionLayout->addWidget(m_evalCombo, 1, 1);
  layout->addLayout(selectionLayout);

  auto *criteriaLayout = new QGridLayout();
  criteriaLayout->addWidget(new QLabel(tr("Dose Diff (%)")), 0, 0);
  m_doseDiffSpin = new QDoubleSpinBox(this);
  m_doseDiffSpin->setRange(0.1, 20.0);
  m_doseDiffSpin->setDecimals(2);
  m_doseDiffSpin->setValue(3.0);
  criteriaLayout->addWidget(m_doseDiffSpin, 0, 1);

  criteriaLayout->addWidget(new QLabel(tr("Distance (mm)")), 1, 0);
  m_distanceSpin = new QDoubleSpinBox(this);
  m_distanceSpin->setRange(0.1, 20.0);
  m_distanceSpin->setDecimals(2);
  m_distanceSpin->setValue(3.0);
  criteriaLayout->addWidget(m_distanceSpin, 1, 1);

  criteriaLayout->addWidget(new QLabel(tr("Threshold (%)")), 2, 0);
  m_thresholdSpin = new QDoubleSpinBox(this);
  m_thresholdSpin->setRange(0.0, 100.0);
  m_thresholdSpin->setDecimals(1);
  m_thresholdSpin->setValue(10.0);
  criteriaLayout->addWidget(m_thresholdSpin, 2, 1);

  criteriaLayout->addWidget(new QLabel(tr("Sample Step (voxel)")), 3, 0);
  m_sampleStepSpin = new QSpinBox(this);
  m_sampleStepSpin->setRange(1, 10);
  m_sampleStepSpin->setValue(1);
  criteriaLayout->addWidget(m_sampleStepSpin, 3, 1);

  layout->addLayout(criteriaLayout);

  m_analyzeButton = new QPushButton(tr("Analyze"), this);
  layout->addWidget(m_analyzeButton);

  m_statusLabel = new QLabel(tr("Ready"), this);
  layout->addWidget(m_statusLabel);

  auto *resultLayout = new QGridLayout();
  resultLayout->addWidget(new QLabel(tr("Pass Rate")), 0, 0);
  m_passRateLabel = new QLabel("-", this);
  resultLayout->addWidget(m_passRateLabel, 0, 1);

  resultLayout->addWidget(new QLabel(tr("Mean Gamma")), 1, 0);
  m_meanGammaLabel = new QLabel("-", this);
  resultLayout->addWidget(m_meanGammaLabel, 1, 1);

  resultLayout->addWidget(new QLabel(tr("Max Gamma")), 2, 0);
  m_maxGammaLabel = new QLabel("-", this);
  resultLayout->addWidget(m_maxGammaLabel, 2, 1);

  resultLayout->addWidget(new QLabel(tr("Points")), 3, 0);
  m_pointsLabel = new QLabel("-", this);
  resultLayout->addWidget(m_pointsLabel, 3, 1);

  resultLayout->addWidget(new QLabel(tr("Invalid Points")), 4, 0);
  m_invalidLabel = new QLabel("-", this);
  resultLayout->addWidget(m_invalidLabel, 4, 1);

  layout->addLayout(resultLayout);

  layout->addWidget(new QLabel(tr("Gamma Pass Histogram"), this));
  m_histogramView = new HistogramWidget(this);
  layout->addWidget(m_histogramView);
  layout->addStretch();

  m_watcher = new QFutureWatcher<GammaResult>(this);
  connect(m_analyzeButton, &QPushButton::clicked, this,
          &GammaAnalysisWindow::onAnalyze);
  connect(m_watcher, &QFutureWatcher<GammaResult>::finished, this,
          &GammaAnalysisWindow::onAnalysisFinished);
}

void GammaAnalysisWindow::setDoseEntries(const std::vector<DoseEntry> &entries,
                                         int calcMode, double alphaBeta) {
  m_entries = entries;
  m_refCombo->clear();
  m_evalCombo->clear();
  m_calcMode = calcMode;
  m_alphaBeta = alphaBeta;

  for (const auto &entry : m_entries) {
    m_refCombo->addItem(entry.name);
    m_evalCombo->addItem(entry.name);
  }

  if (m_entries.size() >= 2) {
    m_refCombo->setCurrentIndex(0);
    m_evalCombo->setCurrentIndex(1);
  }

  if (!m_entries.empty()) {
    m_statusLabel->setText(tr("Ready"));
  } else {
    m_statusLabel->setText(tr("No dose items loaded"));
  }
}

void GammaAnalysisWindow::onAnalyze() {
  if (m_entries.size() < 2) {
    QMessageBox::warning(this, tr("Gamma Analysis"),
                         tr("Please load at least two dose distributions."));
    return;
  }

  int refIndex = m_refCombo->currentIndex();
  int evalIndex = m_evalCombo->currentIndex();
  if (refIndex < 0 || evalIndex < 0 ||
      refIndex >= static_cast<int>(m_entries.size()) ||
      evalIndex >= static_cast<int>(m_entries.size())) {
    QMessageBox::warning(this, tr("Gamma Analysis"),
                         tr("Invalid dose selection."));
    return;
  }
  if (refIndex == evalIndex) {
    QMessageBox::warning(this, tr("Gamma Analysis"),
                         tr("Please select different dose distributions."));
    return;
  }

  AnalysisSettings settings = buildSettings();

  setUiRunning(true);
  m_statusLabel->setText(tr("Running..."));

  const DoseEntry refEntry = m_entries[refIndex];
  const DoseEntry evalEntry = m_entries[evalIndex];

  auto future = QtConcurrent::run([refEntry, evalEntry, settings]() {
    return computeGamma(refEntry, evalEntry, settings);
  });
  m_watcher->setFuture(future);
}

void GammaAnalysisWindow::onAnalysisFinished() {
  setUiRunning(false);
  GammaResult result = m_watcher->result();
  updateResultLabels(result);
  updateHistogram(result);
  if (result.totalPoints == 0) {
    m_statusLabel->setText(tr("No points above threshold"));
  } else {
    m_statusLabel->setText(tr("Completed"));
  }
}

GammaAnalysisWindow::AnalysisSettings GammaAnalysisWindow::buildSettings() const {
  AnalysisSettings settings;
  settings.doseDiffPercent = m_doseDiffSpin->value();
  settings.distanceMm = m_distanceSpin->value();
  settings.thresholdPercent = m_thresholdSpin->value();
  settings.sampleStep = m_sampleStepSpin->value();
  settings.calcMode = m_calcMode;
  settings.alphaBeta = m_alphaBeta;
  return settings;
}

void GammaAnalysisWindow::updateResultLabels(const GammaResult &result) {
  m_passRateLabel->setText(
      QString::number(result.passRate, 'f', 2) + "%");
  if (result.totalPoints > 0 && result.totalPoints > result.invalidPoints) {
    double mean = result.meanGamma;
    m_meanGammaLabel->setText(QString::number(mean, 'f', 3));
  } else {
    m_meanGammaLabel->setText("-");
  }
  if (result.totalPoints > 0 && result.maxGamma > 0.0) {
    m_maxGammaLabel->setText(QString::number(result.maxGamma, 'f', 3));
  } else {
    m_maxGammaLabel->setText("-");
  }
  m_pointsLabel->setText(QString("%1 / %2")
                             .arg(result.passedPoints)
                             .arg(result.totalPoints));
  m_invalidLabel->setText(QString::number(result.invalidPoints));
}

void GammaAnalysisWindow::updateHistogram(const GammaResult &result) {
  if (!m_histogramView)
    return;
  m_histogramView->setData(result.gammaValues, result.maxGamma);
}

void GammaAnalysisWindow::setUiRunning(bool running) {
  m_analyzeButton->setEnabled(!running);
  m_refCombo->setEnabled(!running);
  m_evalCombo->setEnabled(!running);
  m_doseDiffSpin->setEnabled(!running);
  m_distanceSpin->setEnabled(!running);
  m_thresholdSpin->setEnabled(!running);
  m_sampleStepSpin->setEnabled(!running);
}

double GammaAnalysisWindow::transformDose(double rawDose, double dataFractions,
                                          double displayFractions, double factor,
                                          int calcMode, double alphaBeta) {
  double normalized = rawDose / std::max(1.0, dataFractions);
  double physical = normalized * displayFractions;
  double value = physical;
  if (calcMode == 1) {
    value = physical * (1.0 + physical / alphaBeta);
  } else if (calcMode == 2) {
    double bed = physical * (1.0 + physical / alphaBeta);
    value = bed / (1.0 + 2.0 / alphaBeta);
  }
  return value * factor;
}

GammaAnalysisWindow::GammaResult GammaAnalysisWindow::computeGamma(
    const DoseEntry &refEntry, const DoseEntry &evalEntry,
    const AnalysisSettings &settings) {
  GammaResult result;

  RTDoseVolume refDose = refEntry.dose;
  refDose.setPatientShift(refEntry.shift);
  RTDoseVolume evalDose = evalEntry.dose;
  evalDose.setPatientShift(evalEntry.shift);

  const double refMax = transformDose(refDose.maxDose(), refEntry.dataFractions,
                                      refEntry.displayFractions,
                                      refEntry.factor, settings.calcMode,
                                      settings.alphaBeta);
  const double doseCriterion =
      std::max(kEpsilonDose, refMax * settings.doseDiffPercent / 100.0);
  const double threshold = refMax * settings.thresholdPercent / 100.0;
  const double distanceCriterion = settings.distanceMm;

  if (refDose.width() == 0 || refDose.height() == 0 || refDose.depth() == 0) {
    return result;
  }

  const int step = std::max(1, settings.sampleStep);
  const int w = refDose.width();
  const int h = refDose.height();
  const int d = refDose.depth();

  const int rx =
      std::max(1, static_cast<int>(std::ceil(distanceCriterion /
                                             std::max(1e-3, evalDose.spacingX()))));
  const int ry =
      std::max(1, static_cast<int>(std::ceil(distanceCriterion /
                                             std::max(1e-3, evalDose.spacingY()))));
  const int rz =
      std::max(1, static_cast<int>(std::ceil(distanceCriterion /
                                             std::max(1e-3, evalDose.spacingZ()))));

  double sumGamma = 0.0;

  for (int z = 0; z < d; z += step) {
    const float *slice = refDose.data().ptr<float>(z);
    for (int y = 0; y < h; y += step) {
      int row = y * w;
      for (int x = 0; x < w; x += step) {
        float rawRef = slice[row + x];
        double refValue = transformDose(rawRef, refEntry.dataFractions,
                                        refEntry.displayFractions,
                                        refEntry.factor, settings.calcMode,
                                        settings.alphaBeta);
        if (refValue < threshold)
          continue;

        QVector3D patientRef = refDose.voxelToPatient(x + 0.5, y + 0.5, z + 0.5);
        QVector3D evalVoxel = evalDose.patientToVoxelContinuous(patientRef);

        double bestGamma = std::numeric_limits<double>::infinity();
        for (int dz = -rz; dz <= rz; ++dz) {
          for (int dy = -ry; dy <= ry; ++dy) {
            for (int dx = -rx; dx <= rx; ++dx) {
              double cx = evalVoxel.x() + dx;
              double cy = evalVoxel.y() + dy;
              double cz = evalVoxel.z() + dz;
              if (cx < -0.5 || cy < -0.5 || cz < -0.5 ||
                  cx >= evalDose.width() - 0.5 ||
                  cy >= evalDose.height() - 0.5 ||
                  cz >= evalDose.depth() - 0.5) {
                continue;
              }
              QVector3D patientEval = evalDose.voxelToPatient(cx, cy, cz);
              double distance = (patientEval - patientRef).length();
              if (distance > distanceCriterion)
                continue;

              float rawEval = evalDose.sampleDose(cx, cy, cz);
              double evalValue = transformDose(rawEval, evalEntry.dataFractions,
                                               evalEntry.displayFractions,
                                               evalEntry.factor,
                                               settings.calcMode,
                                               settings.alphaBeta);
              double doseDiff = evalValue - refValue;
              double gamma = std::sqrt(
                  (distance * distance) /
                      (distanceCriterion * distanceCriterion) +
                  (doseDiff * doseDiff) / (doseCriterion * doseCriterion));
              if (gamma < bestGamma) {
                bestGamma = gamma;
              }
            }
          }
        }

        result.totalPoints++;
        if (std::isfinite(bestGamma)) {
          sumGamma += bestGamma;
          result.gammaValues.push_back(bestGamma);
          if (bestGamma > result.maxGamma)
            result.maxGamma = bestGamma;
          if (bestGamma <= 1.0)
            result.passedPoints++;
        } else {
          result.invalidPoints++;
        }
      }
    }
  }

  if (result.totalPoints > 0) {
    int validPoints = result.totalPoints - result.invalidPoints;
    if (validPoints > 0) {
      result.meanGamma = sumGamma / static_cast<double>(validPoints);
    }
    result.passRate =
        static_cast<double>(result.passedPoints) /
        static_cast<double>(result.totalPoints) * 100.0;
  }

  return result;
}
