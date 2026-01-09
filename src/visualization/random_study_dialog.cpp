#include "visualization/random_study_dialog.h"

#include <QApplication>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QCheckBox>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QLabel>
#include <QComboBox>
#include <QPushButton>
#include <QProgressBar>
#include <QPlainTextEdit>
#include <QScreen>
#include <QSpinBox>
#include <QSplitter>
#include <QVBoxLayout>

#include "theme_manager.h"
#include <qcustomplot.h>
#include <limits>
#include <algorithm>

RandomStudyDialog::RandomStudyDialog(QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Random Study"));
  setModal(false);
  setAttribute(Qt::WA_DeleteOnClose, true);
  initUi();
  setupHalfScreenSize();
}

RandomStudyDialog::~RandomStudyDialog() {}

void RandomStudyDialog::initUi() {
  m_layout = new QVBoxLayout(this);

  // Splitter: left plot (≈70%), right controls (≈30%)
  m_splitter = new QSplitter(Qt::Horizontal, this);
  m_layout->addWidget(m_splitter);

  // Left: DVH/Histogram plot area
  QWidget *left = new QWidget(this);
  auto *leftLayout = new QVBoxLayout(left);
  leftLayout->setContentsMargins(0, 0, 0, 0);
  m_plot = new QCustomPlot(left);
  leftLayout->addWidget(m_plot, /*stretch*/3);
  // Bottom 1/4 info panel
  m_infoText = new QPlainTextEdit(left);
  m_infoText->setReadOnly(true);
  leftLayout->addWidget(m_infoText, /*stretch*/1);
  m_splitter->addWidget(left);

  // Right: controls panel
  QWidget *right = new QWidget(this);
  auto *rightLayout = new QVBoxLayout(right);

  // ROI selection + View mode
  {
    auto *row = new QWidget(right);
    auto *form = new QFormLayout(row);
    form->setContentsMargins(0, 0, 0, 0);
    m_roiCombo = new QComboBox(row);
    form->addRow(tr("ROI"), m_roiCombo);
    // Changing ROI invalidates cache (must recalc)
    connect(m_roiCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){ clearCache(); });
    m_viewModeCombo = new QComboBox(row);
    m_viewModeCombo->addItem(tr("DVH"), static_cast<int>(ViewMode::DVH));
    m_viewModeCombo->addItem(tr("Histogram"), static_cast<int>(ViewMode::Histogram));
    form->addRow(tr("View"), m_viewModeCombo);
    rightLayout->addWidget(row);
  }

  // Fractions and Iterations
  auto *paramsGroup = new QGroupBox(tr("Parameters"), right);
  auto *paramsForm = new QFormLayout(paramsGroup);
  m_fractionSpin = new QSpinBox(paramsGroup);
  m_fractionSpin->setRange(1, 200);
  m_fractionSpin->setValue(30);
  paramsForm->addRow(tr("Fractions"), m_fractionSpin);

  m_iterationsSpin = new QSpinBox(paramsGroup);
  m_iterationsSpin->setRange(1, 1000000);
  m_iterationsSpin->setSingleStep(100);
  m_iterationsSpin->setValue(1000);
  paramsForm->addRow(tr("Iterations"), m_iterationsSpin);
  // Seed controls
  QWidget *seedRow = new QWidget(paramsGroup);
  auto *seedLayout = new QHBoxLayout(seedRow);
  seedLayout->setContentsMargins(0,0,0,0);
  seedLayout->setSpacing(6);
  m_fixSeedCheck = new QCheckBox(tr("Fix Seed"), seedRow);
  m_seedSpin = new QSpinBox(seedRow);
  m_seedSpin->setRange(0, std::numeric_limits<int>::max());
  m_seedSpin->setValue(12345);
  m_seedSpin->setEnabled(false);
  seedLayout->addWidget(m_fixSeedCheck);
  seedLayout->addWidget(m_seedSpin);
  seedLayout->addStretch();
  connect(m_fixSeedCheck, &QCheckBox::toggled, this, [this](bool on){ m_seedSpin->setEnabled(on); });
  paramsForm->addRow(tr("Reproducibility"), seedRow);
  // X-axis percent controls
  QWidget *xpercRow = new QWidget(paramsGroup);
  auto *xpercLayout = new QHBoxLayout(xpercRow);
  xpercLayout->setContentsMargins(0,0,0,0);
  xpercLayout->setSpacing(6);
  m_xPercentCheck = new QCheckBox(tr("X Axis %"), xpercRow);
  m_xPercentRef = new QDoubleSpinBox(xpercRow);
  m_xPercentRef->setDecimals(2);
  m_xPercentRef->setRange(0.01, 100000.0);
  m_xPercentRef->setValue(1.0);
  m_xPercentRef->setEnabled(false);
  xpercLayout->addWidget(m_xPercentCheck);
  xpercLayout->addWidget(new QLabel(tr("100% ="), xpercRow));
  xpercLayout->addWidget(m_xPercentRef);
  xpercLayout->addWidget(new QLabel(tr("Gy"), xpercRow));
  xpercLayout->addStretch();
  connect(m_xPercentCheck, &QCheckBox::toggled, this, [this](bool){ if (hasCache()) { if (viewMode()==ViewMode::Histogram) plotHistogramFromCache(); else plotDVHsFromCache(); } if (m_xPercentRef) m_xPercentRef->setEnabled(isXAxisPercent()); });
  connect(m_xPercentRef, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double){ if (hasCache()) { if (viewMode()==ViewMode::Histogram) plotHistogramFromCache(); else plotDVHsFromCache(); } });
  paramsForm->addRow(tr("X Scale"), xpercRow);
  rightLayout->addWidget(paramsGroup);

  // Histogram settings
  auto *histGroup = new QGroupBox(tr("Histogram"), right);
  auto *histForm = new QFormLayout(histGroup);
  m_histTypeCombo = new QComboBox(histGroup);
  // Order: D x %, D x cc, then others
  m_histTypeCombo->addItem(tr("D x %"), static_cast<int>(HistType::DxPct));
  m_histTypeCombo->addItem(tr("D x cc"), static_cast<int>(HistType::DxCc));
  m_histTypeCombo->addItem(tr("V x Gy"), static_cast<int>(HistType::VdCc));
  m_histTypeCombo->addItem(tr("Min"), static_cast<int>(HistType::MinDose));
  m_histTypeCombo->addItem(tr("Max"), static_cast<int>(HistType::MaxDose));
  m_histTypeCombo->addItem(tr("Mean"), static_cast<int>(HistType::MeanDose));
  histForm->addRow(tr("Type"), m_histTypeCombo);

  QWidget *paramRow = new QWidget(histGroup);
  auto *paramLayout = new QHBoxLayout(paramRow);
  paramLayout->setContentsMargins(0,0,0,0);
  m_histParamLabel = new QLabel(tr("x"), paramRow);
  m_histParamSpin = new QDoubleSpinBox(paramRow);
  m_histParamSpin->setDecimals(2);
  m_histParamSpin->setRange(0.0, 10000.0);
  m_histParamSpin->setValue(2.0);
  m_histParamSpin->setFixedWidth(90);
  paramLayout->addWidget(m_histParamLabel);
  paramLayout->addWidget(m_histParamSpin);
  paramLayout->addStretch();
  histForm->addRow(tr("Param"), paramRow);

  m_histBinsSpin = new QSpinBox(histGroup);
  m_histBinsSpin->setRange(5, 200);
  m_histBinsSpin->setValue(30);
  histForm->addRow(tr("Bins"), m_histBinsSpin);
  rightLayout->addWidget(histGroup);

  // Redraw histogram from cache when parameters change (if histogram view)
  connect(m_histParamSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double){
    if (viewMode() == ViewMode::Histogram && hasCache()) plotHistogramFromCache();
  });
  connect(m_histBinsSpin, qOverload<int>(&QSpinBox::valueChanged), this, [this](int){
    if (viewMode() == ViewMode::Histogram && hasCache()) plotHistogramFromCache();
  });

  // Systematic Error (value ± value) per axis, units outside
  auto *sysGroup = new QGroupBox(tr("Systematic Error"), right);
  auto *sysGrid = new QGridLayout(sysGroup);
  sysGrid->setHorizontalSpacing(4); // tighten spacing around ±
  auto setupSysRow = [&](int row, const QString &axis, QDoubleSpinBox *&valBox,
                         QDoubleSpinBox *&pmBox) {
    auto *lblAxis = new QLabel(axis, sysGroup);
    auto *lblPM = new QLabel(tr("±"), sysGroup);
    valBox = new QDoubleSpinBox(sysGroup);
    pmBox = new QDoubleSpinBox(sysGroup);
    // Mean (can be negative)
    valBox->setRange(-50.0, 50.0);
    valBox->setDecimals(2);
    valBox->setSingleStep(0.1);
    // Stddev (non-negative)
    pmBox->setRange(0.0, 50.0);
    pmBox->setDecimals(2);
    pmBox->setSingleStep(0.1);
    // No suffix; unit is outside
    const int spinWidth = 80;
    valBox->setFixedWidth(spinWidth);
    pmBox->setFixedWidth(spinWidth);
    sysGrid->addWidget(lblAxis, row, 0);
    sysGrid->addWidget(valBox, row, 1);
    sysGrid->addWidget(lblPM, row, 2);
    sysGrid->addWidget(pmBox, row, 3);
  };
  setupSysRow(0, "X", m_sysX, m_sysXpm);
  setupSysRow(1, "Y", m_sysY, m_sysYpm);
  setupSysRow(2, "Z", m_sysZ, m_sysZpm);
  // Unit label outside spin boxes
  auto *sysUnit = new QLabel(tr("(mm)"), sysGroup);
  sysUnit->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  sysGrid->addWidget(sysUnit, 3, 0, 1, 4);
  rightLayout->addWidget(sysGroup);

  // Random Error (value per axis), units outside
  auto *rndGroup = new QGroupBox(tr("Random Error"), right);
  auto *rndGrid = new QGridLayout(rndGroup);
  rndGrid->setHorizontalSpacing(4);
  auto setupRandRow = [&](int row, const QString &axis, QDoubleSpinBox *&box) {
    auto *lblAxis = new QLabel(axis, rndGroup);
    auto *lblPM = new QLabel(tr("±"), rndGroup);
    box = new QDoubleSpinBox(rndGroup);
    box->setRange(0.0, 50.0);
    box->setDecimals(2);
    box->setSingleStep(0.1);
    box->setFixedWidth(80); // match systematic spin width
    rndGrid->addWidget(lblAxis, row, 0);
    rndGrid->addWidget(lblPM, row, 1);
    rndGrid->addWidget(box, row, 2, 1, 2);
  };
  setupRandRow(0, "X", m_randX);
  setupRandRow(1, "Y", m_randY);
  setupRandRow(2, "Z", m_randZ);
  auto *rndUnit = new QLabel(tr("(mm)"), rndGroup);
  rndUnit->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  rndGrid->addWidget(rndUnit, 3, 0, 1, 4);
  rightLayout->addWidget(rndGroup);

  // Start button and progress at bottom
  QWidget *runRow = new QWidget(right);
  auto *runLayout = new QHBoxLayout(runRow);
  runLayout->setContentsMargins(0,0,0,0);
  runLayout->setSpacing(6);
  m_startButton = new QPushButton(tr("Start Calc"), runRow);
  m_cancelButton = new QPushButton(tr("Cancel"), runRow);
  m_cancelButton->setEnabled(true);
  runLayout->addWidget(m_startButton);
  runLayout->addWidget(m_cancelButton);
  rightLayout->addWidget(runRow);
  m_progress = new QProgressBar(right);
  m_progress->setRange(0, 100);
  m_progress->setValue(0);
  m_progress->setTextVisible(true);
  rightLayout->addWidget(m_progress);
  rightLayout->addStretch();

  m_splitter->addWidget(right);
  m_splitter->setStretchFactor(0, 7);
  m_splitter->setStretchFactor(1, 3);

  // Close button row
  auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Close, this);
  connect(buttonBox, &QDialogButtonBox::rejected, this, &RandomStudyDialog::close);
  m_layout->addWidget(buttonBox);

  // Signal
  connect(m_startButton, &QPushButton::clicked, this, &RandomStudyDialog::startCalculationRequested);
  connect(m_cancelButton, &QPushButton::clicked, this, &RandomStudyDialog::cancelCalculationRequested);

  // Update param unit hint based on histogram type
  connect(m_histTypeCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){
    auto t = histogramType();
    bool needsParam = (t == HistType::DxCc || t == HistType::DxPct || t == HistType::VdCc);
    if (m_histParamLabel) {
      switch (t) {
      case HistType::DxCc: m_histParamLabel->setText(tr("x (cc)")); break;
      case HistType::DxPct: m_histParamLabel->setText(tr("x (%)")); break;
      case HistType::VdCc: m_histParamLabel->setText(tr("D (Gy)")); break;
      default: m_histParamLabel->setText(tr("-")); break;
      }
    }
    if (m_histParamSpin) m_histParamSpin->setEnabled(needsParam);
    // If cached data exists and we are in Histogram view, redraw instantly
    if (viewMode() == ViewMode::Histogram && hasCache()) {
      plotHistogramFromCache();
    }
  });
  // Initialize label
  if (m_histTypeCombo) emit m_histTypeCombo->currentIndexChanged(m_histTypeCombo->currentIndex());

  // React to view mode changes (axes prep only)
  connect(m_viewModeCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){
    if (!m_plot) return;
    if (viewMode() == ViewMode::DVH) {
      prepareDVHAxes();
      // hide markers when not in histogram view
      if (m_meanLine) m_meanLine->setVisible(false);
      if (m_meanLabel) m_meanLabel->setVisible(false);
      if (m_baselineLine) m_baselineLine->setVisible(false);
      if (m_baselineLabel) m_baselineLabel->setVisible(false);
      for (auto *l : m_pctLines) if (l) l->setVisible(false);
      for (auto *t : m_pctLabels) if (t) t->setVisible(false);
      // If we have cached histograms, reconstruct DVH curves quickly
      if (hasCache()) {
        plotDVHsFromCache();
      }
    } else {
      m_plot->clearGraphs();
      m_plot->xAxis->setLabel(tr("Value"));
      m_plot->yAxis->setLabel(tr("Count"));
      m_plot->xAxis->setRange(0.0, 1.0);
      m_plot->yAxis->setRange(0.0, 1.0);
      // ensure markers exist but hidden until first data
      if (!m_meanLine) {
        m_meanLine = new QCPItemLine(m_plot);
        m_meanLine->start->setType(QCPItemPosition::ptPlotCoords);
        m_meanLine->end->setType(QCPItemPosition::ptPlotCoords);
        m_meanLine->start->setAxes(m_plot->xAxis, m_plot->yAxis);
        m_meanLine->end->setAxes(m_plot->xAxis, m_plot->yAxis);
        m_meanLine->setPen(QPen(QColor(255, 220, 0), 2, Qt::DashLine));
        m_meanLine->setVisible(false);
      }
      if (!m_baselineLine) {
        m_baselineLine = new QCPItemLine(m_plot);
        m_baselineLine->start->setType(QCPItemPosition::ptPlotCoords);
        m_baselineLine->end->setType(QCPItemPosition::ptPlotCoords);
        m_baselineLine->start->setAxes(m_plot->xAxis, m_plot->yAxis);
        m_baselineLine->end->setAxes(m_plot->xAxis, m_plot->yAxis);
        m_baselineLine->setPen(QPen(QColor(220, 50, 50), 2, Qt::SolidLine));
        m_baselineLine->setVisible(false);
      }
      if (!m_meanLabel) {
        m_meanLabel = new QCPItemText(m_plot);
        m_meanLabel->setColor(Qt::yellow);
        m_meanLabel->setPositionAlignment(Qt::AlignBottom | Qt::AlignHCenter);
        m_meanLabel->setVisible(false);
      }
      if (!m_baselineLabel) {
        m_baselineLabel = new QCPItemText(m_plot);
        m_baselineLabel->setColor(QColor(255, 120, 120));
        m_baselineLabel->setPositionAlignment(Qt::AlignBottom | Qt::AlignHCenter);
        m_baselineLabel->setVisible(false);
      }
      if (m_pctLines.isEmpty()) {
        for (int i = 0; i < 6; ++i) {
          auto *line = new QCPItemLine(m_plot);
          line->start->setType(QCPItemPosition::ptPlotCoords);
          line->end->setType(QCPItemPosition::ptPlotCoords);
          line->start->setAxes(m_plot->xAxis, m_plot->yAxis);
          line->end->setAxes(m_plot->xAxis, m_plot->yAxis);
          line->setPen(QPen(QColor(200, 200, 200), 1, Qt::DashLine));
          line->setVisible(false);
          m_pctLines.push_back(line);
          auto *txt = new QCPItemText(m_plot);
          txt->applyThemeColor(ThemeManager::instance().textColor());
          txt->setPositionAlignment(Qt::AlignBottom | Qt::AlignHCenter);
          txt->setVisible(false);
          m_pctLabels.push_back(txt);
        }
      }
      m_plot->replot();
      // If cached data exists, draw histogram instantly
      if (hasCache()) {
        plotHistogramFromCache();
      }
    }
  });
}

void RandomStudyDialog::setupHalfScreenSize() {
  QScreen *screen = QGuiApplication::primaryScreen();
  if (!screen) {
    resize(800, 600);
    // Set initial splitter ratio
    if (m_splitter) m_splitter->setSizes({560, 240});
    return;
  }
  QRect avail = screen->availableGeometry();
  QSize sz(avail.width() / 2, avail.height() / 2);
  resize(sz);
  move(avail.center() - QPoint(width() / 2, height() / 2));
  if (m_splitter) {
    int w = width();
    m_splitter->setSizes({static_cast<int>(w * 0.7), static_cast<int>(w * 0.3)});
  }
}

void RandomStudyDialog::setROINames(const QStringList &roiNames) {
  m_roiCombo->clear();
  m_roiCombo->addItems(roiNames);
}

void RandomStudyDialog::setCalculationProgress(int processed, int total) {
  if (!m_progress)
    return;
  if (total <= 0) {
    m_progress->setRange(0, 0); // busy indicator
    return;
  }
  m_progress->setRange(0, 100);
  int pct = static_cast<int>(std::round(100.0 * processed / total));
  pct = std::clamp(pct, 0, 100);
  m_progress->setValue(pct);
}

void RandomStudyDialog::clearPlot() {
  if (!m_plot) return;
  m_plot->clearGraphs();
  m_plot->xAxis->setLabel(isXAxisPercent() ? tr("Dose [%]") : tr("Dose [Gy]"));
  m_plot->yAxis->setLabel(tr("Volume [%]"));
  m_plot->yAxis->setRange(0.0, 100.0);
  m_plot->replot();
}

void RandomStudyDialog::addDVHCurve(const QVector<double> &xGy, const QVector<double> &yPct,
                                    const QColor &color, bool replot, int width) {
  if (!m_plot) return;
  auto *g = m_plot->addGraph();
  QPen pen(color);
  pen.setWidth(std::max(1, width));
  g->setPen(pen);
  g->setData(xGy, yPct);
  if (replot) m_plot->replot();
}

void RandomStudyDialog::setDoseAxisMax(double maxGy) {
  if (!m_plot) return;
  double upper = std::max(1.0, maxGy * 1.05);
  if (m_xPercentRef && !isXAxisPercent()) m_xPercentRef->setValue(maxGy);
  if (isXAxisPercent()) {
    m_plot->xAxis->setLabel(tr("Dose [%]"));
    m_plot->xAxis->setRange(0.0, 100.0);
    m_plot->xAxis->setAutoTicks(false);
    QVector<double> ticks; ticks.reserve(11);
    for (int tck = 0; tck <= 100; tck += 10) ticks.push_back(static_cast<double>(tck));
    m_plot->xAxis->setTickVector(ticks);
  } else {
    m_plot->xAxis->setLabel(tr("Dose [Gy]"));
    m_plot->xAxis->setAutoTicks(true);
    m_plot->xAxis->setRange(0.0, upper);
  }
  m_plot->replot();
}

RandomStudyDialog::ViewMode RandomStudyDialog::viewMode() const {
  if (!m_viewModeCombo) return ViewMode::DVH;
  return static_cast<ViewMode>(m_viewModeCombo->currentData().toInt());
}

RandomStudyDialog::HistType RandomStudyDialog::histogramType() const {
  if (!m_histTypeCombo) return HistType::DxCc;
  return static_cast<HistType>(m_histTypeCombo->currentData().toInt());
}

double RandomStudyDialog::histogramParam() const {
  return m_histParamSpin ? m_histParamSpin->value() : 0.0;
}

int RandomStudyDialog::histogramBins() const {
  return m_histBinsSpin ? m_histBinsSpin->value() : 30;
}

bool RandomStudyDialog::isXAxisPercent() const {
  return m_xPercentCheck && m_xPercentCheck->isChecked();
}

double RandomStudyDialog::xPercentRefGy() const {
  return m_xPercentRef ? m_xPercentRef->value() : 1.0;
}

void RandomStudyDialog::plotHistogram(const QVector<double> &values, int bins,
                                      const QString &xLabel, const QString &yLabel) {
  if (!m_plot) return;
  m_plot->clearGraphs();
  if (values.isEmpty() || bins <= 0) {
    m_plot->replot();
    return;
  }

  double minV = values[0];
  double maxV = values[0];
  for (double v : values) { minV = std::min(minV, v); maxV = std::max(maxV, v); }
  if (!(maxV > minV)) { maxV = minV + 1.0; }
  double width = (maxV - minV) / static_cast<double>(bins);
  QVector<double> keys, counts;
  keys.resize(bins);
  counts.fill(0.0, bins);
  for (int i = 0; i < bins; ++i) keys[i] = minV + (i + 0.5) * width;
  for (double v : values) {
    int b = static_cast<int>(std::floor((v - minV) / width));
    if (b < 0) b = 0;
    if (b >= bins) b = bins - 1;
    counts[b] += 1.0;
  }
  // Draw as filled bars using two graphs (top shape + baseline), to emulate bar chart
  QVector<double> xs, ysTop, ysBase;
  xs.reserve(bins * 4 + 2);
  ysTop.reserve(bins * 4 + 2);
  ysBase.reserve(bins * 4 + 2);
  const double gap = width * 0.05;
  for (int i = 0; i < bins; ++i) {
    double left = minV + i * width;
    double right = left + width - gap; // leave a small gap
    double top = counts[i];
    // baseline start at left
    xs << left;     ysTop << 0.0;    ysBase << 0.0;
    // up
    xs << left;     ysTop << top;     ysBase << 0.0;
    // across
    xs << right;    ysTop << top;     ysBase << 0.0;
    // down
    xs << right;    ysTop << 0.0;     ysBase << 0.0;
  }
  auto *gBase = m_plot->addGraph();
  gBase->setData(xs, ysBase);
  gBase->setPen(Qt::NoPen);
  auto *gTop = m_plot->addGraph();
  gTop->setData(xs, ysTop);
  QPen pen(QColor(50, 90, 140)); pen.setWidth(1);
  gTop->setPen(pen);
  gTop->setBrush(QColor(80, 160, 220, 200));
  gTop->setChannelFillGraph(gBase);
  m_plot->xAxis->setLabel(xLabel);
  m_plot->yAxis->setLabel(yLabel);
  m_plot->xAxis->setRange(minV, maxV);
  double maxC = 0.0; for (double c : counts) maxC = std::max(maxC, c);
  m_plot->yAxis->setRange(0.0, std::max(1.0, maxC * 1.1));
  m_plot->replot();
}

void RandomStudyDialog::prepareDVHAxes() {
  if (!m_plot) return;
  m_plot->clearGraphs();
  m_plot->xAxis->setLabel(isXAxisPercent() ? tr("Dose [%]") : tr("Dose [Gy]"));
  m_plot->yAxis->setLabel(tr("Volume [%]"));
  m_plot->yAxis->setRange(0.0, 100.0);
  m_plot->replot();
}

void RandomStudyDialog::setHistogramMarkers(double mean,
                           const QVector<QPair<double, QString>> &marks) {
  if (!m_plot) return;
  // Ensure items exist
  if (!m_meanLine) {
    m_meanLine = new QCPItemLine(m_plot);
    m_meanLine->start->setType(QCPItemPosition::ptPlotCoords);
    m_meanLine->end->setType(QCPItemPosition::ptPlotCoords);
    m_meanLine->start->setAxes(m_plot->xAxis, m_plot->yAxis);
    m_meanLine->end->setAxes(m_plot->xAxis, m_plot->yAxis);
    m_meanLine->setPen(QPen(QColor(255, 220, 0), 2, Qt::DashLine));
  }
  if (!m_meanLabel) {
    m_meanLabel = new QCPItemText(m_plot);
    m_meanLabel->setColor(Qt::yellow);
    m_meanLabel->setPositionAlignment(Qt::AlignBottom | Qt::AlignHCenter);
  }
  if (m_pctLines.size() < 6) {
    while (m_pctLines.size() < 6) {
      auto *line = new QCPItemLine(m_plot);
      line->start->setType(QCPItemPosition::ptPlotCoords);
      line->end->setType(QCPItemPosition::ptPlotCoords);
      line->start->setAxes(m_plot->xAxis, m_plot->yAxis);
      line->end->setAxes(m_plot->xAxis, m_plot->yAxis);
      line->setPen(QPen(QColor(200, 200, 200), 1, Qt::DashLine));
      m_pctLines.push_back(line);
      auto *txt = new QCPItemText(m_plot);
      txt->applyThemeColor(ThemeManager::instance().textColor());
      txt->setPositionAlignment(Qt::AlignBottom | Qt::AlignHCenter);
      m_pctLabels.push_back(txt);
    }
  }

  double yTop = m_plot->yAxis->max();
  double yMin = m_plot->yAxis->min();
  double ySpan = std::max(1e-9, yTop - yMin);
  // Mean (slightly below top so Baseline can sit at very top)
  const double h1 = std::max(yMin, yTop - 0.02 * ySpan);
  m_meanLine->start->setCoords(mean, yTop);
  m_meanLine->end->setCoords(mean, yMin);
  m_meanLine->setVisible(true);
  m_meanLabel->position->setType(QCPItemPosition::ptPlotCoords);
  m_meanLabel->position->setAxes(m_plot->xAxis, m_plot->yAxis);
  m_meanLabel->position->setCoords(mean, h1);
  m_meanLabel->setText(tr("Mean\n%1").arg(mean, 0, 'f', 2));
  m_meanLabel->setVisible(true);

  // Percentiles markers
  // marks order: [P5, P10, P25, P75, P90, P95]
  // Heights: (2) P25/P75 at 4%, (3) P10/P90 at 8%, (4) P5/P95 at 12%
  // Compute label heights as fractions below top
  const double h2 = std::max(yMin, yTop - 0.06 * ySpan);
  const double h3 = std::max(yMin, yTop - 0.10 * ySpan);
  const double h4 = std::max(yMin, yTop - 0.14 * ySpan);

  int n = std::min(marks.size(), m_pctLines.size());
  for (int i = 0; i < n; ++i) {
    double x = marks[i].first;
    const QString &label = marks[i].second;
    m_pctLines[i]->start->setCoords(x, yTop);
    m_pctLines[i]->end->setCoords(x, yMin);
    m_pctLines[i]->setVisible(true);
    m_pctLabels[i]->position->setType(QCPItemPosition::ptPlotCoords);
    m_pctLabels[i]->position->setAxes(m_plot->xAxis, m_plot->yAxis);
    double yLabel = h4; // default lowest
    if (i == 2 || i == 3) {
      yLabel = h2; // P25/P75
    } else if (i == 1 || i == 4) {
      yLabel = h3; // P10/P90
    } else if (i == 0 || i == 5) {
      yLabel = h4; // P5/P95
    }
    m_pctLabels[i]->position->setCoords(x, yLabel);
    // Show label and value on separate lines
    m_pctLabels[i]->setText(QString("%1\n%2").arg(label).arg(x, 0, 'f', 2));
    m_pctLabels[i]->setVisible(true);
  }
  // Hide any extra markers if previously visible
  for (int i = n; i < m_pctLines.size(); ++i) {
    if (m_pctLines[i]) m_pctLines[i]->setVisible(false);
    if (m_pctLabels[i]) m_pctLabels[i]->setVisible(false);
  }
  m_plot->replot();
}

int RandomStudyDialog::selectedROIIndex() const {
  return m_roiCombo ? m_roiCombo->currentIndex() : -1;
}

int RandomStudyDialog::fractionCount() const {
  return m_fractionSpin ? m_fractionSpin->value() : 1;
}

int RandomStudyDialog::iterationCount() const {
  return m_iterationsSpin ? m_iterationsSpin->value() : 1;
}

void RandomStudyDialog::systematicError(double &mx, double &my, double &mz,
                                        double &sx, double &sy, double &sz) const {
  mx = m_sysX ? m_sysX->value() : 0.0;
  my = m_sysY ? m_sysY->value() : 0.0;
  mz = m_sysZ ? m_sysZ->value() : 0.0;
  sx = m_sysXpm ? m_sysXpm->value() : 0.0;
  sy = m_sysYpm ? m_sysYpm->value() : 0.0;
  sz = m_sysZpm ? m_sysZpm->value() : 0.0;
}

void RandomStudyDialog::randomError(double &sx, double &sy, double &sz) const {
  sx = m_randX ? m_randX->value() : 0.0;
  sy = m_randY ? m_randY->value() : 0.0;
  sz = m_randZ ? m_randZ->value() : 0.0;
}
bool RandomStudyDialog::isSeedFixed() const {
  return m_fixSeedCheck && m_fixSeedCheck->isChecked();
}

quint64 RandomStudyDialog::seedValue() const {
  return m_seedSpin ? static_cast<quint64>(m_seedSpin->value()) : 0ULL;
}

void RandomStudyDialog::setRunning(bool running) {
  // Disable inputs while running
  for (QWidget *w : {static_cast<QWidget*>(m_roiCombo), static_cast<QWidget*>(m_fractionSpin),
                     static_cast<QWidget*>(m_iterationsSpin), static_cast<QWidget*>(m_fixSeedCheck),
                     static_cast<QWidget*>(m_seedSpin), static_cast<QWidget*>(m_sysX), static_cast<QWidget*>(m_sysY), static_cast<QWidget*>(m_sysZ),
                     static_cast<QWidget*>(m_sysXpm), static_cast<QWidget*>(m_sysYpm), static_cast<QWidget*>(m_sysZpm),
                     static_cast<QWidget*>(m_randX), static_cast<QWidget*>(m_randY), static_cast<QWidget*>(m_randZ)}) {
    if (w) w->setEnabled(!running);
  }
  if (m_startButton) m_startButton->setEnabled(!running);
  if (m_cancelButton) m_cancelButton->setEnabled(running);
}

void RandomStudyDialog::closeEvent(QCloseEvent *event) {
  Q_UNUSED(event);
  clearCache();
}

void RandomStudyDialog::clearCache() {
  m_cachedHists.clear();
  m_cachedBinSize = 0.0;
  m_cachedTotalVolume = 0.0;
}

void RandomStudyDialog::setCachedHistograms(const QVector<QVector<float>> &hists,
                                            double binSize,
                                            double totalVolume) {
  m_cachedHists = hists;
  m_cachedBinSize = binSize;
  m_cachedTotalVolume = totalVolume;
}

void RandomStudyDialog::plotHistogramFromCache() {
  if (!m_plot || m_cachedHists.isEmpty() || m_cachedBinSize <= 0.0 || m_cachedTotalVolume <= 0.0)
    return;

  // Compute per-iteration scalar metric based on current UI selection
  const auto t = histogramType();
  const double param = histogramParam();
  const int binsOut = histogramBins();

  QVector<double> values;
  values.reserve(m_cachedHists.size());

  const int binCount = m_cachedHists[0].size();
  const double binSize = m_cachedBinSize;
  const double totalVol = m_cachedTotalVolume;

  // For Dx-type, we need cumulative from high dose to low once per iteration
  auto doseAtVolume = [&](const QVector<float> &hist, double targetVol) -> double {
    double cum = 0.0;
    for (int i = binCount - 1; i >= 0; --i) {
      cum += hist[i];
      if (cum >= targetVol) return i * binSize;
    }
    return 0.0;
  };

  for (const auto &hist : m_cachedHists) {
    double metric = 0.0;
    switch (t) {
    case HistType::DxCc: {
      double targetVol = std::clamp(param, 0.0, totalVol);
      metric = doseAtVolume(hist, targetVol);
      break; }
    case HistType::DxPct: {
      double pct = std::clamp(param, 0.0, 100.0);
      double targetVol = totalVol * (pct / 100.0);
      metric = doseAtVolume(hist, targetVol);
      break; }
    case HistType::VdCc: {
      int thBin = std::max(0, std::min(binCount - 1, static_cast<int>(param / binSize)));
      double volMm3 = 0.0;
      for (int i = thBin; i < binCount; ++i) volMm3 += hist[i];
      metric = volMm3 / 1000.0; // result in cc
      break; }
    case HistType::MinDose: {
      double md = std::numeric_limits<double>::infinity();
      for (int i = 0; i < binCount; ++i) {
        if (hist[i] > 0.0f) { md = i * binSize; break; }
      }
      if (!std::isfinite(md)) md = 0.0;
      metric = md;
      break; }
    case HistType::MaxDose: {
      double Md = 0.0;
      for (int i = binCount - 1; i >= 0; --i) {
        if (hist[i] > 0.0f) { Md = i * binSize; break; }
      }
      metric = Md;
      break; }
    case HistType::MeanDose: {
      double sum = 0.0;
      for (int i = 0; i < binCount; ++i) sum += static_cast<double>(hist[i]) * (i * binSize);
      metric = (totalVol > 0.0) ? (sum / totalVol) : 0.0;
      break; }
    }
    values.push_back(metric);
  }

  QString xLabel = tr("Value");
  const bool xDoseLike = (t != HistType::VdCc);
  if (!xDoseLike) xLabel = tr("Volume [cc]");
  else xLabel = isXAxisPercent() ? tr("Dose [%]") : tr("Dose [Gy]");
  // If dose-like and percent mode, convert values to %
  const double refGy = xPercentRefGy();
  if (xDoseLike && isXAxisPercent() && refGy > 0.0) {
    for (double &v : values) v = v / refGy * 100.0;
  }
  plotHistogram(values, std::max(5, binsOut), xLabel, tr("Count"));
  // Set nice percent ticks in percent mode; otherwise auto ticks
  if (xDoseLike && isXAxisPercent()) {
    if (m_plot && m_plot->xAxis) {
      m_plot->xAxis->setAutoTicks(false);
      QVector<double> ticks; ticks.reserve(11);
      for (int tck = 0; tck <= 100; tck += 10) ticks.push_back(static_cast<double>(tck));
      m_plot->xAxis->setTickVector(ticks);
      m_plot->replot();
    }
  } else if (m_plot && m_plot->xAxis) {
    m_plot->xAxis->setAutoTicks(true);
  }

  // Also compute and overlay mean + percentiles markers
  if (!values.isEmpty()) {
    QVector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    double mean = 0.0; for (double v : values) mean += v; mean /= static_cast<double>(values.size());
    auto pctAt = [&](double p) -> double {
      if (sorted.isEmpty()) return 0.0;
      double pos = p * (sorted.size() - 1) / 100.0;
      int idx = static_cast<int>(std::floor(pos));
      int idx2 = std::min(static_cast<int>(sorted.size() - 1), idx + 1);
      double tlin = pos - idx;
      return sorted[idx] * (1.0 - tlin) + sorted[idx2] * tlin;
    };
    QVector<QPair<double, QString>> marks;
    // pctAt is in the current value units (already % if converted above)
    marks << qMakePair(pctAt(5.0),  tr("P5"))
          << qMakePair(pctAt(10.0), tr("P10"))
          << qMakePair(pctAt(25.0), tr("P25"))
          << qMakePair(pctAt(75.0), tr("P75"))
          << qMakePair(pctAt(90.0), tr("P90"))
          << qMakePair(pctAt(95.0), tr("P95"));
    setHistogramMarkers(mean, marks);
  }

  // Baseline (no-error) metric marker (red)
  if (hasBaseline()) {
    const auto tcur = histogramType();
    const double param = histogramParam();
    const int binCount = m_baselineHist.size();
    const double binSize = m_baselineBinSize;
    const double totalVol = m_baselineTotalVolume;
    auto doseAtVolume = [&](double targetVol) -> double {
      double cum = 0.0;
      for (int i = binCount - 1; i >= 0; --i) {
        cum += m_baselineHist[i];
        if (cum >= targetVol) return i * binSize;
      }
      return 0.0;
    };
    double baseVal = 0.0;
    switch (tcur) {
      case HistType::DxCc: baseVal = doseAtVolume(std::clamp(param, 0.0, totalVol)); break;
      case HistType::DxPct: baseVal = doseAtVolume(totalVol * std::clamp(param, 0.0, 100.0) / 100.0); break;
      case HistType::VdCc: {
        int thBin = std::max(0, std::min(binCount - 1, static_cast<int>(param / binSize)));
        double volMm3 = 0.0; for (int i = thBin; i < binCount; ++i) volMm3 += m_baselineHist[i];
        baseVal = volMm3 / 1000.0; // cc
        break; }
      case HistType::MinDose: {
        double md = 0.0; bool found=false; for (int i=0;i<binCount;++i) if (m_baselineHist[i]>0.0f){ md=i*binSize; found=true; break; }
        baseVal = found?md:0.0; break; }
      case HistType::MaxDose: {
        double Md = 0.0; for (int i=binCount-1;i>=0;--i) if (m_baselineHist[i]>0.0f){ Md=i*binSize; break; }
        baseVal = Md; break; }
      case HistType::MeanDose: {
        double sum=0.0; for (int i=0;i<binCount;++i) sum += static_cast<double>(m_baselineHist[i]) * (i*binSize);
        baseVal = (totalVol>0.0)?(sum/totalVol):0.0; break; }
    }
    // Convert baseline dose to percent if needed for dose-like axes
    if (tcur != HistType::VdCc && isXAxisPercent() && refGy > 0.0) {
      baseVal = baseVal / refGy * 100.0;
    }
    // Draw red line + label
    double yTop = m_plot->yAxis->max();
    double yMin = m_plot->yAxis->min();
    if (!m_baselineLine) {
      m_baselineLine = new QCPItemLine(m_plot);
      m_baselineLine->start->setType(QCPItemPosition::ptPlotCoords);
      m_baselineLine->end->setType(QCPItemPosition::ptPlotCoords);
      m_baselineLine->start->setAxes(m_plot->xAxis, m_plot->yAxis);
      m_baselineLine->end->setAxes(m_plot->xAxis, m_plot->yAxis);
      m_baselineLine->setPen(QPen(QColor(220, 50, 50), 2, Qt::SolidLine));
    }
    if (!m_baselineLabel) {
      m_baselineLabel = new QCPItemText(m_plot);
      m_baselineLabel->setColor(QColor(255, 120, 120));
      m_baselineLabel->setPositionAlignment(Qt::AlignBottom | Qt::AlignHCenter);
    }
    m_baselineLine->start->setCoords(baseVal, yTop);
    m_baselineLine->end->setCoords(baseVal, yMin);
    m_baselineLine->setVisible(true);
    m_baselineLabel->position->setType(QCPItemPosition::ptPlotCoords);
    m_baselineLabel->position->setAxes(m_plot->xAxis, m_plot->yAxis);
    m_baselineLabel->position->setCoords(baseVal, yTop);
    m_baselineLabel->setText(tr("Baseline\n%1").arg(baseVal, 0, 'f', 2));
    m_baselineLabel->setVisible(true);
    m_plot->replot();
    // Update info panel text
    if (m_infoText) {
      QStringList lines;
      lines << tr("ROI: %1").arg(m_roiCombo ? m_roiCombo->currentText() : tr("(none)"));
      lines << tr("Iterations: %1").arg(iterationCount());
      double mx,my,mz,sx,sy,sz; systematicError(mx,my,mz,sx,sy,sz);
      double rx,ry,rz; randomError(rx,ry,rz);
      lines << tr("Systematic: X %1 ± %2 mm, Y %3 ± %4 mm, Z %5 ± %6 mm")
                  .arg(mx,0,'f',2).arg(sx,0,'f',2)
                  .arg(my,0,'f',2).arg(sy,0,'f',2)
                  .arg(mz,0,'f',2).arg(sz,0,'f',2);
      lines << tr("Random σ: X %1 mm, Y %2 mm, Z %3 mm")
                  .arg(rx,0,'f',2).arg(ry,0,'f',2).arg(rz,0,'f',2);
      // Mode string
      auto modeStr = [&](){
        switch (histogramType()) {
          case HistType::DxCc: return tr("D%1cc").arg(histogramParam(),0,'f',2);
          case HistType::DxPct: return tr("D%1%%").arg(histogramParam(),0,'f',2);
          case HistType::VdCc: return tr("V%1 Gy").arg(histogramParam(),0,'f',2);
          case HistType::MinDose: return tr("Min Dose");
          case HistType::MaxDose: return tr("Max Dose");
          case HistType::MeanDose: return tr("Mean Dose");
        }
        return tr("(unknown)");
      }();
      lines << tr("Mode: %1").arg(modeStr);
      // Values summary
      auto fmt = [&](double v){ return QString::number(v, 'f', 2); };
      QString unit = (histogramType()==HistType::VdCc) ? tr(" cc") : (isXAxisPercent()? tr(" %") : tr(" Gy"));
      // Recompute mean and percentiles on possibly scaled values
      QVector<double> vals = values;
      std::sort(vals.begin(), vals.end());
      auto pctAt2 = [&](double p){ if (vals.isEmpty()) return 0.0; double pos=p*(vals.size()-1)/100.0; int i=floor(pos); int j=std::min((int)vals.size()-1,i+1); double t=pos-i; return vals[i]*(1.0-t)+vals[j]*t; };
      double mean2 = 0.0; for (double v: vals) mean2 += v; mean2 /= (vals.isEmpty() ? 1.0 : static_cast<double>(vals.size()));
      lines << tr("Original: %1%2").arg(fmt(baseVal)).arg(unit);
      lines << tr("P5: %1%2, P10: %3%2, P25: %4%2, P75: %5%2, P90: %6%2, P95: %7%2")
                  .arg(fmt(pctAt2(5))).arg(unit)
                  .arg(fmt(pctAt2(10)))
                  .arg(fmt(pctAt2(25)))
                  .arg(fmt(pctAt2(75)))
                  .arg(fmt(pctAt2(90)))
                  .arg(fmt(pctAt2(95)));
      lines << tr("Mean: %1%2").arg(fmt(mean2)).arg(unit);
      m_infoText->setPlainText(lines.join("\n"));
    }
  }
}

void RandomStudyDialog::plotDVHsFromCache(int maxCurves) {
  if (!m_plot || m_cachedHists.isEmpty() || m_cachedBinSize <= 0.0 || m_cachedTotalVolume <= 0.0)
    return;
  prepareDVHAxes();
  m_plot->clearGraphs();

  const int binCount = m_cachedHists[0].size();
  const double binSize = m_cachedBinSize;
  const double totalVol = m_cachedTotalVolume;
  // Ensure X axis range covers max dose or 0..100% if percent
  if (isXAxisPercent()) {
    m_plot->xAxis->setLabel(tr("Dose [%]"));
    m_plot->xAxis->setRange(0.0, 100.0);
    // Set nice percent ticks
    if (m_plot && m_plot->xAxis) {
      m_plot->xAxis->setAutoTicks(false);
      QVector<double> ticks; ticks.reserve(11);
      for (int tck = 0; tck <= 100; tck += 10) ticks.push_back(static_cast<double>(tck));
      m_plot->xAxis->setTickVector(ticks);
    }
  } else {
    setDoseAxisMax(binCount * binSize);
  }

  const int totalIters = static_cast<int>(m_cachedHists.size());
  const int safeMaxCurves = std::max(1, maxCurves);
  const int stride = std::max(1, (totalIters + safeMaxCurves - 1) / safeMaxCurves);
  int plotted = 0;
  // First, if baseline exists, plot it in opaque red
  if (hasBaseline()) {
    QVector<double> x, y;
    x.reserve(binCount); y.reserve(binCount);
    double cumulative = 0.0;
    for (int i = binCount - 1; i >= 0; --i) {
      cumulative += m_baselineHist[i];
      double volPct = totalVol > 0.0 ? (cumulative / totalVol * 100.0) : 0.0;
      double doseVal = i * binSize;
      if (isXAxisPercent()) {
        double refGy = xPercentRefGy();
        doseVal = (refGy > 0.0) ? (doseVal / refGy * 100.0) : 0.0;
      }
      x.push_back(doseVal);
      y.push_back(volPct);
    }
    std::reverse(x.begin(), x.end());
    std::reverse(y.begin(), y.end());
    addDVHCurve(x, y, QColor(255,0,0), false, 2);
  }
  for (int it = 0; it < totalIters; ++it) {
    if (it % stride != 0 || plotted >= maxCurves) continue;
    ++plotted;
    const auto &hist = m_cachedHists[it];
    // Build cumulative DVH (percent)
    QVector<double> x, y;
    x.reserve(binCount);
    y.reserve(binCount);
    double cumulative = 0.0;
    for (int i = binCount - 1; i >= 0; --i) {
      cumulative += hist[i];
      double volPct = totalVol > 0.0 ? (cumulative / totalVol * 100.0) : 0.0;
      double doseVal = i * binSize;
      if (isXAxisPercent()) {
        double refGy = xPercentRefGy();
        doseVal = (refGy > 0.0) ? (doseVal / refGy * 100.0) : 0.0;
      }
      x.push_back(doseVal);
      y.push_back(volPct);
    }
    std::reverse(x.begin(), x.end());
    std::reverse(y.begin(), y.end());
    QColor col = QColor::fromHsv((it * 37) % 360, 200, 255);
    col.setAlpha(70);
    addDVHCurve(x, y, col, false);
  }
  m_plot->replot();
}
