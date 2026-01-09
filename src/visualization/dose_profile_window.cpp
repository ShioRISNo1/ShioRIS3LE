#include "visualization/dose_profile_window.h"
#include <QDebug>
#include <QHBoxLayout>
#include <QLabel>
#include <QMap>
#include <QMessageBox>
#include <QPen>
#include <QPlainTextEdit>
#include <QResizeEvent>
#include <QMouseEvent>
#include <QShowEvent>
#include <QString>
#include <QTimer>
#include <QVBoxLayout>
#include <QtConcurrent>
#include <QFont>
#include <QFontMetrics>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "theme_manager.h"

DoseProfileWindow::DoseProfileWindow(QWidget *parent) : QWidget(parent) {
  QHBoxLayout *mainLayout = new QHBoxLayout(this);
  mainLayout->setContentsMargins(2, 2, 2, 2);
  mainLayout->setSpacing(3);

  m_plot = new QCustomPlot(this);
  const int axisLabelPointSize = 8;
  const int tickLabelPointSize = 7;
  const int overlayPointSize = 8;
  const int infoBoxPointSize = 8;
  QFont axisLabelFont = m_plot->xAxis->labelFont();
  axisLabelFont.setPointSize(axisLabelPointSize);
  m_plot->xAxis->setLabelFont(axisLabelFont);
  m_plot->yAxis->setLabelFont(axisLabelFont);
  QFont tickLabelFont = m_plot->xAxis->tickLabelFont();
  tickLabelFont.setPointSize(tickLabelPointSize);
  m_plot->xAxis->setTickLabelFont(tickLabelFont);
  m_plot->yAxis->setTickLabelFont(tickLabelFont);
  m_plot->setPlotMargins(50, 24, 28, 36);
  QFont overlayFont = m_plot->font();
  overlayFont.setPointSize(overlayPointSize);
  m_plot->xAxis->setLabel("Position (mm)");
  m_plot->yAxis->setLabel("Dose (Gy)");
  m_plot->setMouseTracking(true);
  m_plot->installEventFilter(this);
  m_plot->setContentsMargins(2, 2, 2, 2);
  mainLayout->addWidget(m_plot, 7);

  // DPSD表示用のタイトルとフィルターROIラベル（絶対座標表示）
  m_dpsdTitleLabel = new QLabel(m_plot);
  m_dpsdTitleLabel->setFont(overlayFont);
  ThemeManager &theme = ThemeManager::instance();
  theme.applyTextColor(
      m_dpsdTitleLabel,
      QStringLiteral(
          "QLabel { background: transparent; color: %1; padding:2px; }"));
  m_dpsdTitleLabel->setAttribute(Qt::WA_TransparentForMouseEvents);
  m_dpsdTitleLabel->setVisible(false);

  m_filterRoiLabel = new QLabel(m_plot);
  m_filterRoiLabel->setFont(overlayFont);
  theme.applyTextColor(
      m_filterRoiLabel,
      QStringLiteral(
          "QLabel { background: transparent; color: %1; padding:2px; }"));
  m_filterRoiLabel->setAttribute(Qt::WA_TransparentForMouseEvents);
  m_filterRoiLabel->setVisible(false);

  m_cursorLine = new QCPItemLine(m_plot);
  m_cursorLine->setPen(QPen(Qt::gray, 1, Qt::DashLine));
  m_cursorLine->start->setType(QCPItemPosition::ptPlotCoords);
  m_cursorLine->start->setAxes(m_plot->xAxis, m_plot->yAxis);
  m_cursorLine->end->setType(QCPItemPosition::ptPlotCoords);
  m_cursorLine->end->setAxes(m_plot->xAxis, m_plot->yAxis);
  m_cursorLine->setVisible(false);

  m_cursorInfoLabel = new QLabel(m_plot);
  m_cursorInfoLabel->setFont(overlayFont);
  theme.applyTextColor(
      m_cursorInfoLabel,
      QStringLiteral(
          "QLabel { background: transparent; color: %1; padding:2px; }"));
  m_cursorInfoLabel->setAttribute(Qt::WA_TransparentForMouseEvents);
  m_cursorInfoLabel->setVisible(false);
  m_cursorInfoLabel->raise();

  QWidget *rightPanel = new QWidget(this);
  rightPanel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
  rightPanel->setMinimumWidth(160);
  QVBoxLayout *rightLayout = new QVBoxLayout(rightPanel);
  rightLayout->setContentsMargins(0, 0, 0, 0);
  rightLayout->setSpacing(4);

  QHBoxLayout *modeLayout = new QHBoxLayout();
  m_profileRadio = new QRadioButton(tr("Profile"), rightPanel);
  m_dpsdRadio = new QRadioButton(tr("DPSD"), rightPanel);
  m_profileRadio->setChecked(true);
  modeLayout->addWidget(m_profileRadio);
  modeLayout->addWidget(m_dpsdRadio);
  rightLayout->addLayout(modeLayout);
  connect(m_profileRadio, &QRadioButton::toggled, this, [this](bool checked) {
    if (checked)
      updatePlotForProfile();
  });
  connect(m_dpsdRadio, &QRadioButton::toggled, this, [this](bool checked) {
    if (checked)
      updatePlotForDPSD();
  });
  QLabel *profileLabel = new QLabel(tr("Dose Profile"), rightPanel);
  rightLayout->addWidget(profileLabel);

  m_pickButton = new QPushButton(tr("Select Line"), rightPanel);
  connect(m_pickButton, &QPushButton::clicked, this, [this]() {
    if (m_profileRadio)
      m_profileRadio->setChecked(true);
    emit requestLineSelection();
  });
  rightLayout->addWidget(m_pickButton);

  QHBoxLayout *presetLayout = new QHBoxLayout();
  presetLayout->setContentsMargins(0, 0, 0, 0);
  presetLayout->setSpacing(2);
  m_slotHasData = QVector<bool>(3, false);
  m_slotButtons.reserve(3);
  for (int i = 0; i < 3; ++i) {
    QPushButton *slotButton = new QPushButton(QString::number(i + 1), rightPanel);
    slotButton->setFixedSize(24, 24);
    slotButton->setFocusPolicy(Qt::StrongFocus);
    connect(slotButton, &QPushButton::clicked, this, [this, i]() {
      if (m_pendingSave) {
        emit saveLineRequested(i);
        if (m_saveButton)
          m_saveButton->setChecked(false);
      } else {
        emit loadLineRequested(i);
      }
      refreshSlotButtonStates();
    });
    m_slotButtons.append(slotButton);
    presetLayout->addWidget(slotButton);
  }

  m_saveButton = new QPushButton(tr("Save"), rightPanel);
  m_saveButton->setCheckable(true);
  m_saveButton->setFixedHeight(24);
  connect(m_saveButton, &QPushButton::toggled, this, [this](bool checked) {
    m_pendingSave = checked;
    refreshSlotButtonStates();
  });
  presetLayout->addWidget(m_saveButton);
  rightLayout->addLayout(presetLayout);

  refreshSlotButtonStates();

  rightLayout->addSpacing(8);

  // DPS-D解析用グループ
  m_dpsdGroup = new QGroupBox(tr("DPS-D"), rightPanel);
  QVBoxLayout *dpsdLayout = new QVBoxLayout(m_dpsdGroup);
  dpsdLayout->setContentsMargins(2, 2, 2, 2);
  dpsdLayout->setSpacing(2);
  m_roiCombo = new QComboBox(m_dpsdGroup);
  dpsdLayout->addWidget(m_roiCombo);

  QHBoxLayout *rangeLayout = new QHBoxLayout();
  rangeLayout->setSpacing(2);
  m_startSpin = new QDoubleSpinBox(m_dpsdGroup);
  m_startSpin->setRange(-1000.0, 1000.0);
  m_startSpin->setValue(-20.0);
  m_startSpin->setSuffix(" mm");
  rangeLayout->addWidget(m_startSpin);
  m_endSpin = new QDoubleSpinBox(m_dpsdGroup);
  m_endSpin->setRange(-1000.0, 1000.0);
  m_endSpin->setValue(50.0);
  m_endSpin->setSuffix(" mm");
  rangeLayout->addWidget(m_endSpin);
  dpsdLayout->addLayout(rangeLayout);

  m_modeCombo = new QComboBox(m_dpsdGroup);
  m_modeCombo->addItems({tr("2D"), tr("3D")});
  dpsdLayout->addWidget(m_modeCombo);

  m_sampleCombo = new QComboBox(m_dpsdGroup);
  m_sampleCombo->addItem("-");
  dpsdLayout->addWidget(m_sampleCombo);

  m_calcButton = new QPushButton(tr("Calculate"), m_dpsdGroup);
  connect(m_calcButton, &QPushButton::clicked, this,
          &DoseProfileWindow::onCalculateDPSD);
  dpsdLayout->addWidget(m_calcButton);

  m_exportButton = new QPushButton(tr("Export"), m_dpsdGroup);
  m_exportButton->setEnabled(false);
  connect(m_exportButton, &QPushButton::clicked, this,
          &DoseProfileWindow::onExportDPSD);
  dpsdLayout->addWidget(m_exportButton);

  // カーソル統計表示切替用チェックボックス
  m_showCursorCheck = new QCheckBox(tr("Show cursor stats"), m_dpsdGroup);
  m_showCursorCheck->setChecked(false);
  dpsdLayout->addWidget(m_showCursorCheck);
  connect(m_showCursorCheck, &QCheckBox::toggled, this, [this](bool checked) {
    if (m_cursorLine)
      m_cursorLine->setVisible(checked);
    if (m_cursorInfoLabel)
      m_cursorInfoLabel->setVisible(checked);
    if (m_plot)
      m_plot->replot();
  });

  m_progress = new QProgressBar(m_dpsdGroup);
  m_progress->setRange(0, 100);
  m_progress->setValue(0);
  m_progress->setTextVisible(false);
  m_progress->setFixedHeight(10);
  m_progress->setVisible(false);
  dpsdLayout->addWidget(m_progress);

  rightLayout->addWidget(m_dpsdGroup, 1);

  m_infoBox = new QPlainTextEdit(rightPanel);
  m_infoBox->setReadOnly(true);
  theme.applyTextColor(
      m_infoBox,
      QStringLiteral("QPlainTextEdit {background: #202020; color: %1; border:"
                    " 1px solid #444444;}"));
  m_infoBox->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
  QFont infoFont = m_infoBox->font();
  infoFont.setPointSize(infoBoxPointSize);
  m_infoBox->setFont(infoFont);
#if QT_VERSION >= QT_VERSION_CHECK(5, 10, 0)
  const QFontMetrics infoMetrics(infoFont);
  m_infoBox->setTabStopDistance(infoMetrics.horizontalAdvance(
      QStringLiteral("    ")));
#endif
  rightLayout->addWidget(m_infoBox, 2);

  mainLayout->addWidget(rightPanel, 3);
}

void DoseProfileWindow::setProfile(const QVector<double> &positions,
                                   const QVector<double> &doses,
                                   const QVector<Segment> &segments) {
  m_profilePositions = positions;
  m_profileDoses = doses;
  m_profileSegments = segments;
  if (m_profileRadio && m_profileRadio->isChecked())
    updatePlotForProfile();
}

void DoseProfileWindow::setStats(double lengthMm, double minDoseGy,
                                 double maxDoseGy,
                                 const QVector<Segment> &segments,
                                 const QVector<SamplePoint> &samplePoints) {
  QStringList lines;
  lines << tr("Length: %1 mm").arg(lengthMm, 0, 'f', 1)
        << tr("Min Dose: %1 Gy").arg(minDoseGy, 0, 'f', 2)
        << tr("Max Dose: %1 Gy").arg(maxDoseGy, 0, 'f', 2);
  QMap<QString, QPair<double, double>> roiStats;
  for (const Segment &s : segments) {
    auto it = roiStats.find(s.name);
    if (it == roiStats.end()) {
      roiStats.insert(s.name, qMakePair(s.minDoseGy, s.maxDoseGy));
    } else {
      it.value().first = std::min(it.value().first, s.minDoseGy);
      it.value().second = std::max(it.value().second, s.maxDoseGy);
    }
  }
  for (auto it = roiStats.cbegin(); it != roiStats.cend(); ++it) {
    lines << tr("%1: min %2 Gy, max %3 Gy")
                 .arg(it.key())
                 .arg(it.value().first, 0, 'f', 2)
                 .arg(it.value().second, 0, 'f', 2);
  }
  if (!samplePoints.isEmpty()) {
    lines << QString();
    lines << tr("Position (mm)\tCT (HU)\tDose (Gy)");
    for (const SamplePoint &sample : samplePoints) {
      const QString posText =
          QString::number(sample.positionMm, 'f', 1);
      const QString ctText = sample.ctHu.has_value()
                                 ? QString::number(sample.ctHu.value(), 'f', 0)
                                 : tr("N/A");
      const QString doseText = sample.doseGy.has_value()
                                   ? QString::number(sample.doseGy.value(), 'f', 2)
                                   : tr("N/A");
      lines << QStringList{posText, ctText, doseText}.join(QStringLiteral("\t"));
    }
  }
  m_infoBox->setPlainText(lines.join("\n"));
}

void DoseProfileWindow::setROINames(const QStringList &names) {
  if (m_roiCombo) {
    m_roiCombo->clear();
    m_roiCombo->addItems(names);
    int ptvIndex = -1;
    for (int i = 0; i < names.size(); ++i) {
      if (names[i].contains("PTV", Qt::CaseInsensitive)) {
        ptvIndex = i;
        break;
      }
    }
    if (ptvIndex >= 0)
      m_roiCombo->setCurrentIndex(ptvIndex);
  }
  if (m_sampleCombo) {
    m_sampleCombo->clear();
    m_sampleCombo->addItem("-");
    m_sampleCombo->addItems(names);
    m_sampleCombo->setCurrentIndex(0);
  }
}

void DoseProfileWindow::setDicomData(const DicomVolume *ct,
                                     const DoseResampledVolume *dose,
                                     const RTStructureSet *structures) {
  m_ct = ct;
  m_dose = dose;
  m_structures = structures;
}

void DoseProfileWindow::setLineSlotAvailable(int slotIndex, bool hasLine) {
  if (slotIndex < 0)
    return;
  if (m_slotHasData.size() < m_slotButtons.size())
    m_slotHasData.resize(m_slotButtons.size(), false);
  if (slotIndex >= m_slotHasData.size())
    m_slotHasData.resize(slotIndex + 1, false);
  m_slotHasData[slotIndex] = hasLine;
  updateSlotButtonState(slotIndex);
}

void DoseProfileWindow::updateSlotButtonState(int slotIndex) {
  if (slotIndex < 0 || slotIndex >= m_slotButtons.size())
    return;
  if (slotIndex >= m_slotHasData.size())
    m_slotHasData.resize(slotIndex + 1, false);
  QPushButton *button = m_slotButtons[slotIndex];
  if (!button)
    return;
  const bool hasLine = m_slotHasData.value(slotIndex, false);
  QFont font = button->font();
  font.setBold(hasLine);
  button->setFont(font);
  if (m_pendingSave) {
    button->setToolTip(
        tr("Click to store the current line in slot %1.").arg(slotIndex + 1));
  } else if (hasLine) {
    button->setToolTip(
        tr("Restore saved line %1.").arg(slotIndex + 1));
  } else {
    button->setToolTip(
        tr("No saved line in slot %1.").arg(slotIndex + 1));
  }
  button->setEnabled(m_pendingSave || hasLine);
}

void DoseProfileWindow::refreshSlotButtonStates() {
  for (int i = 0; i < m_slotButtons.size(); ++i)
    updateSlotButtonState(i);
  if (m_saveButton) {
    if (m_pendingSave) {
      m_saveButton->setToolTip(tr("Select a slot to store the current line."));
    } else {
      m_saveButton->setToolTip(
          tr("Click Save, then choose a slot to store the current line."));
    }
  }
}

void DoseProfileWindow::onCalculateDPSD() {
  if (!m_ct || !m_dose || !m_structures)
    return;

  if (m_dpsdRadio)
    m_dpsdRadio->setChecked(true);

  int roiIndex = m_roiCombo ? m_roiCombo->currentIndex() : -1;
  if (roiIndex < 0)
    return;

  double start = m_startSpin ? m_startSpin->value() : -20.0;
  double end = m_endSpin ? m_endSpin->value() : 50.0;
  auto mode = (m_modeCombo && m_modeCombo->currentIndex() == 0)
                  ? DPSDCalculator::Mode::Mode2D
                  : DPSDCalculator::Mode::Mode3D;

  int sampleIndex = -1;
  if (m_sampleCombo && m_sampleCombo->currentIndex() > 0)
    sampleIndex = m_sampleCombo->currentIndex() - 1;

  performDPSDCalculation(roiIndex, start, end, mode, sampleIndex);
}

void DoseProfileWindow::performDPSDCalculation(int roiIndex, double start,
                                               double end,
                                               DPSDCalculator::Mode mode,
                                               int sampleRoiIndex) {
  if (start >= end) {
    qWarning() << "performDPSDCalculation: start" << start
               << "is not less than end" << end;
    QMessageBox::warning(this, tr("DPSD"),
                         tr("Start distance must be less than end distance."));
    return;
  }
  if (!m_ct) {
    qWarning() << "performDPSDCalculation: CT volume not set";
    return;
  }
  if (!m_dose) {
    qWarning() << "performDPSDCalculation: dose volume not set";
    return;
  }
  if (!m_structures) {
    qWarning() << "performDPSDCalculation: structure set not set";
    return;
  }
  if (roiIndex < 0 || roiIndex >= m_structures->roiCount()) {
    qWarning() << "performDPSDCalculation: ROI index" << roiIndex
               << "out of range" << m_structures->roiCount();
    QMessageBox::warning(this, tr("DPSD"),
                         tr("Invalid ROI index: %1").arg(roiIndex));
    return;
  }
  qDebug() << "performDPSDCalculation: starting calculation for ROI"
           << m_roiCombo->itemText(roiIndex) << "index" << roiIndex;
  m_pendingROIIndex = roiIndex;
  m_pendingSampleIndex = sampleRoiIndex;
  m_progress->setRange(0, 100);
  m_progress->setValue(0);
  m_progress->setVisible(true);
  if (!m_progressTimer) {
    m_progressTimer = new QTimer(this);
    connect(m_progressTimer, &QTimer::timeout, this, [this]() {
      int v = m_progress->value();
      if (v < 99)
        m_progress->setValue(v + 1);
    });
  }
  m_progressTimer->start(100);
  if (!m_calcWatcher) {
    m_calcWatcher = new QFutureWatcher<
        QPair<DPSDCalculator::Result, DPSDCalculator::Result>>(this);
    connect(m_calcWatcher,
            &QFutureWatcher<QPair<DPSDCalculator::Result, DPSDCalculator::Result>>::finished,
            this, [this]() {
              if (m_progressTimer)
                m_progressTimer->stop();
              m_progress->setRange(0, 100);
              m_progress->setValue(100);
              m_progress->setVisible(false);
              if (m_roiCombo)
                m_roiCombo->setCurrentIndex(m_pendingROIIndex);
              if (m_sampleCombo)
                m_sampleCombo->setCurrentIndex(m_pendingSampleIndex + 1);
              QPair<DPSDCalculator::Result, DPSDCalculator::Result> res =
                  m_calcWatcher->result();
              setDPSDData(res.first, res.second);
            });
  }
  auto future = QtConcurrent::run(
      [=]() {
        QPair<DPSDCalculator::Result, DPSDCalculator::Result> pair;
        pair.first =
            DPSDCalculator::calculate(*m_ct, *m_dose, *m_structures, roiIndex,
                                      start, end, 2.0, mode, -1, nullptr);
        if (sampleRoiIndex >= 0)
          pair.second = DPSDCalculator::calculate(*m_ct, *m_dose, *m_structures,
                                                 roiIndex, start, end, 2.0,
                                                 mode, sampleRoiIndex,
                                                 nullptr);
        return pair;
      });
  m_calcWatcher->setFuture(future);
}

void DoseProfileWindow::setDPSDData(const DPSDCalculator::Result &data,
                                    const DPSDCalculator::Result &sample) {
    qDebug() << "=== DoseProfileWindow::setDPSDData START ===";
    qDebug() << "DPSD result size" << data.distancesMm.size();
    
    m_dpsdResult = data;
    m_dpsdSampleResult = sample;
    m_dpsdSampleIndex = m_pendingSampleIndex;
    m_hasDPSDResult = !data.distancesMm.empty();
    
    if (!m_hasDPSDResult) {
        m_plot->clearGraphs();
        m_plot->replot();
        QMessageBox::warning(this, tr("DPSD"), tr("計算結果が空です"));
        if (m_exportButton)
            m_exportButton->setEnabled(false);
        return;
    }
    
    if (m_exportButton)
        m_exportButton->setEnabled(true);
    
    // ★ 重要：DPSD表示モードに切り替え
    if (m_dpsdRadio && !m_dpsdRadio->isChecked()) {
        qDebug() << "Switching to DPSD mode";
        m_dpsdRadio->setChecked(true);
    }
    
    // ★ プロットを更新（ここでタイトルが表示される）
    updatePlotForDPSD();
    
    qDebug() << "=== DoseProfileWindow::setDPSDData END ===";
}

void DoseProfileWindow::updatePlotForProfile() {
  m_plot->xAxis->setLabel(tr("Position (mm)"));
  m_plot->yAxis->setLabel(tr("Dose (Gy)"));
  m_plot->xAxis->setAutoTicks(true);
  m_plot->yAxis->setAutoTicks(true);
  m_plot->clearGraphs();
  if (m_zeroAxis)
    m_zeroAxis->setVisible(false);
  if (m_dpsdTitleLabel)
    m_dpsdTitleLabel->setVisible(false);
  if (m_filterRoiLabel)
    m_filterRoiLabel->setVisible(false);
  QCPGraph *graph = m_plot->addGraph();
  QPen pen(QColor(0, 150, 255));
  pen.setWidth(2);
  graph->setPen(pen);
  graph->setData(m_profilePositions, m_profileDoses);

  const double centerX =
      m_profilePositions.isEmpty()
          ? 0.0
          : (m_profilePositions.first() + m_profilePositions.last()) / 2.0;

  for (auto *line : m_roiLines)
    line->setVisible(false);
  for (auto *label : m_roiLabels)
    label->setVisible(false);
  for (const Segment &seg : m_profileSegments) {
    QCPItemLine *line = nullptr;
    QCPItemText *label = nullptr;
    if (!m_roiLines.isEmpty()) {
      for (int i = 0; i < m_roiLines.size(); ++i) {
        if (!m_roiLines[i]->visible()) {
          line = m_roiLines[i];
          if (i < m_roiLabels.size())
            label = m_roiLabels[i];
          break;
        }
      }
    }
    if (!line) {
      line = new QCPItemLine(m_plot);
      m_roiLines.append(line);
    }
    if (!label) {
      label = new QCPItemText(m_plot);
      m_roiLabels.append(label);
    }
    label->setFont(m_plot->xAxis->tickLabelFont());
    bool isSpecial = seg.name.contains("GTV", Qt::CaseInsensitive) ||
                     seg.name.contains("PTV", Qt::CaseInsensitive) ||
                     seg.name.contains("CTV", Qt::CaseInsensitive) ||
                     seg.name.contains("ITV", Qt::CaseInsensitive);
    double y = isSpecial ? seg.minDoseGy : seg.maxDoseGy;
    line->start->setCoords(seg.startMm, y);
    line->end->setCoords(seg.endMm, y);
    line->setPen(QPen(seg.color, 3));
    line->setVisible(true);

    const double midX = (seg.startMm + seg.endMm) / 2.0;
    label->position->setCoords(midX, y);
    if (isSpecial) {
      label->setPositionAlignment(Qt::AlignBottom | Qt::AlignHCenter);
    } else {
      Qt::Alignment horiz = midX >= centerX ? Qt::AlignLeft : Qt::AlignRight;
      label->setPositionAlignment(Qt::AlignTop | horiz);
    }
    label->setText(seg.name);
    label->applyThemeColor(ThemeManager::instance().textColor());
    label->setVisible(true);
  }

  m_plot->rescaleAxes();
  m_plot->replot();
}

void DoseProfileWindow::updatePlotForDPSD() {
  if (!m_hasDPSDResult)
    return;
    
  qDebug() << "=== DoseProfileWindow::updatePlotForDPSD START ===";
    
  m_plot->xAxis->setLabel(tr("Distance (mm)"));
  m_plot->yAxis->setLabel(tr("Dose (Gy)"));
  m_plot->xAxis->setAutoTicks(false);
  
  for (auto *line : m_roiLines)
    line->setVisible(false);
  for (auto *label : m_roiLabels)
    label->setVisible(false);

  QVector<double> dist, minDose, maxDose, meanDose;
  dist.reserve(static_cast<int>(m_dpsdResult.distancesMm.size()));
  minDose.reserve(static_cast<int>(m_dpsdResult.minDoseGy.size()));
  maxDose.reserve(static_cast<int>(m_dpsdResult.maxDoseGy.size()));
  meanDose.reserve(static_cast<int>(m_dpsdResult.meanDoseGy.size()));
  
  for (double v : m_dpsdResult.distancesMm)
    dist.push_back(v);
  for (double v : m_dpsdResult.minDoseGy)
    minDose.push_back(v);
  for (double v : m_dpsdResult.maxDoseGy)
    maxDose.push_back(v);
  for (double v : m_dpsdResult.meanDoseGy)
    meanDose.push_back(v);

  m_plot->clearGraphs();
  QCPGraph *gMin = m_plot->addGraph();
  //gMin->setPen(QPen(Qt::lightblue));
  gMin->setPen(QPen(QColor(173, 216, 230)));
  gMin->setData(dist, minDose);
  QCPGraph *gMax = m_plot->addGraph();
  //gMax->setPen(QPen(Qt::pink));
  gMax->setPen(QPen(QColor(255, 182, 193)));
  gMax->setData(dist, maxDose);
  QCPGraph *gMean = m_plot->addGraph();
  gMean->setPen(QPen(Qt::green));
  gMean->setData(dist, meanDose);

  // サンプルROIデータがある場合の追加表示
  if (!m_dpsdSampleResult.distancesMm.empty()) {
    QVector<double> distS, minS, maxS, meanS;
    distS.reserve(static_cast<int>(m_dpsdSampleResult.distancesMm.size()));
    minS.reserve(static_cast<int>(m_dpsdSampleResult.minDoseGy.size()));
    maxS.reserve(static_cast<int>(m_dpsdSampleResult.maxDoseGy.size()));
    meanS.reserve(static_cast<int>(m_dpsdSampleResult.meanDoseGy.size()));
    for (double v : m_dpsdSampleResult.distancesMm)
      distS.push_back(v);
    for (double v : m_dpsdSampleResult.minDoseGy)
      minS.push_back(v);
    for (double v : m_dpsdSampleResult.maxDoseGy)
      maxS.push_back(v);
    for (double v : m_dpsdSampleResult.meanDoseGy)
      meanS.push_back(v);

    QCPGraph *gRoiMin = m_plot->addGraph();
    int sampleIndex = m_dpsdSampleIndex;
    QColor baseColor = QColor::fromHsv((sampleIndex * 40) % 360, 255, 255);
    QColor roiMinColor = baseColor.lighter(150);
    roiMinColor.setAlpha(180);
    QPen roiMinPen(roiMinColor);
    roiMinPen.setStyle(Qt::DashLine);
    gRoiMin->setPen(roiMinPen);
    gRoiMin->setData(distS, minS);

    QCPGraph *gRoiMax = m_plot->addGraph();
    QColor roiMaxColor = baseColor.lighter(150);
    roiMaxColor.setAlpha(180);
    QPen roiMaxPen(roiMaxColor);
    roiMaxPen.setStyle(Qt::DashLine);
    gRoiMax->setPen(roiMaxPen);
    gRoiMax->setData(distS, maxS);

    QCPGraph *gRoiMean = m_plot->addGraph();
    QColor roiMeanColor = baseColor.lighter(150);
    roiMeanColor.setAlpha(180);
    QPen roiMeanPen(roiMeanColor);
    roiMeanPen.setStyle(Qt::DashLine);
    gRoiMean->setPen(roiMeanPen);
    gRoiMean->setData(distS, meanS);

    QColor fillColor = baseColor;
    fillColor.setAlpha(128);
    gRoiMax->setBrush(QBrush(fillColor));
    gRoiMax->setChannelFillGraph(gRoiMin);
  }

  // ★★★ 既存ラベルを完全にクリア ★★★
  for (auto *label : m_roiLabels) {
    if (label) {
      label->setVisible(false);
      delete label;
    }
  }
  m_roiLabels.clear();
  
  qDebug() << "All existing labels cleared";
  // ★★★ タイトルラベル（上部中央）を設定 ★★★
  QString roiName = "Unknown";
  if (m_roiCombo && m_pendingROIIndex >= 0 &&
      m_pendingROIIndex < m_roiCombo->count()) {
    roiName = m_roiCombo->itemText(m_pendingROIIndex);
  }
  qDebug() << "Setting DPSD title for ROI:" << roiName;
  if (m_dpsdTitleLabel) {
    m_dpsdTitleLabel->setText(
        QString("Distance from %1 Surface-Dose").arg(roiName));
    m_dpsdTitleLabel->adjustSize();
    m_dpsdTitleLabel->setVisible(true);
  }

  // ★★★ フィルターROI名（右下）を設定 ★★★
  if (m_filterRoiLabel)
    m_filterRoiLabel->setVisible(false);
  if (!m_dpsdSampleResult.distancesMm.empty() && m_sampleCombo &&
      m_dpsdSampleIndex >= 0) {
    QString filterRoiName = "Unknown Filter ROI";
    int sampleComboIndex = m_dpsdSampleIndex + 1; // +1 because first item is "-"
    if (sampleComboIndex > 0 && sampleComboIndex < m_sampleCombo->count()) {
      filterRoiName = m_sampleCombo->itemText(sampleComboIndex);
    }
    qDebug() << "Setting filter ROI label:" << filterRoiName;
    m_filterRoiLabel->setText(
        QString("Filter ROI: %1").arg(filterRoiName));
    m_filterRoiLabel->adjustSize();
    m_filterRoiLabel->setVisible(true);
  }

  updateOverlayPositions();

  m_plot->rescaleAxes();
  double yMax = m_plot->yAxis->max();
  m_plot->yAxis->setRange(0, yMax * 1.1);

  double minX = m_plot->xAxis->min();
  double maxX = m_plot->xAxis->max();
  minX = std::floor(std::min(-20.0, minX) / 5.0) * 5.0;
  maxX = std::ceil(std::max(50.0, maxX) / 5.0) * 5.0;
  m_plot->xAxis->setRange(minX, maxX);

  QVector<double> ticks;
  for (int i = static_cast<int>(minX); i < static_cast<int>(maxX); i += 5) {
    ticks.push_back(i);
  }
  m_plot->xAxis->setTickVector(ticks);

  if (!m_zeroAxis)
    m_zeroAxis = new QCPItemLine(m_plot);
  m_zeroAxis->start->setCoords(0, 0);
  m_zeroAxis->end->setCoords(0, yMax * 1.1);
  QPen axisPen(ThemeManager::instance().textColor());
  m_zeroAxis->setPen(axisPen);
  m_zeroAxis->setVisible(true);

  m_plot->replot();

  qDebug() << "=== DoseProfileWindow::updatePlotForDPSD END ===";
}

bool DoseProfileWindow::eventFilter(QObject *obj, QEvent *event) {
  if (obj == m_plot && m_dpsdRadio && m_dpsdRadio->isChecked() &&
      m_hasDPSDResult) {
    if (!m_showCursorCheck || !m_showCursorCheck->isChecked()) {
      if (m_cursorLine)
        m_cursorLine->setVisible(false);
      if (m_cursorInfoLabel)
        m_cursorInfoLabel->setVisible(false);
      if (m_plot)
        m_plot->replot();
      return QWidget::eventFilter(obj, event);
    }

    if (event->type() == QEvent::MouseMove) {
      auto *me = static_cast<QMouseEvent *>(event);
      if (m_cursorLine)
        m_cursorLine->setVisible(true);
      if (m_cursorInfoLabel)
        m_cursorInfoLabel->setVisible(true);
      updateCursorDisplay(me->pos());
      return true;
    } else if (event->type() == QEvent::Leave) {
      if (m_cursorLine)
        m_cursorLine->setVisible(false);
      if (m_cursorInfoLabel)
        m_cursorInfoLabel->setVisible(false);
      if (m_plot)
        m_plot->replot();
      return true;
    }
  }
  return QWidget::eventFilter(obj, event);
}

void DoseProfileWindow::updateCursorDisplay(const QPoint &pos) {
  if (!m_cursorLine || !m_showCursorCheck || !m_showCursorCheck->isChecked() ||
      m_dpsdResult.distancesMm.empty())
    return;

  double xCoord = m_plot->xAxis->pixelToCoord(pos.x());
  double yMax = m_plot->yAxis->max();
  m_cursorLine->start->setCoords(xCoord, 0);
  m_cursorLine->end->setCoords(xCoord, yMax);

  auto interpolate = [](const std::vector<double> &xs,
                        const std::vector<double> &ys, double x,
                        double &out) {
    if (xs.empty())
      return false;
    if (x <= xs.front()) {
      out = ys.front();
      return true;
    }
    if (x >= xs.back()) {
      out = ys.back();
      return true;
    }
    auto it = std::lower_bound(xs.begin(), xs.end(), x);
    if (it == xs.begin() || it == xs.end())
      return false;
    size_t i = static_cast<size_t>(it - xs.begin());
    double x0 = xs[i - 1];
    double x1 = xs[i];
    double y0 = ys[i - 1];
    double y1 = ys[i];
    double t = (x - x0) / (x1 - x0);
    out = y0 + t * (y1 - y0);
    return true;
  };

  double maxDose = 0.0, minDose = 0.0, meanDose = 0.0;
  interpolate(m_dpsdResult.distancesMm, m_dpsdResult.maxDoseGy, xCoord,
              maxDose);
  interpolate(m_dpsdResult.distancesMm, m_dpsdResult.minDoseGy, xCoord,
              minDose);
  interpolate(m_dpsdResult.distancesMm, m_dpsdResult.meanDoseGy, xCoord,
              meanDose);

  QString text = tr("x=%1 mm\nmax=%2 min=%3 mean=%4")
                     .arg(xCoord, 0, 'f', 1)
                     .arg(maxDose, 0, 'f', 2)
                     .arg(minDose, 0, 'f', 2)
                     .arg(meanDose, 0, 'f', 2);

  const auto &fd = m_dpsdSampleResult.distancesMm;
  if (fd.size() >= 2) {
    double stepMm = fd[1] - fd[0];
    auto it = std::lower_bound(fd.begin(), fd.end(), xCoord);
    size_t ridx = (it == fd.end()) ? fd.size() - 1 : static_cast<size_t>(it - fd.begin());
    if (ridx > 0 && it != fd.end() && std::abs(*it - xCoord) > std::abs(fd[ridx - 1] - xCoord)) {
      --ridx;
      it = fd.begin() + static_cast<long>(ridx);
    }
    double diff = std::abs(fd[ridx] - xCoord);
    if (diff <= stepMm / 2.0) {
      double fmax, fmin, fmean;
      if (interpolate(fd, m_dpsdSampleResult.maxDoseGy, xCoord, fmax) &&
          interpolate(fd, m_dpsdSampleResult.minDoseGy, xCoord, fmin) &&
          interpolate(fd, m_dpsdSampleResult.meanDoseGy, xCoord, fmean)) {
        text += tr("\nFilter max=%1 min=%2 mean=%3")
                    .arg(fmax, 0, 'f', 2)
                    .arg(fmin, 0, 'f', 2)
                    .arg(fmean, 0, 'f', 2);
      }
    }
  }

  m_cursorInfoLabel->setText(text);
  m_cursorInfoLabel->adjustSize();
  int x = pos.x() + 10;
  int y = pos.y() + 10;
  if (x + m_cursorInfoLabel->width() > m_plot->width())
    x = m_plot->width() - m_cursorInfoLabel->width() - 5;
  if (y + m_cursorInfoLabel->height() > m_plot->height())
    y = m_plot->height() - m_cursorInfoLabel->height() - 5;
  m_cursorInfoLabel->move(x, y);
  m_cursorInfoLabel->raise();

  m_plot->replot();
}

void DoseProfileWindow::resizeEvent(QResizeEvent *event) {
  QWidget::resizeEvent(event);
  updateOverlayPositions();
}

void DoseProfileWindow::showEvent(QShowEvent *event) {
  QWidget::showEvent(event);
  QTimer::singleShot(0, this, &DoseProfileWindow::updateOverlayPositions);
}

void DoseProfileWindow::updateOverlayPositions() {
  if (!m_plot)
    return;
  int w = m_plot->width();
  int h = m_plot->height();
  if (m_dpsdTitleLabel && m_dpsdTitleLabel->isVisible()) {
    int x = (w - m_dpsdTitleLabel->width()) / 2;
    m_dpsdTitleLabel->move(x, 5);
    m_dpsdTitleLabel->raise();
  }
  if (m_filterRoiLabel && m_filterRoiLabel->isVisible()) {
    int x = w - m_filterRoiLabel->width() - 5;
    int y = h - m_filterRoiLabel->height() - 5;
    m_filterRoiLabel->move(x, y);
    m_filterRoiLabel->raise();
  }
}

void DoseProfileWindow::onExportDPSD() {
  if (m_dpsdResult.distancesMm.empty()) {
    QMessageBox::warning(this, tr("DPSD"), tr("データがありません"));
    return;
  }
  QString text =
      QStringLiteral("distance_mm\tmin_dose_Gy\tmax_dose_Gy\tmean_dose_Gy\n");
  for (size_t i = 0; i < m_dpsdResult.distancesMm.size(); ++i) {
    text += QStringLiteral("%1\t%2\t%3\t%4\n")
                .arg(m_dpsdResult.distancesMm[i])
                .arg(m_dpsdResult.minDoseGy[i])
                .arg(m_dpsdResult.maxDoseGy[i])
                .arg(m_dpsdResult.meanDoseGy[i]);
  }
  if (m_infoBox)
    m_infoBox->setPlainText(text);
}
