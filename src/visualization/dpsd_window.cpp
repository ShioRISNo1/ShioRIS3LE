#include "visualization/dpsd_window.h"
#include <QDebug>
#include <QBrush>
#include <QFont>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPen>
#include <QPlainTextEdit>
#include <QResizeEvent>
#include <QString>
#include <QTimer>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <QtConcurrent>
#include <QMessageBox>
#include "dicom/dicom_volume.h"
#include "dicom/dose_resampled_volume.h"
#include "dicom/rtstruct.h"
#include "theme_manager.h"


DPSDWindow::DPSDWindow(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *mainLayout = new QVBoxLayout(this);

  m_plot = new QCustomPlot(this);
  const int axisLabelPointSize = 9;
  const int tickLabelPointSize = 8;
  const int overlayPointSize = 8;
  QFont axisLabelFont = m_plot->xAxis->labelFont();
  axisLabelFont.setPointSize(axisLabelPointSize);
  m_plot->xAxis->setLabelFont(axisLabelFont);
  m_plot->yAxis->setLabelFont(axisLabelFont);
  QFont tickLabelFont = m_plot->xAxis->tickLabelFont();
  tickLabelFont.setPointSize(tickLabelPointSize);
  m_plot->xAxis->setTickLabelFont(tickLabelFont);
  m_plot->yAxis->setTickLabelFont(tickLabelFont);
  QFont overlayFont = m_plot->font();
  overlayFont.setPointSize(overlayPointSize);
  QFont overlayBoldFont = overlayFont;
  overlayBoldFont.setBold(true);
  m_plot->xAxis->setLabel("Distance (mm)");
  m_plot->yAxis->setLabel("Dose (Gy)");
  m_plot->setMouseTracking(true);
  m_plot->installEventFilter(this);
  m_plot->setStyleSheet(
      "QCustomPlot { background-color: black; border: 1px solid gray; }");
  mainLayout->addWidget(m_plot, 1);

  QFont titleFont = font();
  titleFont.setPointSize(9);
  titleFont.setBold(true);

  m_titleLabel = new QLabel(m_plot);
  m_titleLabel->setFont(titleFont);
  ThemeManager &theme = ThemeManager::instance();
  theme.applyTextColor(
      m_titleLabel,
      QStringLiteral(
          "QLabel { background: transparent; color: %1; padding:2px; }"));
  m_titleLabel->setAttribute(Qt::WA_TransparentForMouseEvents);
  m_titleLabel->setVisible(false);

  m_roiLabel = new QLabel(m_plot);
  m_roiLabel->setFont(overlayBoldFont);
  theme.applyTextColor(
      m_roiLabel,
      QStringLiteral(
          "QLabel { background: transparent; color: %1; padding:2px; }"));
  m_roiLabel->setAttribute(Qt::WA_TransparentForMouseEvents);
  m_roiLabel->setVisible(false);

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
  m_cursorInfoLabel->setAttribute(Qt::WA_TranslucentBackground);
  m_cursorInfoLabel->setVisible(false);
  m_cursorInfoLabel->raise();

  QHBoxLayout *ctrlLayout = new QHBoxLayout();
  m_roiCombo = new QComboBox(this);
  ctrlLayout->addWidget(m_roiCombo);
  QVBoxLayout *modeLayout = new QVBoxLayout();
  m_modeCombo = new QComboBox(this);
  m_modeCombo->addItem("3D", static_cast<int>(DPSDCalculator::Mode::Mode3D));
  m_modeCombo->addItem("2D", static_cast<int>(DPSDCalculator::Mode::Mode2D));
  modeLayout->addWidget(m_modeCombo);
  m_roiFilterCombo = new QComboBox(this);
  m_roiFilterCombo->addItem("-");
  modeLayout->addWidget(m_roiFilterCombo);
  ctrlLayout->addLayout(modeLayout);
  m_startSpin = new QDoubleSpinBox(this);
  m_startSpin->setRange(-1000, 1000);
  m_startSpin->setValue(-20.0);
  ctrlLayout->addWidget(m_startSpin);
  m_endSpin = new QDoubleSpinBox(this);
  m_endSpin->setRange(-1000, 1000);
  m_endSpin->setValue(50.0);
  ctrlLayout->addWidget(m_endSpin);
  m_calcButton = new QPushButton(tr("Calculate"), this);
  ctrlLayout->addWidget(m_calcButton);
  m_exportButton = new QPushButton(tr("Export"), this);
  m_exportButton->setEnabled(false);
  QVBoxLayout *exportLayout = new QVBoxLayout();
  exportLayout->addWidget(m_exportButton);
  m_showDataCheck = new QCheckBox(tr("Show data"), this);
  m_showDataCheck->setChecked(true);
  exportLayout->addWidget(m_showDataCheck);
  ctrlLayout->addLayout(exportLayout);
  ctrlLayout->addStretch();
  mainLayout->addLayout(ctrlLayout);

  m_progress = new QProgressBar(this);
  m_progress->setRange(0, 100);
  m_progress->setValue(0);
  m_progress->setTextVisible(true);
  m_progress->setVisible(false);
  mainLayout->addWidget(m_progress);

  m_infoBox = new QPlainTextEdit(this);
  m_infoBox->setReadOnly(true);
  theme.applyTextColor(
      m_infoBox,
      QStringLiteral("QPlainTextEdit {background: #202020; color: %1; "
                    "border: 1px solid #444444;}"));
  mainLayout->addWidget(m_infoBox, 1);

  connect(m_calcButton, &QPushButton::clicked, this, &DPSDWindow::onCalculate);
  connect(m_exportButton, &QPushButton::clicked, this, &DPSDWindow::onExport);
}

void DPSDWindow::setROINames(const QStringList &names) {
  m_roiCombo->clear();
  m_roiCombo->addItems(names);
  if (m_roiFilterCombo) {
    m_roiFilterCombo->clear();
    m_roiFilterCombo->addItem("-");
    m_roiFilterCombo->addItems(names);
    m_roiFilterCombo->setCurrentIndex(0);
  }
  int ptvIndex = -1;
  for (int i = 0; i < names.size(); ++i) {
    if (names[i].contains("PTV", Qt::CaseInsensitive)) {
      ptvIndex = i;
      break;
    }
  }
  if (ptvIndex >= 0)
    m_roiCombo->setCurrentIndex(ptvIndex);
  updateOverlayPositions();
}

void DPSDWindow::setCurrentROI(int index) {
  m_roiCombo->setCurrentIndex(index);
}

void DPSDWindow::setDPSDData(int roiIndex, const DPSDCalculator::Result &data,
                             const DPSDCalculator::Result &roiData) {
  qDebug() << "=== setDPSDData START ===";
  qDebug() << "DPSD result for ROI" << m_roiCombo->itemText(roiIndex) << "size"
           << data.distancesMm.size();
  
  if (data.distancesMm.empty()) {
    m_plot->clearGraphs();
    m_plot->replot();
    QMessageBox::warning(this, tr("DPSD"), tr("計算結果が空です"));
    return;
  }

  m_currentResult = data;
  m_currentRoiResult = roiData;
  
  // タイトルとROI名の設定
  QString roiName = m_roiCombo->itemText(roiIndex);
  QString titleText = QString("Distance from %1 Surface - Dose Profile").arg(roiName);
  
  qDebug() << "Setting title text:" << titleText;
  qDebug() << "ROI name:" << roiName;
  
  // プロットのサイズ確認
  qDebug() << "Plot size:" << m_plot->size() << "width:" << m_plot->width() << "height:" << m_plot->height();
  
  // タイトルラベルの設定と表示
  m_titleLabel->setText(titleText);
  m_titleLabel->adjustSize();
  qDebug() << "Title label size after adjustSize:" << m_titleLabel->size();
  m_titleLabel->setVisible(true);
  qDebug() << "Title label visible:" << m_titleLabel->isVisible();
  
  // ROI名ラベルの設定と表示  
  QString roiLabelText = QString("ROI: %1").arg(roiName);
  m_roiLabel->setText(roiLabelText);
  m_roiLabel->adjustSize();
  qDebug() << "ROI label size after adjustSize:" << m_roiLabel->size();
  m_roiLabel->setVisible(true);
  qDebug() << "ROI label visible:" << m_roiLabel->isVisible();

  m_plot->clearGraphs();
  
  // X軸の範囲設定
  double minX = *std::min_element(data.distancesMm.begin(), data.distancesMm.end());
  double maxX = *std::max_element(data.distancesMm.begin(), data.distancesMm.end());
  m_plot->xAxis->setRange(minX - 1, maxX + 1);
  
  // Y軸の範囲設定
  double maxDose = 0;
  for (const auto& dose : data.maxDoseGy) {
    maxDose = std::max(maxDose, dose);
  }
  m_plot->yAxis->setRange(0, maxDose * 1.1);

  // メイングラフ（Min、Max、Mean）の描画
  QCPGraph *minGraph = m_plot->addGraph();
  QCPGraph *maxGraph = m_plot->addGraph();
  QCPGraph *meanGraph = m_plot->addGraph();
  
  minGraph->setPen(QPen(Qt::blue, 2));
  maxGraph->setPen(QPen(Qt::red, 2));
  meanGraph->setPen(QPen(Qt::green, 2));
  
  // データポイントの設定
  QVector<double> distances(data.distancesMm.begin(), data.distancesMm.end());
  QVector<double> minDoses(data.minDoseGy.begin(), data.minDoseGy.end());
  QVector<double> maxDoses(data.maxDoseGy.begin(), data.maxDoseGy.end());
  QVector<double> meanDoses(data.meanDoseGy.begin(), data.meanDoseGy.end());
  
  minGraph->setData(distances, minDoses);
  maxGraph->setData(distances, maxDoses);
  meanGraph->setData(distances, meanDoses);
  
  // Min-Max間の塗りつぶし
  QCPGraph *fillGraph = m_plot->addGraph();
  fillGraph->setPen(QPen(Qt::blue, 1, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
  fillGraph->setBrush(QBrush(QColor(0, 0, 255, 50))); // 半透明の青
  fillGraph->setData(distances, maxDoses);
  fillGraph->setChannelFillGraph(minGraph);

  // ROIデータがある場合の追加表示
  if (!roiData.distancesMm.empty()) {
    QCPGraph *roiMeanGraph = m_plot->addGraph();
    roiMeanGraph->setPen(QPen(Qt::cyan, 2, Qt::DashLine));
    
    QVector<double> roiDistances(roiData.distancesMm.begin(), roiData.distancesMm.end());
    QVector<double> roiMeanDoses(roiData.meanDoseGy.begin(), roiData.meanDoseGy.end());
    roiMeanGraph->setData(roiDistances, roiMeanDoses);
  }
  
  // ゼロライン（距離=0の縦線）の追加
  if (!m_zeroAxis) {
    m_zeroAxis = new QCPItemLine(m_plot);
  }
  m_zeroAxis->setPen(
      QPen(ThemeManager::instance().textColor(), 1, Qt::DashLine));
  m_zeroAxis->start->setCoords(0, 0);
  m_zeroAxis->end->setCoords(0, maxDose * 1.1);
  
  // プロット更新
  m_plot->replot();
  
  // ラベル位置の更新（デバッグ情報付き）
  qDebug() << "Before updateOverlayPositions - plot size:" << m_plot->size();
  updateOverlayPositions();
  qDebug() << "After updateOverlayPositions - title pos:" << m_titleLabel->pos() << "roi pos:" << m_roiLabel->pos();
  
  // エクスポートボタンの有効化
  m_exportButton->setEnabled(true);
  
  // 統計情報の表示
  QString stats = QString("ROI: %1\n").arg(roiName);
  stats += QString("Distance Range: %1 to %2 mm\n")
           .arg(minX, 0, 'f', 1)
           .arg(maxX, 0, 'f', 1);
  stats += QString("Max Dose: %1 Gy\n").arg(maxDose, 0, 'f', 2);
  stats += QString("Data Points: %1\n").arg(data.distancesMm.size());
  
  if (m_infoBox) {
    m_infoBox->setPlainText(stats);
  }
  
  qDebug() << "=== setDPSDData END ===";
}

void DPSDWindow::setBusy(bool busy) {
  if (busy) {
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
  } else {
    if (m_progressTimer)
      m_progressTimer->stop();
    m_progress->setRange(0, 100);
    m_progress->setValue(100);
    m_progress->setVisible(false);
  }
}

void DPSDWindow::onCalculate() {
    qDebug() << "=== DPSDWindow::onCalculate START ===";
    
    // DICOMデータの確認
    if (!m_ct || !m_dose || !m_structures) {
        qWarning() << "DPSDWindow::onCalculate: DICOM data not set";
        QMessageBox::warning(this, tr("DPSD"), tr("DICOMデータが設定されていません"));
        return;
    }
    
    // ROI選択の確認
    int roiIndex = m_roiCombo->currentIndex();
    if (roiIndex < 0) {
        qWarning() << "DPSDWindow::onCalculate: No ROI selected";
        QMessageBox::warning(this, tr("DPSD"), tr("ROIが選択されていません"));
        return;
    }
    
    // パラメータの取得
    int filterIndex = -1;
    if (m_roiFilterCombo && m_roiFilterCombo->currentIndex() > 0) {
        filterIndex = m_roiFilterCombo->currentIndex() - 1;
    }
    
    double startMm = m_startSpin->value();
    double endMm = m_endSpin->value();
    auto mode = static_cast<DPSDCalculator::Mode>(m_modeCombo->currentData().toInt());
    
    qDebug() << "Starting DPSD calculation for ROI" << m_roiCombo->itemText(roiIndex);
    qDebug() << "Parameters: start=" << startMm << "mm, end=" << endMm << "mm, mode=" << (int)mode;
    
    // プログレスバー表示
    setBusy(true);
    
    // 非同期計算の設定
    if (!m_calcWatcher) {
        m_calcWatcher = new QFutureWatcher<QPair<DPSDCalculator::Result, DPSDCalculator::Result>>(this);
        connect(m_calcWatcher,
                &QFutureWatcher<QPair<DPSDCalculator::Result, DPSDCalculator::Result>>::finished,
                this, [this]() {
                    qDebug() << "DPSD calculation finished in DPSDWindow";
                    setBusy(false);
                    
                    QPair<DPSDCalculator::Result, DPSDCalculator::Result> result = m_calcWatcher->result();
                    
                    qDebug() << "Calling setDPSDData with result size:" << result.first.distancesMm.size();
                    
                    // 結果をDPSDWindowに設定（ここでタイトル表示される）
                    setDPSDData(m_roiCombo->currentIndex(), result.first, result.second);
                });
    }
    
    // バックグラウンドで計算実行
    auto future = QtConcurrent::run([this, roiIndex, startMm, endMm, mode, filterIndex]() {
        QPair<DPSDCalculator::Result, DPSDCalculator::Result> pair;
        
        qDebug() << "Running DPSD calculation in background thread";
        
        // メイン計算
        pair.first = DPSDCalculator::calculate(*m_ct, *m_dose, *m_structures, roiIndex,
                                              startMm, endMm, 2.0, mode, -1, nullptr);
        
        qDebug() << "Main calculation result size:" << pair.first.distancesMm.size();
        
        // フィルタROIがある場合の追加計算
        if (filterIndex >= 0) {
            qDebug() << "Running filter calculation for ROI index:" << filterIndex;
            pair.second = DPSDCalculator::calculate(*m_ct, *m_dose, *m_structures, roiIndex,
                                                   startMm, endMm, 2.0, mode, filterIndex, nullptr);
            qDebug() << "Filter calculation result size:" << pair.second.distancesMm.size();
        }
        
        return pair;
    });
    
    m_calcWatcher->setFuture(future);
    qDebug() << "=== DPSDWindow::onCalculate END ===";
}

void DPSDWindow::onExport() {
  if (m_currentResult.distancesMm.empty()) {
    QMessageBox::warning(this, tr("DPSD"), tr("データがありません"));
    return;
  }
  QString text = "distance_mm,min_dose_Gy,max_dose_Gy,mean_dose_Gy\n";
  for (size_t i = 0; i < m_currentResult.distancesMm.size(); ++i) {
    text += QString("%1,%2,%3,%4\n")
                .arg(m_currentResult.distancesMm[i])
                .arg(m_currentResult.minDoseGy[i])
                .arg(m_currentResult.maxDoseGy[i])
                .arg(m_currentResult.meanDoseGy[i]);
  }
  if (m_infoBox)
    m_infoBox->setPlainText(text);
}

bool DPSDWindow::eventFilter(QObject *obj, QEvent *event) {
  if (obj == m_plot) {
    if (event->type() == QEvent::MouseMove) {
      auto *me = static_cast<QMouseEvent *>(event);
      if (m_showDataCheck && m_showDataCheck->isChecked()) {
        m_cursorLine->setVisible(true);
        m_cursorInfoLabel->setVisible(true);
        updateCursorDisplay(me->pos());
      } else {
        m_cursorLine->setVisible(false);
        m_cursorInfoLabel->setVisible(false);
        m_plot->replot();
      }
      return true;
    } else if (event->type() == QEvent::Leave) {
      m_cursorLine->setVisible(false);
      m_cursorInfoLabel->setVisible(false);
      m_plot->replot();
      return true;
    }
  }
  return QWidget::eventFilter(obj, event);
}

void DPSDWindow::resizeEvent(QResizeEvent *event) {
  QWidget::resizeEvent(event);
  updateOverlayPositions();
}

void DPSDWindow::showEvent(QShowEvent *event) {
  QWidget::showEvent(event);
  QTimer::singleShot(0, this, &DPSDWindow::updateOverlayPositions);
}

void DPSDWindow::updateOverlayPositions() {
  qDebug() << "=== updateOverlayPositions START ===";
  
  if (!m_plot) {
    qDebug() << "m_plot is null";
    return;
  }
  
  int w = m_plot->width();
  int h = m_plot->height();
  qDebug() << "Plot dimensions: width =" << w << ", height =" << h;
  
  if (w <= 0 || h <= 0) {
    qDebug() << "Invalid plot dimensions";
    return;
  }
  
  if (m_titleLabel && m_titleLabel->isVisible()) {
    qDebug() << "Title label text:" << m_titleLabel->text();
    qDebug() << "Title label size:" << m_titleLabel->size();
    
    int x = (w - m_titleLabel->width()) / 2;
    qDebug() << "Calculated title label x position:" << x;
    
    m_titleLabel->move(x, 5);
    m_titleLabel->raise();
    
    qDebug() << "Title label moved to:" << m_titleLabel->pos();
    qDebug() << "Title label geometry:" << m_titleLabel->geometry();
  } else {
    qDebug() << "Title label not visible or null";
  }
  
  if (m_roiLabel && m_roiLabel->isVisible()) {
    qDebug() << "ROI label text:" << m_roiLabel->text();
    qDebug() << "ROI label size:" << m_roiLabel->size();
    
    int x = w - m_roiLabel->width() - 5;
    int y = h - m_roiLabel->height() - 5;
    qDebug() << "Calculated ROI label position: x =" << x << ", y =" << y;
    
    m_roiLabel->move(x, y);
    m_roiLabel->raise();
    
    qDebug() << "ROI label moved to:" << m_roiLabel->pos();
    qDebug() << "ROI label geometry:" << m_roiLabel->geometry();
  } else {
    qDebug() << "ROI label not visible or null";
  }
  
  qDebug() << "=== updateOverlayPositions END ===";
}

void DPSDWindow::updateCursorDisplay(const QPoint &pos) {
  if (!m_cursorLine || m_currentResult.distancesMm.empty() ||
      (m_showDataCheck && !m_showDataCheck->isChecked()))
    return;
  double xCoord = m_plot->xAxis->pixelToCoord(pos.x());
  double yMax = m_plot->yAxis->max();
  m_cursorLine->start->setCoords(xCoord, 0);
  m_cursorLine->end->setCoords(xCoord, yMax);

  auto interpolate = [](const std::vector<double> &xs,
                        const std::vector<double> &ys, double x) {
    if (xs.empty() || ys.empty())
      return 0.0;
    if (x <= xs.front())
      return ys.front();
    if (x >= xs.back())
      return ys.back();
    auto it = std::lower_bound(xs.begin(), xs.end(), x);
    size_t i1 = std::distance(xs.begin(), it);
    size_t i0 = i1 - 1;
    double x0 = xs[i0], x1 = xs[i1];
    double y0 = ys[i0], y1 = ys[i1];
    double t = (x - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
  };

  double maxDose =
      interpolate(m_currentResult.distancesMm, m_currentResult.maxDoseGy, xCoord);
  double minDose =
      interpolate(m_currentResult.distancesMm, m_currentResult.minDoseGy, xCoord);
  double meanDose =
      interpolate(m_currentResult.distancesMm, m_currentResult.meanDoseGy, xCoord);

  QString text = tr("x=%1 mm\nmax=%2 min=%3 mean=%4")
                     .arg(xCoord, 0, 'f', 1)
                     .arg(maxDose, 0, 'f', 2)
                     .arg(minDose, 0, 'f', 2)
                     .arg(meanDose, 0, 'f', 2);

  if (!m_currentRoiResult.distancesMm.empty()) {
    const auto &d = m_currentRoiResult.distancesMm;
    double step = d.size() >= 2 ? d[1] - d[0] : 0.0;
    auto it = std::lower_bound(d.begin(), d.end(), xCoord);
    double diff = std::numeric_limits<double>::max();
    if (it != d.end()) {
      diff = std::abs(*it - xCoord);
    }
    if (it != d.begin() && (it == d.end() || diff > std::abs(*(it - 1) - xCoord))) {
      --it;
      diff = std::abs(*it - xCoord);
    }
    if (step > 0.0 && diff <= step / 2.0) {
      double fMax = interpolate(m_currentRoiResult.distancesMm,
                                m_currentRoiResult.maxDoseGy, xCoord);
      double fMin = interpolate(m_currentRoiResult.distancesMm,
                                m_currentRoiResult.minDoseGy, xCoord);
      double fMean = interpolate(m_currentRoiResult.distancesMm,
                                 m_currentRoiResult.meanDoseGy, xCoord);
      text += tr("\nFilter max=%1 min=%2 mean=%3")
                  .arg(fMax, 0, 'f', 2)
                  .arg(fMin, 0, 'f', 2)
                  .arg(fMean, 0, 'f', 2);
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

void DPSDWindow::setDicomData(const DicomVolume *ct, const DoseResampledVolume *dose,
                             const RTStructureSet *structures) {
    qDebug() << "DPSDWindow::setDicomData called";
    m_ct = ct;
    m_dose = dose;
    m_structures = structures;
}
