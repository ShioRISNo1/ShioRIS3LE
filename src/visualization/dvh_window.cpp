#include "visualization/dvh_window.h"
#include <QAbstractItemView>
#include <QApplication>
#include <QCoreApplication>
#include <QEvent>
#include <QEventLoop>
#include <QFileDialog>
#include <QHash>
#include <QIcon>
#include <QFont>
#include <QMessageBox>
#include <QPen>
#include <QPixmap>
#include <QSet>
#include <QSignalBlocker>
#include <QSize>
#include <QSizePolicy>
#include <QTextStream>
#include <QtGlobal>
#include <algorithm>
#include <cmath>

#include "theme_manager.h"

DVHWindow::DVHWindow(QWidget *parent) : QWidget(parent) {
  // 多数ウィンドウ表示でも極端に縮小されないよう最小サイズを設定
  setMinimumSize(30, 30);
  QVBoxLayout *mainLayout = new QVBoxLayout(this);
  mainLayout->setContentsMargins(2, 2, 2, 2);
  mainLayout->setSpacing(3);

  m_plot = new QCustomPlot(this);
  const int axisLabelPointSize = 8;
  const int tickLabelPointSize = 7;
  const int overlayPointSize = 8;
  const int detailBoxPointSize = 8;
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
  m_plot->xAxis->setLabel("Dose [Gy]");
  m_plot->yAxis->setLabel("Volume");
  m_plot->setMouseTracking(true);
  m_plot->installEventFilter(this);
  m_plot->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  // QCustomPlot の簡易版では plotLayout が利用できないため、ウィジェットの
  // 内容余白を直接設定してグラフ内のマージンを抑える
  m_plot->setContentsMargins(2, 2, 2, 2);
  m_roiNameLabel = new QLabel(m_plot);
  m_roiNameLabel->setFont(overlayFont);
  // 背景を透明にしてテキストだけを重ねて表示
  ThemeManager &theme = ThemeManager::instance();
  theme.applyTextColor(
      m_roiNameLabel,
      QStringLiteral(
          "QLabel { background-color: transparent; color: %1; padding:2px; }"));
  m_roiNameLabel->setAttribute(Qt::WA_TransparentForMouseEvents);
  m_roiNameLabel->setVisible(false);
  m_cursorVLine = new QCPItemLine(m_plot);
  m_cursorVLine->setPen(QPen(Qt::gray, 1, Qt::DashLine));
  m_cursorVLine->setVisible(false);
  m_cursorHLine = new QCPItemLine(m_plot);
  m_cursorHLine->setPen(QPen(Qt::gray, 1, Qt::DashLine));
  m_cursorHLine->setVisible(false);
  m_cursorInfoLabel = new QLabel(m_plot);
  m_cursorInfoLabel->setFont(overlayFont);
  // マウスポインタの線量/Volume 表示も背景を透明化
  theme.applyTextColor(
      m_cursorInfoLabel,
      QStringLiteral(
          "QLabel { background-color: transparent; color: %1; padding:2px; }"));
  m_cursorInfoLabel->setAttribute(Qt::WA_TransparentForMouseEvents);
  m_cursorInfoLabel->setVisible(false);
  m_patientInfoLabel = new QLabel(m_plot);
  m_patientInfoLabel->setFont(overlayFont);
  theme.applyTextColor(
      m_patientInfoLabel,
      QStringLiteral(
          "QLabel { background-color: transparent; color: %1; padding:2px; }"));
  m_patientInfoLabel->setAttribute(Qt::WA_TransparentForMouseEvents);
  m_patientInfoLabel->setVisible(false);

  QHBoxLayout *plotLayout = new QHBoxLayout();
  plotLayout->setContentsMargins(0, 0, 0, 0);
  plotLayout->setSpacing(4);

  QWidget *controlPanel = new QWidget(this);
  controlPanel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
  controlPanel->setMinimumWidth(160);

  QVBoxLayout *axisLayout = new QVBoxLayout(controlPanel);
  axisLayout->setContentsMargins(0, 0, 0, 0);
  axisLayout->setSpacing(2);
  m_xPercentButton = new QRadioButton(tr("%"), this);
  m_xGyButton = new QRadioButton(tr("Gy"), this);
  m_yCcButton = new QRadioButton(tr("Volume(cc)"), this);
  m_yPercentButton = new QRadioButton(tr("Volume(%)"), this);
  axisLayout->addWidget(m_xPercentButton);
  axisLayout->addWidget(m_xGyButton);
  axisLayout->addWidget(m_yCcButton);
  axisLayout->addWidget(m_yPercentButton);

  // CalcMax control (for clamping high-dose tail)
  QLabel *calcMaxLabel = new QLabel(tr("CalcMax [Gy]"), this);
  m_calcMaxSpin = new QDoubleSpinBox(this);
  m_calcMaxSpin->setRange(0.0, 10000.0);
  m_calcMaxSpin->setDecimals(3);
  m_calcMaxSpin->setSingleStep(0.1);
  m_calcMaxSpin->setSpecialValueText(tr("Auto(Max)"));
  m_calcMaxSpin->setValue(0.0);  // Auto until data arrives
  m_calcMaxSpin->setToolTip(tr("Clamp doses above this value when drawing DVH"));
  // Button: Calc Max = MaxDose
  m_calcMax100Button = new QPushButton(tr("Calc Max = MaxDose"), this);
  m_calcMax100Button->setToolTip(tr("Set CalcMax to current maximum DVH dose"));
  QHBoxLayout *calcMaxLayout = new QHBoxLayout();
  calcMaxLayout->setContentsMargins(0, 0, 0, 0);
  calcMaxLayout->setSpacing(4);
  calcMaxLayout->addWidget(calcMaxLabel);
  calcMaxLayout->addWidget(m_calcMaxSpin);
  // Place the 100% button below the spin, compactly
  QVBoxLayout *calcMaxContainer = new QVBoxLayout();
  calcMaxContainer->setContentsMargins(0, 0, 0, 0);
  calcMaxContainer->setSpacing(2);
  calcMaxContainer->addLayout(calcMaxLayout);
  calcMaxContainer->addWidget(m_calcMax100Button);
  axisLayout->addLayout(calcMaxContainer);
  axisLayout->addStretch();
  m_progressBar = new QProgressBar(this);
  m_progressBar->setTextVisible(false);
  m_progressBar->setFixedHeight(10);
  m_progressBar->setRange(0, 100);
  m_progressBar->setValue(0);
  axisLayout->addWidget(m_progressBar);

  m_xUnitGroup = new QButtonGroup(this);
  m_xUnitGroup->setExclusive(true);
  m_xUnitGroup->addButton(m_xPercentButton);
  m_xUnitGroup->addButton(m_xGyButton);
  m_yUnitGroup = new QButtonGroup(this);
  m_yUnitGroup->setExclusive(true);
  m_yUnitGroup->addButton(m_yCcButton);
  m_yUnitGroup->addButton(m_yPercentButton);
  m_xGyButton->setChecked(true);
  m_yPercentButton->setChecked(true);
  connect(m_xPercentButton, &QRadioButton::toggled, this,
          &DVHWindow::onAxisUnitChanged);
  connect(m_xGyButton, &QRadioButton::toggled, this,
          &DVHWindow::onAxisUnitChanged);
  connect(m_yCcButton, &QRadioButton::toggled, this,
          &DVHWindow::onAxisUnitChanged);
  connect(m_yPercentButton, &QRadioButton::toggled, this,
          &DVHWindow::onAxisUnitChanged);
  connect(m_calcMaxSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
          &DVHWindow::onCalcMaxChanged);
  connect(m_calcMax100Button, &QPushButton::clicked, this,
          &DVHWindow::onCalcMaxTo100);

  plotLayout->addWidget(m_plot, 7);
  plotLayout->addWidget(controlPanel, 3);
  mainLayout->addLayout(plotLayout);

  // --- コントロール領域 ---
  QHBoxLayout *bottomLayout = new QHBoxLayout();
  bottomLayout->setContentsMargins(0, 0, 0, 0);
  // ROI リスト（チェックボックス）とテキストボックスの間隔も 10px
  bottomLayout->setSpacing(10);

  // 左側: ROI リストと All/None ボタン
  QVBoxLayout *roiLayout = new QVBoxLayout();
  roiLayout->setContentsMargins(0, 0, 0, 0);
  roiLayout->setSpacing(5);
  m_roiList = new QListWidget(this);
  // 画面幅がさらに狭い環境でも収まるよう最小幅を縮小
  m_roiList->setMinimumWidth(50);
  m_roiList->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);

  m_roiList->setSelectionMode(QAbstractItemView::SingleSelection);
  roiLayout->addWidget(m_roiList);

  QHBoxLayout *listButtonLayout = new QHBoxLayout();
  listButtonLayout->setContentsMargins(0, 0, 0, 0);
  listButtonLayout->setSpacing(5);
  m_allButton = new QPushButton(tr("All"), this);
  m_noneButton = new QPushButton(tr("None"), this);
  const int btnHeight = 24;
  m_allButton->setFixedHeight(btnHeight);
  m_noneButton->setFixedHeight(btnHeight);
  listButtonLayout->addWidget(m_allButton);
  listButtonLayout->addWidget(m_noneButton);
  roiLayout->addLayout(listButtonLayout);
  roiLayout->setStretch(0, 1);

  // 右側: 詳細表示と Export ボタン
  QVBoxLayout *detailLayout = new QVBoxLayout();
  detailLayout->setContentsMargins(0, 0, 0, 0);
  detailLayout->setSpacing(5);
  m_detailBox = new QPlainTextEdit(this);
  m_detailBox->setReadOnly(true);
  m_detailBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  // 詳細表示も最小幅を縮小してレイアウトを柔軟にする
  m_detailBox->setMinimumWidth(50);
  QFont detailFont = m_detailBox->font();
  detailFont.setPointSize(detailBoxPointSize);
  m_detailBox->setFont(detailFont);
  detailLayout->addWidget(m_detailBox);

  // 横幅が不足する環境ではボタンが隠れてしまうため、
  // 2列2行のグリッドレイアウトに変更してコンパクトに配置する
  QGridLayout *exportLayout = new QGridLayout();
  exportLayout->setContentsMargins(0, 0, 0, 0);
  exportLayout->setHorizontalSpacing(5);
  exportLayout->setVerticalSpacing(5);
  m_exportCsvButton = new QPushButton(tr("Export CSV"), this);
  m_exportPngButton = new QPushButton(tr("Export PNG"), this);
  m_exportPdfButton = new QPushButton(tr("Export PDF"), this);
  m_exportCsvButton->setFixedHeight(btnHeight);
  m_exportPngButton->setFixedHeight(btnHeight);
  m_exportPdfButton->setFixedHeight(btnHeight);
  exportLayout->addWidget(m_exportCsvButton, 0, 0);
  exportLayout->addWidget(m_exportPngButton, 0, 1);
  exportLayout->addWidget(m_exportPdfButton, 1, 0, 1, 2);
  detailLayout->addLayout(exportLayout);
  detailLayout->setStretch(0, 1);

  bottomLayout->addLayout(roiLayout);
  bottomLayout->addLayout(detailLayout);
  bottomLayout->setStretch(0, 0);
  bottomLayout->setStretch(1, 1);

  QWidget *controlWidget = new QWidget(this);
  controlWidget->setLayout(bottomLayout);
  controlWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

  // コントロール領域: 全体の40%を確保
  mainLayout->addWidget(controlWidget);

  // プロット領域とコントロール領域の比率を9:5に変更
  // （グラフ部分の横幅を従来比1.2倍に拡大）
  mainLayout->setStretch(0, 9); // プロット領域 約64%
  mainLayout->setStretch(1, 5); // コントロール領域 約36%

  connect(m_exportCsvButton, &QPushButton::clicked, this,
          &DVHWindow::onExportCSV);
  connect(m_exportPngButton, &QPushButton::clicked, this,
          &DVHWindow::onExportPNG);
  connect(m_exportPdfButton, &QPushButton::clicked, this,
          &DVHWindow::onExportPDF);
  connect(m_allButton, &QPushButton::clicked, this, &DVHWindow::onSelectAll);
  connect(m_noneButton, &QPushButton::clicked, this, &DVHWindow::onSelectNone);
  // ユーザー操作のみで可視性を変更するため、itemChanged を使用
  connect(m_roiList, &QListWidget::itemChanged, this,
          &DVHWindow::onRoiItemChanged);
  connect(m_roiList, &QListWidget::currentRowChanged, this,
          &DVHWindow::onCurrentRoiChanged);
}

void DVHWindow::setROINames(const QStringList &roiNames) {
  if (!m_roiList)
    return;

  qDebug() << QString("=== setROINames called with %1 ROIs ===")
                  .arg(roiNames.size());

  // 【修正】シグナルブロッカーのスコープを明確化
  {
    QSignalBlocker blocker(m_roiList);
    m_roiList->clear();

    for (int i = 0; i < roiNames.size(); ++i) {
      QListWidgetItem *item = new QListWidgetItem(roiNames[i], m_roiList);

      // 【修正】アイテム作成時にすべての必要な設定を一度に行う
      item->setData(Qt::UserRole, -1); // 無効なインデックス（データ未設定状態）
      item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable |
                     Qt::ItemIsUserCheckable);
      item->setCheckState(Qt::Unchecked);

      // アイコンを設定
      QColor color = QColor::fromHsv((i * 40) % 360, 255, 255);
      QPixmap pix(12, 12);
      pix.fill(color);
      item->setIcon(QIcon(pix));

      qDebug() << QString("Created ROI list item: %1 (index %2)")
                      .arg(roiNames[i])
                      .arg(i);
    }
  } // ここでシグナルブロッカーが解除される

  qDebug()
      << QString("setROINames completed for %1 items").arg(m_roiList->count());
}

void DVHWindow::setROIChecked(const QString &roiName, bool checked) {
  if (!m_roiList)
    return;
  QList<QListWidgetItem *> items =
      m_roiList->findItems(roiName, Qt::MatchExactly);
  if (items.isEmpty())
    return;
  QSignalBlocker blocker(m_roiList);
  items.first()->setCheckState(checked ? Qt::Checked : Qt::Unchecked);
}

void DVHWindow::setDVHData(const std::vector<DVHCalculator::DVHData> &dvhData) {
  qDebug() << "=== DVH Window setDVHData ===";
  qDebug() << QString("Received %1 DVH datasets").arg(dvhData.size());

  // 終了処理中は何もしない
  if (m_isClosing) {
    qDebug() << "DVHWindow is closing, ignoring setDVHData";
    return;
  }

  // 基本的なNullチェック
  if (!m_plot || !m_roiList) {
    qCritical() << "setDVHData: Essential components are null";
    return;
  }

  QString currentSelection;
  if (m_roiList->currentItem()) {
    currentSelection = m_roiList->currentItem()->text();
  }

  // データが空の場合の安全な処理
  if (dvhData.empty()) {
    qDebug() << "No DVH data provided, clearing display";

    try {
      // データをクリア
      m_data.clear();
      m_plot->clearGraphs();

      // UIコンポーネントの安全なクリア
      if (m_detailBox)
        m_detailBox->clear();

      if (m_roiNameLabel) {
        m_roiNameLabel->clear();
        m_roiNameLabel->setVisible(false);
      }

      if (m_cursorInfoLabel) {
        m_cursorInfoLabel->clear();
        m_cursorInfoLabel->setVisible(false);
      }

      if (m_cursorVLine)
        m_cursorVLine->setVisible(false);
      if (m_cursorHLine)
        m_cursorHLine->setVisible(false);

      // リストアイテムを安全にリセット（削除はしない）
      {
        QSignalBlocker blocker(m_roiList);
        for (int i = 0; i < m_roiList->count(); ++i) {
          QListWidgetItem *item = m_roiList->item(i);
          if (item) {
            item->setCheckState(Qt::Unchecked);
            item->setData(Qt::UserRole, -1); // 無効なインデックス
          }
        }
      }

      m_plot->replot();
      update();

    } catch (const std::exception &e) {
      qCritical() << QString("Exception clearing DVH data: %1").arg(e.what());
    }

    return;
  }

  try {
    // 現在の可視状態をROI名で保存
    QHash<QString, bool> previousVisibility;
    for (int i = 0; i < m_roiList->count(); ++i) {
      QListWidgetItem *item = m_roiList->item(i);
      if (item) {
        previousVisibility.insert(item->text(),
                                  item->checkState() == Qt::Checked);
      }
    }

    // 既存のデータと表示をクリア
    m_data.clear();
    m_plot->clearGraphs();

    bool hasValidData = false;
    double maxDose = 0.0;
    double maxVolumeCc = 0.0;

    // 名前とアイテムのマッピングを作成し、重複を除去
    QHash<QString, QListWidgetItem *> itemMap;
    {
      QSignalBlocker blocker(m_roiList);
      for (int i = 0; i < m_roiList->count(); ++i) {
        QListWidgetItem *item = m_roiList->item(i);
        if (!item)
          continue;
        QString name = item->text();
        if (itemMap.contains(name)) {
          delete m_roiList->takeItem(i);
          --i;
          continue;
        }
        // Keep current check state to preserve user's intent for yet-to-be-computed ROIs
        // Reset only the data index; it will be reassigned for ROIs with data below.
        itemMap.insert(name, item);
        item->setData(Qt::UserRole, -1);
      }
    }

    // 各ROIのDVHデータを処理
    for (size_t dataIndex = 0; dataIndex < dvhData.size(); ++dataIndex) {
      auto data = dvhData[dataIndex]; // コピーして加工

      qDebug() << QString("Processing ROI %1 (data index %2)")
                      .arg(data.roiName)
                      .arg(dataIndex);

      // データの妥当性チェック
      if (data.points.empty() || data.totalVolume <= 0) {
        qDebug() << QString("  Skipping invalid ROI %1").arg(data.roiName);
        continue;
      }

      try {
        // データポイントの妥当性チェックと準備
        QVector<double> xData, yData;
        xData.reserve(static_cast<int>(data.points.size()));
        yData.reserve(static_cast<int>(data.points.size()));
        double roiMaxDose = 0.0;

        for (const auto &point : data.points) {
          if (std::isfinite(point.dose) && std::isfinite(point.volume) &&
              point.dose >= 0.0 && point.volume >= 0.0) {
            xData.append(point.dose);
            yData.append(point.volume);
            roiMaxDose = std::max(roiMaxDose, point.dose);
          }
        }

        if (xData.isEmpty()) {
          qDebug()
              << QString("  No valid data points for %1").arg(data.roiName);
          continue;
        }

        // グラフの作成
        QCPGraph *graph = m_plot->addGraph();
        if (!graph) {
          qWarning()
              << QString("Failed to create graph for %1").arg(data.roiName);
          continue;
        }

        // グラフデータを設定
        graph->setData(xData, yData);

        QPen pen(data.color);
        pen.setWidth(2);
        graph->setPen(pen);

        // 以前の可視状態を適用（なければデフォルト値）
        bool visible = previousVisibility.contains(data.roiName)
                           ? previousVisibility.value(data.roiName)
                           : data.isVisible;
        graph->setVisible(visible);
        data.isVisible = visible;
        data.maxDose = roiMaxDose;

        hasValidData = true;
        maxDose = std::max(maxDose, roiMaxDose);
        maxVolumeCc = std::max(maxVolumeCc, data.totalVolume / 1000.0);

        // 新しいROIインデックスを計算
        int newRoiIndex = static_cast<int>(m_data.size());

        // データを追加（インデックスと一致させる）
        m_data.push_back(std::move(data));

        // 対応するリストアイテムを取得または新規作成
        QListWidgetItem *item =
            itemMap.value(m_data[newRoiIndex].roiName, nullptr);
        if (!item) {
          item = new QListWidgetItem(m_data[newRoiIndex].roiName, m_roiList);
          item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
          QColor color = m_data[newRoiIndex].color;
          QPixmap pix(12, 12);
          pix.fill(color);
          item->setIcon(QIcon(pix));
          itemMap.insert(m_data[newRoiIndex].roiName, item);
          qDebug() << QString("Created new list item for ROI: %1")
                          .arg(m_data[newRoiIndex].roiName);
        }

        // インデックスと状態を安全に設定
        {
          QSignalBlocker blocker(m_roiList);
          item->setData(Qt::UserRole, newRoiIndex);
          item->setCheckState(visible ? Qt::Checked : Qt::Unchecked);
        }

        qDebug() << QString("  ROI %1 assigned index %2, visible: %3")
                        .arg(m_data[newRoiIndex].roiName)
                        .arg(newRoiIndex)
                        .arg(visible);

      } catch (const std::exception &e) {
        qCritical() << QString("Exception processing ROI %1: %2")
                           .arg(data.roiName)
                           .arg(e.what());
        continue;
      }
    }

    // グローバル最大値を設定
    m_globalMaxDose = maxDose;
    m_globalMaxVolumeCc = maxVolumeCc;

    if (hasValidData) {
      // 軸の更新と再描画
      updatePlotUnits();

      // 選択状態を復元
      if (!currentSelection.isEmpty()) {
        for (int i = 0; i < m_roiList->count(); ++i) {
          QListWidgetItem *item = m_roiList->item(i);
          if (item && item->text() == currentSelection) {
            m_roiList->setCurrentItem(item);
            break;
          }
        }
      }

      if (m_roiList->currentItem())
        onCurrentRoiChanged(m_roiList->currentRow());

      qDebug() << QString("DVH data successfully set for %1 ROIs")
                      .arg(m_data.size());
    } else {
      qWarning() << "No valid DVH data found";
    }

  } catch (const std::exception &e) {
    qCritical() << QString("Exception in setDVHData: %1").arg(e.what());
  }
}

void DVHWindow::onAxisUnitChanged() { updatePlotUnits(); }

void DVHWindow::onCalcMaxChanged(double value) {
  // Auto when value <= 0, otherwise user override
  if (value > 0.0) {
    m_calcMaxGy = value;
    m_calcMaxUserSet = true;
  } else {
    m_calcMaxGy = 0.0;
    m_calcMaxUserSet = false;
  }
  // CalcMax変更時は既存のDVHデータや表示を全て初期化
  try {
    if (m_plot)
      m_plot->clearGraphs();
    m_data.clear();
    m_globalMaxDose = 0.0;
    m_globalMaxVolumeCc = 0.0;

    if (m_detailBox)
      m_detailBox->clear();

    if (m_roiNameLabel) {
      m_roiNameLabel->clear();
      m_roiNameLabel->setVisible(false);
    }

    if (m_cursorInfoLabel) {
      m_cursorInfoLabel->clear();
      m_cursorInfoLabel->setVisible(false);
    }

    if (m_cursorVLine)
      m_cursorVLine->setVisible(false);
    if (m_cursorHLine)
      m_cursorHLine->setVisible(false);

    if (m_progressBar) {
      m_progressBar->setRange(0, 100);
      m_progressBar->setValue(0);
    }

    if (m_roiList) {
      QSignalBlocker blocker(m_roiList);
      for (int i = 0; i < m_roiList->count(); ++i) {
        if (auto *item = m_roiList->item(i)) {
          item->setData(Qt::UserRole, -1);
          item->setCheckState(Qt::Unchecked);
        }
      }
    }

    if (m_plot)
      m_plot->replot();
  } catch (...) {
  }

  // 通知: CalcMax変更に伴い再計算を要求（全体再計算は上位が処理）
  try {
    emit calcMaxChanged(m_calcMaxGy);
  } catch (...) {
  }
}

void DVHWindow::onCalcMaxTo100() {
  // Set CalcMax to the current global maximum dose of all DVHs
  double maxGy = m_globalMaxDose;
  if (maxGy <= 0.0) {
    // Compute from data if not yet cached
    for (const auto &d : m_data) {
      for (const auto &p : d.points) maxGy = std::max(maxGy, p.dose);
    }
  }
  if (maxGy < 0.0) maxGy = 0.0;
  if (m_calcMaxSpin) {
    // Setting the spin value will trigger onCalcMaxChanged and notify parent
    m_calcMaxSpin->setValue(maxGy);
  } else {
    onCalcMaxChanged(maxGy);
  }
}

void DVHWindow::updatePlotUnits() {
  bool xPercent = m_xPercentButton && m_xPercentButton->isChecked();
  bool yCc = m_yCcButton && m_yCcButton->isChecked();

  // 全てのROIの生データから最大線量と体積を再計算し、
  // 表示中ROIの最大体積も併せて取得する
  m_globalMaxDose = 0.0;
  m_globalMaxVolumeCc = 0.0;
  double visibleMaxVolumeCc = 0.0;
  for (auto &d : m_data) {
    double roiMaxDose = 0.0;
    for (const auto &p : d.points)
      roiMaxDose = std::max(roiMaxDose, p.dose);
    d.maxDose = roiMaxDose;
    double volumeCc = d.totalVolume / 1000.0;
    m_globalMaxDose = std::max(m_globalMaxDose, d.maxDose);
    m_globalMaxVolumeCc = std::max(m_globalMaxVolumeCc, volumeCc);
    if (d.isVisible)
      visibleMaxVolumeCc = std::max(visibleMaxVolumeCc, volumeCc);
  }

  // If user hasn't set CalcMax explicitly, default to global max dose
  if (!m_calcMaxUserSet) {
    if (!qFuzzyCompare(m_calcMaxGy + 1.0, m_globalMaxDose + 1.0)) {
      m_calcMaxGy = m_globalMaxDose;
      if (m_calcMaxSpin) {
        QSignalBlocker blocker(m_calcMaxSpin);
        m_calcMaxSpin->setValue(m_calcMaxGy);
      }
    }
  }

  // Ensure a reasonable positive CalcMax for plotting
  double calcMaxGy = (m_calcMaxGy > 0.0) ? m_calcMaxGy : m_globalMaxDose;

  for (int i = 0;
       i < m_plot->graphCount() && i < static_cast<int>(m_data.size()); ++i) {
    QCPGraph *graph = m_plot->graph(i);
    if (!graph)
      continue;
    const auto &d = m_data[i];
    QVector<double> xData, yData;
    xData.reserve(d.points.size());
    yData.reserve(d.points.size());
    for (const auto &p : d.points) {
      double clampedDose = std::min(p.dose, calcMaxGy);
      double x = xPercent && m_prescriptionDose > 0.0
                     ? clampedDose / m_prescriptionDose * 100.0
                     : clampedDose;
      double y = yCc ? (p.volume / 100.0) * (d.totalVolume / 1000.0) : p.volume;
      xData.append(x);
      yData.append(y);
    }
    graph->setData(xData, yData);
  }

  if (xPercent) {
    m_plot->xAxis->setLabel(tr("Dose [%]"));
    double upper = 100.0;
    if (m_prescriptionDose > 0.0) {
      double cmPercent = (calcMaxGy / m_prescriptionDose) * 100.0;
      // Set upper exactly to CalcMax in percent for clear clamp visualization
      upper = std::max(1.0, cmPercent);
    }
    double step = niceTickStep(upper);
    // データ最大値を刻み幅単位で切り上げ、端の補助線を確実に描画
    double tickUpper = std::ceil(upper / step) * step;
    double rangeUpper = tickUpper + step;
    QVector<double> ticks;
    for (double t = 0; t <= rangeUpper + 1e-9; t += step)
      ticks << t;
    m_plot->xAxis->setAutoTicks(false);
    m_plot->xAxis->setTickVector(ticks);
    m_plot->xAxis->setRange(0, rangeUpper);
  } else {
    m_plot->xAxis->setLabel(tr("Dose [Gy]"));
    // Set Gy axis upper limit to CalcMax
    double upper = std::max(1e-6, calcMaxGy);
    double step = niceTickStep(upper);
    // データ最大値を刻み幅単位で切り上げ、端の補助線を確実に描画
    double tickUpper = std::ceil(upper / step) * step;
    double rangeUpper = tickUpper + step;
    QVector<double> ticks;
    for (double t = 0; t <= rangeUpper + 1e-9; t += step)
      ticks << t;
    m_plot->xAxis->setAutoTicks(false);
    m_plot->xAxis->setTickVector(ticks);
    m_plot->xAxis->setRange(0, rangeUpper);
  }

  if (yCc) {
    m_plot->yAxis->setLabel(tr("Volume [cc]"));
    double maxVolume =
        visibleMaxVolumeCc > 0.0 ? visibleMaxVolumeCc : m_globalMaxVolumeCc;
    m_plot->yAxis->setRange(0, maxVolume);
  } else {
    m_plot->yAxis->setLabel(tr("Volume [%]"));
    m_plot->yAxis->setRange(0, 100);
  }

  m_plot->replot();
}

double DVHWindow::niceTickStep(double range) const {
  if (range <= 0)
    return 1.0;
  double exponent = std::floor(std::log10(range));
  double fraction = range / std::pow(10.0, exponent);
  double niceFraction;
  if (fraction < 1.5)
    niceFraction = 1.0;
  else if (fraction < 3.0)
    niceFraction = 2.0;
  else if (fraction < 7.0)
    niceFraction = 5.0;
  else
    niceFraction = 10.0;
  return niceFraction * std::pow(10.0, exponent - 1);
}

void DVHWindow::updateVisibility(int roiIndex, bool visible) {
  qDebug() << QString("=== updateVisibility: ROI %1, visible %2 ===")
                  .arg(roiIndex)
                  .arg(visible);

  // 基本的な妥当性チェック
  if (roiIndex < 0) {
    qWarning()
        << QString("updateVisibility: Invalid ROI index %1").arg(roiIndex);
    return;
  }

  if (static_cast<size_t>(roiIndex) >= m_data.size()) {
    qCritical()
        << QString(
               "updateVisibility: ROI index %1 out of range (data size: %2)")
               .arg(roiIndex)
               .arg(m_data.size());
    return;
  }

  if (!m_plot) {
    qWarning() << "updateVisibility: m_plot is null";
    return;
  }

  try {
    // データの可視性を更新
    m_data[roiIndex].isVisible = visible;
    qDebug() << QString("Updated data visibility for ROI %1: %2")
                    .arg(roiIndex)
                    .arg(visible);

    // グラフの可視性を更新
    if (roiIndex < m_plot->graphCount()) {
      QCPGraph *graph = m_plot->graph(roiIndex);
      if (graph) {
        graph->setVisible(visible);
        qDebug()
            << QString("Updated graph visibility for ROI %1").arg(roiIndex);
      } else {
        qWarning() << QString("Graph for ROI %1 is null").arg(roiIndex);
      }
    } else {
      qWarning() << QString("ROI index %1 exceeds graph count %2")
                        .arg(roiIndex)
                        .arg(m_plot->graphCount());
    }

    // リストアイテムの状態を更新（シグナルを発生させずに）
    if (m_roiList) {
      // 対応するリストアイテムを検索
      QListWidgetItem *targetItem = nullptr;
      for (int i = 0; i < m_roiList->count(); ++i) {
        QListWidgetItem *item = m_roiList->item(i);
        if (item && item->data(Qt::UserRole).toInt() == roiIndex) {
          targetItem = item;
          break;
        }
      }

      if (targetItem) {
        // シグナルをブロックして状態更新
        QSignalBlocker blocker(m_roiList);
        targetItem->setCheckState(visible ? Qt::Checked : Qt::Unchecked);
        qDebug() << QString("Updated list item for ROI %1 check state to %2")
                        .arg(roiIndex)
                        .arg(visible);
      } else {
        qWarning() << QString("List item for ROI %1 not found").arg(roiIndex);
      }
    } else {
      qWarning() << "updateVisibility: m_roiList is null";
    }

    // 可視性変更シグナルを発行
    try {
      emit visibilityChanged(roiIndex, visible);
    } catch (const std::exception &e) {
      qCritical()
          << QString("Exception emitting visibilityChanged: %1").arg(e.what());
    } catch (...) {
      qCritical() << "Unknown exception emitting visibilityChanged";
    }

    // 軸スケールを再計算し再描画
    try {
      updatePlotUnits();
    } catch (const std::exception &e) {
      qCritical() << QString("Exception in updatePlotUnits: %1").arg(e.what());
    } catch (...) {
      qCritical() << "Unknown exception in updatePlotUnits";
    }

    // ROI情報ラベルの可視性を更新
    if (m_roiList && m_roiNameLabel) {
      QListWidgetItem *currentItem = m_roiList->currentItem();
      if (currentItem && currentItem->data(Qt::UserRole).toInt() == roiIndex) {
        m_roiNameLabel->setVisible(visible);
        updateOverlayPosition();

        // 非表示の場合はカーソル情報も非表示
        if (!visible) {
          if (m_cursorInfoLabel)
            m_cursorInfoLabel->setVisible(false);
          if (m_cursorVLine)
            m_cursorVLine->setVisible(false);
          if (m_cursorHLine)
            m_cursorHLine->setVisible(false);
          m_roiNameLabel->setVisible(false);
        }
      }
    }

    qDebug()
        << QString("DVH visibility update completed for ROI %1").arg(roiIndex);

  } catch (const std::exception &e) {
    qCritical() << QString("Error updating DVH visibility for ROI %1: %2")
                       .arg(roiIndex)
                       .arg(e.what());
  } catch (...) {
    qCritical() << QString("Unknown error updating DVH visibility for ROI %1")
                       .arg(roiIndex);
  }
}

void DVHWindow::onRoiItemChanged(QListWidgetItem *item) {
  qDebug() << "=== onRoiItemChanged ===";

  // 終了処理中は何もしない
  if (m_isClosing) {
    qDebug() << "DVHWindow is closing, ignoring item change";
    return;
  }

  // Nullチェック
  if (!item) {
    qWarning() << "onRoiItemChanged: item is null";
    return;
  }

  if (!m_roiList) {
    qWarning() << "onRoiItemChanged: m_roiList is null";
    return;
  }

  QString roiName = item->text();
  bool isChecked = (item->checkState() == Qt::Checked);

  qDebug() << QString("ROI %1 check state changed to: %2")
                  .arg(roiName)
                  .arg(isChecked);

  // データインデックスを取得
  bool ok = false;
  int idx = item->data(Qt::UserRole).toInt(&ok);
  if (!ok) {
    // Treat as not-yet-computed ROI (index = -1)
    idx = -1;
    qWarning() << QString(
                       "Item data missing/invalid for ROI '%1'. Treating as new.")
                      .arg(roiName);
  }

  qDebug() << QString("Processing item: %1, index: %2, data size: %3")
                  .arg(roiName)
                  .arg(idx)
                  .arg(m_data.size());

  // 有効なインデックスの場合（既にDVHデータがある）
  if (idx >= 0) {
    // 配列範囲チェック
    if (static_cast<size_t>(idx) >= m_data.size()) {
      qCritical() << QString("Index %1 out of range for ROI %2 (data size: %3)")
                         .arg(idx)
                         .arg(roiName)
                         .arg(m_data.size());

      // データを初期化してクラッシュを防ぐ
      {
        QSignalBlocker blocker(m_roiList);
        item->setData(Qt::UserRole, -1);
        item->setCheckState(Qt::Unchecked);
      }
      return;
    }

    // ROI名が一致するか確認し、ずれがあれば修正
    if (m_data[idx].roiName != roiName) {
      qWarning() << QString("ROI name mismatch: item %1 points to %2")
                        .arg(roiName)
                        .arg(m_data[idx].roiName);
      bool found = false;
      for (size_t i = 0; i < m_data.size(); ++i) {
        if (m_data[i].roiName == roiName) {
          idx = static_cast<int>(i);
          QSignalBlocker blocker(m_roiList);
          item->setData(Qt::UserRole, idx);
          found = true;
          break;
        }
      }
      if (!found) {
        QSignalBlocker blocker(m_roiList);
        item->setData(Qt::UserRole, -1);
        item->setCheckState(Qt::Unchecked);
        qWarning()
            << QString("ROI %1 not found in data, resetting item").arg(roiName);
        return;
      }
    }

    // 安全な可視性更新
    try {
      qDebug() << QString("Updating visibility for ROI %1 (index %2) to %3")
                      .arg(roiName)
                      .arg(idx)
                      .arg(isChecked);
      updateVisibility(idx, isChecked);
    } catch (const std::exception &e) {
      qCritical() << QString("Exception in updateVisibility for ROI %1: %2")
                         .arg(roiName)
                         .arg(e.what());
    } catch (...) {
      qCritical() << QString("Unknown exception in updateVisibility for ROI %1")
                         .arg(roiName);
    }
    if (m_roiList->currentItem() == item)
      updateRoiInfo(idx);
    return;
  }

  // 無効なインデックス（-1）の場合：新しい計算をリクエスト
  if (isChecked) {
    qDebug() << QString("Requesting recalculation for ROI: %1").arg(roiName);
    try {
      emit recalculateRequested(roiName);
    } catch (const std::exception &e) {
      qCritical()
          << QString("Exception emitting recalculateRequested for ROI %1: %2")
                 .arg(roiName)
                 .arg(e.what());
    } catch (...) {
      qCritical()
          << QString(
                 "Unknown exception emitting recalculateRequested for ROI %1")
                 .arg(roiName);
    }
  } else {
    qDebug() << QString("ROI %1 unchecked but has no data - no action needed")
                    .arg(roiName);
  }

  if (m_roiList->currentItem() == item)
    updateRoiInfo(idx);
}

void DVHWindow::onCurrentRoiChanged(int row) {
  if (m_isClosing || !m_roiList)
    return;

  QListWidgetItem *item = m_roiList->item(row);
  if (!item) {
    updateRoiInfo(-1);
    if (m_roiNameLabel) {
      m_roiNameLabel->clear();
      m_roiNameLabel->setVisible(false);
    }
    return;
  }

  bool ok = false;
  int idx = item->data(Qt::UserRole).toInt(&ok);
  if (!ok)
    idx = -1;

  updateRoiInfo(idx);

  if (m_roiNameLabel) {
    if (ok && static_cast<size_t>(idx) < m_data.size()) {
      m_roiNameLabel->setText(m_data[idx].roiName);
      m_roiNameLabel->setVisible(m_data[idx].isVisible);
      updateOverlayPosition();
    } else {
      m_roiNameLabel->clear();
      m_roiNameLabel->setVisible(false);
    }
  }
}

void DVHWindow::updateRoiInfo(int index) {
  qDebug() << QString("=== updateRoiInfo: index %1, data size %2 ===")
                  .arg(index)
                  .arg(m_data.size());

  // 基本的な妥当性チェック
  if (!m_detailBox) {
    qWarning() << "updateRoiInfo: m_detailBox is null";
    return;
  }

  // 無効なインデックスの場合
  if (index < 0 || static_cast<size_t>(index) >= m_data.size()) {
    qDebug() << QString("updateRoiInfo: Invalid index %1, clearing detail box")
                    .arg(index);
    try {
      m_detailBox->clear();
    } catch (const std::exception &e) {
      qWarning() << QString("Exception clearing detail box: %1").arg(e.what());
    }
    return;
  }

  try {
    const auto &data = m_data[index];

    // Helper lambdas for DVH calculations
    auto doseAtVolume = [](const DVHCalculator::DVHData &d,
                           double volPercent) -> double {
      if (d.points.empty())
        return 0.0;
      const auto &pts = d.points;
      if (volPercent >= pts.front().volume)
        return pts.front().dose;
      if (volPercent <= pts.back().volume)
        return pts.back().dose;
      for (size_t i = 1; i < pts.size(); ++i) {
        double v0 = pts[i - 1].volume;
        double v1 = pts[i].volume;
        if (v0 >= volPercent && v1 <= volPercent) {
          double d0 = pts[i - 1].dose;
          double d1 = pts[i].dose;
          double t = (v0 - volPercent) / (v0 - v1);
          return d0 + t * (d1 - d0);
        }
      }
      return pts.back().dose;
    };

    auto volumeAtDose = [](const DVHCalculator::DVHData &d,
                           double doseGy) -> double {
      if (d.points.empty())
        return 0.0;
      const auto &pts = d.points;
      if (doseGy <= pts.front().dose)
        return pts.front().volume;
      if (doseGy >= pts.back().dose)
        return pts.back().volume;
      for (size_t i = 1; i < pts.size(); ++i) {
        double d0 = pts[i - 1].dose;
        double d1 = pts[i].dose;
        if (d0 <= doseGy && d1 >= doseGy) {
          double v0 = pts[i - 1].volume;
          double v1 = pts[i].volume;
          double t = (doseGy - d0) / (d1 - d0);
          return v0 + t * (v1 - v0);
        }
      }
      return 0.0;
    };

    double totalVolumeCm3 = data.totalVolume / 1000.0;

    QString info;
    info += QString("ROI Name: %1\n").arg(data.roiName);
    info += QString("Total Volume: %1 cm³\n").arg(totalVolumeCm3, 0, 'f', 2);
    info += QString("Maximum Dose: %1 Gy\n").arg(data.maxDose, 0, 'f', 2);
    if (data.minDose >= 0.0) {
      info += QString("Minimum Dose: %1 Gy\n").arg(data.minDose, 0, 'f', 2);
    }
    info += QString("Data Points: %1\n").arg(data.points.size());
    info +=
        QString("Visibility: %1\n").arg(data.isVisible ? "Visible" : "Hidden");

    if (m_prescriptionDose > 0.0) {
      info += QString("\n--- Statistics ---\n");
      info += QString("Prescription Dose: %1 Gy\n")
                  .arg(m_prescriptionDose, 0, 'f', 2);

      double v95 = volumeAtDose(data, m_prescriptionDose * 0.95);
      double v100 = volumeAtDose(data, m_prescriptionDose);
      double v105 = volumeAtDose(data, m_prescriptionDose * 1.05);

      info += QString("V95%%: %1%% (%2 cm³)\n")
                  .arg(v95, 0, 'f', 1)
                  .arg(v95 / 100.0 * totalVolumeCm3, 0, 'f', 2);
      info += QString("V100%%: %1%% (%2 cm³)\n")
                  .arg(v100, 0, 'f', 1)
                  .arg(v100 / 100.0 * totalVolumeCm3, 0, 'f', 2);
      info += QString("V105%%: %1%% (%2 cm³)\n")
                  .arg(v105, 0, 'f', 1)
                  .arg(v105 / 100.0 * totalVolumeCm3, 0, 'f', 2);
    }

    info += QString("\n--- Dose Metrics ---\n");
    double d99 = doseAtVolume(data, 99.0);
    double d98 = doseAtVolume(data, 98.0);
    double d95 = doseAtVolume(data, 95.0);
    double d90 = doseAtVolume(data, 90.0);
    double d50 = doseAtVolume(data, 50.0);
    double d2 = doseAtVolume(data, 2.0);
    double d1 = doseAtVolume(data, 1.0);

    info += QString("D99: %1 Gy\n").arg(d99, 0, 'f', 2);
    info += QString("D98: %1 Gy\n").arg(d98, 0, 'f', 2);
    info += QString("D95: %1 Gy\n").arg(d95, 0, 'f', 2);
    info += QString("D90: %1 Gy\n").arg(d90, 0, 'f', 2);
    info += QString("D50: %1 Gy\n").arg(d50, 0, 'f', 2);
    info += QString("D2: %1 Gy\n").arg(d2, 0, 'f', 2);
    info += QString("D1: %1 Gy\n").arg(d1, 0, 'f', 2);
    if (data.meanDose >= 0.0) {
      info += QString("Mean Dose: %1 Gy\n").arg(data.meanDose, 0, 'f', 2);
    }

    m_detailBox->setPlainText(info);

    qDebug() << QString("Successfully updated ROI info for index %1 (%2)")
                    .arg(index)
                    .arg(data.roiName);

  } catch (const std::exception &e) {
    qCritical() << QString("Exception updating ROI info for index %1: %2")
                       .arg(index)
                       .arg(e.what());
    try {
      m_detailBox->setPlainText(
          QString("Error displaying ROI information\nIndex: %1").arg(index));
    } catch (...) {
      qCritical() << "Failed to set error message in detail box";
    }
  } catch (...) {
    qCritical() << QString("Unknown exception updating ROI info for index %1")
                       .arg(index);
    try {
      m_detailBox->setPlainText(
          QString("Unknown error displaying ROI information\nIndex: %1")
              .arg(index));
    } catch (...) {
      qCritical() << "Failed to set error message in detail box";
    }
  }
}

void DVHWindow::updateOverlayPosition() {
  if (!m_plot) {
    qWarning() << "updateOverlayPosition: m_plot is null";
    return;
  }

  try {
    // 患者情報ラベルの位置更新
    if (m_patientInfoLabel && m_patientInfoLabel->isVisible()) {
      m_patientInfoLabel->adjustSize();
      int x = m_plot->width() - m_patientInfoLabel->width() -
              10; // 右端から10px余白
      int y = 5;  // 上端から5px余白

      // 範囲チェック
      if (x >= 0 && y >= 0) {
        m_patientInfoLabel->move(x, y);
      }
    }

    // ROI名ラベルの位置更新
    if (m_roiNameLabel && m_roiNameLabel->isVisible()) {
      m_roiNameLabel->adjustSize();
      int x = (m_plot->width() - m_roiNameLabel->width()) / 2; // 中央配置
      int y = 5; // 上端から5px余白

      // 範囲チェック
      if (x >= 0 && y >= 0) {
        m_roiNameLabel->move(x, y);
      }
    }

  } catch (const std::exception &e) {
    qWarning()
        << QString("Exception in updateOverlayPosition: %1").arg(e.what());
  } catch (...) {
    qWarning() << "Unknown exception in updateOverlayPosition";
  }
}

void DVHWindow::onPlotMouseMove(QMouseEvent *event) {
  qDebug() << "=== onPlotMouseMove ===";

  // 基本的なチェック
  if (!event || !m_plot || !m_roiList) {
    qDebug() << "Basic checks failed in onPlotMouseMove";
    return;
  }

  if (m_data.empty()) {
    qDebug() << "No DVH data available";
    // カーソル情報を非表示
    if (m_cursorInfoLabel)
      m_cursorInfoLabel->setVisible(false);
    if (m_cursorVLine)
      m_cursorVLine->setVisible(false);
    if (m_cursorHLine)
      m_cursorHLine->setVisible(false);
    if (m_roiNameLabel)
      m_roiNameLabel->setVisible(false);
    return;
  }

  // 現在選択されているROIのインデックスを取得
  QListWidgetItem *currentItem = m_roiList->currentItem();
  if (!currentItem) {
    qDebug() << "No current item selected";
    return;
  }

  // インデックスの妥当性チェック
  bool ok = false;
  int idx = currentItem->data(Qt::UserRole).toInt(&ok);

  if (!ok || idx < 0 || static_cast<size_t>(idx) >= m_data.size()) {
    qDebug() << QString("Invalid ROI index: %1 (ok=%2, data size=%3)")
                    .arg(idx)
                    .arg(ok)
                    .arg(m_data.size());
    // カーソル情報を非表示
    if (m_cursorInfoLabel)
      m_cursorInfoLabel->setVisible(false);
    if (m_cursorVLine)
      m_cursorVLine->setVisible(false);
    if (m_cursorHLine)
      m_cursorHLine->setVisible(false);
    return;
  }

  const auto &data = m_data[idx];

  // 非表示のROIの場合はカーソル情報を表示しない
  if (!data.isVisible) {
    qDebug() << QString("ROI %1 is not visible, hiding cursor info")
                    .arg(data.roiName);
    if (m_cursorInfoLabel)
      m_cursorInfoLabel->setVisible(false);
    if (m_cursorVLine)
      m_cursorVLine->setVisible(false);
    if (m_cursorHLine)
      m_cursorHLine->setVisible(false);
    if (m_roiNameLabel)
      m_roiNameLabel->setVisible(false);
    m_plot->replot();
    return;
  }

  try {
    // マウス座標からプロット座標に変換
    QRect plotRect = m_plot->plotRect();
    if (plotRect.width() <= 0 || plotRect.height() <= 0) {
      qDebug() << "Invalid plot rectangle";
      return;
    }

    // X軸座標の計算
    double coordX = m_plot->xAxis->pixelToCoord(event->pos().x());
    double coordY = 0.0; // Y座標は後で補間計算

    // 単位変換の設定
    bool xPercent = m_xPercentButton && m_xPercentButton->isChecked();
    bool yCc = m_yCcButton && m_yCcButton->isChecked();

    // 線量値の計算
    double doseGy = xPercent ? coordX / 100.0 * m_prescriptionDose : coordX;

    qDebug() << QString("Mouse coordinates: (%1, %2), Dose: %3 Gy")
                    .arg(event->pos().x())
                    .arg(event->pos().y())
                    .arg(doseGy);

    // DVHカーブからの体積値の補間計算
    double volumePercent = 0.0;
    bool foundValue = false;

    if (!data.points.empty()) {
      // 範囲チェック
      if (doseGy <= data.points.front().dose) {
        volumePercent = data.points.front().volume;
        foundValue = true;
      } else if (doseGy >= data.points.back().dose) {
        volumePercent = data.points.back().volume;
        foundValue = true;
      } else {
        // 線形補間
        for (size_t i = 1; i < data.points.size(); ++i) {
          if (data.points[i].dose >= doseGy) {
            const auto &p1 = data.points[i - 1];
            const auto &p2 = data.points[i];
            double t = (doseGy - p1.dose) / (p2.dose - p1.dose);
            volumePercent = p1.volume + t * (p2.volume - p1.volume);
            foundValue = true;
            break;
          }
        }
      }
    }

    if (!foundValue) {
      qDebug() << "Could not find/interpolate volume value";
      return;
    }

    // 表示用の体積値を計算
    double displayVolume =
        yCc ? (volumePercent / 100.0) * (data.totalVolume / 1000.0)
            : volumePercent;
    coordY = displayVolume; // カーソルライン用のY座標

    // カーソル情報テキストの作成
    if (m_cursorInfoLabel) {
      auto formatValue = [](double v) {
        QString text = QString::number(v, 'f', 2);
        while (text.endsWith('0') && text.contains('.'))
          text.chop(1);
        if (text.endsWith('.'))
          text.chop(1);
        return text;
      };

      QString doseStr = formatValue(xPercent ? coordX : doseGy);
      QString volumeStr = formatValue(displayVolume);
      QString doseUnit = xPercent ? "%" : "Gy";
      QString volumeUnit = yCc ? "cm³" : "%";

      QString cursorText = QString("Dose: %1%2\nVolume: %3%4")
                               .arg(doseStr)
                               .arg(doseUnit)
                               .arg(volumeStr)
                               .arg(volumeUnit);

      m_cursorInfoLabel->setText(cursorText);
      m_cursorInfoLabel->adjustSize();

      // カーソル情報ラベルの位置調整
      int labelX = event->pos().x() + 10;
      int labelY = event->pos().y() - m_cursorInfoLabel->height() - 10;

      // 画面外に出ないように調整
      if (labelX + m_cursorInfoLabel->width() > m_plot->width()) {
        labelX = event->pos().x() - m_cursorInfoLabel->width() - 10;
      }
      if (labelY < 0) {
        labelY = event->pos().y() + 10;
      }

      m_cursorInfoLabel->move(labelX, labelY);
      m_cursorInfoLabel->setVisible(true);
    }

    // カーソルラインの表示
    if (m_cursorVLine && m_cursorHLine) {
      // 垂直線
      m_cursorVLine->start->setCoords(coordX, m_plot->yAxis->min());
      m_cursorVLine->end->setCoords(coordX, m_plot->yAxis->max());
      m_cursorVLine->setVisible(true);

      // 水平線
      m_cursorHLine->start->setCoords(m_plot->xAxis->min(), coordY);
      m_cursorHLine->end->setCoords(m_plot->xAxis->max(), coordY);
      m_cursorHLine->setVisible(true);
    }

    // ROI名ラベルの表示
    if (m_roiNameLabel) {
      m_roiNameLabel->setText(data.roiName);
      m_roiNameLabel->setVisible(true);
      updateOverlayPosition();
    }

    // プロット更新
    m_plot->replot();

    qDebug() << QString("Cursor info updated successfully for ROI %1")
                    .arg(data.roiName);

  } catch (const std::exception &e) {
    qCritical() << QString("Exception in onPlotMouseMove: %1").arg(e.what());
  } catch (...) {
    qCritical() << "Unknown exception in onPlotMouseMove";
  }
}

void DVHWindow::onExportCSV() {
  QString fileName = QFileDialog::getSaveFileName(
      this, tr("Export CSV"), QString(), "CSV Files (*.csv)");
  if (fileName.isEmpty())
    return;
  QFile file(fileName);
  if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
    QMessageBox::warning(this, tr("Error"), tr("Cannot write file"));
    return;
  }
  QTextStream out(&file);
  for (const auto &data : m_data) {
    out << data.roiName << "\n";
    for (const auto &p : data.points)
      out << p.dose << "," << p.volume << "\n";
    out << "\n";
  }
}

void DVHWindow::onExportPNG() {
  QString fileName = QFileDialog::getSaveFileName(
      this, tr("Export PNG"), QString(), "PNG Files (*.png)");
  if (!fileName.isEmpty())
    m_plot->savePng(fileName);
}

void DVHWindow::onExportPDF() {
  QString fileName = QFileDialog::getSaveFileName(
      this, tr("Export PDF"), QString(), "PDF Files (*.pdf)");
  if (!fileName.isEmpty())
    m_plot->savePdf(fileName);
}
void DVHWindow::onSelectAll() {
  qDebug() << "=== onSelectAll ===";

  if (!m_roiList) {
    qWarning() << "onSelectAll: m_roiList is null";
    return;
  }

  QSignalBlocker blocker(m_roiList);

  for (int i = 0; i < m_roiList->count(); ++i) {
    QListWidgetItem *item = m_roiList->item(i);
    if (!item)
      continue;

    try {
      bool ok = false;
      int idx = item->data(Qt::UserRole).toInt(&ok);

      // チェック状態を設定
      item->setCheckState(Qt::Checked);

      if (ok && idx >= 0 && static_cast<size_t>(idx) < m_data.size()) {
        // 既存データがある場合は可視性を更新
        updateVisibility(idx, true);
      } else {
        // データがない場合は再計算をリクエスト
        emit recalculateRequested(item->text());
      }

    } catch (const std::exception &e) {
      qWarning() << QString("Exception in onSelectAll for item %1: %2")
                        .arg(item->text())
                        .arg(e.what());
    } catch (...) {
      qWarning() << QString("Unknown exception in onSelectAll for item %1")
                        .arg(item->text());
    }
  }

  qDebug() << "onSelectAll completed";
}

void DVHWindow::onSelectNone() {
  qDebug() << "=== onSelectNone ===";

  if (!m_roiList) {
    qWarning() << "onSelectNone: m_roiList is null";
    return;
  }

  QSignalBlocker blocker(m_roiList);

  for (int i = 0; i < m_roiList->count(); ++i) {
    QListWidgetItem *item = m_roiList->item(i);
    if (!item)
      continue;

    try {
      // チェック状態を解除
      item->setCheckState(Qt::Unchecked);

      bool ok = false;
      int idx = item->data(Qt::UserRole).toInt(&ok);

      if (ok && idx >= 0 && static_cast<size_t>(idx) < m_data.size()) {
        // 既存データがある場合は可視性を更新
        updateVisibility(idx, false);
      }

    } catch (const std::exception &e) {
      qWarning() << QString("Exception in onSelectNone for item %1: %2")
                        .arg(item->text())
                        .arg(e.what());
    } catch (...) {
      qWarning() << QString("Unknown exception in onSelectNone for item %1")
                        .arg(item->text());
    }
  }

  qDebug() << "onSelectNone completed";
}

bool DVHWindow::eventFilter(QObject *obj, QEvent *event) {
  if (obj == m_plot) {
    if (event->type() == QEvent::MouseButtonDblClick) {
      qDebug() << "DVHWindow: Double-click detected on plot";

      // DVHデータが設定されているかチェック
      if (m_data.empty()) {
        qDebug() << "DVHWindow: No DVH data available, ignoring double-click";
        return true; // イベントを消費して処理終了
      }

      // プロットの更新中かどうかをチェック
      if (m_plot->graphCount() == 0) {
        qDebug() << "DVHWindow: Plot is empty, ignoring double-click";
        return true; // イベントを消費して処理終了
      }

      // 安全にシグナルを発行
      qDebug() << "DVHWindow: Emitting doubleClicked signal";
      emit doubleClicked();

      return true; // イベントを消費してQCustomPlotの処理を防ぐ
    } else if (event->type() == QEvent::Leave) {
      if (m_cursorInfoLabel)
        m_cursorInfoLabel->setVisible(false);
      if (m_cursorVLine)
        m_cursorVLine->setVisible(false);
      if (m_cursorHLine)
        m_cursorHLine->setVisible(false);
      if (m_roiNameLabel)
        m_roiNameLabel->setVisible(false);
      m_plot->replot();
    } else if (event->type() == QEvent::MouseMove) {
      onPlotMouseMove(static_cast<QMouseEvent *>(event));
    } else if (event->type() == QEvent::Resize) {
      updateOverlayPosition();
    }
  }

  return QWidget::eventFilter(obj, event);
}

const std::vector<DVHCalculator::DVHData> &DVHWindow::dvhData() const {
  return m_data;
}

int DVHWindow::currentRoiIndex() const {
  if (!m_roiList)
    return -1;
  QListWidgetItem *item = m_roiList->currentItem();
  return item ? item->data(Qt::UserRole).toInt() : -1;
}

void DVHWindow::setCurrentRoiIndex(int index) {
  if (!m_roiList)
    return;
  for (int i = 0; i < m_roiList->count(); ++i) {
    QListWidgetItem *item = m_roiList->item(i);
    if (item->data(Qt::UserRole).toInt() == index) {
      m_roiList->setCurrentRow(i);
      updateRoiInfo(index);
      if (m_roiNameLabel && static_cast<size_t>(index) < m_data.size()) {
        m_roiNameLabel->setText(m_data[index].roiName);
        m_roiNameLabel->setVisible(m_data[index].isVisible);
        updateOverlayPosition();
      }
      break;
    }
  }
}

void DVHWindow::setPatientInfo(const QString &text) {
  if (!m_patientInfoLabel)
    return;
  m_patientInfoLabel->setText(text);
  m_patientInfoLabel->setVisible(!text.isEmpty());
  updateOverlayPosition();
}

void DVHWindow::setPrescriptionDose(double doseGy) {
  // 処方線量が変化した場合のみ再計算を行う
  if (doseGy <= 0.0 || qFuzzyCompare(doseGy, m_prescriptionDose))
    return;
  m_prescriptionDose = doseGy;
  updatePlotUnits();
}

void DVHWindow::setCalculationProgress(int processed, int total) {
  if (!m_progressBar)
    return;
  if (total <= 0) {
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);
    return;
  }
  m_progressBar->setRange(0, total);
  m_progressBar->setValue(processed);
}

void DVHWindow::setAxisUnits(bool xPercent, bool yCc) {
  QSignalBlocker b1(m_xPercentButton);
  QSignalBlocker b2(m_xGyButton);
  QSignalBlocker b3(m_yCcButton);
  QSignalBlocker b4(m_yPercentButton);
  if (m_xPercentButton)
    m_xPercentButton->setChecked(xPercent);
  if (m_xGyButton)
    m_xGyButton->setChecked(!xPercent);
  if (m_yCcButton)
    m_yCcButton->setChecked(yCc);
  if (m_yPercentButton)
    m_yPercentButton->setChecked(!yCc);
  updatePlotUnits();
}

void DVHWindow::setCalcMaxAuto() {
  m_calcMaxGy = 0.0;
  m_calcMaxUserSet = false;
  if (m_calcMaxSpin) {
    QSignalBlocker blk(m_calcMaxSpin);
    m_calcMaxSpin->setValue(0.0);
  }
  updatePlotUnits();
}

void DVHWindow::setCalcMaxGyNoRecalc(double gy) {
  if (gy <= 0.0) {
    setCalcMaxAuto();
    return;
  }
  m_calcMaxGy = gy;
  m_calcMaxUserSet = true;
  if (m_calcMaxSpin) {
    QSignalBlocker blk(m_calcMaxSpin);
    m_calcMaxSpin->setValue(gy);
  }
  updatePlotUnits();
}

DVHWindow::~DVHWindow() {
  qDebug() << "DVHWindow destructor called";
  cleanup();
}

void DVHWindow::closeEvent(QCloseEvent *event) {
  qDebug() << "DVHWindow closeEvent called";
  m_isClosing = true;
  cleanup();
  QWidget::closeEvent(event);
}

void DVHWindow::cleanup() {
  qDebug() << "DVHWindow cleanup started";

  try {
    // イベントフィルタを安全に削除
    if (m_plot) {
      m_plot->removeEventFilter(this);
    }

    // シグナル接続を全て切断
    disconnect();

    // データを安全にクリア
    m_data.clear();

    // プロットを安全にクリア
    if (m_plot) {
      m_plot->clearGraphs();
    }

    // ROIリストを安全にクリア
    if (m_roiList) {
      QSignalBlocker blocker(m_roiList);
      m_roiList->clear();
    }

    qDebug() << "DVHWindow cleanup completed successfully";

  } catch (const std::exception &e) {
    qCritical() << "Exception during DVHWindow cleanup:" << e.what();
  } catch (...) {
    qCritical() << "Unknown exception during DVHWindow cleanup";
  }
}
