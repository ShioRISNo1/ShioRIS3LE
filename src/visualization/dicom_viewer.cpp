#include "visualization/dicom_viewer.h"
#include "web/web_server.h"
#include "export/usdz_exporter.h"
#include "visualization/collapsible_group_box.h"
#include "visualization/dose_profile_window.h"
#include "visualization/gamma_analysis_window.h"
#include "visualization/opengl_3d_widget.h"
#include "visualization/opengl_image_widget.h"
#include "visualization/random_study_dialog.h"
#include "dicom/structure_surface.h"
#include "database/database_manager.h"
#include <QApplication>
#include <QAbstractItemView>
#include <QCoreApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QColor>
#include <QCursor>
#include <QDebug>
#include <QDateTime>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QFontMetrics>
#include <QFormLayout>
#include <QFuture>
#include <QFutureWatcher>
#include <QGridLayout>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QJsonValue>
#include <QDialogButtonBox>
#include <QLocale>
#include <QLineF>
#include <QLayoutItem>
#include <QMatrix4x4>
#include <QMenu>
#include <QSettings>
#include <QMessageBox>
#include <QPainter>
#include <QPen>
#include <QPair>
#include <QSizePolicy>
#include <QInputDialog>
#include <QBrush>
#include <QPalette>
#include <QMetaObject>
#include <QPixmap>
#include <QPointer>
#include <QProgressBar>
#include <QProgressDialog>
#include <QPushButton>
#include <QScrollBar>
#include <QSignalBlocker>
#include <QSize>
#include <QSizePolicy>
#include <QSplitter>
#include <QGroupBox>
#include <QLineEdit>
#include <QComboBox>

#include "theme_manager.h"
#include <QStackedWidget>
#include <QStandardPaths>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVariant>
#include <QUrl>
#include <QStringList>
#include <QStringView>
#include <QRegularExpression>
#include <array>
#include <cmath>
#include <initializer_list>
#include <QMap>
#include <QTextCursor>
#include <QThread>
#include <QTimer>
#include <QVector>
#include <QVector3D>
#include <QtConcurrent>
#include <QtGlobal>
#include <utility>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <dcmtk/dcmdata/dctk.h>
#include <functional>
#include <limits>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <random>
#if __has_include(<opencv2/wechat_qrcode.hpp>)
#include <opencv2/wechat_qrcode.hpp>
#define HAS_WECHAT_QR 1
#else
#define HAS_WECHAT_QR 0
#endif
#include "qrcodegen/qrcodegen.hpp"
#include <set>
#include <vector>
#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

class MultiRowTabWidget : public QWidget {
public:
  explicit MultiRowTabWidget(QWidget *parent = nullptr)
      : QWidget(parent), m_buttonGroup(new QButtonGroup(this)),
        m_stack(new QStackedWidget(this)), m_buttonLayout(new QGridLayout) {
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    m_buttonLayout->setContentsMargins(0, 0, 0, 0);
    m_buttonLayout->setHorizontalSpacing(6);
    m_buttonLayout->setVerticalSpacing(6);
    layout->addLayout(m_buttonLayout);
    layout->addWidget(m_stack);
    m_buttonGroup->setExclusive(true);
    connect(m_buttonGroup, &QButtonGroup::idToggled, this,
            [this](int id, bool checked) {
              if (!checked)
                return;
              m_stack->setCurrentIndex(id);
              for (int i = 0; i < m_buttons.size(); ++i) {
                if (i == id)
                  continue;
                QSignalBlocker blocker(m_buttons[i]);
                m_buttons[i]->setChecked(false);
              }
            });
  }

  int addTab(QWidget *widget, const QString &label) {
    if (!widget)
      return -1;
    const int index = m_stack->addWidget(widget);
    auto *button = new QToolButton(this);
    button->setText(label);
    button->setCheckable(true);
    button->setAutoRaise(true);
    button->setToolButtonStyle(Qt::ToolButtonTextOnly);
    button->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_buttonGroup->addButton(button, index);
    m_buttons.append(button);
    updateButtonLayout();
    if (index == 0) {
      QSignalBlocker blocker(m_buttonGroup);
      button->setChecked(true);
      m_stack->setCurrentIndex(0);
    }
    return index;
  }

  void setCurrentWidget(QWidget *widget) {
    if (!widget)
      return;
    const int index = m_stack->indexOf(widget);
    if (index < 0)
      return;
    m_stack->setCurrentIndex(index);
    QSignalBlocker blocker(m_buttonGroup);
    for (int i = 0; i < m_buttons.size(); ++i) {
      QSignalBlocker buttonBlocker(m_buttons[i]);
      m_buttons[i]->setChecked(i == index);
    }
  }

  QWidget *widget(int index) const { return m_stack->widget(index); }

  int count() const { return m_stack->count(); }

private:
  void updateButtonLayout() {
    while (QLayoutItem *item = m_buttonLayout->takeAt(0)) {
      delete item;
    }

    const int count = m_buttons.size();
    if (count == 0)
      return;

    const int rows = count > 2 ? 2 : 1;
    const int columns = rows == 0 ? 0 : (count + rows - 1) / rows;

    for (int i = 0; i < count; ++i) {
      QToolButton *button = m_buttons.at(i);
      const int row = rows == 1 ? 0 : i / columns;
      const int column = rows == 1 ? i : i % columns;
      button->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
      m_buttonLayout->addWidget(button, row, column);
    }

    if (columns > 0) {
      for (int c = 0; c < columns; ++c)
        m_buttonLayout->setColumnStretch(c, 1);
    }
    for (int r = 0; r <= rows; ++r)
      m_buttonLayout->setRowStretch(r, 0);
  }

  QButtonGroup *m_buttonGroup{nullptr};
  QStackedWidget *m_stack{nullptr};
  QGridLayout *m_buttonLayout{nullptr};
  QVector<QToolButton *> m_buttons;
};

namespace {
constexpr double DEFAULT_ZOOM = 1.5;
constexpr double DEFAULT_ZOOM_3D = 2.25;
constexpr double ZOOM_3D_RATIO = DEFAULT_ZOOM_3D / DEFAULT_ZOOM;

QRgb labelColor(int label) {
  static const QVector<QRgb> colors = {
      qRgba(255, 0, 0, 100),   qRgba(0, 255, 0, 100),
      qRgba(0, 0, 255, 100),   qRgba(255, 255, 0, 100),
      qRgba(255, 0, 255, 100), qRgba(0, 255, 255, 100)};
  return colors[label % colors.size()];
}
} // namespace

namespace {
// Try multiple preprocessing variants to improve QR detection robustness
static std::vector<std::string>
decodeQrRobust(const cv::Mat &bgr,
               std::vector<std::vector<cv::Point>> *outCorners = nullptr) {
  auto appendCornersFromMat =
      [](const cv::Mat &m, std::vector<std::vector<cv::Point>> *cornersOut) {
        if (!cornersOut)
          return;
        try {
          if (m.total() >= 4) {
            std::vector<cv::Point> poly;
            for (int i = 0; i < m.total(); ++i) {
              cv::Point2f p = m.at<cv::Point2f>(i);
              poly.emplace_back(cv::Point(static_cast<int>(std::lround(p.x)),
                                          static_cast<int>(std::lround(p.y))));
            }
            cornersOut->push_back(std::move(poly));
          }
        } catch (...) {
        }
      };

  auto runVariant = [&](const cv::Mat &img, std::vector<std::string> &acc,
                        std::vector<std::vector<cv::Point>> *cornersOut) {
    try {
      cv::QRCodeDetector qrd;
      // Slightly relax approximation if supported (OpenCV >= 4.5)
      // try/catch to be safe across versions
      try {
        qrd.setEpsX(0.2);
      } catch (...) {
      }
      try {
        qrd.setEpsY(0.2);
      } catch (...) {
      }
      // First try single-code path (often more robust than multi on clean
      // codes)
      try {
        cv::Mat pts;
        std::string single = qrd.detectAndDecode(img, pts);
        if (!single.empty()) {
          acc.push_back(single);
          appendCornersFromMat(pts, cornersOut);
        }
      } catch (...) {
      }
      // Try detect + decode path
      try {
        std::vector<cv::Point> quad;
        bool found = qrd.detect(img, quad);
        if (found && quad.size() >= 4) {
          std::string v = qrd.decode(img, quad);
          if (!v.empty()) {
            acc.push_back(v);
            if (cornersOut)
              cornersOut->push_back(quad);
          }
        }
      } catch (...) {
      }
      std::vector<std::string> decoded;
      std::vector<std::vector<cv::Point>> corners;
      bool ok = qrd.detectAndDecodeMulti(img, decoded, corners);
      if (ok && !decoded.empty()) {
        acc.insert(acc.end(), decoded.begin(), decoded.end());
        if (cornersOut)
          cornersOut->insert(cornersOut->end(), corners.begin(), corners.end());
      }
    } catch (...) {
      // ignore
    }
  };

  std::vector<std::string> results;
  std::vector<std::vector<cv::Point>> gatheredCorners;

  // 1) Original
  runVariant(bgr, results, &gatheredCorners);

  // 2) Grayscale
  cv::Mat gray;
  try {
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  } catch (...) {
  }
  if (!gray.empty()) {
    runVariant(gray, results, &gatheredCorners);
  }

  // 3) CLAHE (contrast enhance) then try
  if (!gray.empty()) {
    try {
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
      cv::Mat eq;
      clahe->apply(gray, eq);
      runVariant(eq, results, &gatheredCorners);
    } catch (...) {
    }
  }

  // 4) Adaptive thresholded binary
  if (!gray.empty()) {
    try {
      cv::Mat bw;
      cv::adaptiveThreshold(gray, bw, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv::THRESH_BINARY, 31, 2);
      runVariant(bw, results, &gatheredCorners);
    } catch (...) {
    }
  }

  // 4b) Inverted variants
  if (!gray.empty()) {
    try {
      cv::Mat invGray;
      cv::bitwise_not(gray, invGray);
      runVariant(invGray, results, &gatheredCorners);
    } catch (...) {
    }
  }
  if (!gray.empty()) {
    try {
      cv::Mat bw2;
      cv::adaptiveThreshold(gray, bw2, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                            cv::THRESH_BINARY, 25, 5);
      cv::Mat invBw;
      cv::bitwise_not(bw2, invBw);
      runVariant(invBw, results, &gatheredCorners);
    } catch (...) {
    }
  }

  // 4c) Morphological closing to connect modules
  if (!gray.empty()) {
    try {
      cv::Mat bw;
      cv::threshold(gray, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
      cv::Mat closed;
      cv::morphologyEx(
          bw, closed, cv::MORPH_CLOSE,
          cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
      runVariant(closed, results, &gatheredCorners);
    } catch (...) {
    }
  }

  // 5) Upscale for small QR codes
  cv::Mat big;
  try {
    double maxDim = std::max(bgr.cols, bgr.rows);
    std::vector<double> scales;
    if (maxDim < 1000)
      scales = {1.5, 2.0, 3.0};
    else
      scales = {1.25, 1.5};
    for (double s : scales) {
      cv::resize(bgr, big, cv::Size(), s, s, cv::INTER_CUBIC);
      runVariant(big, results, &gatheredCorners);
    }
  } catch (...) {
  }

  // 6) Rotations (in case camera orientation confuses detection)
  auto tryRotate = [&](const cv::Mat &src) {
    try {
      cv::Mat rot;
      cv::rotate(src, rot, cv::ROTATE_90_CLOCKWISE);
      runVariant(rot, results, &gatheredCorners);
      cv::rotate(src, rot, cv::ROTATE_180);
      runVariant(rot, results, &gatheredCorners);
      cv::rotate(src, rot, cv::ROTATE_90_COUNTERCLOCKWISE);
      runVariant(rot, results, &gatheredCorners);
    } catch (...) {
    }
  };
  tryRotate(bgr);
  if (!gray.empty())
    tryRotate(gray);

  // Deduplicate
  std::set<std::string> uniq(results.begin(), results.end());
  std::vector<std::string> out(uniq.begin(), uniq.end());
  if (outCorners && !gatheredCorners.empty()) {
    *outCorners = gatheredCorners;
  }

#if HAS_WECHAT_QR
  // Fallback to WeChat QR if nothing found via native detector
  if (out.empty()) {
    auto findModelDir = []() -> QString {
      QString env = QString::fromUtf8(qgetenv("WECHAT_QR_MODEL_DIR"));
      if (!env.isEmpty())
        return env;
      QString appDir = QCoreApplication::applicationDirPath();
      QStringList cands = {appDir + "/models/wechat_qrcode",
                           QDir::currentPath() + "/models/wechat_qrcode"};
      for (const QString &p : cands) {
        if (QDir(p).exists())
          return p;
      }
      return QString();
    };

    QString dir = findModelDir();
    if (!dir.isEmpty()) {
      QString dp = QDir(dir).filePath("detect.prototxt");
      QString dm = QDir(dir).filePath("detect.caffemodel");
      QString sp = QDir(dir).filePath("sr.prototxt");
      QString sm = QDir(dir).filePath("sr.caffemodel");
      if (QFileInfo::exists(dp) && QFileInfo::exists(dm) &&
          QFileInfo::exists(sp) && QFileInfo::exists(sm)) {
        try {
          cv::wechat_qrcode::WeChatQRCode we(dp.toStdString(), dm.toStdString(),
                                             sp.toStdString(),
                                             sm.toStdString());
          std::vector<cv::Mat> pts;
          std::vector<std::string> res = we.detectAndDecode(bgr, pts);
          if (!res.empty()) {
            // Merge
            for (const auto &s : res)
              uniq.insert(s);
            out.assign(uniq.begin(), uniq.end());
            if (outCorners) {
              for (const auto &m : pts) {
                if (m.total() >= 4) {
                  std::vector<cv::Point> poly;
                  for (int i = 0; i < m.total(); ++i) {
                    cv::Point2f p = m.at<cv::Point2f>(i);
                    poly.emplace_back(
                        cv::Point(static_cast<int>(std::lround(p.x)),
                                  static_cast<int>(std::lround(p.y))));
                  }
                  outCorners->push_back(std::move(poly));
                }
              }
            }
          }
        } catch (...) {
          // ignore
        }
      }
    }
  }
#endif
  return out;
}
} // namespace

namespace {
// Fast path: grayscale + detectAndDecode (+ fallback detect/decode), optionally
// on downscaled image
static std::vector<std::string>
decodeQrFast(const cv::Mat &bgr,
             std::vector<std::vector<cv::Point>> *outCorners = nullptr) {
  auto appendCornersFromMat =
      [](const cv::Mat &m, std::vector<std::vector<cv::Point>> *cornersOut) {
        if (!cornersOut)
          return;
        try {
          if (m.total() >= 4) {
            std::vector<cv::Point> poly;
            for (int i = 0; i < m.total(); ++i) {
              cv::Point2f p = m.at<cv::Point2f>(i);
              poly.emplace_back(cv::Point(static_cast<int>(std::lround(p.x)),
                                          static_cast<int>(std::lround(p.y))));
            }
            cornersOut->push_back(std::move(poly));
          }
        } catch (...) {
        }
      };
  std::vector<std::string> acc;
  try {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    // downscale moderately for speed
    cv::Mat small;
    double scale = 0.75;
    cv::resize(gray, small, cv::Size(), scale, scale, cv::INTER_AREA);
    cv::QRCodeDetector qrd;
    // direct
    cv::Mat pts;
    std::string s = qrd.detectAndDecode(small, pts);
    if (!s.empty()) {
      acc.push_back(s);
      // rescale corners back
      if (outCorners && pts.total() >= 4) {
        std::vector<cv::Point> poly;
        for (int i = 0; i < pts.total(); ++i) {
          cv::Point2f p = pts.at<cv::Point2f>(i);
          poly.emplace_back(
              cv::Point(static_cast<int>(std::lround(p.x / scale)),
                        static_cast<int>(std::lround(p.y / scale))));
        }
        outCorners->push_back(std::move(poly));
      }
      return acc;
    }
    // detect + decode
    std::vector<cv::Point> quad;
    if (qrd.detect(small, quad) && quad.size() >= 4) {
      std::string v = qrd.decode(small, quad);
      if (!v.empty()) {
        acc.push_back(v);
        if (outCorners) {
          std::vector<cv::Point> poly;
          for (const auto &pt : quad) {
            poly.emplace_back(
                cv::Point(static_cast<int>(std::lround(pt.x / scale)),
                          static_cast<int>(std::lround(pt.y / scale))));
          }
          outCorners->push_back(std::move(poly));
        }
      }
    }
  } catch (...) {
  }
  return acc;
}
} // namespace

namespace {
double sampleVolumeTrilinear(const cv::Mat &volume, double x, double y,
                             double z) {
  if (volume.empty() || volume.dims != 3)
    return 0.0;

  const int depth = volume.size[0];
  const int height = volume.size[1];
  const int width = volume.size[2];

  if (depth <= 0 || height <= 0 || width <= 0)
    return 0.0;

  x = std::clamp(x, 0.0, static_cast<double>(width - 1));
  y = std::clamp(y, 0.0, static_cast<double>(height - 1));
  z = std::clamp(z, 0.0, static_cast<double>(depth - 1));

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int z0 = static_cast<int>(std::floor(z));
  const int x1 = std::min(x0 + 1, width - 1);
  const int y1 = std::min(y0 + 1, height - 1);
  const int z1 = std::min(z0 + 1, depth - 1);

  const double fx = x - x0;
  const double fy = y - y0;
  const double fz = z - z0;

  auto fetch = [&](int zi, int yi, int xi) -> double {
    zi = std::clamp(zi, 0, depth - 1);
    yi = std::clamp(yi, 0, height - 1);
    xi = std::clamp(xi, 0, width - 1);
    switch (volume.type()) {
    case CV_16SC1: {
      const short *plane = volume.ptr<short>(zi);
      return static_cast<double>(plane[yi * width + xi]);
    }
    case CV_32FC1: {
      const float *plane = volume.ptr<float>(zi);
      return static_cast<double>(plane[yi * width + xi]);
    }
    case CV_8UC1: {
      const uchar *plane = volume.ptr<uchar>(zi);
      return static_cast<double>(plane[yi * width + xi]);
    }
    default:
      return 0.0;
    }
  };

  const double c000 = fetch(z0, y0, x0);
  const double c100 = fetch(z0, y0, x1);
  const double c010 = fetch(z0, y1, x0);
  const double c110 = fetch(z0, y1, x1);
  const double c001 = fetch(z1, y0, x0);
  const double c101 = fetch(z1, y0, x1);
  const double c011 = fetch(z1, y1, x0);
  const double c111 = fetch(z1, y1, x1);

  const double c00 = c000 * (1.0 - fx) + c100 * fx;
  const double c10 = c010 * (1.0 - fx) + c110 * fx;
  const double c01 = c001 * (1.0 - fx) + c101 * fx;
  const double c11 = c011 * (1.0 - fx) + c111 * fx;
  const double c0 = c00 * (1.0 - fy) + c10 * fy;
  const double c1 = c01 * (1.0 - fy) + c11 * fy;
  return c0 * (1.0 - fz) + c1 * fz;
}

cv::Mat resampleVolumeToReference(
    const DicomVolume &reference, const DicomVolume &source,
    const std::function<void(int, int)> &progressCallback = {}) {
  if (reference.width() <= 0 || reference.height() <= 0 ||
      reference.depth() <= 0)
    return cv::Mat();

  const cv::Mat &srcData = source.data();
  if (srcData.empty() || srcData.dims != 3)
    return cv::Mat();

  const int depth = reference.depth();
  const int height = reference.height();
  const int width = reference.width();

  int sizes[3] = {depth, height, width};
  cv::Mat result(3, sizes, CV_16SC1);

  std::atomic<int> completed{0};
  if (progressCallback)
    progressCallback(0, depth);

  auto processSlice = [&](int z) {
    short *dstPlane = result.ptr<short>(z);
    for (int y = 0; y < height; ++y) {
      short *dstRow = dstPlane + y * width;
      for (int x = 0; x < width; ++x) {
        QVector3D patient = reference.voxelToPatient(x, y, z);
        QVector3D srcVoxel = source.patientToVoxelContinuous(patient);
        const double sampled =
            sampleVolumeTrilinear(srcData, srcVoxel.x(), srcVoxel.y(),
                                  srcVoxel.z());
        dstRow[x] = static_cast<short>(std::lround(sampled));
      }
    }
    if (progressCallback) {
      int done = ++completed;
      progressCallback(done, depth);
    }
  };

#if QT_CONFIG(concurrent)
  if (depth > 1) {
    QVector<int> zIndices(depth);
    std::iota(zIndices.begin(), zIndices.end(), 0);
    QFuture<void> future = QtConcurrent::map(zIndices, [&](int &zIndex) {
      processSlice(zIndex);
    });
    while (!future.isFinished()) {
      if (progressCallback)
        QCoreApplication::processEvents();
      QThread::yieldCurrentThread();
    }
  } else {
    processSlice(0);
  }
#else
  for (int z = 0; z < depth; ++z) {
    processSlice(z);
    if (progressCallback)
      QCoreApplication::processEvents();
  }
#endif

  return result;
}
} // namespace

DoseColorBar::DoseColorBar(QWidget *parent) : QWidget(parent) {
  setFixedWidth(80); // カラーバーの幅を固定
  // 多数のビュー表示でも収まるよう最小高さを縮小
  setMinimumHeight(50);
  setAttribute(Qt::WA_OpaquePaintEvent, false);
  setStyleSheet("background: transparent;");
}

void DoseColorBar::setDoseRange(double minDose, double maxDose) {
  if (m_minDose != minDose || m_maxDose != maxDose) {
    m_minDose = minDose;
    m_maxDose = maxDose;
    update();
  }
}

void DoseColorBar::setDisplayMode(DoseResampledVolume::DoseDisplayMode mode) {
  if (m_displayMode != mode) {
    m_displayMode = mode;
    update();
  }
}

void DoseColorBar::setReferenceDose(double referenceDose) {
  if (m_referenceDose != referenceDose) {
    m_referenceDose = referenceDose;
    update();
  }
}

void DoseColorBar::setVisible(bool visible) {
  if (m_isVisible != visible) {
    m_isVisible = visible;
    QWidget::setVisible(visible);
    update();
  }
}

int DoseColorBar::preferredWidth() const { return sizeHint().width(); }

QSize DoseColorBar::sizeHint() const { return QSize(80, 200); }

// カラーマッピング関数（dose_resampled_volume.cppと同じロジック）
QRgb DoseColorBar::mapDoseToColor(float doseRatio) const {
  // マイナス線量の場合は寒色系（紫～青）で表示
  if (doseRatio < 0.0f) {
    float negRatio = std::max(-1.0f, doseRatio);
    float absRatio = std::abs(negRatio);

    float hue = 270.0f - absRatio * 30.0f;
    float saturation = 0.7f + absRatio * 0.3f;
    float value = 0.5f + absRatio * 0.5f;

    QColor color = QColor::fromHsvF(hue / 360.0f, saturation, value);
    return qRgba(color.red(), color.green(), color.blue(), 255);
  }

  if (doseRatio == 0.0f) {
    return qRgba(0, 0, 0, 0);
  }

  float hue = 0.0f;
  float saturation = 1.0f;
  float value = 1.0f;

  if (doseRatio <= 0.2f) {
    hue = 240.0f - (doseRatio / 0.2f) * 60.0f;
    saturation = 0.8f + (doseRatio / 0.2f) * 0.2f;
  } else if (doseRatio <= 0.4f) {
    float t = (doseRatio - 0.2f) / 0.2f;
    hue = 180.0f - t * 60.0f;
    saturation = 1.0f;
  } else if (doseRatio <= 0.6f) {
    float t = (doseRatio - 0.4f) / 0.2f;
    hue = 120.0f - t * 60.0f;
    saturation = 1.0f;
  } else if (doseRatio <= 0.8f) {
    float t = (doseRatio - 0.6f) / 0.2f;
    hue = 60.0f - t * 30.0f;
    saturation = 1.0f;
    value = 1.0f;
  } else if (doseRatio <= 1.0f) {
    float t = (doseRatio - 0.8f) / 0.2f;
    hue = 30.0f - t * 30.0f;
    saturation = 1.0f;
    value = 1.0f;
  } else {
    float t = std::min(1.0f, (doseRatio - 1.0f) / 0.5f);
    hue = 360.0f - t * 60.0f;
    saturation = 1.0f;
    value = 1.0f - t * 0.2f;
  }

  QColor color = QColor::fromHsvF(hue / 360.0f, saturation, value);
  return qRgba(color.red(), color.green(), color.blue(), 255);
}

QRgb DoseColorBar::mapDoseToIsodose(float doseRatio) const {
  // マイナス線量の場合は暗い青で表示
  if (doseRatio < 0.0f) {
    float absRatio = std::min(1.0f, std::abs(doseRatio));
    int blue = static_cast<int>(100 + absRatio * 155);
    return qRgba(50, 50, blue, 255);
  }

  if (doseRatio == 0.0f) {
    return qRgba(0, 0, 0, 0);
  }

  static const float isodoseLevels[] = {0.95f, 0.90f, 0.80f, 0.70f,
                                        0.50f, 0.30f, 0.10f};

  static const QRgb isodoseColors[] = {
      qRgba(255, 0, 0, 255),   // 赤
      qRgba(255, 128, 0, 255), // オレンジ
      qRgba(255, 255, 0, 255), // 黄
      qRgba(0, 255, 0, 255),   // 緑
      qRgba(0, 255, 255, 255), // シアン
      qRgba(0, 0, 255, 255),   // 青
      qRgba(128, 0, 255, 255)  // 紫
  };

  for (int i = 0; i < 7; ++i) {
    if (doseRatio >= isodoseLevels[i]) {
      return isodoseColors[i];
    }
  }

  return qRgba(0, 0, 0, 0);
}

QRgb DoseColorBar::mapDoseToHot(float doseRatio) const {
  // マイナス線量の場合は暗い青から黒へ
  if (doseRatio < 0.0f) {
    float absRatio = std::min(1.0f, std::abs(doseRatio));
    int blue = static_cast<int>(absRatio * 200);
    return qRgba(0, 0, blue, 255);
  }

  doseRatio = std::clamp(doseRatio, 0.0f, 1.0f);
  float r = std::min(1.0f, doseRatio * 3.0f);
  float g = std::clamp((doseRatio - 0.33f) * 3.0f, 0.0f, 1.0f);
  float b = std::clamp((doseRatio - 0.66f) * 3.0f, 0.0f, 1.0f);
  return qRgba(static_cast<int>(r * 255), static_cast<int>(g * 255),
               static_cast<int>(b * 255), 255);
}

void DoseColorBar::paintEvent(QPaintEvent *event) {
  // マイナス値も許容するため、minDose < maxDoseの条件のみチェック
  if (!m_isVisible || m_minDose >= m_maxDose) {
    return;
  }

  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing);

  // 描画領域の設定
  QRect colorBarRect =
      rect().adjusted(10, 20, -35, -20); // テキスト用のマージンを確保

  // 背景を半透明の黒で描画
  painter.fillRect(rect(), QColor(0, 0, 0, 120));

  // フォント設定
  QFont font = painter.font();
  font.setPointSize(8);
  painter.setFont(font);
  const QColor textColor = ThemeManager::instance().textColor();
  painter.setPen(textColor);

  // タイトル描画
  painter.drawText(rect().adjusted(2, 2, -2, 0),
                   Qt::AlignTop | Qt::AlignHCenter, QString("Dose (Gy)"));

  if (m_displayMode == DoseResampledVolume::DoseDisplayMode::Colorful) {
    // カラフルモード：グラデーションバーを描画
    int barHeight = colorBarRect.height();
    int barWidth = 20;

    for (int y = 0; y < barHeight; ++y) {
      // 各ピクセル位置の実際の線量値を計算（上が最大値）
      float pixelRatio = 1.0f - (float)y / barHeight;
      double dose = m_minDose + pixelRatio * (m_maxDose - m_minDose);

      // 線量値を正規化してカラーマッピング用のratioを計算
      float ratio = 0.0f;
      if (dose < 0.0) {
        // マイナス線量の場合はminDoseを基準に正規化
        ratio = static_cast<float>(dose / std::abs(m_minDose));
      } else {
        // 正の線量の場合は0～maxDoseの範囲で正規化
        ratio = static_cast<float>((dose - std::max(0.0, m_minDose)) /
                                    (m_maxDose - std::max(0.0, m_minDose)));
      }

      QRgb color = mapDoseToColor(ratio);

      painter.setPen(QPen(QColor(color), 1));
      painter.drawLine(colorBarRect.left(), colorBarRect.top() + y,
                       colorBarRect.left() + barWidth, colorBarRect.top() + y);
    }

    // 線量値のラベルを描画
    QFontMetrics fm(font);
    int numLabels = 6;
    for (int i = 0; i <= numLabels; ++i) {
      float ratio = (float)i / numLabels;
      double dose = m_minDose + ratio * (m_maxDose - m_minDose);
      int y = colorBarRect.bottom() - (int)(ratio * barHeight);

      // ラベルテキスト（小数点以下4桁まで表示）
      QString label = QString::number(dose, 'f', 4);
      int textX = colorBarRect.left() + barWidth + 5;

      painter.setPen(textColor);
      painter.drawText(textX, y + fm.height() / 4, label);

      // 目盛り線
      painter.setPen(Qt::lightGray);
      painter.drawLine(colorBarRect.left() + barWidth, y,
                       colorBarRect.left() + barWidth + 3, y);
    }

  } else if (m_displayMode == DoseResampledVolume::DoseDisplayMode::Isodose) {
    // 等線量モード：レベル別の色を描画
    static const float isodoseLevels[] = {0.95f, 0.90f, 0.80f, 0.70f,
                                          0.50f, 0.30f, 0.10f};

    static const QString isodoseLabels[] = {"95%", "90%", "80%", "70%",
                                            "50%", "30%", "10%"};

    int numLevels = 7;
    int levelHeight = colorBarRect.height() / numLevels;
    int barWidth = 20;

    for (int i = 0; i < numLevels; ++i) {
      float ratio = isodoseLevels[i];
      QRgb color = mapDoseToIsodose(ratio);

      QRect levelRect(colorBarRect.left(), colorBarRect.top() + i * levelHeight,
                      barWidth, levelHeight);

      painter.fillRect(levelRect, QColor(color));

      // 境界線
      painter.setPen(textColor);
      painter.drawRect(levelRect);

      // ラベル
      double dose = ratio * m_referenceDose;
      QString label = QString("%1\n(%2Gy)")
                          .arg(isodoseLabels[i])
                          .arg(QString::number(dose, 'f', 1));

      int textX = colorBarRect.left() + barWidth + 5;
      int textY = levelRect.center().y();

      painter.setPen(textColor);
      painter.drawText(textX, textY - 5, label);
    }
  } else if (m_displayMode ==
             DoseResampledVolume::DoseDisplayMode::IsodoseLines) {
    static const float isodoseLevels[] = {0.95f, 0.90f, 0.80f, 0.70f,
                                          0.50f, 0.30f, 0.10f};
    static const QString isodoseLabels[] = {"95%", "90%", "80%", "70%",
                                            "50%", "30%", "10%"};

    int numLevels = 7;
    int levelHeight = colorBarRect.height() / numLevels;
    int barWidth = 20;

    painter.setBrush(Qt::NoBrush);

    for (int i = 0; i < numLevels; ++i) {
      float ratio = isodoseLevels[i];
      QRgb color = mapDoseToIsodose(ratio);

      int yCenter = colorBarRect.top() + i * levelHeight + levelHeight / 2;

      QPen pen{QColor::fromRgb(color)};
      pen.setWidth(2);
      painter.setPen(pen);
      painter.drawLine(colorBarRect.left(), yCenter,
                       colorBarRect.left() + barWidth, yCenter);

      QString label =
          QString("%1\n(%2Gy)")
              .arg(isodoseLabels[i])
              .arg(QString::number(ratio * m_referenceDose, 'f', 1));

      QRect textRect(colorBarRect.left() + barWidth + 5,
                     colorBarRect.top() + i * levelHeight,
                     colorBarRect.width() - barWidth - 5, levelHeight);

      painter.setPen(textColor);
      painter.drawText(textRect, Qt::AlignVCenter | Qt::AlignLeft, label);
    }
  } else if (m_displayMode == DoseResampledVolume::DoseDisplayMode::Hot) {
    // Hotモード：黒→赤→黄→白のグラデーション
    int barHeight = colorBarRect.height();
    int barWidth = 20;
    for (int y = 0; y < barHeight; ++y) {
      float ratio = 1.0f - (float)y / barHeight;
      QRgb color = mapDoseToHot(ratio);
      painter.setPen(QPen(QColor(color), 1));
      painter.drawLine(colorBarRect.left(), colorBarRect.top() + y,
                       colorBarRect.left() + barWidth, colorBarRect.top() + y);
    }
    QFontMetrics fm(font);
    int numLabels = 6;
    for (int i = 0; i <= numLabels; ++i) {
      float ratio = (float)i / numLabels;
      double dose = m_minDose + ratio * (m_maxDose - m_minDose);
      int y = colorBarRect.bottom() - (int)(ratio * barHeight);
      QString label = QString::number(dose, 'f', 1);
      int textX = colorBarRect.left() + barWidth + 5;
      painter.setPen(textColor);
      painter.drawText(textX, y + fm.height() / 4, label);
      painter.setPen(Qt::lightGray);
      painter.drawLine(colorBarRect.left() + barWidth, y,
                       colorBarRect.left() + barWidth + 3, y);
    }
  }

  // 外枠を描画
  painter.setPen(QPen(textColor, 1));
  painter.drawRect(colorBarRect.adjusted(0, 0, 20, 0));
}

// RT Dose項目用ウィジェット
class DoseItemWidget : public CollapsibleGroupBox {
  Q_OBJECT
public:
  explicit DoseItemWidget(const QString &filename, double maxDose,
                          QWidget *parent = nullptr)
      : CollapsibleGroupBox(filename, parent), m_filename(filename), m_isSaved(false) {
    m_check = new QCheckBox();
    m_check->setChecked(true);
    addHeaderWidget(m_check, true);

    // Add Save button
    m_saveButton = new QPushButton(tr("Save"));
    m_saveButton->setMaximumWidth(60);
    m_saveButton->setToolTip(tr("Save dose distribution to DICOM RT-Dose file"));
    connect(m_saveButton, &QPushButton::clicked, this, &DoseItemWidget::saveRequested);
    addHeaderWidget(m_saveButton, false);

    auto *main = new QVBoxLayout();
    main->setSizeConstraint(QLayout::SetMinAndMaxSize);

    ThemeManager &theme = ThemeManager::instance();
    QObject::connect(&theme, &ThemeManager::textColorChanged, this,
                     [this](const QColor &color) {
                       bool isDefault = qFuzzyIsNull(m_dataFr->value() - 1.0) &&
                                        qFuzzyIsNull(m_displayFr->value() - 1.0) &&
                                        qFuzzyIsNull(m_factor->value() - 1.0) &&
                                        qFuzzyIsNull(m_shiftX->value()) &&
                                        qFuzzyIsNull(m_shiftY->value()) &&
                                        qFuzzyIsNull(m_shiftZ->value());
                       if (isDefault)
                         setTitleColor(color);
                     });

    m_maxLabel = new QLabel(QString("MaxDose: %1 Gy").arg(maxDose, 0, 'f', 2));
    theme.applyTextColor(m_maxLabel);
    main->addWidget(m_maxLabel);

    QGroupBox *box = new QGroupBox(tr("Settings"));
    theme.applyTextColor(box, QStringLiteral(
                                 "QGroupBox { color: %1; } QGroupBox::title { "
                                 "color: %1; }"));
    box->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
    QFormLayout *form = new QFormLayout(box);
    form->setLabelAlignment(Qt::AlignLeft);
    m_dataFr = new QDoubleSpinBox();
    m_dataFr->setDecimals(1);
    m_dataFr->setRange(0.1, 1000.0);
    m_dataFr->setValue(1.0);
    m_dataFr->setSingleStep(1);
    m_dataFr->setMaximumWidth(90);
    form->addRow(tr("Data Fr"), m_dataFr);
    m_displayFr = new QDoubleSpinBox();
    m_displayFr->setDecimals(1);
    m_displayFr->setRange(0.1, 1000.0);
    m_displayFr->setValue(1.0);
    m_displayFr->setSingleStep(1);
    m_displayFr->setMaximumWidth(90);
    form->addRow(tr("Display Fr"), m_displayFr);
    m_factor = new QDoubleSpinBox();
    m_factor->setDecimals(4);
    m_factor->setRange(-1000.0, 1000.0);
    m_factor->setValue(1.0);
    m_factor->setSingleStep(0.1);
    m_factor->setMaximumWidth(90);
    form->addRow(tr("Factor"), m_factor);
    main->addWidget(box);

    // Collapsible Shift section
    auto *shiftBox = new CollapsibleGroupBox(tr("Dose Shift (mm)"));
    shiftBox->setTitleColor(theme.textColor());
    QObject::connect(&theme, &ThemeManager::textColorChanged, shiftBox,
                     [shiftBox](const QColor &color) { shiftBox->setTitleColor(color); });
    shiftBox->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
    auto *shiftLayout = new QFormLayout();
    m_shiftX = new QDoubleSpinBox();
    m_shiftY = new QDoubleSpinBox();
    m_shiftZ = new QDoubleSpinBox();
    for (auto *sb : {m_shiftX, m_shiftY, m_shiftZ}) {
      sb->setDecimals(2);
      sb->setRange(-1000.0, 1000.0);
      sb->setValue(0.0);
      sb->setSingleStep(1.0);
      sb->setMaximumWidth(90);
    }
    QLabel *xLabel = new QLabel(tr("X"));
    theme.applyTextColor(xLabel);
    shiftLayout->addRow(xLabel, m_shiftX);
    QLabel *yLabel = new QLabel(tr("Y"));
    theme.applyTextColor(yLabel);
    shiftLayout->addRow(yLabel, m_shiftY);
    QLabel *zLabel = new QLabel(tr("Z"));
    theme.applyTextColor(zLabel);
    shiftLayout->addRow(zLabel, m_shiftZ);
    shiftBox->setContentLayout(shiftLayout);
    shiftBox->setCollapsed(true); // hidden by default as a special feature
    main->addWidget(shiftBox);

    setContentLayout(main);

    // Notify parent to resize list item when expanded/collapsed
    auto notify = [this]() {
      if (layout())
        layout()->invalidate();
      updateGeometry();
      emit uiExpandedChanged();
    };
    connect(this, &CollapsibleGroupBox::toggled, this, notify);
    connect(shiftBox, &CollapsibleGroupBox::toggled, this, notify);

    connect(m_check, &QCheckBox::toggled, this,
            &DoseItemWidget::visibilityChanged);
    auto emitChanged = [this]() {
      emit settingsChanged();
      updateTitleColor();
    };
    connect(m_dataFr, qOverload<double>(&QDoubleSpinBox::valueChanged),
            emitChanged);
    connect(m_displayFr, qOverload<double>(&QDoubleSpinBox::valueChanged),
            emitChanged);
    connect(m_factor, qOverload<double>(&QDoubleSpinBox::valueChanged),
            emitChanged);
    connect(m_shiftX, qOverload<double>(&QDoubleSpinBox::valueChanged),
            emitChanged);
    connect(m_shiftY, qOverload<double>(&QDoubleSpinBox::valueChanged),
            emitChanged);
    connect(m_shiftZ, qOverload<double>(&QDoubleSpinBox::valueChanged),
            emitChanged);

    updateTitleColor();
  }

  bool isChecked() const { return m_check->isChecked(); }
  double dataFractions() const { return m_dataFr->value(); }
  double displayFractions() const { return m_displayFr->value(); }
  double factor() const { return m_factor->value(); }
  double shiftX() const { return m_shiftX->value(); }
  double shiftY() const { return m_shiftY->value(); }
  double shiftZ() const { return m_shiftZ->value(); }
  QVector3D shift() const { return QVector3D(shiftX(), shiftY(), shiftZ()); }

  QString name() const { return m_filename; }
  void setChecked(bool checked) {
    QSignalBlocker b(m_check);
    m_check->setChecked(checked);
  }
  void setDataFractions(double value) {
    QSignalBlocker b(m_dataFr);
    m_dataFr->setValue(value);
    updateTitleColor();
  }
  void setDisplayFractions(double value) {
    QSignalBlocker b(m_displayFr);
    m_displayFr->setValue(value);
    updateTitleColor();
  }
  void setFactor(double value) {
    QSignalBlocker b(m_factor);
    m_factor->setValue(value);
    updateTitleColor();
  }
  void setShift(const QVector3D &s) {
    QSignalBlocker bx(m_shiftX), by(m_shiftY), bz(m_shiftZ);
    m_shiftX->setValue(s.x());
    m_shiftY->setValue(s.y());
    m_shiftZ->setValue(s.z());
    updateTitleColor();
  }

  void setSaved(bool saved) {
    m_isSaved = saved;
    updateSaveButtonState();
  }

  bool isSaved() const { return m_isSaved; }

signals:
  void visibilityChanged(bool visible);
  void settingsChanged();
  void uiExpandedChanged();
  void saveRequested();

private:
  QString m_filename;
  QCheckBox *m_check{nullptr};
  QPushButton *m_saveButton{nullptr};
  QLabel *m_maxLabel{nullptr};
  QDoubleSpinBox *m_dataFr{nullptr};
  QDoubleSpinBox *m_displayFr{nullptr};
  QDoubleSpinBox *m_factor{nullptr};
  QDoubleSpinBox *m_shiftX{nullptr};
  QDoubleSpinBox *m_shiftY{nullptr};
  QDoubleSpinBox *m_shiftZ{nullptr};
  bool m_isSaved{false};

  void updateSaveButtonState() {
    if (m_saveButton) {
      if (m_isSaved) {
        m_saveButton->setText(tr("Saved"));
        m_saveButton->setEnabled(false);
        m_saveButton->setStyleSheet("QPushButton { color: gray; }");
      } else {
        m_saveButton->setText(tr("Save"));
        m_saveButton->setEnabled(true);
        m_saveButton->setStyleSheet("");
      }
    }
  }

  void updateTitleColor() {
    bool isDefault = qFuzzyIsNull(m_dataFr->value() - 1.0) &&
                     qFuzzyIsNull(m_displayFr->value() - 1.0) &&
                     qFuzzyIsNull(m_factor->value() - 1.0) &&
                     qFuzzyIsNull(m_shiftX->value()) &&
                     qFuzzyIsNull(m_shiftY->value()) &&
                     qFuzzyIsNull(m_shiftZ->value());
    if (isDefault) {
      setTitleColor(ThemeManager::instance().textColor());
    } else {
      setTitleColor(QColor(255, 80, 80));
    }
  }
};

DicomViewer::DicomViewer(QWidget *parent, bool showControls)
    : QWidget(parent), m_dicomReader(std::make_unique<DicomReader>()),
      m_zoomFactor(DEFAULT_ZOOM), m_windowLevelDragActive(false),
      m_dragStartWindow(256.0), m_dragStartLevel(128.0),
      m_windowLevelTimer(new QTimer(this)), m_panTimer(new QTimer(this)),
      m_zoomTimer(new QTimer(this)), m_panDragActive(false), m_panMode(false),
      m_zoomDragActive(false), m_zoomMode(false),
      m_zoomStartFactor(DEFAULT_ZOOM), m_gridWidget(nullptr),
      m_showControls(showControls), m_viewMode(ViewMode::Single),
      m_fourViewMode(false),
      m_syncScale(true) // ★新規追加: デフォルトでスケール同期ON
{
  // 5画面表示などで極端に縮小されないよう最小サイズを設定
  setMinimumSize(50, 50);
  for (int i = 0; i < VIEW_COUNT; ++i) {
    m_currentIndices[i] = 0;
    m_originalImages[i] = QImage();
    m_viewScrollAreas[i] = nullptr;
    m_panOffsets[i] = QPointF(0.0, 0.0);
    m_orientationButtons[i] = nullptr; // ★新規追加
    m_colorBarPersistentVisibility[i] = false;
  }
  setupUI();

  // Initialize WebServer for Vision Pro integration
  m_webServer = new WebServer(this, this);
  if (m_webServer) {
    connect(m_webServer, &WebServer::serverStarted, this, [](quint16 port) {
      qInfo() << "Vision Pro Web Server started on port" << port;
    });
    connect(m_webServer, &WebServer::serverStopped, this, []() {
      qInfo() << "Vision Pro Web Server stopped";
    });
    // Auto-start web server with HTTPS enabled for Vision Pro WebXR support
    // Use port 8443 for HTTPS (default SSL port for development)
    // If SSL certificates are not available, it will fall back to HTTP on port 8080
    bool useSSL = true;
    quint16 port = useSSL ? 8443 : 8080;
    m_webServer->start(port, useSSL);
  }

  m_viewOrientations[0] = DicomVolume::Orientation::Axial;
  m_viewOrientations[1] = DicomVolume::Orientation::Sagittal;
  m_viewOrientations[2] = DicomVolume::Orientation::Coronal;
  m_viewOrientations[3] = DicomVolume::Orientation::Coronal;
  m_viewOrientations[4] = DicomVolume::Orientation::Axial;

  loadProfileLinePresets();

  // タイマー設定
  m_windowLevelTimer->setSingleShot(true);
  connect(m_windowLevelTimer, &QTimer::timeout, this,
          &DicomViewer::onWindowLevelTimeout);

  m_panTimer->setSingleShot(true);
  connect(m_panTimer, &QTimer::timeout, this, &DicomViewer::onPanTimeout);
  m_zoomTimer->setSingleShot(true);
  connect(m_zoomTimer, &QTimer::timeout, this, &DicomViewer::onZoomTimeout);

  // Auto-load Ir source data if previously saved
  autoLoadBrachySourceData();

  // キーボードフォーカスを有効にする
  setFocusPolicy(Qt::StrongFocus);
}

DicomViewer::~DicomViewer() {
  // Stop web server
  if (m_webServer) {
    m_webServer->stop();
    delete m_webServer;
    m_webServer = nullptr;
  }
}

void DicomViewer::setDatabaseManager(DatabaseManager *dbManager) {
  m_databaseManager = dbManager;
}

void DicomViewer::setBrachyStatusStyle(const QString &borderColor) {
  m_brachyStatusBorderColor = borderColor;
  if (!m_brachyDataStatus)
    return;
  const QColor textColor = ThemeManager::instance().textColor();
  m_brachyDataStatus->setStyleSheet(
      QStringLiteral("QLabel { border: 1px solid %1; border-radius: 4px; padding: 4px; color: %2; }")
          .arg(borderColor, textColor.name()));
}

void DicomViewer::setFourViewMode(bool enabled) {
  setViewMode(enabled ? ViewMode::Quad : ViewMode::Single);
  m_fourViewMode = enabled;
}

void DicomViewer::setViewMode(ViewMode mode) {
  if (m_viewMode == mode)
    return;
  if (m_fusionViewActive && m_restoreViewModeAfterFusion &&
      mode != ViewMode::Dual) {
    m_restoreViewModeAfterFusion = false;
  }
  m_viewMode = mode;

  updateViewLayout();
  m_activeViewIndex = clampToVisibleViewIndex(m_activeViewIndex);

  if (mode == ViewMode::Five) {
    // 画面切替後に時間差で各ビューを設定
    auto setupImageView = [this](int idx, DicomVolume::Orientation ori) {
      if (m_viewMode != ViewMode::Five)
        return;
      setViewToImage(idx);
      m_viewOrientations[idx] = ori;
      int count = sliceCountForOrientation(ori);
      m_sliceSliders[idx]->setRange(0, count > 0 ? count - 1 : 0);
      int mid = count > 0 ? count / 2 : 0;
      m_currentIndices[idx] = mid;
      m_sliceSliders[idx]->setValue(mid);
      loadVolumeSlice(idx, mid);
    };

    QTimer::singleShot(0, this, [this, setupImageView]() {
      setupImageView(0, DicomVolume::Orientation::Axial); // 左上
    });
    QTimer::singleShot(100, this, [this]() {
      if (m_viewMode != ViewMode::Five)
        return;
      setViewTo3D(1); // 右上
    });
    QTimer::singleShot(200, this, [this, setupImageView]() {
      setupImageView(2, DicomVolume::Orientation::Sagittal); // 左下左
    });
    QTimer::singleShot(300, this, [this, setupImageView]() {
      setupImageView(3, DicomVolume::Orientation::Coronal); // 左下右
    });
    QTimer::singleShot(400, this, [this]() {
      if (m_viewMode != ViewMode::Five)
        return;
      setViewToDVH(4); // 右下
    });
    QTimer::singleShot(500, this, [this]() {
      if (m_viewMode != ViewMode::Five)
        return;
      updateSliceLabels();
      updateSliderPosition();
      updateImage();
      updateColorBars();
      updateOrientationButtonTexts();
    });
  } else if (mode == ViewMode::Quad) {
    auto setupImageView = [this](int idx, DicomVolume::Orientation ori) {
      if (m_viewMode != ViewMode::Quad)
        return;
      setViewToImage(idx);
      m_viewOrientations[idx] = ori;
      int count = sliceCountForOrientation(ori);
      m_sliceSliders[idx]->setRange(0, count > 0 ? count - 1 : 0);
      int mid = count > 0 ? count / 2 : 0;
      m_currentIndices[idx] = mid;
      m_sliceSliders[idx]->setValue(mid);
      loadVolumeSlice(idx, mid);
    };

    QTimer::singleShot(0, this, [this, setupImageView]() {
      setupImageView(0, DicomVolume::Orientation::Axial); // 左上
    });
    QTimer::singleShot(100, this, [this, setupImageView]() {
      setupImageView(1, DicomVolume::Orientation::Sagittal); // 右上
    });
    QTimer::singleShot(200, this, [this, setupImageView]() {
      setupImageView(2, DicomVolume::Orientation::Coronal); // 左下
    });
    QTimer::singleShot(300, this, [this]() {
      if (m_viewMode != ViewMode::Quad)
        return;
      setViewToDVH(3); // 右下
    });
    QTimer::singleShot(400, this, [this]() {
      if (m_viewMode != ViewMode::Quad)
        return;
      updateSliceLabels();
      updateSliderPosition();
      updateImage();
      updateColorBars();
      updateOrientationButtonTexts();
    });
  } else {
    updateSliderPosition();
    updateImage();
    updateColorBars();
  }
}

void DicomViewer::setupUI() {
  m_mainLayout = new QVBoxLayout(this);
  ThemeManager &theme = ThemeManager::instance();

  // メインスプリッター（画像表示エリアとコントロールエリア）
  QSplitter *mainSplitter = new QSplitter(Qt::Horizontal, this);

  // 画像表示エリアとスライダーコンテナ
  m_imageContainer = new QWidget();
  QHBoxLayout *imageLayout = new QHBoxLayout(m_imageContainer);
  imageLayout->setContentsMargins(0, 0, 0, 0);

  m_scrollArea = new QScrollArea();
  m_scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  m_scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  m_scrollArea->setAlignment(Qt::AlignCenter);

  m_gridWidget = new QWidget();
  m_imageLayout = new QGridLayout(m_gridWidget);
  m_imageLayout->setSpacing(2);
  m_imageLayout->setContentsMargins(0, 0, 0, 0);

  for (int i = 0; i < VIEW_COUNT; ++i) {
    m_viewContainers[i] = new QWidget();
    QVBoxLayout *cellMainLayout = new QVBoxLayout(m_viewContainers[i]);
    cellMainLayout->setContentsMargins(0, 0, 0, 0);
    cellMainLayout->setSpacing(2);

    // ★変更: 方向変更ボタンと画像切り替えボタンをオーバーレイとして配置
    m_orientationButtons[i] = new QPushButton(m_viewContainers[i]);
    m_orientationButtons[i]->setMaximumHeight(25);
    const QString topButtonStyle =
        QStringLiteral(
            "QPushButton {"
            "    background-color: rgba(0,0,0,180);"
            "    color: %1;"
            "    border: 1px solid #444444;"
            "    padding: 2px 8px;"
            "    font-size: 10px;"
            "}"
            "QPushButton:hover {"
            "    background-color: rgba(50,50,150,200);"
            "}");
    const QString interactionButtonStyle =
        QStringLiteral(
            "QPushButton {"
            "    background-color: rgba(0,0,0,180);"
            "    color: %1;"
            "    border: 1px solid #444444;"
            "    padding: 2px 6px;"
            "    font-size: 10px;"
            "}"
            "QPushButton:hover {"
            "    background-color: rgba(50,50,150,200);"
            "}"
            "QPushButton:checked {"
            "    background-color: #FF6B35;"
            "    border: 1px solid #FF6B35;"
            "}");
    theme.applyTextColor(m_orientationButtons[i], topButtonStyle);
    connect(m_orientationButtons[i], &QPushButton::clicked, this,
            [this, i]() { showOrientationMenuForView(i); });
    m_orientationButtons[i]->raise();

    // Image toggle button (for 3D view only)
    m_imageToggleButtons[i] = new QPushButton(m_viewContainers[i]);
    m_imageToggleButtons[i]->setText("Image");
    m_imageToggleButtons[i]->setMaximumHeight(25);
    theme.applyTextColor(m_imageToggleButtons[i], topButtonStyle);
    connect(m_imageToggleButtons[i], &QPushButton::clicked, this,
            [this, i]() { onImageToggleClicked(i); });
    m_imageToggleButtons[i]->setVisible(false); // Hidden by default, shown only in 3D view
    m_imageToggleButtons[i]->raise();

    // Line toggle button (for 3D view only - Structure Lines)
    m_lineToggleButtons[i] = new QPushButton(m_viewContainers[i]);
    m_lineToggleButtons[i]->setText("Line");
    m_lineToggleButtons[i]->setMaximumHeight(25);
    theme.applyTextColor(m_lineToggleButtons[i], topButtonStyle);
    connect(m_lineToggleButtons[i], &QPushButton::clicked, this,
            [this, i]() { onLineToggleClicked(i); });
    m_lineToggleButtons[i]->setVisible(false); // Hidden by default, shown only in 3D view
    m_lineToggleButtons[i]->raise();

    // Surface toggle button (for 3D view only - Structure Surfaces)
    m_surfaceToggleButtons[i] = new QPushButton(m_viewContainers[i]);
    m_surfaceToggleButtons[i]->setText("Surface");
    m_surfaceToggleButtons[i]->setMaximumHeight(25);
    theme.applyTextColor(m_surfaceToggleButtons[i], topButtonStyle);
    connect(m_surfaceToggleButtons[i], &QPushButton::clicked, this,
            [this, i]() { onSurfaceToggleClicked(i); });
    m_surfaceToggleButtons[i]->setVisible(false); // Hidden by default, shown only in 3D view
    m_surfaceToggleButtons[i]->raise();

    // Export button (for 3D view only - VisionPro USDZ export)
    m_exportButtons[i] = new QPushButton(m_viewContainers[i]);
    m_exportButtons[i]->setText("Export");
    m_exportButtons[i]->setMaximumHeight(25);
    theme.applyTextColor(m_exportButtons[i], topButtonStyle);
    connect(m_exportButtons[i], &QPushButton::clicked, this,
            [this, i]() { onExportButtonClicked(i); });
    m_exportButtons[i]->setVisible(false); // Hidden by default, shown only in 3D view
    m_exportButtons[i]->raise();

    m_imageSeriesButtons[i] = new QPushButton(m_viewContainers[i]);
    m_imageSeriesButtons[i]->setMaximumHeight(25);
    theme.applyTextColor(m_imageSeriesButtons[i], topButtonStyle);
    m_imageSeriesButtons[i]->setVisible(false);
    connect(m_imageSeriesButtons[i], &QPushButton::clicked, this,
            [this, i]() { showImageSeriesMenu(i); });
    m_imageSeriesButtons[i]->raise();

    m_viewWindowLevelButtons[i] =
        new QPushButton(tr("W/L"), m_viewContainers[i]);
    m_viewWindowLevelButtons[i]->setCheckable(true);
    theme.applyTextColor(m_viewWindowLevelButtons[i], interactionButtonStyle);
    m_viewWindowLevelButtons[i]->hide();
    m_viewWindowLevelButtons[i]->raise();
    connect(m_viewWindowLevelButtons[i], &QPushButton::toggled, this,
            [this, i](bool checked) { onViewWindowLevelToggled(i, checked); });

    m_viewPanButtons[i] = new QPushButton(tr("Pan"), m_viewContainers[i]);
    m_viewPanButtons[i]->setCheckable(true);
    theme.applyTextColor(m_viewPanButtons[i], interactionButtonStyle);
    m_viewPanButtons[i]->hide();
    m_viewPanButtons[i]->raise();
    connect(m_viewPanButtons[i], &QPushButton::toggled, this,
            [this, i](bool checked) { onViewPanToggled(i, checked); });

    m_viewZoomButtons[i] = new QPushButton(tr("Zoom"), m_viewContainers[i]);
    m_viewZoomButtons[i]->setCheckable(true);
    theme.applyTextColor(m_viewZoomButtons[i], interactionButtonStyle);
    m_viewZoomButtons[i]->hide();
    m_viewZoomButtons[i]->raise();
    connect(m_viewZoomButtons[i], &QPushButton::toggled, this,
            [this, i](bool checked) { onViewZoomToggled(i, checked); });

    // 画像表示エリアのレイアウトをスタックに配置
    m_imagePanels[i] = new QWidget();
    QHBoxLayout *cellLayout = new QHBoxLayout(m_imagePanels[i]);
    cellLayout->setContentsMargins(0, 0, 0, 0);
    cellLayout->setSpacing(0);

    m_imageWidgets[i] = new OpenGLImageWidget();
    m_imageWidgets[i]->setStructureLineWidth(m_structureLineWidth);
    m_imageWidgets[i]->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(m_imageWidgets[i], &OpenGLImageWidget::doubleClicked, this,
            [this, i]() { onImageDoubleClicked(i); });
    connect(m_imageWidgets[i], &OpenGLImageWidget::customContextMenuRequested,
            this, [this, i](const QPoint &pos) { showJumpToMenu(i, pos); });

    m_viewScrollAreas[i] = new QScrollArea(m_viewContainers[i]);
    m_viewScrollAreas[i]->setWidget(m_imageWidgets[i]);
    m_viewScrollAreas[i]->setWidgetResizable(true);
    m_viewScrollAreas[i]->setFrameShape(QFrame::NoFrame);
    m_viewScrollAreas[i]->setAlignment(Qt::AlignCenter);
    m_viewScrollAreas[i]->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_viewScrollAreas[i]->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_imageWidgets[i]->installEventFilter(this);
    cellLayout->addWidget(m_viewScrollAreas[i], 1);

    // カラーバーを追加
    m_colorBars[i] = new DoseColorBar(m_viewContainers[i]);
    m_colorBars[i]->setVisible(false);
    cellLayout->addWidget(m_colorBars[i]);

    m_sliceSliders[i] = new QSlider(Qt::Vertical);
    m_sliceSliders[i]->setRange(0, 0);
    m_sliceSliders[i]->setEnabled(false);
    cellLayout->addWidget(m_sliceSliders[i]);

    m_dvhWidgets[i] = new DVHWindow(m_viewContainers[i]);
    connect(m_dvhWidgets[i], &DVHWindow::recalculateRequested, this,
            &DicomViewer::onDvhCalculationRequested, Qt::QueuedConnection);
    connect(m_dvhWidgets[i], &DVHWindow::doubleClicked, this,
            [this, i]() { onImageDoubleClicked(i); });
    connect(m_dvhWidgets[i], &DVHWindow::visibilityChanged, this,
            &DicomViewer::onDvhVisibilityChanged);
    connect(m_dvhWidgets[i], &DVHWindow::calcMaxChanged, this,
            &DicomViewer::onDvhCalcMaxChanged);
    m_dvhWidgets[i]->hide();

    m_3dWidgets[i] = new OpenGL3DWidget(m_viewContainers[i]);
    m_3dWidgets[i]->hide();

    m_profileWidgets[i] = new DoseProfileWindow(m_viewContainers[i]);
    connect(m_profileWidgets[i], &DoseProfileWindow::requestLineSelection, this,
            [this, i]() { onProfileLineSelection(i); });
    connect(m_profileWidgets[i], &DoseProfileWindow::saveLineRequested, this,
            [this, i](int slot) { onProfileLineSaveRequested(i, slot); });
    connect(m_profileWidgets[i], &DoseProfileWindow::loadLineRequested, this,
            [this, i](int slot) { onProfileLineLoadRequested(i, slot); });
    m_profileWidgets[i]->hide();

    m_viewStacks[i] = new QStackedLayout();
    m_viewStacks[i]->addWidget(m_imagePanels[i]);
    m_viewStacks[i]->addWidget(m_dvhWidgets[i]);
    m_viewStacks[i]->addWidget(m_3dWidgets[i]);
    m_viewStacks[i]->addWidget(m_profileWidgets[i]);
    cellMainLayout->addLayout(m_viewStacks[i], 1);
    m_viewStacks[i]->setCurrentWidget(m_imagePanels[i]);
    // ラベル類
    m_sliceIndexLabels[i] = new QLabel(m_viewContainers[i]);
    // Ax/Sag/Cor などの情報ラベルは背景を透明化
    theme.applyTextColor(
        m_sliceIndexLabels[i],
        QStringLiteral(
            "QLabel { background-color: transparent; color: %1; font-size: "
            "10px; }"));
    m_sliceIndexLabels[i]->hide();

    m_infoOverlays[i] = new QLabel(m_viewContainers[i]);
    theme.applyTextColor(
        m_infoOverlays[i],
        QStringLiteral(
            "QLabel { background-color: transparent; color: %1; font-size: "
            "10px; }"));
    m_infoOverlays[i]->hide();
    m_infoOverlays[i]->installEventFilter(this);

    // Dose information indicator (Dose Shift / BED / EqD2)
    m_doseShiftLabels[i] = new QLabel(m_viewContainers[i]);
    m_doseShiftLabels[i]->setStyleSheet(
        "QLabel { background-color: transparent; color: red; font-weight: "
        "bold; font-size: 12px; }");
    m_doseShiftLabels[i]->hide();

    m_coordLabels[i] = new QLabel(m_viewContainers[i]);
    theme.applyTextColor(
        m_coordLabels[i],
        QStringLiteral(
            "QLabel { background-color: transparent; color: %1; font-size: "
            "10px; }"));
    m_coordLabels[i]->hide();

    m_cursorDoseLabels[i] = new QLabel(m_viewContainers[i]);
    theme.applyTextColor(
        m_cursorDoseLabels[i],
        QStringLiteral(
            "QLabel { background-color: transparent; color: %1; font-size: "
            "10px; }"));
    m_cursorDoseLabels[i]->hide();

    m_imageLayout->addWidget(m_viewContainers[i], i / 2, i % 2);
  }

  // 初期の方向ボタンテキストを設定
  updateOrientationButtonTexts();

  m_scrollArea->setWidget(m_gridWidget);
  m_scrollArea->setWidgetResizable(true);
  m_scrollArea->viewport()->installEventFilter(this);
  imageLayout->addWidget(m_scrollArea, 1);

  updateSliderPosition();

  // コントロールパネル
  QWidget *controlPanel = new QWidget(this);
  controlPanel->setFixedWidth(300);
  QVBoxLayout *controlLayout = new QVBoxLayout(controlPanel);

  // Window Level controls
  m_windowLevelGroup = new CollapsibleGroupBox("Window Level");
  m_windowLevelGroup->setTitleColor(theme.textColor());
  QObject::connect(&theme, &ThemeManager::textColorChanged,
                   m_windowLevelGroup,
                   [group = m_windowLevelGroup](const QColor &color) {
                     group->setTitleColor(color);
                   });
  QGridLayout *wlLayout = new QGridLayout();
  wlLayout->setContentsMargins(5, 5, 5, 5);
  m_windowLevelGroup->setContentLayout(wlLayout);

  // Window controls
  QLabel *windowLabel = new QLabel("Window:");
  theme.applyTextColor(windowLabel);
  wlLayout->addWidget(windowLabel, 0, 0);
  m_windowSlider = new QSlider(Qt::Horizontal);
  m_windowSlider->setRange(1, 4096);
  m_windowSlider->setValue(256);
  m_windowSpinBox = new QSpinBox();
  m_windowSpinBox->setRange(1, 4096);
  m_windowSpinBox->setValue(256);
  wlLayout->addWidget(m_windowSlider, 0, 1);
  wlLayout->addWidget(m_windowSpinBox, 0, 2);

  // Level controls
  QLabel *levelLabel = new QLabel("Level:");
  theme.applyTextColor(levelLabel);
  wlLayout->addWidget(levelLabel, 1, 0);
  m_levelSlider = new QSlider(Qt::Horizontal);
  m_levelSlider->setRange(-1024, 3072);
  m_levelSlider->setValue(128);
  m_levelSpinBox = new QSpinBox();
  m_levelSpinBox->setRange(-1024, 3072);
  m_levelSpinBox->setValue(128);
  wlLayout->addWidget(m_levelSlider, 1, 1);
  wlLayout->addWidget(m_levelSpinBox, 1, 2);

  // Slice Position checkbox
  m_slicePositionCheck = new QCheckBox("Slice Position");
  theme.applyTextColor(m_slicePositionCheck);
  wlLayout->addWidget(m_slicePositionCheck, 2, 0, 1, 3);
  connect(m_slicePositionCheck, &QCheckBox::toggled, this,
          &DicomViewer::onSlicePositionToggled);
  // Grid + Dose Guide (same row)
  QHBoxLayout *gridGuideRow = new QHBoxLayout();
  m_gridCheck = new QCheckBox("Grid");
  theme.applyTextColor(m_gridCheck);
  gridGuideRow->addWidget(m_gridCheck);
  connect(m_gridCheck, &QCheckBox::toggled, this, [this](bool on) {
    m_showGrid = on;
    updateImage();
  });
  m_doseGuideCheck = new QCheckBox("Guide");
  theme.applyTextColor(m_doseGuideCheck);
  m_doseGuideCheck->setChecked(m_showDoseGuide);
  gridGuideRow->addWidget(m_doseGuideCheck);
  connect(m_doseGuideCheck, &QCheckBox::toggled, this, [this](bool on) {
    m_showDoseGuide = on;
    updateImage();
  });
  gridGuideRow->addStretch();
  wlLayout->addLayout(gridGuideRow, 3, 0, 1, 3);

  // Window/Level調整・Pan・Zoom ボタン
  m_windowLevelButton = new QPushButton("W/L Adjust");
  m_windowLevelButton->setCheckable(true);
  const QString wlButtonStyle =
      QStringLiteral(
          "QPushButton {"
          "    background-color: #444444;"
          "    color: %1;"
          "    border: 1px solid #666666;"
          "    padding: 5px;"
          "    border-radius: 3px;"
          "}"
          "QPushButton:hover {"
          "    background-color: #555555;"
          "}"
          "QPushButton:checked {"
          "    background-color: #FF6B35;"
          "    border: 1px solid #FF6B35;"
          "}");
  theme.applyTextColor(m_windowLevelButton, wlButtonStyle);
  m_panButton = new QPushButton("Pan");
  m_panButton->setCheckable(true);
  theme.applyTextColor(m_panButton, wlButtonStyle);
  m_zoomButton = new QPushButton("Zoom");
  m_zoomButton->setCheckable(true);
  theme.applyTextColor(m_zoomButton, wlButtonStyle);

  // 操作ボタンを横並びで配置
  QHBoxLayout *wlButtonLayout = new QHBoxLayout();
  wlButtonLayout->addWidget(m_windowLevelButton);
  wlButtonLayout->addWidget(m_panButton);
  wlButtonLayout->addWidget(m_zoomButton);
  // Place operation buttons below grid toggle
  wlLayout->addLayout(wlButtonLayout, 4, 0, 1, 3);

  // Dose display controls
  if (m_showControls) {
    m_doseGroup = new CollapsibleGroupBox("Dose Display");
    m_doseGroup->setTitleColor(theme.textColor());
    QObject::connect(&theme, &ThemeManager::textColorChanged, m_doseGroup,
                     [group = m_doseGroup](const QColor &color) {
                       group->setTitleColor(color);
                     });
    QVBoxLayout *doseLayout = new QVBoxLayout();
    doseLayout->setContentsMargins(5, 5, 5, 5);

    QHBoxLayout *modeLayout = new QHBoxLayout();
    QLabel *colorMapLabel = new QLabel("Color Map:");
    theme.applyTextColor(colorMapLabel);
    modeLayout->addWidget(colorMapLabel);
    m_doseColorMapCombo = new QComboBox();
    m_doseColorMapCombo->addItem("Colorful");
    m_doseColorMapCombo->addItem("Isodose");
    m_doseColorMapCombo->addItem("Isodose Lines");
    m_doseColorMapCombo->addItem("Simple");
    m_doseColorMapCombo->addItem("Hot");
    connect(m_doseColorMapCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &DicomViewer::onDoseDisplayModeChanged);
    modeLayout->addWidget(m_doseColorMapCombo);
    doseLayout->addLayout(modeLayout);

    QGridLayout *rangeLayout = new QGridLayout();
    QLabel *minLabel = new QLabel("Min:");
    theme.applyTextColor(minLabel);
    rangeLayout->addWidget(minLabel, 0, 0);
    m_doseMinSpinBox = new QDoubleSpinBox();
    m_doseMinSpinBox->setDecimals(4);
    m_doseMinSpinBox->setRange(-1000.0, 1000.0);
    rangeLayout->addWidget(m_doseMinSpinBox, 0, 1);
    QLabel *maxLabel = new QLabel("Max:");
    theme.applyTextColor(maxLabel);
    rangeLayout->addWidget(maxLabel, 1, 0);
    m_doseMaxSpinBox = new QDoubleSpinBox();
    m_doseMaxSpinBox->setDecimals(4);
    m_doseMaxSpinBox->setRange(-1000.0, 1000.0);
    rangeLayout->addWidget(m_doseMaxSpinBox, 1, 1);
    connect(m_doseMinSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &DicomViewer::updateColorBars);
    connect(m_doseMaxSpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &DicomViewer::updateColorBars);
    connect(m_doseMinSpinBox, &QDoubleSpinBox::editingFinished, this,
            &DicomViewer::onDoseRangeEditingFinished);
    connect(m_doseMaxSpinBox, &QDoubleSpinBox::editingFinished, this,
            &DicomViewer::onDoseRangeEditingFinished);
    doseLayout->addLayout(rangeLayout);

    QHBoxLayout *opacityLayout = new QHBoxLayout();
    QLabel *opacityLabel = new QLabel("Opacity:");
    theme.applyTextColor(opacityLabel);
    opacityLayout->addWidget(opacityLabel);
    m_doseOpacitySlider = new QSlider(Qt::Horizontal);
    m_doseOpacitySlider->setRange(0, 100);
    m_doseOpacitySlider->setValue(80);
    m_doseOpacitySlider->setFixedHeight(20);
    connect(m_doseOpacitySlider, &QSlider::valueChanged, this,
            &DicomViewer::onDoseOpacityChanged);
    opacityLayout->addWidget(m_doseOpacitySlider);
    doseLayout->addLayout(opacityLayout);
    // Always native geometry (toggle removed)
    m_doseGroup->setContentLayout(doseLayout);
  }

  // Zoom controls (only reset and fit)
  m_zoomGroup = new CollapsibleGroupBox("Zoom");
  m_zoomGroup->setTitleColor(theme.textColor());
  QObject::connect(&theme, &ThemeManager::textColorChanged, m_zoomGroup,
                   [group = m_zoomGroup](const QColor &color) {
                     group->setTitleColor(color);
                   });
  QGridLayout *zoomLayout = new QGridLayout();
  zoomLayout->setContentsMargins(5, 5, 5, 5);
  m_resetZoomButton = new QPushButton("Reset");
  m_fitToWindowButton = new QPushButton("Fit to Window");
  zoomLayout->addWidget(m_resetZoomButton, 0, 0);
  zoomLayout->addWidget(m_fitToWindowButton, 0, 1);
  m_zoomGroup->setContentLayout(zoomLayout);

  // 画像情報
  m_infoGroup = new CollapsibleGroupBox("Image Information");
  m_infoGroup->setTitleColor(theme.textColor());
  QObject::connect(&theme, &ThemeManager::textColorChanged, m_infoGroup,
                   [group = m_infoGroup](const QColor &color) {
                     group->setTitleColor(color);
                   });
  m_privacyButton = new QToolButton();
  m_privacyButton->setText("P");
  m_privacyButton->setAutoRaise(true);
  m_privacyButton->setFixedSize(16, 16);
  m_privacyButton->setToolTip("Toggle Privacy Mode");
  m_privacyButton->setStyleSheet(
      "border: 1px solid gray; color: gray;" // ← 文字色をグレーに指定
  );
  // m_privacyButton->setStyleSheet("border: 1px solid gray;");
  m_infoGroup->addHeaderWidget(m_privacyButton);
  connect(m_privacyButton, &QToolButton::clicked, this, [this]() {
    m_privacyMode = !m_privacyMode;
    updateImageInfo();
    updateInfoOverlays();
  });
  QVBoxLayout *infoLayout = new QVBoxLayout();
  m_infoTextBox = new QPlainTextEdit();
  m_infoTextBox->setReadOnly(true);
  m_infoTextBox->setPlainText("Patient: -\n"
                              "Modality: -\n"
                              "Study Date: -\n"
                              "Size: -\n"
                              "Study Desc: -\n"
                              "Slice Thk: -\n"
                              "Pixel Spacing: - x -\n"
                              "CT File: -\n\n"
                              "RT Dose: Not Loaded");
  QPalette infoPal = m_infoTextBox->palette();
  infoPal.setColor(QPalette::Base, QColor(32, 32, 32));
  infoPal.setColor(QPalette::Text, theme.textColor());
  m_infoTextBox->setPalette(infoPal);
  QObject::connect(&theme, &ThemeManager::textColorChanged, m_infoTextBox,
                   [textBox = m_infoTextBox](const QColor &color) {
                     QPalette pal = textBox->palette();
                     pal.setColor(QPalette::Text, color);
                     textBox->setPalette(pal);
                   });
  m_infoTextBox->installEventFilter(this);
  infoLayout->addWidget(m_infoTextBox);

  QHBoxLayout *doseRefLayout = new QHBoxLayout();
  doseRefLayout->addWidget(new QLabel("100% ="));
  m_doseRefSpinBox = new QDoubleSpinBox();
  m_doseRefSpinBox->setDecimals(4);
  m_doseRefSpinBox->setRange(-1000.0, 1000.0);
  m_doseRefSpinBox->setSuffix(" Gy");
  doseRefLayout->addWidget(m_doseRefSpinBox);
  doseRefLayout->addStretch();
  infoLayout->addLayout(doseRefLayout);

  m_doseRefMaxButton = new QPushButton("100% = MaxDose");
  infoLayout->addWidget(m_doseRefMaxButton);

  connect(m_doseRefSpinBox,
          QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
          [this](double v) {
            m_doseReference = v;
            updateColorBars();
            updateImage();
            for (int i = 0; i < VIEW_COUNT; ++i) {
              if (m_dvhWidgets[i])
                m_dvhWidgets[i]->setPrescriptionDose(v);
            }
          });

  connect(m_doseRefMaxButton, &QPushButton::clicked, this, [this]() {
    double maxDose = m_resampledDose.maxDose();
    m_doseRefSpinBox->setValue(maxDose);
  });
  m_infoGroup->setContentLayout(infoLayout);

  // Structures
  m_structureGroup = new CollapsibleGroupBox("Structures");
  m_structureGroup->setTitleColor(theme.textColor());
  QObject::connect(&theme, &ThemeManager::textColorChanged, m_structureGroup,
                   [group = m_structureGroup](const QColor &color) {
                     group->setTitleColor(color);
                   });
  QVBoxLayout *structLayout = new QVBoxLayout();
  m_structureList = new QListWidget();
  structLayout->addWidget(m_structureList);

  QHBoxLayout *pointWidthLayout = new QHBoxLayout();
  m_showPointsCheck = new QCheckBox("Points");
  theme.applyTextColor(m_showPointsCheck);
  m_showPointsCheck->setChecked(m_showStructurePoints);
  pointWidthLayout->addWidget(m_showPointsCheck);

  QLabel *lineWidthLabel = new QLabel("Width");
  theme.applyTextColor(
      lineWidthLabel, QStringLiteral("font-size:10px; color:%1;"));
  m_structureLineWidthSpin = new QSpinBox();
  m_structureLineWidthSpin->setRange(1, 3);
  m_structureLineWidthSpin->setValue(m_structureLineWidth);
  m_structureLineWidthSpin->setFixedWidth(40);
  pointWidthLayout->addWidget(lineWidthLabel);
  pointWidthLayout->addWidget(m_structureLineWidthSpin);
  pointWidthLayout->addStretch();
  structLayout->addLayout(pointWidthLayout);
  QHBoxLayout *structBtnLayout = new QHBoxLayout();
  m_structureAllButton = new QPushButton("All");
  m_structureNoneButton = new QPushButton("None");
  m_dvhButton = new QPushButton("DVH");
  structBtnLayout->addWidget(m_structureAllButton);
  structBtnLayout->addWidget(m_structureNoneButton);
  structBtnLayout->addWidget(m_dvhButton);
  structLayout->addLayout(structBtnLayout);
  m_structureGroup->setContentLayout(structLayout);

  // コントロールパネルにウィジェットを追加
  controlLayout->addWidget(m_infoGroup);
  controlLayout->addWidget(m_windowLevelGroup);
  if (m_showControls && m_doseGroup) {
    controlLayout->addWidget(m_doseGroup);
  }
  controlLayout->addWidget(m_zoomGroup);
  controlLayout->addWidget(m_structureGroup);
  controlLayout->addStretch();

  // DoseManagerパネル（右側タブ）
  m_doseManagerPanel = new QWidget(this);
  QVBoxLayout *doseMgrLayout = new QVBoxLayout(m_doseManagerPanel);
  doseMgrLayout->setContentsMargins(0, 0, 0, 0);
  QHBoxLayout *doseHeader = new QHBoxLayout();
  doseHeader->addWidget(new QLabel("RT Dose"));
  m_doseModeCombo = new QComboBox();
  m_doseModeCombo->addItem(tr("Physical Dose"),
                           QVariant(int(DoseCalcMode::Physical)));
  m_doseModeCombo->addItem(tr("BED"), QVariant(int(DoseCalcMode::BED)));
  m_doseModeCombo->addItem(tr("EqD2"), QVariant(int(DoseCalcMode::EqD2)));
  doseHeader->addWidget(m_doseModeCombo);
  doseHeader->addWidget(new QLabel(QString::fromUtf8("α/β=")));
  m_doseAlphaBetaSpin = new QDoubleSpinBox();
  m_doseAlphaBetaSpin->setRange(0.1, 100.0);
  m_doseAlphaBetaSpin->setValue(10.0);
  m_doseAlphaBetaSpin->setEnabled(false);
  doseHeader->addWidget(m_doseAlphaBetaSpin);
  doseMgrLayout->addLayout(doseHeader);

  m_doseCalcButton = new QPushButton(tr("Start Calculation"));
  doseMgrLayout->addWidget(m_doseCalcButton);
  m_doseIsosurfaceButton = new QPushButton(tr("3D Isosurface"));
  doseMgrLayout->addWidget(m_doseIsosurfaceButton);
  m_doseListWidget = new QListWidget();
  m_doseListWidget->setWordWrap(true);
  m_doseListWidget->setContextMenuPolicy(Qt::CustomContextMenu);
  m_doseListWidget->setSizePolicy(QSizePolicy::Preferred,
                                  QSizePolicy::Expanding);
  connect(m_doseListWidget, &QListWidget::customContextMenuRequested, this,
          &DicomViewer::onDoseListContextMenu);
  doseMgrLayout->addWidget(m_doseListWidget, 1);
  m_gammaAnalysisButton = new QPushButton(tr("Gamma Analysis"));
  doseMgrLayout->addWidget(m_gammaAnalysisButton);
  // Add Random Study button under the DoseList
  m_randomStudyButton = new QPushButton(tr("Random Study"));
  doseMgrLayout->addWidget(m_randomStudyButton);

  // Brachy パネル（右側タブ）
  m_brachyPanel = new QWidget(this);
  QVBoxLayout *brachyLayout = new QVBoxLayout(m_brachyPanel);
  brachyLayout->setContentsMargins(0, 0, 0, 0);

  // Load RT-Plan button
  m_brachyReadButton = new QPushButton(tr("Load RT-Plan"), m_brachyPanel);
  brachyLayout->addWidget(m_brachyReadButton);

  // Source list
  m_brachyListWidget = new QListWidget(m_brachyPanel);
  brachyLayout->addWidget(m_brachyListWidget);

  // Load Ir source data button
  m_brachyLoadDataButton = new QPushButton(tr("Load Ir Source Data"), m_brachyPanel);
  brachyLayout->addWidget(m_brachyLoadDataButton);

  // Data status label with improved formatting
  m_brachyDataStatus = new QLabel(tr("Status: No source data loaded"), m_brachyPanel);
  m_brachyDataStatus->setWordWrap(true);
  m_brachyDataStatus->setTextInteractionFlags(Qt::TextSelectableByMouse);
  setBrachyStatusStyle(m_brachyStatusBorderColor);
  QObject::connect(&theme, &ThemeManager::textColorChanged, m_brachyDataStatus,
                   [this](const QColor &) {
                     setBrachyStatusStyle(m_brachyStatusBorderColor);
                   });
  brachyLayout->addWidget(m_brachyDataStatus);

  // Voxel size setting
  QHBoxLayout *voxelLayout = new QHBoxLayout();
  voxelLayout->addWidget(new QLabel(tr("Voxel Size (mm):"), m_brachyPanel));
  m_brachyVoxelSizeSpinBox = new QDoubleSpinBox(m_brachyPanel);
  m_brachyVoxelSizeSpinBox->setRange(0.5, 10.0);
  m_brachyVoxelSizeSpinBox->setSingleStep(0.5);
  m_brachyVoxelSizeSpinBox->setValue(2.0);
  m_brachyVoxelSizeSpinBox->setDecimals(1);
  voxelLayout->addWidget(m_brachyVoxelSizeSpinBox);
  brachyLayout->addLayout(voxelLayout);

  // Calculate dose button
  m_brachyCalcDoseButton = new QPushButton(tr("Calculate Dose"), m_brachyPanel);
  m_brachyCalcDoseButton->setEnabled(false);
  brachyLayout->addWidget(m_brachyCalcDoseButton);

  // Random source generation button
  m_brachyRandomSourceButton = new QPushButton(tr("Generate Random Sources (Test)"), m_brachyPanel);
  brachyLayout->addWidget(m_brachyRandomSourceButton);

  // Test source at origin button
  m_brachyTestSourceButton = new QPushButton(tr("Generate Test Source at Origin"), m_brachyPanel);
  brachyLayout->addWidget(m_brachyTestSourceButton);

  // Progress bar
  m_brachyProgressBar = new QProgressBar(m_brachyPanel);
  m_brachyProgressBar->setVisible(false);
  brachyLayout->addWidget(m_brachyProgressBar);

  // Dose Optimization Section
  brachyLayout->addWidget(new QLabel(tr("<b>Dose Optimization</b>"), m_brachyPanel));

  // Add evaluation point button
  m_brachyAddEvalPointButton = new QPushButton(tr("Add Dose Point at Cursor"), m_brachyPanel);
  m_brachyAddEvalPointButton->setEnabled(false);
  brachyLayout->addWidget(m_brachyAddEvalPointButton);

  // Evaluation points list
  m_brachyEvalPointsList = new QListWidget(m_brachyPanel);
  m_brachyEvalPointsList->setMaximumHeight(100);
  brachyLayout->addWidget(m_brachyEvalPointsList);

  // Clear evaluation points button
  m_brachyClearEvalPointsButton = new QPushButton(tr("Clear Dose Points"), m_brachyPanel);
  brachyLayout->addWidget(m_brachyClearEvalPointsButton);

  // Optimization settings
  QHBoxLayout *iterLayout = new QHBoxLayout();
  iterLayout->addWidget(new QLabel(tr("Max Iterations:"), m_brachyPanel));
  m_brachyOptimizationIterations = new QSpinBox(m_brachyPanel);
  m_brachyOptimizationIterations->setRange(10, 1000);
  m_brachyOptimizationIterations->setValue(100);
  m_brachyOptimizationIterations->setSingleStep(10);
  iterLayout->addWidget(m_brachyOptimizationIterations);
  brachyLayout->addLayout(iterLayout);

  QHBoxLayout *tolLayout = new QHBoxLayout();
  tolLayout->addWidget(new QLabel(tr("Tolerance:"), m_brachyPanel));
  m_brachyOptimizationTolerance = new QDoubleSpinBox(m_brachyPanel);
  m_brachyOptimizationTolerance->setRange(1e-6, 1e-2);
  m_brachyOptimizationTolerance->setValue(1e-4);
  m_brachyOptimizationTolerance->setDecimals(6);
  m_brachyOptimizationTolerance->setSingleStep(1e-5);
  tolLayout->addWidget(m_brachyOptimizationTolerance);
  brachyLayout->addLayout(tolLayout);

  // Optimize button
  m_brachyOptimizeButton = new QPushButton(tr("Optimize Dwell Times"), m_brachyPanel);
  m_brachyOptimizeButton->setEnabled(false);
  brachyLayout->addWidget(m_brachyOptimizeButton);

  // Reference Points Section
  brachyLayout->addWidget(new QLabel(tr("<b>Reference Points (from Plan)</b>"), m_brachyPanel));

  // Reference points list
  m_brachyRefPointsList = new QListWidget(m_brachyPanel);
  m_brachyRefPointsList->setMaximumHeight(80);
  brachyLayout->addWidget(m_brachyRefPointsList);

  // Show reference points checkbox
  m_brachyShowRefPointsCheck = new QCheckBox(tr("Show on Visualization"), m_brachyPanel);
  m_brachyShowRefPointsCheck->setChecked(true);
  brachyLayout->addWidget(m_brachyShowRefPointsCheck);

  brachyLayout->addStretch();

  // QRコードパネル（右側タブ）
  m_qrPanel = new QWidget(this);
  QVBoxLayout *qrLayout = new QVBoxLayout(m_qrPanel);
  qrLayout->setContentsMargins(0, 0, 0, 0);
  qrLayout->addWidget(new QLabel(tr("QR Generator")));
  m_qrTextEdit = new QPlainTextEdit(m_qrPanel);
  m_qrTextEdit->setPlaceholderText(
      tr("Enter text to encode (supports newlines)"));
  m_qrTextEdit->setMinimumHeight(80);
  qrLayout->addWidget(m_qrTextEdit);
  // Optional escaping for WiFi/MECARD reserved characters
  // Show a literal backslash before parenthesis: "\ (WiFi/MECARD)"
  m_qrEscapeCheck =
      new QCheckBox(tr("Escape ; , : \\ (WiFi/MECARD)"), m_qrPanel);
  m_qrEscapeCheck->setToolTip(
      tr("Adds backslashes before ; , : and \\ if not already escaped. Useful "
         "for iPhone parsing of WiFi/MECARD strings."));
  qrLayout->addWidget(m_qrEscapeCheck);
  m_qrUtf8EciCheck = new QCheckBox(tr("Add UTF-8 ECI"), m_qrPanel);
  m_qrUtf8EciCheck->setToolTip(
      tr("Prepend UTF-8 ECI segment for cross-device decoding. Some decoders "
         "(OpenCV) may fail with ECI."));
  m_qrUtf8EciCheck->setChecked(false);
  qrLayout->addWidget(m_qrUtf8EciCheck);
  m_qrHighAccuracyCheck =
      new QCheckBox(tr("High accuracy (slower)"), m_qrPanel);
  m_qrHighAccuracyCheck->setToolTip(tr("Use multi-variant robust decoding. "
                                       "Uncheck for faster camera scanning."));
  m_qrHighAccuracyCheck->setChecked(false);
  qrLayout->addWidget(m_qrHighAccuracyCheck);
  m_qrGenerateButton = new QPushButton(tr("Generate"), m_qrPanel);
  m_qrDecodeImageButton = new QPushButton(tr("Decode Image"), m_qrPanel);
  m_qrSaveButton = new QPushButton(tr("Save"), m_qrPanel);
  m_qrClearButton = new QPushButton(tr("Clear"), m_qrPanel);
  {
    // Arrange buttons in 2 columns
    QGridLayout *grid = new QGridLayout();
    grid->setHorizontalSpacing(8);
    grid->setVerticalSpacing(6);
    grid->addWidget(m_qrGenerateButton, 0, 0);
    grid->addWidget(m_qrDecodeImageButton, 0, 1);
    grid->addWidget(m_qrSaveButton, 1, 0);
    grid->addWidget(m_qrClearButton, 1, 1);
    qrLayout->addLayout(grid);
  }
  m_qrImageLabel = new QLabel(m_qrPanel);
  m_qrImageLabel->setAlignment(Qt::AlignCenter);
  m_qrImageLabel->setMinimumSize(240, 240);
  m_qrImageLabel->setScaledContents(false); // avoid smooth scaling
  m_qrImageLabel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
  qrLayout->addWidget(m_qrImageLabel, 1);

  // Camera controls row
  {
    QHBoxLayout *row = new QHBoxLayout();
    m_qrStartCamButton = new QPushButton(tr("Start Camera"), m_qrPanel);
    m_qrStopCamButton = new QPushButton(tr("Stop"), m_qrPanel);
    m_qrStopCamButton->setEnabled(false);
    row->addWidget(m_qrStartCamButton);
    row->addWidget(m_qrStopCamButton);
    row->addStretch();
    qrLayout->addLayout(row);
  }

  // Timer for camera grabbing
  m_qrCamTimer = new QTimer(this);
  m_qrCamTimer->setInterval(33); // ~30 FPS
  m_qrTimer.start();

  m_rightTabWidget = new MultiRowTabWidget(this);
  m_rightTabWidget->setFixedWidth(300);
  m_rightTabWidget->addTab(m_doseManagerPanel, tr("DoseManager"));
  m_rightTabWidget->addTab(m_brachyPanel, tr("Brachy"));
  m_rightTabWidget->addTab(m_qrPanel, tr("QR"));

  if (m_showControls) {
    mainSplitter->addWidget(controlPanel); // 左側
  }
  mainSplitter->addWidget(m_imageContainer); // 中央
  mainSplitter->addWidget(m_rightTabWidget); // 右側

  if (m_showControls) {
    mainSplitter->setSizes({300, 700, 300});
  } else {
    mainSplitter->setSizes({700, 300});
  }

  m_mainLayout->addWidget(mainSplitter);
  updateViewLayout();

  // シグナル・スロット接続（既存部分は省略...同じ）
  connect(m_windowSlider, &QSlider::valueChanged, this,
          &DicomViewer::onWindowChanged);
  connect(m_levelSlider, &QSlider::valueChanged, this,
          &DicomViewer::onLevelChanged);
  connect(m_windowSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
          m_windowSlider, &QSlider::setValue);
  connect(m_levelSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
          m_levelSlider, &QSlider::setValue);

  connect(m_resetZoomButton, &QPushButton::clicked, this,
          &DicomViewer::onResetZoom);
  connect(m_fitToWindowButton, &QPushButton::clicked, this,
          &DicomViewer::onFitToWindow);

  for (int i = 0; i < VIEW_COUNT; ++i) {
    connect(m_sliceSliders[i], &QSlider::valueChanged, this,
            &DicomViewer::onImageSliderChanged);
    connect(m_sliceSliders[i], &QSlider::sliderPressed, this,
            [this, i]() { m_activeViewIndex = i; });
  }

  connect(m_windowLevelButton, &QPushButton::clicked, this,
          &DicomViewer::onWindowLevelButtonClicked);
  connect(m_panButton, &QPushButton::toggled, this,
          &DicomViewer::onPanModeToggled);
  connect(m_zoomButton, &QPushButton::toggled, this,
          &DicomViewer::onZoomModeToggled);
  connect(m_qrGenerateButton, &QPushButton::clicked, this,
          &DicomViewer::onGenerateQr);
  connect(m_qrDecodeImageButton, &QPushButton::clicked, this,
          &DicomViewer::onDecodeQrFromImage);
  connect(m_qrClearButton, &QPushButton::clicked, this,
          &DicomViewer::onClearQr);
  connect(m_qrSaveButton, &QPushButton::clicked, this,
          &DicomViewer::onSaveQrImage);
  connect(m_qrStartCamButton, &QPushButton::clicked, this,
          &DicomViewer::onStartQrCamera);
  connect(m_qrStopCamButton, &QPushButton::clicked, this,
          &DicomViewer::onStopQrCamera);
  connect(m_qrCamTimer, &QTimer::timeout, this, &DicomViewer::onQrCameraTick);
  connect(m_brachyReadButton, &QPushButton::clicked, this,
          &DicomViewer::onReadBrachyPlan);
  connect(m_brachyLoadDataButton, &QPushButton::clicked, this,
          &DicomViewer::onLoadBrachyData);
  connect(m_brachyCalcDoseButton, &QPushButton::clicked, this,
          &DicomViewer::onCalculateBrachyDose);
  connect(m_brachyRandomSourceButton, &QPushButton::clicked, this,
          &DicomViewer::onGenerateRandomSources);
  connect(m_brachyTestSourceButton, &QPushButton::clicked, this,
          &DicomViewer::onGenerateTestSource);
  connect(m_brachyAddEvalPointButton, &QPushButton::clicked, this,
          &DicomViewer::onAddDoseEvaluationPoint);
  connect(m_brachyClearEvalPointsButton, &QPushButton::clicked, this,
          &DicomViewer::onClearDoseEvaluationPoints);
  connect(m_brachyOptimizeButton, &QPushButton::clicked, this,
          &DicomViewer::onOptimizeDwellTimes);
  connect(m_brachyShowRefPointsCheck, &QCheckBox::stateChanged, this,
          &DicomViewer::onShowRefPointsChanged);
  connect(m_structureList, &QListWidget::itemChanged, this,
          &DicomViewer::onStructureVisibilityChanged);
  connect(m_structureAllButton, &QPushButton::clicked, this,
          &DicomViewer::onShowAllStructures);
  connect(m_structureNoneButton, &QPushButton::clicked, this,
          &DicomViewer::onHideAllStructures);
  connect(m_showPointsCheck, &QCheckBox::toggled, this,
          &DicomViewer::onStructurePointsToggled);
  connect(m_structureLineWidthSpin, QOverload<int>::of(&QSpinBox::valueChanged),
          this, &DicomViewer::onStructureLineWidthChanged);
  connect(m_dvhButton, &QPushButton::clicked, this, &DicomViewer::onShowDVH);

  connect(m_doseModeCombo, qOverload<int>(&QComboBox::currentIndexChanged),
          this, [this](int) {
            m_doseCalcMode = static_cast<DoseCalcMode>(
                m_doseModeCombo->currentData().toInt());
            bool phys = m_doseCalcMode == DoseCalcMode::Physical;
            m_doseAlphaBetaSpin->setEnabled(!phys);
            updateDoseShiftLabels();
          });
  connect(m_doseAlphaBetaSpin, qOverload<double>(&QDoubleSpinBox::valueChanged),
          [this](double v) {
            m_doseAlphaBeta = v;
            updateDoseShiftLabels();
          });
  connect(m_doseCalcButton, &QPushButton::clicked, this,
          &DicomViewer::onDoseCalculateClicked);
  connect(m_doseIsosurfaceButton, &QPushButton::clicked, this,
          &DicomViewer::onDoseIsosurfaceClicked);
  connect(m_gammaAnalysisButton, &QPushButton::clicked, this,
          &DicomViewer::onGammaAnalysisClicked);
  connect(m_randomStudyButton, &QPushButton::clicked, this,
          &DicomViewer::onRandomStudyClicked);

  for (int i = 0; i < VIEW_COUNT; ++i)
    updateInteractionButtonVisibility(i);
  updateOverlayInteractionStates();
}

bool DicomViewer::loadDicomFile(const QString &filename) {
  clearFusionPreviewImage();
  if (m_dicomReader->loadDicomFile(filename)) {
    m_ctFilename = QFileInfo(filename).fileName();
    QImage img = m_dicomReader->getImage();
    for (int i = 0; i < VIEW_COUNT; ++i) {
      m_currentIndices[i] = 0;
      m_originalImages[i] = img;
    }
    updateImage();
    updateImageInfo();
    updateInfoOverlays();

    // デフォルトのWindow/Levelを設定
    double window, level;
    m_dicomReader->getWindowLevel(window, level);

    m_windowSlider->setValue(static_cast<int>(window));
    m_levelSlider->setValue(static_cast<int>(level));
    m_windowSpinBox->setValue(static_cast<int>(window));
    m_levelSpinBox->setValue(static_cast<int>(level));
    m_seriesWindowValues = QVector<double>(1, window);
    m_seriesLevelValues = QVector<double>(1, level);
    m_seriesWindowLevelInitialized = QVector<bool>(1, true);
    m_activeImageSeriesIndex = 0;

    emit imageLoaded(filename);
    for (int i = 0; i < VIEW_COUNT; ++i) {
      m_sliceSliders[i]->blockSignals(true);
      if (m_dicomFiles.isEmpty()) {
        m_sliceSliders[i]->setRange(0, 0);
        m_sliceSliders[i]->setValue(0);
        m_sliceSliders[i]->setEnabled(false);
      } else {
        m_sliceSliders[i]->setRange(0, m_dicomFiles.size() - 1);
        m_sliceSliders[i]->setValue(m_currentIndices[i]);
        m_sliceSliders[i]->setEnabled(true);
      }
      m_sliceSliders[i]->blockSignals(false);
    }
    updateSliceLabels();
    if (!m_aiSuppressSourceTracking) {
      m_aiHasDicomSource = true;
      m_aiCurrentDicomSourceIsDirectory = false;
      m_aiCurrentDicomSourcePath = QFileInfo(filename).absoluteFilePath();
    }
    return true;
  }
  return false;
}

void DicomViewer::setFusionPreviewImage(const QImage &image, double spacingX,
                                        double spacingY) {
  if (image.isNull())
    return;
  m_fusionViewImage = image;
  m_fusionSpacingX = spacingX > 0.0 ? spacingX : 1.0;
  m_fusionSpacingY = spacingY > 0.0 ? spacingY : 1.0;
  if (!m_fusionViewActive) {
    m_viewModeBeforeFusion = m_viewMode;
  }
  m_fusionViewActive = true;
  if (m_viewMode != ViewMode::Dual) {
    m_restoreViewModeAfterFusion = true;
    setViewMode(ViewMode::Dual);
  } else {
    m_restoreViewModeAfterFusion = false;
  }
  setViewToImage(1);
  updateImage(1);
  updateSliceLabels();
  updateImageSeriesButtons();
}

void DicomViewer::clearFusionPreviewImage() {
  if (!m_fusionViewActive)
    return;
  m_fusionViewActive = false;
  m_fusionViewImage = QImage();
  m_fusionSpacingX = 1.0;
  m_fusionSpacingY = 1.0;
  if (m_restoreViewModeAfterFusion) {
    ViewMode restoreMode = m_viewModeBeforeFusion;
    m_restoreViewModeAfterFusion = false;
    setViewMode(restoreMode);
  }
  updateImage();
  updateSliceLabels();
  updateImageSeriesButtons();
}

void DicomViewer::loadSlice(int viewIndex, int sliceIndex) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return;
  if (isVolumeLoaded()) {
    loadVolumeSlice(viewIndex, sliceIndex);
    return;
  }
  if (sliceIndex < 0 || sliceIndex >= m_dicomFiles.size())
    return;
  if (m_dicomReader->loadDicomFile(m_dicomFiles[sliceIndex])) {
    m_dicomReader->setWindowLevel(m_windowSlider->value(),
                                  m_levelSlider->value());
    m_originalImages[viewIndex] = m_dicomReader->getImage();
    m_currentIndices[viewIndex] = sliceIndex;
    if (m_showSlicePosition)
      updateImage();
    else
      updateImage(viewIndex);
  }

  // スライス変更時にカーソル情報をリセット
  if (viewIndex >= 0 && viewIndex < VIEW_COUNT) {
    if (m_cursorDoseLabels[viewIndex])
      m_cursorDoseLabels[viewIndex]->hide();
    m_imageWidgets[viewIndex]->clearCursorCross();
  }
}

namespace {
struct FusionSeriesMetadata {
  double window{std::numeric_limits<double>::quiet_NaN()};
  double level{std::numeric_limits<double>::quiet_NaN()};
};

bool loadFusionVolumeFromDirectory(const QString &directory, DicomVolume &volume,
                                   FusionSeriesMetadata *outMeta = nullptr) {
  QDir dir(directory);
  if (!dir.exists())
    return false;

  const QString metaPath = dir.filePath(QStringLiteral("fusion_meta.json"));
  if (!QFile::exists(metaPath))
    return false;

  QFile metaFile(metaPath);
  if (!metaFile.open(QIODevice::ReadOnly))
    return false;

  const QByteArray raw = metaFile.readAll();
  QJsonParseError parseError{};
  const QJsonDocument doc = QJsonDocument::fromJson(raw, &parseError);
  if (parseError.error != QJsonParseError::NoError || !doc.isObject())
    return false;

  const QJsonObject obj = doc.object();
  QString volumeFile =
      obj.value(QStringLiteral("volume_file")).toString().trimmed();
  if (volumeFile.isEmpty())
    volumeFile = QStringLiteral("fusion_volume.bin");

  const QString volumePath = dir.filePath(volumeFile);
  if (!QFile::exists(volumePath))
    return false;
  if (!volume.loadFromFile(volumePath))
    return false;

  if (outMeta) {
    const QJsonValue winVal = obj.value(QStringLiteral("window"));
    const QJsonValue lvlVal = obj.value(QStringLiteral("level"));
    if (winVal.isDouble())
      outMeta->window = winVal.toDouble();
    if (lvlVal.isDouble())
      outMeta->level = lvlVal.toDouble();
  }

  return true;
}
} // namespace

bool DicomViewer::loadDicomDirectory(const QString &directory, bool loadCt,
                                     bool loadRtss, bool loadRtdose,
                                     const QStringList &imageSeries,
                                     const QStringList &modalities,
                                     int activeSeriesIndex) {
  clearFusionPreviewImage();
  auto normalizePath = [](const QString &path) {
    if (path.isEmpty())
      return QString();
    QDir dir(path);
    QString absolute = QDir::cleanPath(dir.absolutePath());
    if (absolute == QLatin1String("."))
      return QString();
    return absolute;
  };

  QStringList seriesDirs;
  QSet<QString> seenDirs;
  if (imageSeries.isEmpty()) {
    QString normalized = normalizePath(directory);
    if (!normalized.isEmpty())
      seriesDirs << normalized;
    else if (!directory.isEmpty())
      seriesDirs << directory;
  } else {
    for (const QString &entry : imageSeries) {
      QString normalized = normalizePath(entry);
      if (normalized.isEmpty())
        continue;
      const QString key = normalized.toLower();
      if (seenDirs.contains(key))
        continue;
      seenDirs.insert(key);
      seriesDirs << normalized;
    }
    QString normalizedDir = normalizePath(directory);
    if (!normalizedDir.isEmpty()) {
      bool exists = false;
      for (const QString &dirEntry : std::as_const(seriesDirs)) {
        if (dirEntry.compare(normalizedDir, Qt::CaseInsensitive) == 0) {
          exists = true;
          break;
        }
      }
      if (!exists)
        seriesDirs.prepend(normalizedDir);
    }
    if (seriesDirs.isEmpty() && !directory.isEmpty()) {
      QString normalized = normalizePath(directory);
      if (!normalized.isEmpty())
        seriesDirs << normalized;
      else
        seriesDirs << directory;
    }
  }

  QStringList seriesModalities = modalities;
  if (seriesModalities.size() < seriesDirs.size())
    seriesModalities.resize(seriesDirs.size());

  int resolvedIndex = activeSeriesIndex;
  if (resolvedIndex < 0 || resolvedIndex >= seriesDirs.size()) {
    QString normalizedDir = normalizePath(directory);
    if (!normalizedDir.isEmpty()) {
      int idx = seriesDirs.indexOf(normalizedDir);
      if (idx >= 0)
        resolvedIndex = idx;
    }
    if (resolvedIndex < 0 || resolvedIndex >= seriesDirs.size())
      resolvedIndex = 0;
  }

  // New study: ensure all previous patient data is completely reset
  clearImage();

  const int primarySeriesIndex = seriesDirs.isEmpty() ? -1 : 0;

  m_imageSeriesDirs = seriesDirs;
  m_imageSeriesModalities = seriesModalities;
  m_activeImageSeriesIndex = resolvedIndex;
  m_primaryImageSeriesIndex =
      (primarySeriesIndex < 0) ? 0 : primarySeriesIndex;
  m_seriesVolumeCache.clear();
  m_seriesVolumeCache.resize(m_imageSeriesDirs.size());
  double defaultWindow = m_windowSlider ? m_windowSlider->value() : 256.0;
  double defaultLevel = m_levelSlider ? m_levelSlider->value() : 128.0;
  m_seriesWindowValues =
      QVector<double>(m_imageSeriesDirs.size(), defaultWindow);
  m_seriesLevelValues =
      QVector<double>(m_imageSeriesDirs.size(), defaultLevel);
  m_seriesWindowLevelInitialized =
      QVector<bool>(m_imageSeriesDirs.size(), false);

  QDir dir(directory);
  QFileInfoList fileInfos = dir.entryInfoList(QDir::Files, QDir::Name);

  QStringList ctFiles;
  QStringList rtDoseFiles;
  QStringList rtStructFiles;

  FusionSeriesMetadata primaryFusionMeta;
  bool primarySeriesIsFusion = false;

  for (const QFileInfo &info : fileInfos) {
    QString path = info.absoluteFilePath();
    DcmFileFormat ff;
    if (ff.loadFile(path.toLocal8Bit().data()).bad()) {
      continue;
    }
    OFString modality;
    if (ff.getDataset()->findAndGetOFString(DCM_Modality, modality).good()) {
      QString mod = QString::fromLatin1(modality.c_str());
      if (mod == "RTDOSE") {
        rtDoseFiles << path;
        continue;
      } else if (mod == "RTSTRUCT") {
        rtStructFiles << path;
        continue;
      }
    }
    ctFiles << path;
  }

  DicomVolume baseVolumeForAlignment;
  bool baseVolumeAvailable = false;
  DicomVolume activeOriginalVolume;
  bool haveActiveOriginal = false;

  bool volOk = false;
  bool loaded = false;
  if (loadCt) {
    struct Entry {
      double fallbackPos;
      QString path;
      QVector3D pos3;
      double loc;
      double sortKey;
    };
    std::vector<Entry> entries;
    QVector3D rowDir, colDir;
    bool orientationSet = false;
    for (const QString &path : ctFiles) {
      double pos = DicomReader::getSlicePosition(path);
      QVector3D pos3;
      double locVal = std::numeric_limits<double>::quiet_NaN();
      DicomReader r;
      if (r.loadDicomFile(path)) {
        double x, y, z;
        if (r.getImagePositionPatient(x, y, z)) {
          pos3 = QVector3D(x, y, z);
        }
        double lv;
        if (r.getImageLocation(lv))
          locVal = lv;
        if (!orientationSet) {
          double r1, r2, r3, c1, c2, c3;
          if (r.getImageOrientationPatient(r1, r2, r3, c1, c2, c3)) {
            rowDir = QVector3D(r1, r2, r3).normalized();
            colDir = QVector3D(c1, c2, c3).normalized();
            orientationSet = true;
          }
        }
      }
      entries.push_back({pos, path, pos3, locVal, 0.0});
    }
    QVector3D sliceDir;
    if (orientationSet)
      sliceDir = QVector3D::crossProduct(rowDir, colDir).normalized();
    for (auto &e : entries) {
      if (orientationSet && !e.pos3.isNull()) {
        e.sortKey = QVector3D::dotProduct(e.pos3, sliceDir);
      } else if (!std::isnan(e.loc)) {
        e.sortKey = e.loc;
      } else {
        e.sortKey = e.fallbackPos;
      }
    }
    std::sort(
        entries.begin(), entries.end(),
        [](const Entry &a, const Entry &b) { return a.sortKey < b.sortKey; });

    m_dicomFiles.clear();
    for (const Entry &e : entries) {
      m_dicomFiles << e.path;
    }
    for (int i = 0; i < VIEW_COUNT; ++i) {
      m_currentIndices[i] = 0;
      m_viewOrientations[i] = DicomVolume::Orientation::Axial;
    }

    volOk = m_volume.loadFromDirectory(directory);
    if (!volOk) {
      FusionSeriesMetadata meta;
      if (loadFusionVolumeFromDirectory(directory, m_volume, &meta)) {
        volOk = true;
        primarySeriesIsFusion = true;
        primaryFusionMeta = meta;
      }
    }
    if (volOk && m_volume.depth() > 0) {
      haveActiveOriginal = true;
      activeOriginalVolume = m_volume;

      if (primarySeriesIndex >= 0 &&
          primarySeriesIndex < seriesDirs.size()) {
        if (primarySeriesIndex == resolvedIndex) {
          baseVolumeForAlignment = m_volume;
          baseVolumeAvailable = true;
        } else {
          const QString baseDir = seriesDirs.at(primarySeriesIndex);
          if (!baseDir.isEmpty()) {
            DicomVolume loadedBase;
            if (loadedBase.loadFromDirectory(baseDir)) {
              baseVolumeForAlignment = loadedBase;
              baseVolumeAvailable = true;
            }
          }
        }
      }

      if (!baseVolumeAvailable) {
        baseVolumeForAlignment = m_volume;
        baseVolumeAvailable = true;
      }

      if (baseVolumeAvailable && haveActiveOriginal &&
          primarySeriesIndex >= 0 && resolvedIndex != primarySeriesIndex) {
        const int sliceCount = baseVolumeForAlignment.depth();
        std::unique_ptr<QProgressDialog> progressDialog;
        if (sliceCount > 0) {
          auto dialog = std::make_unique<QProgressDialog>(
              tr("Image%1 を座標合わせ中...")
                  .arg(resolvedIndex + 1),
              QString(), 0, sliceCount, this);
          dialog->setWindowModality(Qt::ApplicationModal);
          dialog->setCancelButton(nullptr);
          dialog->setMinimumDuration(0);
          dialog->setValue(0);
          progressDialog = std::move(dialog);
        }
        QProgressDialog *progressPtr = progressDialog.get();
        std::function<void(int, int)> progressCallback;
        if (progressPtr) {
          progressCallback = [progressPtr](int done, int total) {
            QMetaObject::invokeMethod(
                progressPtr,
                [progressPtr, done, total]() {
                  if (!progressPtr)
                    return;
                  if (progressPtr->maximum() != total)
                    progressPtr->setMaximum(total);
                  progressPtr->setValue(done);
                },
                Qt::QueuedConnection);
          };
        }
        cv::Mat resampled = resampleVolumeToReference(
            baseVolumeForAlignment, activeOriginalVolume, progressCallback);
        if (progressPtr) {
          progressPtr->setValue(progressPtr->maximum());
          progressPtr->close();
          QCoreApplication::processEvents();
        }
        if (!resampled.empty()) {
          DicomVolume converted;
          if (converted.createFromReference(baseVolumeForAlignment,
                                            resampled)) {
            m_volume = converted;
            invalidateStructureSurfaceCache();
          }
        }
      }
    }
    if (volOk && m_doseLoaded) {
      updateDoseAlignment();
      // Enforce native geometry: zero patientShift before first resample
      m_doseVolume.setPatientShift(QVector3D(0, 0, 0));
      QFuture<bool> future = QtConcurrent::run([this]() {
        return m_resampledDose.resampleFromRTDose(
            m_volume, m_doseVolume,
            [this](int current, int total) {
              emit doseLoadProgress(current, total);
            },
            true);
      });
      while (!future.isFinished()) {
        QApplication::processEvents();
        QThread::yieldCurrentThread();
      }
      bool ok = future.result();
      if (ok) {
        resetDoseRange();
        updateImage();
      }
    }
    if (!m_dicomFiles.isEmpty()) {
      m_dicomReader->loadDicomFile(m_dicomFiles[0]);
      m_ctFilename = QFileInfo(m_dicomFiles[0]).fileName();
      double window, level;
      m_dicomReader->getWindowLevel(window, level);
      m_windowSlider->setValue(static_cast<int>(window));
      m_levelSlider->setValue(static_cast<int>(level));
      m_windowSpinBox->setValue(static_cast<int>(window));
      m_levelSpinBox->setValue(static_cast<int>(level));
      if (m_activeImageSeriesIndex >= 0 &&
          m_activeImageSeriesIndex < m_seriesWindowValues.size())
        m_seriesWindowValues[m_activeImageSeriesIndex] = window;
      if (m_activeImageSeriesIndex >= 0 &&
          m_activeImageSeriesIndex < m_seriesLevelValues.size())
        m_seriesLevelValues[m_activeImageSeriesIndex] = level;
      if (m_activeImageSeriesIndex >= 0 &&
          m_activeImageSeriesIndex < m_seriesWindowLevelInitialized.size())
        m_seriesWindowLevelInitialized[m_activeImageSeriesIndex] = true;
      updateImageInfo();
      updateInfoOverlays();
    }

    if (primarySeriesIsFusion) {
      double window = primaryFusionMeta.window;
      double level = primaryFusionMeta.level;
      if (!std::isfinite(window))
        window = m_windowSlider ? m_windowSlider->value() : 256.0;
      if (!std::isfinite(level))
        level = m_levelSlider ? m_levelSlider->value() : 128.0;

      const int windowInt = static_cast<int>(std::lround(window));
      const int levelInt = static_cast<int>(std::lround(level));

      if (m_windowSlider)
        m_windowSlider->setValue(windowInt);
      if (m_windowSpinBox)
        m_windowSpinBox->setValue(windowInt);
      if (m_levelSlider)
        m_levelSlider->setValue(levelInt);
      if (m_levelSpinBox)
        m_levelSpinBox->setValue(levelInt);

      if (m_activeImageSeriesIndex >= 0 &&
          m_activeImageSeriesIndex < m_seriesWindowValues.size())
        m_seriesWindowValues[m_activeImageSeriesIndex] = window;
      if (m_activeImageSeriesIndex >= 0 &&
          m_activeImageSeriesIndex < m_seriesLevelValues.size())
        m_seriesLevelValues[m_activeImageSeriesIndex] = level;
      if (m_activeImageSeriesIndex >= 0 &&
          m_activeImageSeriesIndex < m_seriesWindowLevelInitialized.size())
        m_seriesWindowLevelInitialized[m_activeImageSeriesIndex] = true;

      updateImageInfo();
      updateInfoOverlays();
    }

    for (int i = 0; i < VIEW_COUNT; ++i) {
      int count = isVolumeLoaded()
                      ? sliceCountForOrientation(m_viewOrientations[i])
                      : m_dicomFiles.size();
      int mid = count > 0 ? count / 2 : 0;
      m_sliceSliders[i]->blockSignals(true);
      m_sliceSliders[i]->setRange(0, count > 0 ? count - 1 : 0);
      m_sliceSliders[i]->setValue(mid);
      m_sliceSliders[i]->setEnabled(count > 0);
      m_sliceSliders[i]->blockSignals(false);
      m_currentIndices[i] = mid;
    }

    if (volOk && isVolumeLoaded()) {
      for (int i = 0; i < VIEW_COUNT; ++i) {
        int count = sliceCountForOrientation(m_viewOrientations[i]);
        int mid = count > 0 ? count / 2 : 0;
        loadSlice(i, mid);
      }
      try {
        double window = m_windowSlider ? m_windowSlider->value() : 0.0;
        double level = m_levelSlider ? m_levelSlider->value() : 0.0;
        struct OriMap {
          DicomVolume::Orientation ori;
          int idx;
        } om[3] = {{DicomVolume::Orientation::Axial, 0},
                   {DicomVolume::Orientation::Sagittal, 1},
                   {DicomVolume::Orientation::Coronal, 2}};
        for (const auto &e : om) {
          int cnt = sliceCountForOrientation(e.ori);
          int mid = cnt > 0 ? cnt / 2 : 0;
          m_orientationImages[e.idx] = m_volume.getSlice(
              e.ori == DicomVolume::Orientation::Axial ? mid : mid, e.ori,
              window, level);
          m_orientationIndices[e.idx] = mid;
        }
      } catch (...) {
      }
      updateSliceLabels();
      updateColorBars();
      updateViewLayout();
      for (int i = 0; i < VIEW_COUNT; ++i) {
        if (m_is3DView[i])
          update3DView(i);
      }
      loaded = true;
    } else if (!m_dicomFiles.isEmpty()) {
      int mid = m_dicomFiles.size() / 2;
      {
        QScopedValueRollback<bool> guard(m_aiSuppressSourceTracking, true);
        loaded = loadDicomFile(m_dicomFiles[mid]);
      }
      for (int i = 0; i < VIEW_COUNT; ++i) {
        m_currentIndices[i] = mid;
        m_sliceSliders[i]->setValue(mid);
      }
      updateSliceLabels();
      updateColorBars();
      updateViewLayout();
    }
  } else {
    // CTを読み込まない場合でもUIをリセット
    m_dicomFiles.clear();
    for (int i = 0; i < VIEW_COUNT; ++i) {
      m_sliceSliders[i]->blockSignals(true);
      m_sliceSliders[i]->setRange(0, 0);
      m_sliceSliders[i]->setValue(0);
      m_sliceSliders[i]->setEnabled(false);
      m_sliceSliders[i]->blockSignals(false);
      m_currentIndices[i] = 0;
    }
    updateSliceLabels();
    updateColorBars();
    updateViewLayout();
    loaded = true;
  }

  if (!m_seriesVolumeCache.empty()) {
    if (!baseVolumeAvailable && isVolumeLoaded()) {
      baseVolumeForAlignment = m_volume;
      baseVolumeAvailable = true;
    }
    if (baseVolumeAvailable && m_primaryImageSeriesIndex >= 0 &&
        m_primaryImageSeriesIndex <
            static_cast<int>(m_seriesVolumeCache.size())) {
      m_seriesVolumeCache[m_primaryImageSeriesIndex].volume =
          baseVolumeForAlignment;
      m_seriesVolumeCache[m_primaryImageSeriesIndex].prepared = true;
    }
    if (isVolumeLoaded() && m_activeImageSeriesIndex >= 0 &&
        m_activeImageSeriesIndex <
            static_cast<int>(m_seriesVolumeCache.size())) {
      m_seriesVolumeCache[m_activeImageSeriesIndex].volume = m_volume;
      m_seriesVolumeCache[m_activeImageSeriesIndex].prepared = true;
    }
  }

  if (loaded) {
    if (loadRtdose && !rtDoseFiles.isEmpty()) {
      loadRTDoseFile(rtDoseFiles.first());
    }
    if (loadRtss && !rtStructFiles.isEmpty()) {
      loadRTStructFile(rtStructFiles.first());
    }
  }

  updateImageSeriesButtons();
  updateSliderPosition();

  if (loaded) {
    m_aiHasDicomSource = true;
    m_aiCurrentDicomSourceIsDirectory = true;
    m_aiCurrentDicomSourcePath =
        QDir(directory).absolutePath();
  }
  return loaded;
}

bool DicomViewer::showExternalImageSeries(const QString &directory,
                                          const QString &modality,
                                          const DicomVolume &volume,
                                          double window, double level) {
  Q_UNUSED(directory)
  Q_UNUSED(modality)
  if (volume.depth() <= 0 || volume.width() <= 0 || volume.height() <= 0)
    return false;

  const double useWindow = window > 0.0
                               ? window
                               : (m_windowSlider ? m_windowSlider->value()
                                                 : 256.0);
  const double useLevel =
      std::isfinite(level)
          ? level
          : (m_levelSlider ? m_levelSlider->value() : 128.0);

  int mid = volume.depth() / 2;
  QImage slice = volume.getSlice(mid, DicomVolume::Orientation::Axial, useWindow,
                                 useLevel);
  if (slice.isNull())
    return false;

  setFusionPreviewImage(slice, volume.spacingX(), volume.spacingY());
  return true;
}

bool DicomViewer::loadRTDoseFile(const QString &filename, bool activate) {
  qDebug() << "Loading RT-Dose:" << filename;

  RTDoseVolume loadedDose;
  QFuture<bool> loadFuture = QtConcurrent::run([&]() {
    return loadedDose.loadFromFile(filename, [this](int current, int total) {
      emit doseLoadProgress(current, total);
    });
  });
  while (!loadFuture.isFinished()) {
    QApplication::processEvents();
    QThread::yieldCurrentThread();
  }
  bool loaded = loadFuture.result();
  if (!loaded)
    return false;

  if (!isVolumeLoaded()) {
    qWarning() << "CT volume not loaded; dose stored but not calculated";
    return false;
  }

  QString displayName = QFileInfo(filename).fileName();

  RTDoseVolume alignedDose = loadedDose;
  bool hasIOP = alignedDose.hasIOP();
  bool adoptIOP = !hasIOP;
  qDebug() << "Dose IOP present?" << hasIOP << ", adopt CT IOP?" << adoptIOP;
  if (adoptIOP)
    alignedDose.adoptOrientationFrom(m_volume);

  QMatrix4x4 id;
  id.setToIdentity();
  alignedDose.setCtToDoseTransform(id);
  alignedDose.setPatientShift(QVector3D(0, 0, 0));

  const RTDoseVolume *resampleSource = nullptr;
  if (activate) {
    m_rtDoseFilename = displayName;
    m_doseVolume = alignedDose;
    updateDoseAlignment();
    // Ensure native shift reset before resample
    m_doseVolume.setPatientShift(QVector3D(0, 0, 0));
    resampleSource = &m_doseVolume;
  } else {
    resampleSource = &alignedDose;
  }

  if (resampleSource) {
    QVector3D shift = resampleSource->patientShift();
    qDebug() << QString("[Dose Align] Final shift: (%1, %2, %3)")
                    .arg(shift.x(), 0, 'f', 2)
                    .arg(shift.y(), 0, 'f', 2)
                    .arg(shift.z(), 0, 'f', 2);
    QVector3D ctCenter =
        m_volume.voxelToPatient(m_volume.width() / 2.0, m_volume.height() / 2.0,
                                m_volume.depth() / 2.0);
    QVector3D doseVoxel =
        resampleSource->patientToVoxelContinuous(ctCenter);
    bool inBounds =
        (doseVoxel.x() >= -0.5 && doseVoxel.x() < resampleSource->width() - 0.5 &&
         doseVoxel.y() >= -0.5 && doseVoxel.y() < resampleSource->height() - 0.5 &&
         doseVoxel.z() >= -0.5 && doseVoxel.z() < resampleSource->depth() - 0.5);
    qDebug() << QString("[Dose Align] CT center -> Dose voxel (%1,%2,%3) | "
                        "in-bounds: %4")
                    .arg(doseVoxel.x(), 0, 'f', 2)
                    .arg(doseVoxel.y(), 0, 'f', 2)
                    .arg(doseVoxel.z(), 0, 'f', 2)
                    .arg(inBounds ? "YES" : "NO");
  }

  DoseResampledVolume resampled;
  QFuture<bool> resampleFuture = QtConcurrent::run([&]() {
    return resampled.resampleFromRTDose(
        m_volume, *resampleSource,
        [this](int c, int t) { emit doseLoadProgress(c, t); }, true);
  });
  while (!resampleFuture.isFinished()) {
    QApplication::processEvents();
    QThread::yieldCurrentThread();
  }
  if (!resampleFuture.result()) {
    qWarning() << "Dose resampling failed";
    return false;
  }

  QListWidgetItem *item = nullptr;
  DoseItemWidget *widget = nullptr;
  if (m_doseListWidget) {
    item = new QListWidgetItem(m_doseListWidget);
    widget = new DoseItemWidget(displayName, resampleSource->maxDose());
    widget->setChecked(activate);
    item->setSizeHint(widget->sizeHint());
    m_doseListWidget->addItem(item);
    m_doseListWidget->setItemWidget(item, widget);

    // Adjust row height when UI expands/collapses
    connect(widget, &DoseItemWidget::uiExpandedChanged, this,
            [this, item, widget]() {
              QTimer::singleShot(0, this, [this, item, widget]() {
                widget->adjustSize();
                item->setSizeHint(widget->sizeHint());
                m_doseListWidget->setItemWidget(item, widget);
                m_doseListWidget->doItemsLayout();
                if (m_doseListWidget->viewport())
                  m_doseListWidget->viewport()->update();
              });
            });

    // 設定変更時はキャッシュ済み線量を無効化し、再計算を要求
    connect(widget, &DoseItemWidget::settingsChanged, this, [this]() {
      m_resampledDose.clear();
      m_doseLoaded = false;
      updateColorBars();
      updateImage();
    });
    connect(widget, &DoseItemWidget::visibilityChanged, this,
            [this](bool) {
              onDoseCalculateClicked();
              updateDoseShiftLabels();
            });

    // Connect save button
    connect(widget, &DoseItemWidget::saveRequested, this, [this, widget]() {
      onDoseSaveRequested(widget);
    });
  }

  if (activate)
    m_resampledDose = resampled;

  RTDoseVolume storedDose = activate ? m_doseVolume : alignedDose;
  DoseItem doseItem;
  doseItem.volume = std::move(resampled);
  doseItem.widget = widget;
  doseItem.dose = storedDose;
  doseItem.isSaved = true; // Loaded from file
  doseItem.savedFilePath = filename;
  if (widget) {
    widget->setSaved(true);
  }
  m_doseItems.push_back(std::move(doseItem));

  if (activate) {
    resetDoseRange();
    m_doseLoaded = true;
    if (m_doseRefSpinBox) {
      double maxDose = m_resampledDose.maxDose();
      m_doseRefSpinBox->setValue(maxDose);
      m_doseReference = maxDose;
    }
    // Calculate and display dose immediately after loading
    onDoseCalculateClicked();
    updateImageInfo();
  }

  updateDoseShiftLabels();
  return true;
}

bool DicomViewer::loadRTStructFile(const QString &filename) {
  qDebug() << "Loading RTSTRUCT:" << filename;
  m_rtstructLoaded =
      m_rtstruct.loadFromFile(filename, [this](int current, int total) {
        emit structureLoadProgress(current, total);
      });
  invalidateStructureSurfaceCache();
  if (m_rtstructLoaded) {
    m_lastRtStructPath = QFileInfo(filename).absoluteFilePath();
    m_structureList->clear();
    QStringList roiNames;
    for (int r = 0; r < m_rtstruct.roiCount(); ++r) {
      QString name = m_rtstruct.roiName(r).trimmed();
      roiNames << name;
      QListWidgetItem *it = new QListWidgetItem(name);
      bool isDefaultHidden = name.compare("Outer Contour", Qt::CaseInsensitive) == 0 ||
                             name.compare("Outour Contour", Qt::CaseInsensitive) == 0 ||
                             name.compare("Body", Qt::CaseInsensitive) == 0;
      it->setCheckState(isDefaultHidden ? Qt::Unchecked : Qt::Checked);
      if (isDefaultHidden) {
        m_rtstruct.setROIVisible(r, false);
      }
      QColor color = QColor::fromHsv((r * 40) % 360, 255, 255);
      QPixmap pix(12, 12);
      pix.fill(color);
      it->setIcon(QIcon(pix));
      m_structureList->addItem(it);
    }
    for (int i = 0; i < VIEW_COUNT; ++i) {
      if (m_dvhWidgets[i])
        m_dvhWidgets[i]->setROINames(roiNames);
      if (m_profileWidgets[i]) {
        m_profileWidgets[i]->setROINames(roiNames);
        if (isVolumeLoaded() && m_doseLoaded)
          m_profileWidgets[i]->setDicomData(&m_volume, &m_resampledDose,
                                            &m_rtstruct);
      }
      if (m_viewContainers[i]->isVisible()) {
        loadSlice(i, m_currentIndices[i]);
      }
    }
  } else {
    m_lastRtStructPath.clear();
    QStringList empty;
    for (int i = 0; i < VIEW_COUNT; ++i) {
      if (m_dvhWidgets[i])
        m_dvhWidgets[i]->setROINames(empty);
      if (m_profileWidgets[i])
        m_profileWidgets[i]->setROINames(empty);
    }
  }
  return m_rtstructLoaded;
}

bool DicomViewer::loadBrachyPlanFile(const QString &filename) {
  m_brachyLoaded = m_brachyPlan.loadFromFile(filename);
  m_brachyListWidget->clear();
  if (m_brachyLoaded) {
    m_brachyListWidget->addItems(m_brachyPlan.dwellTimeStrings());
    for (int i = 0; i < VIEW_COUNT; ++i) {
      if (m_viewContainers[i]->isVisible()) {
        loadSlice(i, m_currentIndices[i]);
      }
    }
    for (int i = 0; i < VIEW_COUNT; ++i) {
      if (m_is3DView[i]) {
        update3DView(i);
      }
    }

    // Enable Calculate button if source data is already loaded
    bool canCalculate = m_brachyLoaded && m_brachyDoseCalc &&
                       m_brachyDoseCalc->isInitialized();
    if (m_brachyCalcDoseButton) {
      m_brachyCalcDoseButton->setEnabled(canCalculate);
    }

    // Enable Add Evaluation Point button if volume is loaded
    if (m_brachyAddEvalPointButton && isVolumeLoaded()) {
      m_brachyAddEvalPointButton->setEnabled(true);
    }

    // Update reference points display
    updateReferencePointsDisplay();
  }
  return m_brachyLoaded;
}

void DicomViewer::onReadBrachyPlan() {
  QString filename = QFileDialog::getOpenFileName(
      this, tr("Open Brachy RT Plan"), QDir::homePath(),
      tr("DICOM Files (*.dcm *.DCM *.dicom *.DICOM);;All Files (*.*)"));
  if (!filename.isEmpty()) {
    if (!loadBrachyPlanFile(filename)) {
      QMessageBox::warning(
          this, tr("Error"),
          QString("Failed to load RT Plan file:\n%1").arg(filename));
    } else {
      // Try to auto-load Ir source data if not already loaded
      if (!m_brachyDoseCalc || !m_brachyDoseCalc->isInitialized()) {
        QSettings settings("ShioRIS3", "ShioRIS3");
        QString savedPath = settings.value("Brachy/IrSourceDataPath").toString();
        if (!savedPath.isEmpty() && QFile::exists(savedPath)) {
          qDebug() << "RT-Plan loaded, auto-loading Ir source data...";
          loadBrachySourceData(savedPath);
        }
      }

      // Update Calculate button state
      bool canCalculate = m_brachyLoaded && m_brachyDoseCalc &&
                         m_brachyDoseCalc->isInitialized();
      if (m_brachyCalcDoseButton) {
        m_brachyCalcDoseButton->setEnabled(canCalculate);
      }
    }
  }
}

void DicomViewer::onLoadBrachyData() {
  QString filename = QFileDialog::getOpenFileName(
      this, tr("Open Ir Source Data"), QDir::homePath(),
      tr("Binary Files (*.bin *.dat);;All Files (*.*)"));

  if (filename.isEmpty()) {
    return;
  }

  if (loadBrachySourceData(filename)) {
    QMessageBox::information(this, tr("Success"),
                            tr("Ir source data loaded successfully"));
  } else {
    QMessageBox::warning(this, tr("Error"),
                        tr("Failed to load Ir source data from:\n%1").arg(filename));
  }
}

bool DicomViewer::loadBrachySourceData(const QString &filename) {
  // Initialize calculator if needed
  if (!m_brachyDoseCalc) {
    m_brachyDoseCalc = std::make_unique<Brachy::BrachyDoseCalculator>();
  }

  // Load source data
  bool success = m_brachyDoseCalc->initialize(filename);

  if (success) {
    // Save path to settings for auto-load next time
    QSettings settings("ShioRIS3", "ShioRIS3");
    settings.setValue("Brachy/IrSourceDataPath", filename);

    if (m_brachyDataStatus) {
      QString displayText = tr("✓ Ir source data loaded\n"
                              "File: %1\n"
                              "Path: %2")
                              .arg(QFileInfo(filename).fileName())
                              .arg(QDir::toNativeSeparators(filename));
      m_brachyDataStatus->setText(displayText);
      setBrachyStatusStyle(QStringLiteral("#00ff00"));
    }

    // Enable Calculate button if plan is also loaded
    bool canCalculate = m_brachyLoaded && m_brachyDoseCalc->isInitialized();
    if (m_brachyCalcDoseButton) {
      m_brachyCalcDoseButton->setEnabled(canCalculate);
    }

    qDebug() << "Ir source data loaded and saved to settings:" << filename;
    return true;
  } else {
    if (m_brachyDataStatus) {
      m_brachyDataStatus->setText(tr("✗ Failed to load source data\nPath: %1")
                                   .arg(QDir::toNativeSeparators(filename)));
      setBrachyStatusStyle(QStringLiteral("#ff0000"));
    }

    qWarning() << "Failed to load Ir source data from:" << filename;
    return false;
  }
}

void DicomViewer::autoLoadBrachySourceData() {
  QSettings settings("ShioRIS3", "ShioRIS3");
  QString savedPath = settings.value("Brachy/IrSourceDataPath").toString();

  if (!savedPath.isEmpty() && QFile::exists(savedPath)) {
    qDebug() << "Auto-loading Ir source data from:" << savedPath;

    if (loadBrachySourceData(savedPath)) {
      qDebug() << "Auto-load successful";
      if (m_brachyDataStatus) {
        // Update status to show it was auto-loaded
        QString currentText = m_brachyDataStatus->text();
        m_brachyDataStatus->setText(currentText + tr("\n(Auto-loaded on startup)"));
      }
    } else {
      qDebug() << "Auto-load failed, user can manually load if needed";
      if (m_brachyDataStatus) {
        m_brachyDataStatus->setText(tr("⚠ Auto-load failed\n"
                                      "Previous file: %1\n"
                                      "Please load source data manually")
                                     .arg(QFileInfo(savedPath).fileName()));
        setBrachyStatusStyle(QStringLiteral("#ffaa00"));
      }
    }
  } else {
    qDebug() << "No saved Ir source data path or file not found";
    if (m_brachyDataStatus && !savedPath.isEmpty()) {
      // File path was saved but file no longer exists
      m_brachyDataStatus->setText(tr("⚠ Previously saved file not found\n"
                                    "Path: %1\n"
                                    "Please load source data")
                                   .arg(QDir::toNativeSeparators(savedPath)));
      setBrachyStatusStyle(QStringLiteral("#ffaa00"));
    }
  }
}

void DicomViewer::onCalculateBrachyDose() {
  if (!m_brachyDoseCalc || !m_brachyDoseCalc->isInitialized()) {
    QMessageBox::warning(this, tr("Error"),
                        tr("Please load Ir source data first"));
    return;
  }

  if (!m_brachyLoaded || m_brachyPlan.sources().isEmpty()) {
    QMessageBox::warning(this, tr("Error"),
                        tr("Please load RT-Plan first"));
    return;
  }

  // Get voxel size
  double voxelSize = 2.0;
  if (m_brachyVoxelSizeSpinBox) {
    voxelSize = m_brachyVoxelSizeSpinBox->value();
  }

  // Disable button and show progress
  if (m_brachyCalcDoseButton) {
    m_brachyCalcDoseButton->setEnabled(false);
  }
  if (m_brachyProgressBar) {
    m_brachyProgressBar->setVisible(true);
    m_brachyProgressBar->setRange(0, 100);
    m_brachyProgressBar->setValue(0);
  }

  // Set CT volume for the calculator (optional)
  if (isVolumeLoaded()) {
    m_brachyDoseCalc->setCtVolume(&m_volume);
  }

  // IMPORTANT: Dose calculation does NOT modify dwell times
  // It uses the current dwell times from the plan to calculate dose distribution only
  // Dwell times are only modified when user explicitly clicks "Optimize Dwell Times" button

  // Capture a copy of the current brachy plan to ensure thread safety
  // This guarantees that the dose calculation uses the current dwell times
  // even if m_brachyPlan is modified on the main thread while calculation is running
  BrachyPlan planCopy = m_brachyPlan;

  // Calculate dose in a separate thread with exception handling
  // NOTE: This is PURE dose calculation - no optimization is performed here
  QFuture<BrachyDoseResult> future = QtConcurrent::run([this, voxelSize, planCopy]() -> BrachyDoseResult {
    BrachyDoseResult result;
    try {
      auto progressCallback = [this](int current, int total) {
        int percent = (current * 100) / total;
        QMetaObject::invokeMethod(this, [this, percent]() {
          if (m_brachyProgressBar) {
            m_brachyProgressBar->setValue(percent);
          }
        }, Qt::QueuedConnection);
      };

      qDebug() << "Starting brachytherapy dose calculation...";
      qDebug() << "NOTE: Dose calculation ONLY - dwell times will NOT be modified";
      qDebug() << "Using plan with" << planCopy.sources().size() << "sources";

      // Calculate normalization factor from reference points
      // This is just a scaling factor for dose display, does NOT change dwell times
      double normalizationFactor = m_brachyDoseCalc->calculateNormalizationFactor(planCopy);

      // Calculate dose with normalization
      // This calculates dose distribution using CURRENT dwell times (unchanged)
      RTDoseVolume doseVolume = m_brachyDoseCalc->calculateVolumeDoseNormalized(
          planCopy, normalizationFactor, voxelSize, {}, progressCallback);

      qDebug() << "Dose calculation returned. Dimensions:"
               << doseVolume.width() << "x" << doseVolume.height() << "x" << doseVolume.depth();

      if (doseVolume.width() > 0 && doseVolume.height() > 0 && doseVolume.depth() > 0) {
        result.success = true;
        result.dose = std::move(doseVolume);
        result.normalizationFactor = normalizationFactor;
        qDebug() << "Dose calculation successful. Max dose:" << result.dose.maxDose();

        // Verify reference point doses
        result.referencePointErrors = m_brachyDoseCalc->verifyReferencePointDoses(
            planCopy, normalizationFactor);

        if (!result.referencePointErrors.isEmpty()) {
          qDebug() << "\n=== Reference Point Dose Error Summary ===";
          for (const auto &error : result.referencePointErrors) {
            qDebug() << QString("%1: Prescribed=%2 Gy, Calculated=%3 Gy, Error=%4 Gy (%5%)")
                          .arg(error.label)
                          .arg(error.prescribedDose, 0, 'f', 3)
                          .arg(error.calculatedDose, 0, 'f', 3)
                          .arg(error.absoluteError, 0, 'f', 3)
                          .arg(error.relativeError, 0, 'f', 2);
          }
          qDebug() << "==========================================\n";
        }
      } else {
        result.success = false;
        result.errorMessage = "Invalid dose volume dimensions";
        qWarning() << result.errorMessage;
      }
    } catch (const std::exception &e) {
      result.success = false;
      result.errorMessage = QString("Exception during dose calculation: %1").arg(e.what());
      qWarning() << result.errorMessage;
    } catch (...) {
      result.success = false;
      result.errorMessage = "Unknown exception during dose calculation";
      qWarning() << result.errorMessage;
    }
    return result;
  });

  // Watch for completion
  auto *watcher = new QFutureWatcher<BrachyDoseResult>(this);
  connect(watcher, &QFutureWatcher<BrachyDoseResult>::finished, this, [this, watcher]() {
    BrachyDoseResult result = watcher->result();
    watcher->deleteLater();

    // Hide progress
    if (m_brachyProgressBar) {
      m_brachyProgressBar->setVisible(false);
    }

    // Re-enable button
    if (m_brachyCalcDoseButton) {
      m_brachyCalcDoseButton->setEnabled(true);
    }

    // Check if calculation succeeded
    if (!result.success) {
      QMessageBox::warning(this, tr("Error"),
                          tr("Dose calculation failed:\n%1").arg(result.errorMessage));
      return;
    }

    RTDoseVolume doseVolume = std::move(result.dose);
    qDebug() << "Brachy dose volume dimensions:" << doseVolume.width()
             << "x" << doseVolume.height() << "x" << doseVolume.depth();
    qDebug() << "Max dose:" << doseVolume.maxDose() << "Gy";

    // Update reference points display with error information if available
    if (!result.referencePointErrors.isEmpty()) {
      updateReferencePointsDisplay(result.referencePointErrors);
    }

    // Load the dose volume
    QString label = tr("Brachy Dose");
    DoseResampledVolume resampledDose;

    if (isVolumeLoaded()) {
      // Resample dose to CT grid
      qDebug() << "Resampling dose to CT grid...";
      bool resampleOk = resampledDose.resampleFromRTDose(
          m_volume, doseVolume,
          nullptr,  // no progress callback needed here
          true      // use native dose geometry
      );

      if (!resampleOk) {
        QMessageBox::warning(this, tr("Error"),
                            tr("Failed to resample dose to CT grid"));
        return;
      }
      qDebug() << "Resampling completed. Max dose:" << resampledDose.maxDose();
    } else {
      // No CT volume - use dose volume directly without resampling
      qDebug() << "No CT volume, using dose directly";
      cv::Mat doseCopy = doseVolume.data().clone();  // Deep copy
      resampledDose.setFromMat(
          doseCopy,
          doseVolume.spacingX(),
          doseVolume.spacingY(),
          doseVolume.spacingZ(),
          doseVolume.originX(),
          doseVolume.originY(),
          doseVolume.originZ()
      );
      resampledDose.updateMaxDose();
      qDebug() << "Dose set directly. Max dose:" << resampledDose.maxDose();
    }

    // Add to dose list
    if (m_doseItems.size() >= 5) {
      QMessageBox::StandardButton reply = QMessageBox::question(
          this, tr("Dose Limit"),
          tr("Maximum 5 dose volumes. Replace oldest?"),
          QMessageBox::Yes | QMessageBox::No);
      if (reply == QMessageBox::No) {
        return;
      }
      // Remove oldest
      if (!m_doseItems.empty() && m_doseListWidget) {
        delete m_doseListWidget->takeItem(0);
        m_doseItems.erase(m_doseItems.begin());
      }
    } else {
      m_doseItems.clear();
    }

    // Deactivate all existing dose items before adding new calculated dose
    for (auto &doseItem : m_doseItems) {
      if (doseItem.widget) {
        doseItem.widget->setChecked(false);
      }
    }

    DoseItemWidget *widget = nullptr;
    QListWidgetItem *item = nullptr;
    if (m_doseListWidget) {
      item = new QListWidgetItem(m_doseListWidget);
      widget = new DoseItemWidget(label, doseVolume.maxDose());
      widget->setChecked(true);  // Ensure dose is checked by default
      item->setSizeHint(widget->sizeHint());
      m_doseListWidget->addItem(item);
      m_doseListWidget->setItemWidget(item, widget);

      connect(widget, &DoseItemWidget::uiExpandedChanged, this,
              [this, item, widget]() {
                QTimer::singleShot(0, this, [this, item, widget]() {
                  if (!m_doseListWidget)
                    return;
                  widget->adjustSize();
                  item->setSizeHint(widget->sizeHint());
                  m_doseListWidget->setItemWidget(item, widget);
                  m_doseListWidget->doItemsLayout();
                  if (m_doseListWidget->viewport())
                    m_doseListWidget->viewport()->update();
                });
              });
      connect(widget, &DoseItemWidget::settingsChanged, this, [this]() {
        m_resampledDose.clear();
        m_doseLoaded = false;
        updateColorBars();
        updateImage();
      });
      connect(widget, &DoseItemWidget::visibilityChanged, this,
              [this](bool) {
                onDoseCalculateClicked();
                updateDoseShiftLabels();
              });

      // Connect save button
      connect(widget, &DoseItemWidget::saveRequested, this, [this, widget]() {
        onDoseSaveRequested(widget);
      });
    }

    DoseItem newItem;
    newItem.volume = resampledDose;
    newItem.widget = widget;
    newItem.dose = doseVolume;
    // isSaved=false by default (newly calculated)
    m_doseItems.push_back(std::move(newItem));

    m_resampledDose = resampledDose;
    m_doseLoaded = true;
    resetDoseRange();
    if (m_doseRefSpinBox) {
      double maxDose = m_resampledDose.maxDose();
      m_doseRefSpinBox->setValue(maxDose);
      m_doseReference = maxDose;
    }

    // Calculate and display dose immediately after calculation
    onDoseCalculateClicked();
    updateImageInfo();
    updateDoseShiftLabels();

    QMessageBox::information(this, tr("Success"),
                            tr("Brachytherapy dose calculation completed\nMax dose: %1 Gy")
                                .arg(doseVolume.maxDose(), 0, 'f', 2));
  });

  watcher->setFuture(future);
}

void DicomViewer::onGenerateRandomSources() {
  if (!m_brachyDoseCalc || !m_brachyDoseCalc->isInitialized()) {
    QMessageBox::warning(this, tr("Error"),
                        tr("Please load Ir source data first"));
    return;
  }

  // Confirm action
  QMessageBox::StandardButton reply = QMessageBox::question(
      this, tr("Generate Random Sources"),
      tr("This will clear the current plan and generate 10 random sources.\n"
         "Continue?"),
      QMessageBox::Yes | QMessageBox::No);

  if (reply == QMessageBox::No) {
    return;
  }

  // Clear current plan
  m_brachyPlan.clearSources();
  m_brachyLoaded = false;

  // Generate random sources
  // Use smaller spatial range (±20mm) for better visualization
  m_brachyPlan.generateRandomSources(
      10,     // count
      20.0,   // spatial range ±20mm (smaller for concentrated dose)
      5.0,    // min dwell time
      15.0    // max dwell time
  );

  // Update UI
  m_brachyListWidget->clear();
  m_brachyListWidget->addItems(m_brachyPlan.dwellTimeStrings());

  // DEBUG: Log generated sources
  qDebug() << "=== Generated" << m_brachyPlan.sources().size() << "random sources ===";
  for (int i = 0; i < m_brachyPlan.sources().size() && i < 3; ++i) {
    const auto &src = m_brachyPlan.sources()[i];
    qDebug() << "Source" << i << ": pos=" << src.position()
             << ", dir=" << src.direction()
             << ", dwell=" << src.dwellTime() << "s";
  }

  // Set loaded flag
  m_brachyLoaded = true;

  // Enable calculate button
  if (m_brachyCalcDoseButton) {
    m_brachyCalcDoseButton->setEnabled(true);
  }

  // Refresh views to show source positions
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_viewContainers[i]->isVisible()) {
      loadSlice(i, m_currentIndices[i]);
    }
  }
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_is3DView[i]) {
      update3DView(i);
    }
  }

  QMessageBox::information(this, tr("Success"),
                          tr("Generated 10 random sources.\n"
                             "Click 'Calculate Dose' to compute dose distribution."));
}

void DicomViewer::onGenerateTestSource() {
  if (!m_brachyDoseCalc || !m_brachyDoseCalc->isInitialized()) {
    QMessageBox::warning(this, tr("Error"),
                        tr("Please load Ir source data first"));
    return;
  }

  // Confirm action
  QMessageBox::StandardButton reply = QMessageBox::question(
      this, tr("Generate Test Source"),
      tr("This will clear the current plan and generate a single test source at origin (0,0,0).\n"
         "The dose distribution should be symmetric.\n"
         "Continue?"),
      QMessageBox::Yes | QMessageBox::No);

  if (reply == QMessageBox::No) {
    return;
  }

  // Generate test source at origin
  m_brachyPlan.generateTestSourceAtOrigin();
  m_brachyLoaded = true;

  // Update UI
  m_brachyListWidget->clear();
  m_brachyListWidget->addItems(m_brachyPlan.dwellTimeStrings());

  // Enable calculate button
  if (m_brachyCalcDoseButton) {
    m_brachyCalcDoseButton->setEnabled(true);
  }

  // Refresh views to show source position
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_viewContainers[i]->isVisible()) {
      loadSlice(i, m_currentIndices[i]);
    }
  }
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_is3DView[i]) {
      update3DView(i);
    }
  }

  QMessageBox::information(this, tr("Success"),
                          tr("Generated test source at origin (0,0,0) with Z-axis direction.\n"
                             "Dose distribution should be symmetric (circular in XY plane).\n"
                             "Click 'Calculate Dose' to verify alignment."));
}

void DicomViewer::onAddDoseEvaluationPoint() {
  if (!isVolumeLoaded()) {
    QMessageBox::warning(this, tr("Error"), tr("Please load a CT volume first"));
    return;
  }

  // Get active view
  int viewIndex = m_activeViewIndex;

  // Use the last displayed patient coordinates from the coordinate label
  QVector3D cursorPos = m_lastPatientCoordinates[viewIndex];

  // Check if the position is valid (coordinates are being displayed)
  if (std::isnan(cursorPos.x()) || std::isnan(cursorPos.y()) || std::isnan(cursorPos.z())) {
    QMessageBox::warning(this, tr("Error"),
                        tr("No coordinates available.\n"
                           "Please move the mouse cursor over the image to see coordinates."));
    return;
  }

  // Ask user for target dose
  bool ok;
  double targetDose = QInputDialog::getDouble(
      this, tr("Target Dose"),
      tr("Enter target dose at this point (Gy):"),
      10.0,  // default value
      0.0,   // min
      1000.0, // max
      2,     // decimals
      &ok
  );

  if (!ok) {
    return;
  }

  // Ask for weight (optional)
  double weight = QInputDialog::getDouble(
      this, tr("Point Weight"),
      tr("Enter optimization weight for this point:"),
      1.0,   // default
      0.01,  // min
      100.0, // max
      2,     // decimals
      &ok
  );

  if (!ok) {
    weight = 1.0;
  }

  // Create evaluation point
  DoseEvaluationPoint point(cursorPos, targetDose, weight);
  point.setLabel(QString("Point %1").arg(m_brachyPlan.evaluationPoints().size() + 1));

  // Add to plan
  m_brachyPlan.addEvaluationPoint(point);

  // Update UI
  QString itemText = QString("%1: (%2, %3, %4) - Target: %5 Gy, Weight: %6")
                        .arg(point.label())
                        .arg(cursorPos.x(), 0, 'f', 1)
                        .arg(cursorPos.y(), 0, 'f', 1)
                        .arg(cursorPos.z(), 0, 'f', 1)
                        .arg(targetDose, 0, 'f', 2)
                        .arg(weight, 0, 'f', 2);
  m_brachyEvalPointsList->addItem(itemText);

  // Enable optimize button if we have sources
  if (m_brachyLoaded && m_brachyPlan.sources().size() > 0) {
    m_brachyOptimizeButton->setEnabled(true);
  }

  qDebug() << "Added dose evaluation point:" << itemText;
}

void DicomViewer::onClearDoseEvaluationPoints() {
  m_brachyPlan.clearEvaluationPoints();
  m_brachyEvalPointsList->clear();
  m_brachyOptimizeButton->setEnabled(false);
  qDebug() << "Cleared all dose evaluation points";
}

void DicomViewer::onOptimizeDwellTimes() {
  // IMPORTANT: This function is ONLY called when user explicitly clicks "Optimize Dwell Times" button
  // It WILL modify the dwell times to achieve target doses at evaluation points
  // WARNING: This optimization does NOT preserve original dwell time ratios
  //          Each source's dwell time is independently optimized

  if (!m_brachyLoaded || m_brachyPlan.sources().isEmpty()) {
    QMessageBox::warning(this, tr("Error"), tr("Please load a brachytherapy plan first"));
    return;
  }

  if (m_brachyPlan.evaluationPoints().isEmpty()) {
    QMessageBox::warning(this, tr("Error"), tr("Please add dose evaluation points first"));
    return;
  }

  if (!m_brachyDoseCalc || !m_brachyDoseCalc->isInitialized()) {
    QMessageBox::warning(this, tr("Error"), tr("Please load Ir source data first"));
    return;
  }

  // Create optimizer if not exists
  if (!m_brachyOptimizer) {
    m_brachyOptimizer = std::make_unique<Brachy::DwellTimeOptimizer>(m_brachyDoseCalc.get());
  }

  // Set up optimization settings
  Brachy::DwellTimeOptimizer::OptimizationSettings settings;
  settings.maxIterations = m_brachyOptimizationIterations->value();
  settings.convergenceTolerance = m_brachyOptimizationTolerance->value();
  settings.minDwellTime = 0.0;
  settings.maxDwellTime = 100.0;
  settings.learningRate = 0.5;

  // Set progress callback
  m_brachyOptimizer->setProgressCallback([this](int iter, int maxIter, double error) {
    // Update progress bar
    m_brachyProgressBar->setVisible(true);
    m_brachyProgressBar->setRange(0, maxIter);
    m_brachyProgressBar->setValue(iter);
    QApplication::processEvents();
  });

  // Disable buttons during optimization
  m_brachyOptimizeButton->setEnabled(false);
  m_brachyCalcDoseButton->setEnabled(false);

  // Run optimization
  qDebug() << "Starting dwell time optimization...";
  auto result = m_brachyOptimizer->optimize(m_brachyPlan, m_brachyPlan.evaluationPoints(), settings);

  // Hide progress bar
  m_brachyProgressBar->setVisible(false);

  // Re-enable buttons
  m_brachyOptimizeButton->setEnabled(true);
  if (m_brachyDoseCalc && m_brachyDoseCalc->isInitialized()) {
    m_brachyCalcDoseButton->setEnabled(true);
  }

  // Check result
  if (!result.converged && result.iterations >= settings.maxIterations) {
    QMessageBox::warning(this, tr("Optimization"),
                        tr("Optimization did not converge within maximum iterations.\n"
                           "Initial error: %1 Gy\n"
                           "Final error: %2 Gy\n"
                           "Consider increasing max iterations or adjusting tolerance.")
                        .arg(result.initialError, 0, 'f', 4)
                        .arg(result.finalError, 0, 'f', 4));
  } else {
    QMessageBox::information(this, tr("Optimization Complete"),
                            tr("Dwell time optimization completed successfully!\n\n"
                               "Iterations: %1\n"
                               "Initial RMS error: %2 Gy\n"
                               "Final RMS error: %3 Gy\n"
                               "Improvement: %4 Gy")
                            .arg(result.iterations)
                            .arg(result.initialError, 0, 'f', 4)
                            .arg(result.finalError, 0, 'f', 4)
                            .arg(result.initialError - result.finalError, 0, 'f', 4));
  }

  // Update plan with optimized dwell times
  qDebug() << "=== Updating Plan with Optimized Dwell Times ===";
  qDebug() << "Number of optimized dwell times:" << result.optimizedDwellTimes.size();
  for (int i = 0; i < qMin(10, result.optimizedDwellTimes.size()); ++i) {
    qDebug() << QString("  Optimized dwell time [%1] = %2 s").arg(i).arg(result.optimizedDwellTimes[i], 0, 'f', 3);
  }
  if (result.optimizedDwellTimes.size() > 10) {
    qDebug() << QString("  ... (%1 more dwell times)").arg(result.optimizedDwellTimes.size() - 10);
  }
  m_brachyPlan.setDwellTimes(result.optimizedDwellTimes);
  qDebug() << "Dwell times updated in plan";
  qDebug() << "==============================================";

  // Update UI to show new dwell times
  m_brachyListWidget->clear();
  m_brachyListWidget->addItems(m_brachyPlan.dwellTimeStrings());

  // Update evaluation points list with calculated doses
  auto updatedPoints = m_brachyOptimizer->calculateDosesAtPoints(
      m_brachyPlan, result.optimizedDwellTimes, m_brachyPlan.evaluationPoints());

  m_brachyEvalPointsList->clear();
  for (int i = 0; i < updatedPoints.size(); ++i) {
    const auto& pt = updatedPoints[i];
    QString itemText = QString("%1: (%2, %3, %4) - Target: %5 Gy, Calculated: %6 Gy (Error: %7%)")
                          .arg(pt.label())
                          .arg(pt.position().x(), 0, 'f', 1)
                          .arg(pt.position().y(), 0, 'f', 1)
                          .arg(pt.position().z(), 0, 'f', 1)
                          .arg(pt.targetDose(), 0, 'f', 2)
                          .arg(pt.calculatedDose(), 0, 'f', 2)
                          .arg(pt.relativeError() * 100.0, 0, 'f', 1);
    m_brachyEvalPointsList->addItem(itemText);
  }

  qDebug() << "Dwell time optimization complete";
}

void DicomViewer::updateReferencePointsDisplay(
    const QVector<Brachy::ReferencePointError> &errors) {
  if (!m_brachyRefPointsList) {
    return;
  }

  m_brachyRefPointsList->clear();

  if (!m_brachyLoaded) {
    return;
  }

  const auto& refPoints = m_brachyPlan.referencePoints();
  if (refPoints.isEmpty()) {
    m_brachyRefPointsList->addItem(tr("(No reference points in plan)"));
    return;
  }

  // Check if we have error information
  bool hasErrors = !errors.isEmpty() && errors.size() == refPoints.size();

  for (int i = 0; i < refPoints.size(); ++i) {
    const auto& rp = refPoints[i];

    // Always display basic information: coordinates and prescribed dose
    QString itemText = QString("Ref %1: (%2, %3, %4) - %5 Gy")
                          .arg(i + 1)
                          .arg(rp.position.x(), 0, 'f', 1)
                          .arg(rp.position.y(), 0, 'f', 1)
                          .arg(rp.position.z(), 0, 'f', 1)
                          .arg(rp.prescribedDose, 0, 'f', 2);

    if (!rp.label.isEmpty()) {
      itemText += QString(" [%1]").arg(rp.label);
    }

    // Append error information if available
    if (hasErrors) {
      const auto& error = errors[i];
      itemText += QString(" | Calc=%1 Gy, Err=%2 Gy (%3%)")
                      .arg(error.calculatedDose, 0, 'f', 2)
                      .arg(error.absoluteError, 0, 'f', 3)
                      .arg(error.relativeError, 0, 'f', 1);
    }

    m_brachyRefPointsList->addItem(itemText);
  }

  qDebug() << "Reference points display updated:" << refPoints.size() << "points"
           << (hasErrors ? "with error info" : "without error info");
}

void DicomViewer::showReferencePointErrorDialog(
    const QVector<Brachy::ReferencePointError> &errors,
    double normalizationFactor) {

  // Create dialog
  QDialog *dialog = new QDialog(this);
  dialog->setWindowTitle(tr("Reference Point Dose Verification"));
  dialog->resize(900, 400);

  QVBoxLayout *layout = new QVBoxLayout(dialog);

  // Add information label
  QLabel *infoLabel = new QLabel(
      tr("Normalization factor: %1\n"
         "Dose values after normalization:").arg(normalizationFactor, 0, 'f', 6),
      dialog);
  layout->addWidget(infoLabel);

  // Create table widget
  QTableWidget *table = new QTableWidget(dialog);
  table->setColumnCount(6);
  table->setRowCount(errors.size());

  // Set headers
  QStringList headers;
  headers << tr("Label") << tr("Position (mm)") << tr("Prescribed (Gy)")
          << tr("Calculated (Gy)") << tr("Abs Error (Gy)") << tr("Rel Error (%)");
  table->setHorizontalHeaderLabels(headers);

  // Populate table
  for (int i = 0; i < errors.size(); ++i) {
    const auto &error = errors[i];

    // Label
    QTableWidgetItem *labelItem = new QTableWidgetItem(error.label);
    labelItem->setFlags(labelItem->flags() & ~Qt::ItemIsEditable);
    table->setItem(i, 0, labelItem);

    // Position
    QString posStr = QString("(%1, %2, %3)")
                         .arg(error.position.x(), 0, 'f', 1)
                         .arg(error.position.y(), 0, 'f', 1)
                         .arg(error.position.z(), 0, 'f', 1);
    QTableWidgetItem *posItem = new QTableWidgetItem(posStr);
    posItem->setFlags(posItem->flags() & ~Qt::ItemIsEditable);
    table->setItem(i, 1, posItem);

    // Prescribed dose
    QTableWidgetItem *prescribedItem = new QTableWidgetItem(
        QString::number(error.prescribedDose, 'f', 3));
    prescribedItem->setFlags(prescribedItem->flags() & ~Qt::ItemIsEditable);
    table->setItem(i, 2, prescribedItem);

    // Calculated dose
    QTableWidgetItem *calculatedItem = new QTableWidgetItem(
        QString::number(error.calculatedDose, 'f', 3));
    calculatedItem->setFlags(calculatedItem->flags() & ~Qt::ItemIsEditable);
    table->setItem(i, 3, calculatedItem);

    // Absolute error
    QTableWidgetItem *absErrorItem = new QTableWidgetItem(
        QString::number(error.absoluteError, 'f', 4));
    absErrorItem->setFlags(absErrorItem->flags() & ~Qt::ItemIsEditable);
    // Color code based on error magnitude
    if (std::abs(error.absoluteError) > 0.1) {
      absErrorItem->setBackground(QBrush(QColor(255, 200, 200))); // Light red
    } else if (std::abs(error.absoluteError) > 0.05) {
      absErrorItem->setBackground(QBrush(QColor(255, 255, 200))); // Light yellow
    } else {
      absErrorItem->setBackground(QBrush(QColor(200, 255, 200))); // Light green
    }
    table->setItem(i, 4, absErrorItem);

    // Relative error
    QTableWidgetItem *relErrorItem = new QTableWidgetItem(
        QString::number(error.relativeError, 'f', 2));
    relErrorItem->setFlags(relErrorItem->flags() & ~Qt::ItemIsEditable);
    // Color code based on error percentage
    if (std::abs(error.relativeError) > 2.0) {
      relErrorItem->setBackground(QBrush(QColor(255, 200, 200))); // Light red
    } else if (std::abs(error.relativeError) > 1.0) {
      relErrorItem->setBackground(QBrush(QColor(255, 255, 200))); // Light yellow
    } else {
      relErrorItem->setBackground(QBrush(QColor(200, 255, 200))); // Light green
    }
    table->setItem(i, 5, relErrorItem);
  }

  // Adjust column widths
  table->resizeColumnsToContents();
  table->horizontalHeader()->setStretchLastSection(true);

  layout->addWidget(table);

  // Add close button
  QPushButton *closeButton = new QPushButton(tr("Close"), dialog);
  connect(closeButton, &QPushButton::clicked, dialog, &QDialog::accept);
  layout->addWidget(closeButton);

  // Show dialog
  dialog->exec();
  dialog->deleteLater();
}

void DicomViewer::onShowRefPointsChanged(int state) {
  // Update visualization when checkbox state changes
  bool show = (state == Qt::Checked);

  // Update all visible views
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_viewContainers[i]->isVisible()) {
      if (m_is3DView[i]) {
        update3DView(i);
      } else {
        loadSlice(i, m_currentIndices[i]);
      }
    }
  }

  qDebug() << "Reference points visualization:" << (show ? "enabled" : "disabled");
}

void DicomViewer::showNextImage() {
  if (!isVolumeLoaded() && m_dicomFiles.isEmpty())
    return;
  int i = m_activeViewIndex; // アクティブビューのみ更新
  int count = isVolumeLoaded() ? sliceCountForOrientation(m_viewOrientations[i])
                               : m_dicomFiles.size();
  if (i >= 0 && i < VIEW_COUNT && m_currentIndices[i] + 1 < count) {
    loadSlice(i, m_currentIndices[i] + 1);
    m_sliceSliders[i]->blockSignals(true);
    m_sliceSliders[i]->setValue(m_currentIndices[i]);
    m_sliceSliders[i]->blockSignals(false);
    updateSliceLabels();
  }
}

void DicomViewer::showPreviousImage() {
  if (!isVolumeLoaded() && m_dicomFiles.isEmpty())
    return;
  int i = m_activeViewIndex; // アクティブビューのみ更新
  int count = isVolumeLoaded() ? sliceCountForOrientation(m_viewOrientations[i])
                               : m_dicomFiles.size();
  if (i >= 0 && i < VIEW_COUNT && m_currentIndices[i] > 0) {
    loadSlice(i, m_currentIndices[i] - 1);
    m_sliceSliders[i]->blockSignals(true);
    m_sliceSliders[i]->setValue(m_currentIndices[i]);
    m_sliceSliders[i]->blockSignals(false);
    updateSliceLabels();
  }
}

void DicomViewer::clearImage() {
  // Also reset study-level state to avoid stale DVH/RTSTRUCT tasks
  resetStudyState();
  clearFusionPreviewImage();

  for (int i = 0; i < VIEW_COUNT; ++i) {
    m_imageWidgets[i]->setImage(QImage());
    m_imageWidgets[i]->setStructureLines(StructureLineList());
    m_imageWidgets[i]->setStructurePoints(StructurePointList());
    m_panOffsets[i] = QPointF(0.0, 0.0);
    m_imageWidgets[i]->setPan(m_panOffsets[i]);
  }
  for (int i = 0; i < VIEW_COUNT; ++i) {
    m_originalImages[i] = QImage();
  }
  m_volume = DicomVolume();
  invalidateStructureSurfaceCache();
  m_resampledDose.clear();
  m_dicomFiles.clear();
  m_ctFilename.clear();
  m_rtDoseFilename.clear();
  m_doseLoaded = false;
  m_doseVisible = false;
  if (m_doseListWidget) {
    m_doseListWidget->clear();
  }
  for (int i = 0; i < VIEW_COUNT; ++i) {
    m_currentIndices[i] = 0;
    m_viewOrientations[i] = DicomVolume::Orientation::Axial;
  }
  for (int i = 0; i < VIEW_COUNT; ++i) {
    m_sliceSliders[i]->blockSignals(true);
    m_sliceSliders[i]->setRange(0, 0);
    m_sliceSliders[i]->setValue(0);
    m_sliceSliders[i]->setEnabled(false);
    m_sliceSliders[i]->blockSignals(false);
    m_sliceIndexLabels[i]->hide();
    m_infoOverlays[i]->hide();
    if (m_coordLabels[i])
      m_coordLabels[i]->hide();
    if (m_cursorDoseLabels[i])
      m_cursorDoseLabels[i]->hide();
    m_imageWidgets[i]->clearCursorCross();
  }

  // 情報をクリア
  m_infoTextBox->setPlainText("Patient: -\n"
                              "Modality: -\n"
                              "Study Date: -\n"
                              "Size: -\n"
                              "Study Desc: -\n"
                              "Slice Thk: -\n"
                              "Pixel Spacing: - x -\n"
                              "CT File: -\n\n"
                              "RT Dose: Not Loaded");

  m_imageSeriesDirs.clear();
  m_imageSeriesModalities.clear();
  m_activeImageSeriesIndex = 0;
  m_primaryImageSeriesIndex = 0;
  m_seriesVolumeCache.clear();
  m_seriesWindowValues.clear();
  m_seriesLevelValues.clear();
  m_seriesWindowLevelInitialized.clear();
  updateImageSeriesButtons();
  updateSliderPosition();

  // Reset brachytherapy data
  m_brachyPlan.clearSources();
  m_brachyPlan.clearEvaluationPoints();
  m_brachyPlan.clearReferencePoints();
  m_brachyLoaded = false;
  if (m_brachyListWidget) {
    m_brachyListWidget->clear();
  }
  if (m_brachyEvalPointsList) {
    m_brachyEvalPointsList->clear();
  }
  if (m_brachyRefPointsList) {
    m_brachyRefPointsList->clear();
  }
  if (m_brachyProgressBar) {
    m_brachyProgressBar->setVisible(false);
    m_brachyProgressBar->setValue(0);
  }

  // Reset dose volume and shift
  m_doseVolume = RTDoseVolume();
  m_doseShift = QVector3D(0.0, 0.0, 0.0);

  // Reset 3D view widgets and profile widgets
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_3dWidgets[i]) {
      // Clear all 3D visualization data
      m_3dWidgets[i]->setStructureLines(StructureLine3DList());
      m_3dWidgets[i]->setActiveSourcePoints(QVector<QVector3D>());
      m_3dWidgets[i]->setInactiveSourcePoints(QVector<QVector3D>());
      m_3dWidgets[i]->setActiveSourceSegments(QVector<QPair<QVector3D, QVector3D>>());
      m_3dWidgets[i]->setInactiveSourceSegments(QVector<QPair<QVector3D, QVector3D>>());
      m_3dWidgets[i]->setDoseIsosurfaces(QVector<DoseIsosurface>());
      m_3dWidgets[i]->setStructureSurfaces(QVector<StructureSurface>());
      // Reset slices to empty images
      m_3dWidgets[i]->setSlices(QImage(), 0, QImage(), 0, QImage(), 0,
                                 1, 1, 1, 1.0, 1.0, 1.0);
    }
    if (m_profileWidgets[i]) {
      // Clear all dose profile data
      m_profileWidgets[i]->setROINames(QStringList());
      m_profileWidgets[i]->setDicomData(nullptr, nullptr, nullptr);
      m_profileWidgets[i]->setProfile(QVector<double>(), QVector<double>(),
                                       QVector<DoseProfileWindow::Segment>());
    }
  }

  // Window/Level調整を停止
  stopWindowLevelDrag();
  updateColorBars();
}

void DicomViewer::resetStudyState() {
  // Cancel and delete any pending DVH computations
  if (!m_dvhWatchers.isEmpty()) {
    for (auto it = m_dvhWatchers.begin(); it != m_dvhWatchers.end(); ++it) {
      if (it.value()) {
        // Disconnect to avoid invoking finished handlers for old study
        it.value()->disconnect(this);
        it.value()->deleteLater();
      }
    }
    m_dvhWatchers.clear();
  }

  // Clear DVH data and reset widgets
  m_dvhData.clear();
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_dvhWidgets[i]) {
      // Clear ROI list and plotted data
      m_dvhWidgets[i]->setROINames(QStringList());
      m_dvhWidgets[i]->setDVHData({});
      m_dvhWidgets[i]->setPatientInfo(QString());
      m_dvhWidgets[i]->setPrescriptionDose(0.0);
    }
  }

  // Reset RTSTRUCT and related UI lists
  m_rtstruct = RTStructureSet();
  m_rtstructLoaded = false;
  m_lastRtStructPath.clear();
  invalidateStructureSurfaceCache();
  if (m_structureList) {
    m_structureList->clear();
  }

  // Reset dose items list and flags; resampled dose will be reset by callers
  if (m_doseListWidget) {
    m_doseListWidget->clear();
  }
  m_doseItems.clear();

  m_showDoseGuide = false;
  if (m_doseGuideCheck) {
    QSignalBlocker blocker(m_doseGuideCheck);
    m_doseGuideCheck->setChecked(false);
  }
  if (!m_aiSuppressSourceTracking) {
    m_aiHasDicomSource = false;
    m_aiCurrentDicomSourceIsDirectory = false;
    m_aiCurrentDicomSourcePath.clear();
  }
}

void DicomViewer::onWindowChanged(int value) {
  if (!m_windowLevelDragActive) {
    m_windowSpinBox->setValue(value);
    setWindowLevel(value, m_levelSlider->value());
    emit windowLevelChanged(value, m_levelSlider->value());
  }
}

void DicomViewer::onLevelChanged(int value) {
  if (!m_windowLevelDragActive) {
    m_levelSpinBox->setValue(value);
    setWindowLevel(m_windowSlider->value(), value);
    emit windowLevelChanged(m_windowSlider->value(), value);
  }
}

void DicomViewer::onZoomIn() { setZoomFactor(m_zoomFactor * ZOOM_STEP); }

void DicomViewer::onZoomOut() { setZoomFactor(m_zoomFactor / ZOOM_STEP); }

void DicomViewer::onResetZoom() {
  setZoomFactor(DEFAULT_ZOOM);

  // 全ビューのパンをリセット
  for (int i = 0; i < VIEW_COUNT; ++i) {
    m_panOffsets[i] = QPointF(0.0, 0.0);
    m_imageWidgets[i]->setPan(m_panOffsets[i]);
  }
}

void DicomViewer::onFitToWindow() {
  if (m_originalImages[0].isNull())
    return;

  QSize availableSize;
  if (m_viewScrollAreas[m_activeViewIndex])
    availableSize = m_viewScrollAreas[m_activeViewIndex]->viewport()->size();
  else
    availableSize = m_scrollArea->viewport()->size();
  QSize imageSize = m_originalImages[m_activeViewIndex].size();

  double scaleX =
      static_cast<double>(availableSize.width()) / imageSize.width();
  double scaleY =
      static_cast<double>(availableSize.height()) / imageSize.height();

  setZoomFactor(qMin(scaleX, scaleY));

  // 全ビューのパンをリセット
  for (int i = 0; i < VIEW_COUNT; ++i) {
    m_panOffsets[i] = QPointF(0.0, 0.0);
    m_imageWidgets[i]->setPan(m_panOffsets[i]);
  }
}

void DicomViewer::onWindowLevelButtonClicked() {
  if (m_windowLevelButton->isChecked()) {
    // disable other modes
    m_panButton->setChecked(false);
    m_zoomButton->setChecked(false);
    m_panMode = false;
    m_zoomMode = false;
    startWindowLevelDrag();
  } else {
    stopWindowLevelDrag();
  }
}

void DicomViewer::onWindowLevelTimeout() { stopWindowLevelDrag(); }

void DicomViewer::onPanTimeout() { m_panButton->setChecked(false); }

void DicomViewer::onZoomTimeout() { m_zoomButton->setChecked(false); }

void DicomViewer::onPanModeToggled(bool checked) {
  if (checked) {
    m_zoomButton->setChecked(false);
    m_windowLevelButton->setChecked(false);
    if (m_windowLevelDragActive)
      stopWindowLevelDrag();
    m_zoomMode = false;
    m_panMode = true;
    m_panTimer->start(WINDOW_LEVEL_TIMEOUT);
    setCursor(Qt::OpenHandCursor);
  } else {
    m_panMode = false;
    m_panDragActive = false;
    m_rotationDragActive = false;
    m_panTimer->stop();
    if (!m_zoomMode && !m_windowLevelDragActive)
      setCursor(Qt::ArrowCursor);
  }
  updateOverlayInteractionStates();
}

void DicomViewer::onZoomModeToggled(bool checked) {
  if (checked) {
    m_panButton->setChecked(false);
    m_windowLevelButton->setChecked(false);
    if (m_windowLevelDragActive)
      stopWindowLevelDrag();
    m_panMode = false;
    m_zoomMode = true;
    m_zoomTimer->start(WINDOW_LEVEL_TIMEOUT);
    setCursor(Qt::OpenHandCursor);
  } else {
    m_zoomMode = false;
    m_zoomDragActive = false;
    m_rotationDragActive = false;
    m_zoomTimer->stop();
    if (!m_panMode && !m_windowLevelDragActive)
      setCursor(Qt::ArrowCursor);
  }
  updateOverlayInteractionStates();
}

void DicomViewer::wheelEvent(QWheelEvent *event) {
  if (m_structureList && m_structureList->hasFocus()) {
    event->accept();
    return;
  }
  if (m_zoomMode) {
    const double scaleFactor = 1.15;
    if (event->angleDelta().y() > 0)
      setZoomFactor(m_zoomFactor * scaleFactor);
    else
      setZoomFactor(m_zoomFactor / scaleFactor);
    m_zoomTimer->start(WINDOW_LEVEL_TIMEOUT);
    event->accept();
  } else if (event->modifiers() & Qt::ControlModifier) {
    // Ctrl + ホイールでズーム
    const double scaleFactor = 1.15;
    if (event->angleDelta().y() > 0) {
      setZoomFactor(m_zoomFactor * scaleFactor);
    } else {
      setZoomFactor(m_zoomFactor / scaleFactor);
    }
    event->accept();
  } else if (isVolumeLoaded() || !m_dicomFiles.isEmpty()) {
    m_activeViewIndex =
        viewIndexFromGlobalPos(event->globalPosition().toPoint());
    if (event->angleDelta().y() > 0) {
      showPreviousImage();
    } else if (event->angleDelta().y() < 0) {
      showNextImage();
    }
    event->accept();
  } else {
    // Prevent the scroll areas from scrolling when pan mode is disabled
    event->accept();
  }
}

void DicomViewer::mousePressEvent(QMouseEvent *event) {
  m_activeViewIndex = viewIndexFromGlobalPos(event->globalPosition().toPoint());
  setFocus();

  if (m_selectingProfileLine && event->button() == Qt::LeftButton) {
    int view = m_activeViewIndex;
    if (view >= 0 && view < VIEW_COUNT && !m_isDVHView[view] &&
        !m_is3DView[view] && !m_isProfileView[view]) {
      QPoint wpos = m_imageWidgets[view]->mapFromGlobal(
          event->globalPosition().toPoint());
      QVector3D patient = patientCoordinateAt(view, wpos);
      if (!std::isnan(patient.x())) {
        m_profileStartPatient = patient;
        m_profileEndPatient = patient;
        m_profileLine.points.clear();
        QPointF plane = planeCoordinateFromPatient(view, patient);
        m_profileLine.points.append(plane);
        m_profileLine.points.append(plane);
        m_profileLine.color = Qt::yellow;
        m_profileLineView = view;
        m_profileLineVisible = true;
        m_profileLineHasStart = true;
        updateImage(view);
      }
    }
    event->accept();
    return;
  }

  if (!m_selectingProfileLine && event->button() == Qt::LeftButton &&
      m_profileLineVisible && m_activeViewIndex == m_profileLineView) {
    int view = m_activeViewIndex;
    QPoint wpos =
        m_imageWidgets[view]->mapFromGlobal(event->globalPosition().toPoint());
    QVector3D patient = patientCoordinateAt(view, wpos);
    if (!std::isnan(patient.x())) {
      QPointF plane = planeCoordinateFromPatient(view, patient);
      double distStart = QLineF(plane, m_profileLine.points.value(0)).length();
      double distEnd = QLineF(plane, m_profileLine.points.value(1)).length();
      const double threshold = 5.0; // mm
      if (distStart < threshold)
        m_dragProfileStart = true;
      else if (distEnd < threshold)
        m_dragProfileEnd = true;
      if (m_dragProfileStart || m_dragProfileEnd) {
        event->accept();
        return;
      }
    }
  }

  if (m_windowLevelDragActive && event->button() == Qt::LeftButton) {
    m_dragStartPos = event->pos();
    m_dragStartWindow = m_windowSlider->value();
    m_dragStartLevel = m_levelSlider->value();
    setCursor(Qt::ClosedHandCursor);
    m_windowLevelTimer->start(WINDOW_LEVEL_TIMEOUT);
  } else if (event->button() == Qt::RightButton && m_windowLevelDragActive) {
    stopWindowLevelDrag();
  } else if (m_panMode && event->button() == Qt::LeftButton) {
    m_panDragActive = true;
    // store global position so delta is unaffected by scrolling
    m_panStartPos = event->globalPosition().toPoint();
    m_panTimer->start(WINDOW_LEVEL_TIMEOUT);
    setCursor(Qt::ClosedHandCursor);
    event->accept();
    return;
  } else if (m_zoomMode && event->button() == Qt::LeftButton) {
    m_zoomDragActive = true;
    m_zoomStartPos = event->pos();
    m_zoomStartFactor = m_zoomFactor;
    m_zoomTimer->start(WINDOW_LEVEL_TIMEOUT);
    setCursor(Qt::ClosedHandCursor);
    event->accept();
    return;
  } else if (!m_panMode && !m_zoomMode && event->button() == Qt::LeftButton &&
             m_activeViewIndex >= 0 && m_is3DView[m_activeViewIndex]) {
    m_rotationDragActive = true;
    m_rotationStartPos = event->pos();
    setCursor(Qt::ClosedHandCursor);
    event->accept();
    return;
  }

  QWidget::mousePressEvent(event);
}

void DicomViewer::mouseMoveEvent(QMouseEvent *event) {
  int view = viewIndexFromGlobalPos(event->globalPosition().toPoint());
  m_activeViewIndex = view;
  if (view >= 0 && view < VIEW_COUNT && !m_isDVHView[view] &&
      !m_is3DView[view]) {
    QPoint wpos =
        m_imageWidgets[view]->mapFromGlobal(event->globalPosition().toPoint());
    updateCoordLabel(view, wpos);
  }

  if (m_windowLevelDragActive && (event->buttons() & Qt::LeftButton)) {
    updateWindowLevelFromMouse(event->pos());
    m_windowLevelTimer->start(WINDOW_LEVEL_TIMEOUT);
  } else if (m_selectingProfileLine && m_profileLineHasStart &&
             (event->buttons() & Qt::LeftButton) && view == m_profileLineView) {
    QPoint wpos =
        m_imageWidgets[view]->mapFromGlobal(event->globalPosition().toPoint());
    QVector3D patient = patientCoordinateAt(view, wpos);
    if (!std::isnan(patient.x())) {
      m_profileEndPatient = patient;
      QPointF plane = planeCoordinateFromPatient(view, patient);
      if (m_profileLine.points.size() < 2)
        m_profileLine.points.append(plane);
      else
        m_profileLine.points[1] = plane;
      updateImage(view);
    }
    event->accept();
    return;
  } else if ((m_dragProfileStart || m_dragProfileEnd) &&
             (event->buttons() & Qt::LeftButton) && view == m_profileLineView) {
    QPoint wpos =
        m_imageWidgets[view]->mapFromGlobal(event->globalPosition().toPoint());
    QVector3D patient = patientCoordinateAt(view, wpos);
    if (!std::isnan(patient.x())) {
      QPointF plane = planeCoordinateFromPatient(view, patient);
      if (m_dragProfileStart) {
        m_profileStartPatient = patient;
        if (m_profileLine.points.size() >= 2)
          m_profileLine.points[0] = plane;
      } else if (m_dragProfileEnd) {
        m_profileEndPatient = patient;
        if (m_profileLine.points.size() >= 2)
          m_profileLine.points[1] = plane;
      }
      updateImage(view);
    }
    event->accept();
    return;
  } else if (m_panDragActive && (event->buttons() & Qt::LeftButton)) {
    QPoint currentGlobal = event->globalPosition().toPoint();
    QPoint delta = currentGlobal - m_panStartPos;

    // アクティブビューのみパンを更新
    m_panOffsets[m_activeViewIndex] += QPointF(delta);
    if (m_is3DView[m_activeViewIndex])
      m_3dWidgets[m_activeViewIndex]->setPan(m_panOffsets[m_activeViewIndex]);
    else
      m_imageWidgets[m_activeViewIndex]->setPan(
          m_panOffsets[m_activeViewIndex]);

    m_panStartPos = currentGlobal;
    m_panTimer->start(WINDOW_LEVEL_TIMEOUT);
    event->accept();
    return;
  } else if (m_zoomDragActive && (event->buttons() & Qt::LeftButton)) {
    int dy = event->pos().y() - m_zoomStartPos.y();
    double factor = m_zoomStartFactor * (1.0 - dy * 0.01);
    setZoomFactor(factor);
    m_zoomTimer->start(WINDOW_LEVEL_TIMEOUT);
    event->accept();
    return;
  } else if (m_rotationDragActive && (event->buttons() & Qt::LeftButton) &&
             m_activeViewIndex >= 0 && m_is3DView[m_activeViewIndex]) {
    QPoint delta = event->pos() - m_rotationStartPos;
    m_3dWidgets[m_activeViewIndex]->addRotation(delta.x(), delta.y());
    m_rotationStartPos = event->pos();
    event->accept();
    return;
  }

  QWidget::mouseMoveEvent(event);
}

void DicomViewer::onGenerateQr() {
  if (!m_qrTextEdit || !m_qrImageLabel)
    return;

  auto escapeWifiLike = [](const QString &s) -> QString {
    QString out;
    out.reserve(s.size() * 2);
    for (int i = 0; i < s.size(); ++i) {
      const QChar c = s[i];
      if (c == QChar('\n') || c == QChar('\r')) {
        // preserve newlines as-is
        out.append(c);
        continue;
      }
      if (c == QChar('\\')) {
        // Preserve existing escape sequence: copy backslash and next char if
        // any
        if (i + 1 < s.size()) {
          out.append('\\');
          out.append(s[i + 1]);
          ++i;
        } else {
          // Trailing backslash, escape it
          out.append("\\\\");
        }
        continue;
      }
      if (c == QChar(';') || c == QChar(',') || c == QChar(':')) {
        out.append('\\');
        out.append(c);
      } else {
        out.append(c);
      }
    }
    return out;
  };

  QString text = m_qrTextEdit->toPlainText();
  if (m_qrEscapeCheck && m_qrEscapeCheck->isChecked()) {
    text = escapeWifiLike(text);
  }
  if (text.isEmpty()) {
    m_qrImageLabel->clear();
    return;
  }

  try {
    QByteArray utf8 = text.toUtf8();
    using qrcodegen::QrCode;
    using qrcodegen::QrSegment;

    QrCode qr = [&]() {
      // Optionally add UTF-8 ECI; otherwise use bytes-only or encodeText for
      // ASCII
      bool addEci = m_qrUtf8EciCheck && m_qrUtf8EciCheck->isChecked();
      auto isAscii = [&utf8]() {
        for (unsigned char c : utf8)
          if (c >= 0x80)
            return false;
        return true;
      }();
      if (!addEci) {
        if (isAscii) {
          return QrCode::encodeText(utf8.constData(), QrCode::Ecc::MEDIUM);
        } else {
          std::vector<std::uint8_t> bytes(
              reinterpret_cast<const std::uint8_t *>(utf8.constData()),
              reinterpret_cast<const std::uint8_t *>(utf8.constData()) +
                  utf8.size());
          std::vector<QrSegment> segs;
          segs.push_back(QrSegment::makeBytes(bytes));
          return QrCode::encodeSegments(segs, QrCode::Ecc::MEDIUM);
        }
      } else {
        std::vector<std::uint8_t> bytes(
            reinterpret_cast<const std::uint8_t *>(utf8.constData()),
            reinterpret_cast<const std::uint8_t *>(utf8.constData()) +
                utf8.size());
        std::vector<QrSegment> segs;
        segs.push_back(QrSegment::makeEci(26)); // UTF-8 ECI designator
        segs.push_back(QrSegment::makeBytes(bytes));
        return QrCode::encodeSegments(segs, QrCode::Ecc::MEDIUM);
      }
    }();

    const int border = 8; // generous quiet zone in modules
    const int modules = qr.getSize();
    // Render at a steady base scale, then fit to label with nearest-neighbor
    const int baseScale = 4;
    const int sizePx = (modules + border * 2) * baseScale;

    QImage img(sizePx, sizePx, QImage::Format_ARGB32);
    img.fill(Qt::white);

    QPainter p(&img);
    p.setPen(Qt::NoPen);
    p.setBrush(Qt::black);
    for (int y = 0; y < modules; ++y) {
      for (int x = 0; x < modules; ++x) {
        if (qr.getModule(x, y)) {
          QRect r((x + border) * baseScale, (y + border) * baseScale, baseScale,
                  baseScale);
          p.drawRect(r);
        }
      }
    }
    p.end();
    // Fit to label while keeping module edges crisp (nearest-neighbor)
    int destW = m_qrImageLabel->width() > 0 ? m_qrImageLabel->width() : 260;
    int destH = m_qrImageLabel->height() > 0 ? m_qrImageLabel->height() : 260;
    QImage out =
        img.scaled(destW, destH, Qt::KeepAspectRatio, Qt::FastTransformation);
    m_lastQrImage =
        img; // keep original pixel-perfect image for saving/decoding tests
    m_qrImageLabel->setPixmap(QPixmap::fromImage(out));
  } catch (const std::exception &e) {
    m_qrImageLabel->setText(tr("Failed to generate QR: %1").arg(e.what()));
  }
}

void DicomViewer::onClearQr() {
  if (m_qrTextEdit)
    m_qrTextEdit->clear();
  if (m_qrImageLabel)
    m_qrImageLabel->clear();
  m_lastQrImage = QImage();
}

void DicomViewer::onDecodeQrFromImage() {
  QString file = QFileDialog::getOpenFileName(
      this, tr("Open Image for QR"), QString(),
      tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*.*)"));
  if (file.isEmpty())
    return;

  cv::Mat img = cv::imread(file.toStdString(), cv::IMREAD_COLOR);
  if (img.empty()) {
    QMessageBox::warning(this, tr("QR Decode"), tr("Failed to load image."));
    return;
  }

  try {
    std::vector<std::vector<cv::Point>> corners;
    std::vector<std::string> decoded = decodeQrRobust(img, &corners);
    QString outText;
    for (size_t i = 0; i < decoded.size(); ++i) {
      if (i)
        outText.append('\n');
      outText.append(QString::fromUtf8(decoded[i].c_str()));
    }

    if (outText.isEmpty()) {
      QMessageBox::information(this, tr("QR Decode"),
                               tr("No QR code detected."));
      return;
    }

    if (m_qrTextEdit)
      m_qrTextEdit->setPlainText(outText);

    // Optionally show the source image in the preview
    if (m_qrImageLabel) {
      // Convert BGR -> RGB for QImage
      cv::Mat rgb;
      cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
      QImage qimg(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                  QImage::Format_RGB888);
      m_qrImageLabel->setPixmap(QPixmap::fromImage(qimg.copy())
                                    .scaled(m_qrImageLabel->size(),
                                            Qt::KeepAspectRatio,
                                            Qt::FastTransformation));
    }
  } catch (const std::exception &e) {
    QMessageBox::warning(this, tr("QR Decode"),
                         tr("Failed to decode: %1").arg(e.what()));
  }
}

void DicomViewer::onSaveQrImage() {
  if (m_lastQrImage.isNull()) {
    QMessageBox::information(this, tr("Save QR"),
                             tr("No QR image to save. Generate first."));
    return;
  }
  QString fn = QFileDialog::getSaveFileName(this, tr("Save QR Image"),
                                            QString(), tr("PNG Image (*.png)"));
  if (fn.isEmpty())
    return;
  if (!fn.endsWith(".png", Qt::CaseInsensitive))
    fn += ".png";
  if (!m_lastQrImage.save(fn, "PNG")) {
    QMessageBox::warning(this, tr("Save QR"), tr("Failed to save image."));
  }
}

// Self Test removed per request

void DicomViewer::onStartQrCamera() {
  if (m_qrCamRunning)
    return;
  try {
    m_qrCapture = std::make_unique<cv::VideoCapture>(0);
  } catch (...) {
    m_qrCapture.reset();
  }
  if (!m_qrCapture || !m_qrCapture->isOpened()) {
    QMessageBox::warning(this, tr("QR Camera"), tr("Failed to open camera."));
    m_qrCapture.reset();
    return;
  }
  // Set a modest resolution for performance
  m_qrCapture->set(cv::CAP_PROP_FRAME_WIDTH, 1280);
  m_qrCapture->set(cv::CAP_PROP_FRAME_HEIGHT, 720);
  m_qrCamRunning = true;
  if (m_qrStartCamButton)
    m_qrStartCamButton->setEnabled(false);
  if (m_qrStopCamButton)
    m_qrStopCamButton->setEnabled(true);
  m_qrCamTimer->start();
}

void DicomViewer::onStopQrCamera() {
  if (!m_qrCamRunning)
    return;
  m_qrCamTimer->stop();
  m_qrCamRunning = false;
  if (m_qrCapture) {
    m_qrCapture->release();
    m_qrCapture.reset();
  }
  if (m_qrStartCamButton)
    m_qrStartCamButton->setEnabled(true);
  if (m_qrStopCamButton)
    m_qrStopCamButton->setEnabled(false);
}

void DicomViewer::onQrCameraTick() {
  if (!m_qrCapture || !m_qrCapture->isOpened()) {
    onStopQrCamera();
    return;
  }
  cv::Mat frame;
  if (!m_qrCapture->read(frame) || frame.empty()) {
    return;
  }
  // Decide whether to decode this frame based on time budget
  bool doDecode = (m_qrTimer.elapsed() >= m_qrDecodeIntervalMs);
  std::vector<std::vector<cv::Point>> corners;
  std::vector<std::string> decoded;
  if (doDecode) {
    m_qrTimer.restart();
    if (m_qrHighAccuracyCheck && m_qrHighAccuracyCheck->isChecked())
      decoded = decodeQrRobust(frame, &corners);
    else
      decoded = decodeQrFast(frame, &corners);
  }
  // Draw detections (guard sizes)
  if (!corners.empty()) {
    for (const auto &poly : corners) {
      if (poly.size() < 2)
        continue;
      for (size_t i = 0; i < poly.size(); ++i) {
        const cv::Point &a = poly[i];
        const cv::Point &b = poly[(i + 1) % poly.size()];
        cv::line(frame, a, b, cv::Scalar(0, 255, 0), 2);
      }
    }
  }
  // Update text if any
  if (!decoded.empty() && m_qrTextEdit) {
    QString out;
    for (size_t i = 0; i < decoded.size(); ++i) {
      if (i)
        out.append('\n');
      out.append(QString::fromUtf8(decoded[i].c_str()));
    }
    m_qrTextEdit->setPlainText(out);
  }
  // Show frame
  if (m_qrImageLabel) {
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    QImage qimg(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                QImage::Format_RGB888);
    m_qrImageLabel->setPixmap(QPixmap::fromImage(qimg.copy())
                                  .scaled(m_qrImageLabel->size(),
                                          Qt::KeepAspectRatio,
                                          Qt::FastTransformation));
  }
}




bool DicomViewer::ensureDvhDataReady(int roiIndex,
                                     DVHCalculator::DVHData **outData) {
  if (!isVolumeLoaded() || !m_doseLoaded || !m_rtstructLoaded ||
      !m_resampledDose.isResampled()) {
    return false;
  }
  if (roiIndex < 0 || roiIndex >= m_rtstruct.roiCount()) {
    qWarning() << QString("Invalid ROI index for DVH request: %1").arg(roiIndex);
    return false;
  }
  if (m_dvhWatchers.contains(roiIndex)) {
    qWarning() << QString("DVH calculation already running for ROI: %1")
                      .arg(m_rtstruct.roiName(roiIndex));
    return false;
  }

  if (static_cast<size_t>(m_rtstruct.roiCount()) > m_dvhData.size())
    m_dvhData.resize(m_rtstruct.roiCount());

  if (m_dvhData[roiIndex].points.empty()) {
    double maxDose = m_resampledDose.maxDose();
    double binSize = maxDose > 0.0 ? maxDose / 200.0 : 0.25;
    try {
      auto data = DVHCalculator::calculateSingleROI(
          m_volume, m_resampledDose, m_rtstruct, roiIndex, binSize, nullptr, {});
      data.isVisible = true;
      m_dvhData[roiIndex] = std::move(data);
    } catch (const std::exception &e) {
      qWarning() << QString("DVH calculation failed for ROI %1: %2")
                        .arg(roiIndex)
                        .arg(QString::fromUtf8(e.what()));
      return false;
    }
  } else {
    m_dvhData[roiIndex].isVisible = true;
  }

  refreshDvhWidgets();
  if (outData)
    *outData = &m_dvhData[roiIndex];
  return true;
}

int DicomViewer::findRoiIndex(const QString &roiName) const {
  if (!m_rtstructLoaded)
    return -1;
  for (int i = 0; i < m_rtstruct.roiCount(); ++i) {
    if (m_rtstruct.roiName(i).compare(roiName, Qt::CaseInsensitive) == 0)
      return i;
  }
  return -1;
}

double DicomViewer::doseAtVolumePercent(const DVHCalculator::DVHData &data,
                                        double volumePercent) const {
  if (data.points.empty())
    return 0.0;
  const auto &pts = data.points;
  if (volumePercent >= pts.front().volume)
    return pts.front().dose;
  if (volumePercent <= pts.back().volume)
    return pts.back().dose;
  for (size_t i = 1; i < pts.size(); ++i) {
    double v0 = pts[i - 1].volume;
    double v1 = pts[i].volume;
    if (v0 >= volumePercent && v1 <= volumePercent) {
      double d0 = pts[i - 1].dose;
      double d1 = pts[i].dose;
      double t = (v0 - volumePercent) / (v0 - v1);
      return d0 + t * (d1 - d0);
    }
  }
  return pts.back().dose;
}

double DicomViewer::volumeAtDoseGy(const DVHCalculator::DVHData &data,
                                   double doseGy) const {
  if (data.points.empty())
    return 0.0;
  const auto &pts = data.points;
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
}

QString DicomViewer::formatDvhMetrics(const DVHCalculator::DVHData &data,
                                      const QStringList &metrics) const {
  if (metrics.isEmpty())
    return QString();
  QStringList parts;
  for (const QString &metric : metrics) {
    const QString trimmed = metric.trimmed();
    if (trimmed.isEmpty())
      continue;
    const QString lower = trimmed.toLower();
    if (lower.startsWith(QLatin1Char('d')) && lower.length() > 1) {
      bool ok = false;
      double vol = trimmed.mid(1).toDouble(&ok);
      if (ok) {
        double val = doseAtVolumePercent(data, vol);
        parts << QStringLiteral("%1=%2 Gy")
                      .arg(trimmed.toUpper(),
                           QString::number(val, 'f', 2));
      }
    } else if (lower.startsWith(QLatin1Char('v')) && lower.length() > 1) {
      bool ok = false;
      double dose = trimmed.mid(1).toDouble(&ok);
      if (ok) {
        double vol = volumeAtDoseGy(data, dose);
        parts << QStringLiteral("%1=%2%%")
                      .arg(trimmed.toUpper(),
                           QString::number(vol, 'f', 1));
      }
    } else if (lower == QStringLiteral("dmean") ||
               lower == QStringLiteral("mean")) {
      parts << QStringLiteral("DMEAN=%1 Gy")
                    .arg(QString::number(data.meanDose, 'f', 2));
    } else if (lower == QStringLiteral("dmax") ||
               lower == QStringLiteral("max")) {
      parts << QStringLiteral("DMAX=%1 Gy")
                    .arg(QString::number(data.maxDose, 'f', 2));
    } else if (lower == QStringLiteral("dmin") ||
               lower == QStringLiteral("min")) {
      parts << QStringLiteral("DMIN=%1 Gy")
                    .arg(QString::number(data.minDose, 'f', 2));
    }
  }
  return parts.join(QStringLiteral(", "));
}

void DicomViewer::refreshDvhWidgets() {
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (!m_dvhWidgets[i])
      continue;
    m_dvhWidgets[i]->setDVHData(m_dvhData);
    m_dvhWidgets[i]->setPatientInfo(patientInfoText());
    m_dvhWidgets[i]->setPrescriptionDose(m_doseReference);
  }
}

bool DicomViewer::runDpsdAnalysis(const QString &roiName,
                                   const QString &sampleRoiName,
                                   double startMm, double endMm,
                                   DPSDCalculator::Mode mode) {
  if (!isVolumeLoaded() || !m_doseLoaded || !m_rtstructLoaded ||
      !m_resampledDose.isResampled()) {
    return false;
  }
  if (startMm >= endMm) {
    return false;
  }
  int roiIndex = findRoiIndex(roiName);
  if (roiIndex < 0) {
    qWarning() << QString("Unknown ROI for DPSD analysis: %1").arg(roiName);
    return false;
  }
  int sampleIndex = -1;
  if (!sampleRoiName.trimmed().isEmpty()) {
    sampleIndex = findRoiIndex(sampleRoiName);
    if (sampleIndex < 0) {
      qWarning() << QString("Unknown sample ROI for DPSD analysis: %1")
                        .arg(sampleRoiName);
      return false;
    }
  }

  DPSDCalculator::Result roiResult;
  DPSDCalculator::Result sampleResult;
  try {
    roiResult = DPSDCalculator::calculate(m_volume, m_resampledDose, m_rtstruct,
                                          roiIndex, startMm, endMm, 2.0, mode,
                                          -1, nullptr);
    if (sampleIndex >= 0) {
      sampleResult = DPSDCalculator::calculate(
          m_volume, m_resampledDose, m_rtstruct, roiIndex, startMm, endMm, 2.0,
          mode, sampleIndex, nullptr);
    }
  } catch (const std::exception &e) {
    qWarning() << QString("DPSD analysis failed: %1")
                      .arg(QString::fromUtf8(e.what()));
    return false;
  }

  if (roiResult.distancesMm.empty()) {
    return false;
  }

  QStringList lines;
  lines << tr("DPSD解析結果: ROI=%1, モード=%2")
               .arg(roiName,
                    mode == DPSDCalculator::Mode::Mode2D ? QStringLiteral("2D")
                                                         : QStringLiteral("3D"));
  lines << tr("距離範囲: %1 mm ～ %2 mm (ステップ 2.0 mm)")
               .arg(QString::number(startMm, 'f', 1),
                    QString::number(endMm, 'f', 1));
  if (sampleIndex >= 0) {
    lines << tr("サンプルROI: %1").arg(sampleRoiName);
  }

  const int count = static_cast<int>(roiResult.distancesMm.size());
  const int samples = std::clamp(count / 6, 1, std::max(1, count));
  for (int i = 0; i < count; i += samples) {
    double dist = roiResult.distancesMm[static_cast<size_t>(i)];
    double dmin = roiResult.minDoseGy[static_cast<size_t>(i)];
    double dmax = roiResult.maxDoseGy[static_cast<size_t>(i)];
    double dmean = roiResult.meanDoseGy[static_cast<size_t>(i)];
    QString line =
        tr("距離%1mm: min=%2 Gy, max=%3 Gy, mean=%4 Gy")
            .arg(QString::number(dist, 'f', 1),
                 QString::number(dmin, 'f', 2),
                 QString::number(dmax, 'f', 2),
                 QString::number(dmean, 'f', 2));
    if (!sampleResult.distancesMm.empty() &&
        sampleResult.distancesMm.size() == roiResult.distancesMm.size()) {
      double smean = sampleResult.meanDoseGy[static_cast<size_t>(i)];
      line += tr(" (sample mean=%1 Gy)")
                  .arg(QString::number(smean, 'f', 2));
    }
    lines << line;
  }
  // ensure最後のサンプルが含まれる
  if ((count - 1) % samples != 0) {
    int i = count - 1;
    double dist = roiResult.distancesMm[static_cast<size_t>(i)];
    double dmin = roiResult.minDoseGy[static_cast<size_t>(i)];
    double dmax = roiResult.maxDoseGy[static_cast<size_t>(i)];
    double dmean = roiResult.meanDoseGy[static_cast<size_t>(i)];
    QString line =
        tr("距離%1mm: min=%2 Gy, max=%3 Gy, mean=%4 Gy")
            .arg(QString::number(dist, 'f', 1),
                 QString::number(dmin, 'f', 2),
                 QString::number(dmax, 'f', 2),
                 QString::number(dmean, 'f', 2));
    if (!sampleResult.distancesMm.empty() &&
        sampleResult.distancesMm.size() == roiResult.distancesMm.size()) {
      double smean = sampleResult.meanDoseGy.back();
      line += tr(" (sample mean=%1 Gy)")
                  .arg(QString::number(smean, 'f', 2));
    }
    lines << line;
  }

  return true;
}

void DicomViewer::mouseReleaseEvent(QMouseEvent *event) {
  if (m_selectingProfileLine && m_profileLineHasStart &&
      event->button() == Qt::LeftButton) {
    int view = viewIndexFromGlobalPos(event->globalPosition().toPoint());
    if (view == m_profileLineView) {
      QPoint wpos = m_imageWidgets[view]->mapFromGlobal(
          event->globalPosition().toPoint());
      QVector3D patient = patientCoordinateAt(view, wpos);
      if (!std::isnan(patient.x()))
        m_profileEndPatient = patient;
      m_selectingProfileLine = false;
      m_profileLineHasStart = false;
      computeDoseProfile();
      updateImage(view);
    }
    event->accept();
    return;
  }
  if ((m_dragProfileStart || m_dragProfileEnd) &&
      event->button() == Qt::LeftButton) {
    int view = viewIndexFromGlobalPos(event->globalPosition().toPoint());
    if (view == m_profileLineView) {
      QPoint wpos = m_imageWidgets[view]->mapFromGlobal(
          event->globalPosition().toPoint());
      QVector3D patient = patientCoordinateAt(view, wpos);
      if (!std::isnan(patient.x())) {
        QPointF plane = planeCoordinateFromPatient(view, patient);
        if (m_dragProfileStart) {
          m_profileStartPatient = patient;
          if (m_profileLine.points.size() >= 2)
            m_profileLine.points[0] = plane;
        } else if (m_dragProfileEnd) {
          m_profileEndPatient = patient;
          if (m_profileLine.points.size() >= 2)
            m_profileLine.points[1] = plane;
        }
      }
      computeDoseProfile();
      updateImage(view);
    }
    m_dragProfileStart = m_dragProfileEnd = false;
    event->accept();
    return;
  }
  if (m_windowLevelDragActive && event->button() == Qt::LeftButton) {
    setCursor(Qt::OpenHandCursor);
  } else if (m_panDragActive && event->button() == Qt::LeftButton) {
    m_panDragActive = false;
    setCursor(Qt::OpenHandCursor);
    event->accept();
    return;
  } else if (m_zoomDragActive && event->button() == Qt::LeftButton) {
    m_zoomDragActive = false;
    setCursor(Qt::OpenHandCursor);
    event->accept();
    return;
  } else if (m_rotationDragActive && event->button() == Qt::LeftButton) {
    m_rotationDragActive = false;
    setCursor(m_panMode || m_zoomMode ? Qt::OpenHandCursor : Qt::ArrowCursor);
    event->accept();
    return;
  }

  QWidget::mouseReleaseEvent(event);
}

void DicomViewer::keyPressEvent(QKeyEvent *event) {
  if (m_structureList && m_structureList->hasFocus()) {
    QWidget::keyPressEvent(event);
    return;
  }
  if (event->key() == Qt::Key_Escape && m_windowLevelDragActive) {
    stopWindowLevelDrag();
    event->accept();
  } else if (event->key() == Qt::Key_Right || event->key() == Qt::Key_Down) {
    showNextImage();
    event->accept();
  } else if (event->key() == Qt::Key_Left || event->key() == Qt::Key_Up) {
    showPreviousImage();
    event->accept();
  } else {
    QWidget::keyPressEvent(event);
  }
}

void DicomViewer::enterEvent(QEnterEvent *event) {
  if (m_windowLevelDragActive || m_panMode || m_zoomMode) {
    setCursor(Qt::OpenHandCursor);
  }
  QWidget::enterEvent(event);
}

void DicomViewer::leaveEvent(QEvent *event) {
  if (m_windowLevelDragActive || m_panMode || m_zoomMode) {
    setCursor(Qt::ArrowCursor);
  }
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_coordLabels[i])
      m_coordLabels[i]->hide();
    if (m_cursorDoseLabels[i])
      m_cursorDoseLabels[i]->hide();
    m_imageWidgets[i]->clearCursorCross();
  }
  QWidget::leaveEvent(event);
}

void DicomViewer::updateImage() {
  int count = 1;
  if (m_viewMode == ViewMode::Dual)
    count = 2;
  else if (m_viewMode == ViewMode::Quad)
    count = 4;
  else if (m_viewMode == ViewMode::Five)
    count = VIEW_COUNT;
  for (int i = 0; i < count; ++i) {
    if (m_isDVHView[i] || m_isProfileView[i])
      continue;
    if (m_is3DView[i]) {
      update3DView(i);
    } else {
      updateImage(i);
      m_imageWidgets[i]->setZoom(m_zoomFactor);
      m_imageWidgets[i]->setPan(m_panOffsets[i]);
    }
  }
  // Update Dose Shift overlay visibility after image updates
  updateDoseShiftLabels();
}

void DicomViewer::updateImage(int viewIndex, bool updateStructure) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT || m_isDVHView[viewIndex] ||
      m_is3DView[viewIndex] || m_isProfileView[viewIndex])
    return;
  if (m_fusionViewActive && viewIndex == 1) {
    if (m_fusionViewImage.isNull()) {
      m_imageWidgets[viewIndex]->setImage(QImage());
    } else {
      m_imageWidgets[viewIndex]->setImage(m_fusionViewImage);
    }
    m_imageWidgets[viewIndex]->setStructureLines(StructureLineList());
    m_imageWidgets[viewIndex]->setStructurePoints(StructurePointList());
    m_imageWidgets[viewIndex]->setSlicePositionLines(StructureLineList());
    m_imageWidgets[viewIndex]->setPixelSpacing(
        static_cast<float>(m_fusionSpacingX),
        static_cast<float>(m_fusionSpacingY));
    m_imageWidgets[viewIndex]->setZoom(m_zoomFactor);
    m_imageWidgets[viewIndex]->setPan(m_panOffsets[viewIndex]);
    return;
  }

  QImage img = m_originalImages[viewIndex];
  if (img.isNull())
    return;

  QImage displayImg = img.convertToFormat(QImage::Format_ARGB32);

  StructureLineList doseLines;
  if (m_doseLoaded && isVolumeLoaded() && m_doseVisible) {
    qDebug()
        << QString("=== Updating dose overlay for view %1 ===").arg(viewIndex);

    DicomVolume::Orientation ori = m_viewOrientations[viewIndex];
    int ctIndex = m_currentIndices[viewIndex];

    QString oriStr = (ori == DicomVolume::Orientation::Axial)      ? "Axial"
                     : (ori == DicomVolume::Orientation::Sagittal) ? "Sagittal"
                                                                   : "Coronal";
    qDebug() << QString("Orientation: %1, CT slice index: %2")
                    .arg(oriStr)
                    .arg(ctIndex);

    if (m_doseDisplayMode ==
        DoseResampledVolume::DoseDisplayMode::IsodoseLines) {
      auto isoLines = m_resampledDose.getIsodoseLines(
          ctIndex, ori, m_doseMinRange, m_doseMaxRange, m_doseReference);
      for (const auto &line : isoLines) {
        StructureLine converted;
        converted.points = line.points;
        converted.color = line.color;
        doseLines.append(converted);
      }
      qDebug() << QString("Isodose line paths: %1")
                      .arg(doseLines.size());
    } else {
      // 線量オーバーレイを取得
      QImage overlay = m_resampledDose.getSlice(
          ctIndex, ori, m_doseMinRange, m_doseMaxRange, m_doseDisplayMode,
          m_doseReference);

      if (!overlay.isNull()) {
        QPainter painter(&displayImg);
        painter.setCompositionMode(QPainter::CompositionMode_SourceOver);

        // オーバーレイ画像のサイズ調整
        if (overlay.size() != displayImg.size()) {
          qDebug() << QString("Scaling colorful overlay from %1x%2 to %3x%4")
                          .arg(overlay.width())
                          .arg(overlay.height())
                          .arg(displayImg.width())
                          .arg(displayImg.height());
          overlay = overlay.scaled(displayImg.size(), Qt::IgnoreAspectRatio,
                                   Qt::SmoothTransformation);
        }

        // 美しい線量オーバーレイを描画
        painter.setOpacity(m_doseOpacity);
        painter.drawImage(0, 0, overlay);
        painter.setOpacity(1.0);

        // デバッグ: カラフルピクセル数をカウント
        int colorfulPixels = 0;
        for (int y = 0; y < overlay.height(); ++y) {
          for (int x = 0; x < overlay.width(); ++x) {
            QRgb pixel = overlay.pixel(x, y);
            if (qAlpha(pixel) > 10) {
              colorfulPixels++;
            }
          }
        }
        qDebug() << QString("Colorful dose pixels: %1 out of %2 total (%3%)")
                        .arg(colorfulPixels)
                        .arg(overlay.width() * overlay.height())
                        .arg(100.0 * colorfulPixels /
                                 (overlay.width() * overlay.height()),
                             0, 'f', 1);

        painter.end();

      } else {
        qDebug() << "Warning: colorful dose overlay is null";
      }
    }
  }

  if (updateStructure) {
    StructureLineList sLines;
    StructurePointList sPoints;
    int ctIndex = m_currentIndices[viewIndex];
    if (m_rtstructLoaded && isVolumeLoaded()) {
      int overlayStride = 1;
      switch (m_viewOrientations[viewIndex]) {
      case DicomVolume::Orientation::Axial:
        sLines = m_rtstruct.axialContours(m_volume, ctIndex);
        break;
      case DicomVolume::Orientation::Sagittal:
        sLines = m_rtstruct.sagittalContours(m_volume, ctIndex, overlayStride);
        if (m_showStructurePoints)
          sPoints =
              m_rtstruct.sagittalVertices(m_volume, ctIndex, overlayStride);
        break;
      case DicomVolume::Orientation::Coronal:
        sLines = m_rtstruct.coronalContours(m_volume, ctIndex, overlayStride);
        if (m_showStructurePoints)
          sPoints =
              m_rtstruct.coronalVertices(m_volume, ctIndex, overlayStride);
        break;
      }
    }
    if (m_brachyLoaded && isVolumeLoaded()) {
      switch (m_viewOrientations[viewIndex]) {
      case DicomVolume::Orientation::Axial: {
        // Use voxel center (+0.5) to match dose calculation coordinate system
        double target = m_volume
                            .voxelToPatient(m_volume.width() / 2.0,
                                            m_volume.height() / 2.0, ctIndex + 0.5)
                            .z();
        double tol = m_volume.spacingZ() / 2.0;
        for (const auto &d : m_brachyPlan.sources()) {
          if (std::abs(d.position().z() - target) <= tol) {
            QColor c =
                (d.dwellTime() > 0.0) ? QColor(Qt::red) : QColor(Qt::yellow);
            sPoints.append(
                {planeCoordinateFromPatient(viewIndex, d.position()), c});
          }
        }
        break;
      }
      case DicomVolume::Orientation::Sagittal: {
        // Use voxel center (+0.5) to match dose calculation coordinate system
        double target = m_volume.voxelToPatient(ctIndex + 0.5, 0.0, 0.0).x();
        double tol = m_volume.spacingX() / 2.0;
        for (const auto &d : m_brachyPlan.sources()) {
          if (std::abs(d.position().x() - target) <= tol) {
            QColor c =
                (d.dwellTime() > 0.0) ? QColor(Qt::red) : QColor(Qt::yellow);
            sPoints.append(
                {planeCoordinateFromPatient(viewIndex, d.position()), c});
          }
        }
        break;
      }
      case DicomVolume::Orientation::Coronal: {
        // Use voxel center (+0.5) to match dose calculation coordinate system
        double target = m_volume.voxelToPatient(0.0, ctIndex + 0.5, 0.0).y();
        double tol = m_volume.spacingY() / 2.0;
        for (const auto &d : m_brachyPlan.sources()) {
          if (std::abs(d.position().y() - target) <= tol) {
            QColor c =
                (d.dwellTime() > 0.0) ? QColor(Qt::red) : QColor(Qt::yellow);
            sPoints.append(
                {planeCoordinateFromPatient(viewIndex, d.position()), c});
          }
        }
        break;
      }
      }
    }
    if (m_profileLineVisible && viewIndex == m_profileLineView &&
        m_profileLine.points.size() >= 2) {
      sPoints.append({m_profileLine.points[0], m_profileLine.color});
      sPoints.append({m_profileLine.points[1], m_profileLine.color});
  }
  m_imageWidgets[viewIndex]->setStructureLines(sLines);
  m_imageWidgets[viewIndex]->setStructurePoints(sPoints);

  // Extract and display reference points on current slice
  StructurePointList refPoints;
  if (m_brachyLoaded && m_brachyShowRefPointsCheck && m_brachyShowRefPointsCheck->isChecked()) {
    const auto& brachyRefPoints = m_brachyPlan.referencePoints();
    if (isVolumeLoaded() && !brachyRefPoints.isEmpty()) {
      switch (m_viewOrientations[viewIndex]) {
      case DicomVolume::Orientation::Axial: {
        double target = m_volume.voxelToPatient(m_volume.width() / 2.0,
                                                m_volume.height() / 2.0, ctIndex + 0.5).z();
        double tol = m_volume.spacingZ() / 2.0;
        for (const auto &rp : brachyRefPoints) {
          if (std::abs(rp.position.z() - target) <= tol) {
            refPoints.append({planeCoordinateFromPatient(viewIndex, rp.position), QColor(0, 0, 255)});
          }
        }
        break;
      }
      case DicomVolume::Orientation::Sagittal: {
        double target = m_volume.voxelToPatient(ctIndex + 0.5, 0.0, 0.0).x();
        double tol = m_volume.spacingX() / 2.0;
        for (const auto &rp : brachyRefPoints) {
          if (std::abs(rp.position.x() - target) <= tol) {
            refPoints.append({planeCoordinateFromPatient(viewIndex, rp.position), QColor(0, 0, 255)});
          }
        }
        break;
      }
      case DicomVolume::Orientation::Coronal: {
        double target = m_volume.voxelToPatient(0.0, ctIndex + 0.5, 0.0).y();
        double tol = m_volume.spacingY() / 2.0;
        for (const auto &rp : brachyRefPoints) {
          if (std::abs(rp.position.y() - target) <= tol) {
            refPoints.append({planeCoordinateFromPatient(viewIndex, rp.position), QColor(0, 0, 255)});
          }
        }
        break;
      }
      }
    }
  }
  m_imageWidgets[viewIndex]->setReferencePoints(refPoints);
  }
  // Prepare slice position lines container early so guides can append
  StructureLineList sliceLines;

  // Optional mm-based grid overlay
  if (m_showGrid && isVolumeLoaded()) {
    auto upd = [](double &mn, double &mx, double v) {
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    };
    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double minZ = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    double maxZ = -std::numeric_limits<double>::infinity();
    int xs[2] = {0, m_volume.width() - 1};
    int ys[2] = {0, m_volume.height() - 1};
    int zs[2] = {0, m_volume.depth() - 1};
    for (int ix : xs)
      for (int iy : ys)
        for (int iz : zs) {
          QVector3D p = m_volume.voxelToPatient(ix + 0.5, iy + 0.5, iz + 0.5);
          upd(minX, maxX, p.x());
          upd(minY, maxY, p.y());
          upd(minZ, maxZ, p.z());
        }
    const double grid = 50.0; // 50 mm spacing
    QColor gridColor(128, 128, 128, 120);
    auto addLine = [&](double x1, double y1, double z1, double x2, double y2,
                       double z2) {
      StructureLine l;
      l.color = gridColor;
      l.points.append(
          planeCoordinateFromPatient(viewIndex, QVector3D(x1, y1, z1)));
      l.points.append(
          planeCoordinateFromPatient(viewIndex, QVector3D(x2, y2, z2)));
      sliceLines.append(l);
    };
    switch (m_viewOrientations[viewIndex]) {
    case DicomVolume::Orientation::Axial: {
      double z =
          m_volume
              .voxelToPatient(static_cast<double>(m_volume.width()) / 2.0,
                              static_cast<double>(m_volume.height()) / 2.0,
                              static_cast<double>(m_currentIndices[viewIndex]))
              .z();
      for (double x = std::ceil(minX / grid) * grid; x <= maxX; x += grid)
        addLine(x, minY, z, x, maxY, z);
      for (double y = std::ceil(minY / grid) * grid; y <= maxY; y += grid)
        addLine(minX, y, z, maxX, y, z);
      break;
    }
    case DicomVolume::Orientation::Sagittal: {
      double x =
          m_volume
              .voxelToPatient(static_cast<double>(m_currentIndices[viewIndex]),
                              static_cast<double>(m_volume.height()) / 2.0,
                              static_cast<double>(m_volume.depth()) / 2.0)
              .x();
      for (double y = std::ceil(minY / grid) * grid; y <= maxY; y += grid)
        addLine(x, y, minZ, x, y, maxZ);
      for (double z = std::ceil(minZ / grid) * grid; z <= maxZ; z += grid)
        addLine(x, minY, z, x, maxY, z);
      break;
    }
    case DicomVolume::Orientation::Coronal: {
      double y =
          m_volume
              .voxelToPatient(static_cast<double>(m_volume.width()) / 2.0,
                              static_cast<double>(m_currentIndices[viewIndex]),
                              static_cast<double>(m_volume.depth()) / 2.0)
              .y();
      for (double x = std::ceil(minX / grid) * grid; x <= maxX; x += grid)
        addLine(x, y, minZ, x, y, maxZ);
      for (double z = std::ceil(minZ / grid) * grid; z <= maxZ; z += grid)
        addLine(minX, y, z, maxX, y, z);
      break;
    }
    }
  }

  // Add RTDose bounds guide (native/aligned) as overlay lines
  if (m_doseLoaded && isVolumeLoaded() && m_showDoseGuide) {
    double minX = 0, maxX = 0, minY = 0, maxY = 0, minZ = 0, maxZ = 0;
    bool has = m_doseVolume.nativeExtents(minX, maxX, minY, maxY, minZ, maxZ);
    if (has) {
      // Draw only if current slice intersects dose extent in the orthogonal
      // axis
      bool draw = false;
      // Four corners in patient space for all orientations
      auto addRect = [&](double zfixed, double xfixed, double yfixed,
                         DicomVolume::Orientation ori) {
        StructureLine rect;
        rect.color = QColor(255, 0, 255, 180); // magenta
        auto addPt = [&](double px, double py, double pz) {
          QPointF q =
              planeCoordinateFromPatient(viewIndex, QVector3D(px, py, pz));
          rect.points.append(q);
        };
        if (ori == DicomVolume::Orientation::Axial) {
          addPt(minX, minY, zfixed);
          addPt(maxX, minY, zfixed);
          addPt(maxX, maxY, zfixed);
          addPt(minX, maxY, zfixed);
          addPt(minX, minY, zfixed);
        } else if (ori == DicomVolume::Orientation::Sagittal) {
          addPt(xfixed, minY, minZ);
          addPt(xfixed, maxY, minZ);
          addPt(xfixed, maxY, maxZ);
          addPt(xfixed, minY, maxZ);
          addPt(xfixed, minY, minZ);
        } else { // Coronal
          addPt(minX, yfixed, minZ);
          addPt(maxX, yfixed, minZ);
          addPt(maxX, yfixed, maxZ);
          addPt(minX, yfixed, maxZ);
          addPt(minX, yfixed, minZ);
        }
        // append to existing slice position lines to render as guide
        sliceLines.append(rect);
      };

      switch (m_viewOrientations[viewIndex]) {
      case DicomVolume::Orientation::Axial: {
        double zslice =
            m_volume
                .voxelToPatient(
                    static_cast<double>(m_volume.width()) / 2.0,
                    static_cast<double>(m_volume.height()) / 2.0,
                    static_cast<double>(m_currentIndices[viewIndex]))
                .z();
        draw =
            (zslice >= std::min(minZ, maxZ) && zslice <= std::max(minZ, maxZ));
        if (draw)
          addRect(zslice, 0, 0, DicomVolume::Orientation::Axial);
        break;
      }
      case DicomVolume::Orientation::Sagittal: {
        double xslice =
            m_volume
                .voxelToPatient(
                    static_cast<double>(m_currentIndices[viewIndex]),
                    static_cast<double>(m_volume.height()) / 2.0,
                    static_cast<double>(m_volume.depth()) / 2.0)
                .x();
        draw =
            (xslice >= std::min(minX, maxX) && xslice <= std::max(minX, maxX));
        if (draw)
          addRect(0, xslice, 0, DicomVolume::Orientation::Sagittal);
        break;
      }
      case DicomVolume::Orientation::Coronal: {
        double yslice =
            m_volume
                .voxelToPatient(
                    static_cast<double>(m_volume.width()) / 2.0,
                    static_cast<double>(m_currentIndices[viewIndex]),
                    static_cast<double>(m_volume.depth()) / 2.0)
                .y();
        draw =
            (yslice >= std::min(minY, maxY) && yslice <= std::max(minY, maxY));
        if (draw)
          addRect(0, 0, yslice, DicomVolume::Orientation::Coronal);
        break;
      }
      }
    }
  }
  if (m_showSlicePosition && isVolumeLoaded()) {
    QVector<int> axIndices, sagIndices, corIndices;
    for (int i = 0; i < VIEW_COUNT; ++i) {
      if (!m_viewContainers[i]->isVisible() || m_isDVHView[i] ||
          m_is3DView[i] || m_isProfileView[i])
        continue;
      switch (m_viewOrientations[i]) {
      case DicomVolume::Orientation::Axial:
        axIndices.append(m_currentIndices[i]);
        break;
      case DicomVolume::Orientation::Sagittal:
        sagIndices.append(m_currentIndices[i]);
        break;
      case DicomVolume::Orientation::Coronal:
        corIndices.append(m_currentIndices[i]);
        break;
      }
    }

    double sx = m_volume.spacingX();
    double sy = m_volume.spacingY();
    double sz = m_volume.spacingZ();
    QColor lineColor = Qt::gray;

    switch (m_viewOrientations[viewIndex]) {
    case DicomVolume::Orientation::Axial: {
      double w = m_volume.width() * sx;
      double h = m_volume.height() * sy;
      for (int idx : sagIndices) {
        double x = idx * sx - w / 2.0;
        sliceLines.append(
            {{QPointF(x, h / 2.0), QPointF(x, -h / 2.0)}, lineColor});
      }
      for (int idx : corIndices) {
        double y = h / 2.0 - idx * sy;
        sliceLines.append(
            {{QPointF(-w / 2.0, y), QPointF(w / 2.0, y)}, lineColor});
      }
      break;
    }
    case DicomVolume::Orientation::Sagittal: {
      double w = m_volume.height() * sy;
      double h = m_volume.depth() * sz;
      for (int idx : corIndices) {
        double x = idx * sy - w / 2.0;
        sliceLines.append(
            {{QPointF(x, -h / 2.0), QPointF(x, h / 2.0)}, lineColor});
      }
      for (int idx : axIndices) {
        // Axialスライス位置の表示が頭尾方向で反転していたため、
        // インデックスをそのまま使用して座標を算出する。
        double y = idx * sz - h / 2.0;
        sliceLines.append(
            {{QPointF(-w / 2.0, y), QPointF(w / 2.0, y)}, lineColor});
      }
      break;
    }
    case DicomVolume::Orientation::Coronal: {
      double w = m_volume.width() * sx;
      double h = m_volume.depth() * sz;
      for (int idx : sagIndices) {
        double x = idx * sx - w / 2.0;
        sliceLines.append(
            {{QPointF(x, -h / 2.0), QPointF(x, h / 2.0)}, lineColor});
      }
      for (int idx : axIndices) {
        // Axialスライス位置の表示が頭尾方向で反転していたため修正
        double y = idx * sz - h / 2.0;
        sliceLines.append(
            {{QPointF(-w / 2.0, y), QPointF(w / 2.0, y)}, lineColor});
      }
      break;
    }
    }
  }
  if (m_profileLineVisible && viewIndex == m_profileLineView) {
    sliceLines.append(m_profileLine);
  }
  m_imageWidgets[viewIndex]->setSlicePositionLines(sliceLines);

  m_imageWidgets[viewIndex]->setDoseLines(doseLines);

  m_imageWidgets[viewIndex]->setImage(displayImg);

  float sx = 1.0f, sy = 1.0f;
  if (isVolumeLoaded()) {
    switch (m_viewOrientations[viewIndex]) {
    case DicomVolume::Orientation::Axial:
      sx = static_cast<float>(m_volume.spacingX());
      sy = static_cast<float>(m_volume.spacingY());
      break;
    case DicomVolume::Orientation::Sagittal:
      sx = static_cast<float>(m_volume.spacingY());
      sy = static_cast<float>(m_volume.spacingZ());
      break;
    case DicomVolume::Orientation::Coronal:
      sx = static_cast<float>(m_volume.spacingX());
      sy = static_cast<float>(m_volume.spacingZ());
      break;
    }
  } else {
    double row, col;
    m_dicomReader->getPixelSpacing(row, col);
    sx = static_cast<float>(col);
    sy = static_cast<float>(row);
  }
  m_imageWidgets[viewIndex]->setPixelSpacing(sx, sy);
  m_imageWidgets[viewIndex]->setZoom(m_zoomFactor);
  m_imageWidgets[viewIndex]->setPan(m_panOffsets[viewIndex]);
}

void DicomViewer::regenerateStructureSurfaceCache() {
  m_cachedStructureSurfaces.clear();

  if (!(m_rtstructLoaded && isVolumeLoaded())) {
    m_structureSurfacesDirty = false;
    return;
  }

  int roiCount = m_rtstruct.roiCount();
  m_cachedStructureSurfaces.resize(roiCount);

  m_structureSurfacesDirty = false;
}

void DicomViewer::invalidateStructureSurfaceCache() {
  m_structureSurfacesDirty = true;
  m_cachedStructureSurfaces.clear();
}

void DicomViewer::update3DView(int viewIndex) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT || !m_3dWidgets[viewIndex])
    return;
  if (!isVolumeLoaded())
    return;

  m_3dWidgets[viewIndex]->setSlices(
      m_orientationImages[0], m_orientationIndices[0], m_orientationImages[1],
      m_orientationIndices[1], m_orientationImages[2], m_orientationIndices[2],
      m_volume.width(), m_volume.height(), m_volume.depth(),
      m_volume.spacingX(), m_volume.spacingY(), m_volume.spacingZ());

  StructureLine3DList lines3D;
  if (m_rtstructLoaded && isVolumeLoaded()) {
    lines3D = m_rtstruct.allContours3D(m_volume);
  }
  m_3dWidgets[viewIndex]->setStructureLines(lines3D);

  if (m_structureSurfacesDirty) {
    regenerateStructureSurfaceCache();
  }

  QVector<StructureSurface> structureSurfaces;
  if (m_rtstructLoaded && isVolumeLoaded()) {
    int roiCount = m_rtstruct.roiCount();
    int cachedCount = m_cachedStructureSurfaces.size();
    for (int r = 0; r < roiCount && r < cachedCount; ++r) {
      if (!m_rtstruct.isROIVisible(r)) {
        continue;
      }

      auto &surface = m_cachedStructureSurfaces[r];
      if (surface.isEmpty()) {
        QVector<QVector<QVector3D>> contours = m_rtstruct.roiContoursPatient(r);
        if (!contours.isEmpty()) {
          QColor roiColor = QColor::fromHsv((r * 40) % 360, 255, 255, 255);
          surface.setColor(roiColor);
          surface.setOpacity(0.3f);
          surface.generateFromContours(contours, m_volume);
          surface.transformTo3DWidgetSpace(m_volume);
        }
      }

      if (!surface.isEmpty()) {
        structureSurfaces.append(surface);
      }
    }
  }
  m_3dWidgets[viewIndex]->setStructureSurfaces(structureSurfaces);

  // Build separate lists for time>0 and ==0 to enforce coloring.
  // Map patient coordinates to 3D widget's centered-mm space
  QVector<QVector3D> activePts;
  QVector<QVector3D> inactivePts;
  QVector<QPair<QVector3D, QVector3D>> activeSegs;
  QVector<QPair<QVector3D, QVector3D>> inactiveSegs;
  auto to3Dmm = [&](const QVector3D &patient) {
    QVector3D vox = m_volume.patientToVoxelContinuous(patient);
    double px_mm = m_volume.width() * m_volume.spacingX();
    double py_mm = m_volume.height() * m_volume.spacingY();
    double pz_mm = m_volume.depth() * m_volume.spacingZ();
    double x_mm = vox.x() * m_volume.spacingX() - px_mm / 2.0;
    double y_mm = vox.y() * m_volume.spacingY() - py_mm / 2.0;
    double z_mm = 0.0;
    if (m_volume.depth() > 1) {
      z_mm = ((vox.z() / (m_volume.depth() - 1.0)) - 0.5) * pz_mm;
    }
    return QVector3D(x_mm, y_mm, z_mm);
  };
  if (m_brachyLoaded && isVolumeLoaded()) {
    // Build per-channel ordered index list
    const auto &srcs = m_brachyPlan.sources();
    const float halfLen = 1.25f; // mm (half of 2.5mm)
    for (const auto &s : srcs) {
      QVector3D p = s.position();
      QVector3D dir = s.direction();
      if (dir.lengthSquared() > 0.0f) {
        QVector3D a = p - dir * halfLen;
        QVector3D b = p + dir * halfLen;
        QVector3D a3 = to3Dmm(a);
        QVector3D b3 = to3Dmm(b);
        if (s.dwellTime() > 0.0)
          activeSegs.append(qMakePair(a3, b3));
        else
          inactiveSegs.append(qMakePair(a3, b3));
      }
      QVector3D p3 = to3Dmm(p);
      if (s.dwellTime() > 0.0)
        activePts.append(p3);
      else
        inactivePts.append(p3);
    }
  }
  m_3dWidgets[viewIndex]->setActiveSourcePoints(activePts);
  m_3dWidgets[viewIndex]->setInactiveSourcePoints(inactivePts);
  m_3dWidgets[viewIndex]->setActiveSourceSegments(activeSegs);
  m_3dWidgets[viewIndex]->setInactiveSourceSegments(inactiveSegs);
  m_3dWidgets[viewIndex]->setStructureLineWidth(m_structureLineWidth);
  m_3dWidgets[viewIndex]->setZoom(m_zoomFactor * ZOOM_3D_RATIO);
  m_3dWidgets[viewIndex]->setPan(m_panOffsets[viewIndex]);
}

void DicomViewer::updateImageInfo() {
  QString name = "-";
  QString id = "-";
  QString modality = "-";
  QString studyDate = "-";
  QString size = "-";
  QString size3d = "-";
  QString studyDesc = "-";
  QString sliceThk = "-";
  QString pixelSpacing = "- x -";
  QString ctExtents = "-";

  if (isVolumeLoaded()) {
    name =
        m_privacyMode ? QStringLiteral("-") : m_dicomReader->getPatientName();
    id = m_privacyMode ? QStringLiteral("-") : m_dicomReader->getPatientID();
    double row, col;
    m_dicomReader->getPixelSpacing(row, col);
    pixelSpacing = QString("%1 x %2").arg(row).arg(col);
    modality = m_dicomReader->getModality();
    studyDate = m_dicomReader->getStudyDate();
    size = QString("%1 x %2")
               .arg(m_dicomReader->getWidth())
               .arg(m_dicomReader->getHeight());
    size3d = QString("%1 x %2 x %3")
                 .arg(m_volume.width())
                 .arg(m_volume.height())
                 .arg(m_volume.depth());
    studyDesc = m_dicomReader->getStudyDescription();
    sliceThk = QString::number(m_dicomReader->getSliceThickness());

    // CT Extents in patient space (mm)
    if (m_volume.width() > 0 && m_volume.height() > 0 && m_volume.depth() > 0) {
      auto upd = [](double &mn, double &mx, double v) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
      };
      double minX = std::numeric_limits<double>::infinity();
      double minY = std::numeric_limits<double>::infinity();
      double minZ = std::numeric_limits<double>::infinity();
      double maxX = -std::numeric_limits<double>::infinity();
      double maxY = -std::numeric_limits<double>::infinity();
      double maxZ = -std::numeric_limits<double>::infinity();
      int xs[2] = {0, m_volume.width() - 1};
      int ys[2] = {0, m_volume.height() - 1};
      int zs[2] = {0, m_volume.depth() - 1};
      for (int ix : xs)
        for (int iy : ys)
          for (int iz : zs) {
            QVector3D p = m_volume.voxelToPatient(ix + 0.5, iy + 0.5, iz + 0.5);
            upd(minX, maxX, p.x());
            upd(minY, maxY, p.y());
            upd(minZ, maxZ, p.z());
          }
      ctExtents = QString("X:[%1,%2] Y:[%3,%4] Z:[%5,%6]")
                      .arg(minX, 0, 'f', 2)
                      .arg(maxX, 0, 'f', 2)
                      .arg(minY, 0, 'f', 2)
                      .arg(maxY, 0, 'f', 2)
                      .arg(minZ, 0, 'f', 2)
                      .arg(maxZ, 0, 'f', 2);
    }
  }

  QString ctFile = m_ctFilename.isEmpty() ? QStringLiteral("-") : m_ctFilename;
  QString text = QString("Patient: %1 (%2)\n"
                         "Modality: %3\n"
                         "Study Date: %4\n"
                         "Size: %5\n"
                         "Size (vox 3D): %6\n"
                         "Study Desc: %6\n"
                         "Slice Thk: %7\n"
                         "Pixel Spacing: %8\n"
                         "CT Extents (mm): %9\n"
                         "CT File: %10")
                     .arg(name)
                     .arg(id)
                     .arg(modality)
                     .arg(studyDate)
                     .arg(size)
                     .arg(size3d)
                     .arg(studyDesc)
                     .arg(sliceThk)
                     .arg(pixelSpacing)
                     .arg(ctExtents)
                     .arg(ctFile);

  if (m_doseLoaded) {
    QString doseSize = QString("%1 x %2 x %3")
                           .arg(m_doseVolume.width())
                           .arg(m_doseVolume.height())
                           .arg(m_doseVolume.depth());
    QString doseSpacing = QString("%1 x %2 x %3")
                              .arg(m_doseVolume.spacingX())
                              .arg(m_doseVolume.spacingY())
                              .arg(m_doseVolume.spacingZ());
    // Show both native origin (IPP) and aligned origin (after patientShift)
    QString doseOrigin = QString("(%1, %2, %3)")
                             .arg(m_doseVolume.originX())
                             .arg(m_doseVolume.originY())
                             .arg(m_doseVolume.originZ());
    QVector3D alignedOriginV(
        m_doseVolume.originX() + m_doseVolume.patientShift().x(),
        m_doseVolume.originY() + m_doseVolume.patientShift().y(),
        m_doseVolume.originZ() + m_doseVolume.patientShift().z());
    QString doseOriginAligned = QString("(%1, %2, %3)")
                                    .arg(alignedOriginV.x())
                                    .arg(alignedOriginV.y())
                                    .arg(alignedOriginV.z());
    QString doseFile =
        m_rtDoseFilename.isEmpty() ? QStringLiteral("-") : m_rtDoseFilename;

    // RTDose extents in patient space (mm) - use native (no patientShift)
    QString doseExtents = "-";
    if (m_doseVolume.width() > 0 && m_doseVolume.height() > 0 &&
        m_doseVolume.depth() > 0) {
      auto upd = [](double &mn, double &mx, double v) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
      };
      double minX = std::numeric_limits<double>::infinity();
      double minY = std::numeric_limits<double>::infinity();
      double minZ = std::numeric_limits<double>::infinity();
      double maxX = -std::numeric_limits<double>::infinity();
      double maxY = -std::numeric_limits<double>::infinity();
      double maxZ = -std::numeric_limits<double>::infinity();
      int xs[2] = {0, m_doseVolume.width() - 1};
      int ys[2] = {0, m_doseVolume.height() - 1};
      int zs[2] = {0, m_doseVolume.depth() - 1};
      for (int ix : xs)
        for (int iy : ys)
          for (int iz : zs) {
            // Origin is now at voxel center, so use integer indices directly
            QVector3D p =
                m_doseVolume.voxelToPatientNative(ix, iy, iz);
            upd(minX, maxX, p.x());
            upd(minY, maxY, p.y());
            upd(minZ, maxZ, p.z());
          }
      doseExtents = QString("X:[%1,%2] Y:[%3,%4] Z:[%5,%6]")
                        .arg(minX, 0, 'f', 2)
                        .arg(maxX, 0, 'f', 2)
                        .arg(minY, 0, 'f', 2)
                        .arg(maxY, 0, 'f', 2)
                        .arg(minZ, 0, 'f', 2)
                        .arg(maxZ, 0, 'f', 2);
    }
    text += QString("\n\nRT Dose File: %1\n"
                    " Size: %2\n"
                    " Spacing: %3\n"
                    " Max Dose: %4\n"
                    " Origin (native): %5\n"
                    " Origin (aligned): %6\n"
                    " Extents (mm, native): %7")
                .arg(doseFile)
                .arg(doseSize)
                .arg(doseSpacing)
                .arg(m_doseVolume.maxDose())
                .arg(doseOrigin)
                .arg(doseOriginAligned)
                .arg(doseExtents);
  } else {
    text += "\n\nRT Dose: Not Loaded";
  }

  m_infoTextBox->setPlainText(text);
}

void DicomViewer::setZoomFactor(double factor) {
  double maxZoom =
      m_is3DView[m_activeViewIndex] ? (MAX_ZOOM_3D / ZOOM_3D_RATIO) : MAX_ZOOM;
  m_zoomFactor = qBound(MIN_ZOOM, factor, maxZoom);

  if (m_syncScale) {
    syncZoomToAllViews(m_zoomFactor);
  } else {
    int count = 1;
    if (m_viewMode == ViewMode::Dual)
      count = 2;
    else if (m_viewMode == ViewMode::Quad)
      count = 4;
    else if (m_viewMode == ViewMode::Five)
      count = VIEW_COUNT;
    for (int i = 0; i < count; ++i) {
      if (m_is3DView[i])
        m_3dWidgets[i]->setZoom(m_zoomFactor * ZOOM_3D_RATIO);
      else
        updateImage(i);
    }
  }
  updateSliderPosition();
}

DicomStudyInfo DicomViewer::currentStudyInfo() const {
  DicomStudyInfo info;
  if (m_dicomReader) {
    if (!m_privacyMode) {
      info.patientID = m_dicomReader->getPatientID();
      info.patientName = m_dicomReader->getPatientName();
    }
    info.modality = m_dicomReader->getModality();
    info.studyDescription = m_dicomReader->getStudyDescription();
    info.studyDate = m_dicomReader->getStudyDate();
    info.frameOfReferenceUID = m_dicomReader->getFrameOfReferenceUID();
  }
  if (isVolumeLoaded()) {
    const QString frameUid = m_volume.frameOfReferenceUID();
    if (!frameUid.isEmpty())
      info.frameOfReferenceUID = frameUid;
  }
  if (!m_dicomFiles.isEmpty()) {
    const QFileInfo infoFile(m_dicomFiles.first());
    info.seriesDirectory = infoFile.absolutePath();
  } else if (m_activeImageSeriesIndex >= 0 &&
             m_activeImageSeriesIndex < m_imageSeriesDirs.size()) {
    info.seriesDirectory = m_imageSeriesDirs.at(m_activeImageSeriesIndex);
  }
  return info;
}

QString DicomViewer::patientInfoText() const {
  if (m_privacyMode) {
    return QStringLiteral("Privacy Mode");
  }
  return QString("ID: %1\nName: %2")
      .arg(m_dicomReader->getPatientID())
      .arg(m_dicomReader->getPatientName());
}

void DicomViewer::updateInfoOverlays() {
  QString text = patientInfoText();
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_isDVHView[i] || m_is3DView[i] || m_isProfileView[i]) {
      m_infoOverlays[i]->hide();
      if (m_dvhWidgets[i])
        m_dvhWidgets[i]->setPatientInfo(text);
    } else {
      m_infoOverlays[i]->setText(text);
      m_infoOverlays[i]->adjustSize();
      m_infoOverlays[i]->show();
    }
  }
  updateSliderPosition();
}

QVector3D DicomViewer::patientCoordinateAt(int viewIndex,
                                           const QPoint &pos) const {
  if (!isVolumeLoaded())
    return QVector3D(qQNaN(), qQNaN(), qQNaN());

  const OpenGLImageWidget *widget = m_imageWidgets[viewIndex];
  if (!widget)
    return QVector3D(qQNaN(), qQNaN(), qQNaN());

  int ww = widget->width();
  int wh = widget->height();
  if (ww <= 0 || wh <= 0)
    return QVector3D(qQNaN(), qQNaN(), qQNaN());

  QPointF pan = widget->pan();
  float zoom = widget->zoom();

  float spacingX = static_cast<float>(m_volume.spacingX());
  float spacingY = static_cast<float>(m_volume.spacingY());
  float spacingZ = static_cast<float>(m_volume.spacingZ());

  float w_mm = 0.0f, h_mm = 0.0f;
  switch (m_viewOrientations[viewIndex]) {
  case DicomVolume::Orientation::Axial:
    w_mm = m_volume.width() * spacingX;
    h_mm = m_volume.height() * spacingY;
    break;
  case DicomVolume::Orientation::Sagittal:
    w_mm = m_volume.height() * spacingY;
    h_mm = m_volume.depth() * spacingZ;
    break;
  case DicomVolume::Orientation::Coronal:
    w_mm = m_volume.width() * spacingX;
    h_mm = m_volume.depth() * spacingZ;
    break;
  }
  float maxDim = std::max(w_mm, h_mm);
  if (maxDim <= 0.0f)
    return QVector3D(qQNaN(), qQNaN(), qQNaN());

  // ウィジェット座標 -> 正規化デバイス座標
  float ndcX = 2.0f * pos.x() / ww - 1.0f;
  float ndcY = 1.0f - 2.0f * pos.y() / wh;

  // パン/ズーム補正
  ndcX -= 2.0f * pan.x() / ww;
  ndcY += 2.0f * pan.y() / wh;
  ndcX /= zoom;
  ndcY /= zoom;

  // アスペクト比補正
  float aspect = static_cast<float>(ww) / static_cast<float>(wh);
  if (aspect > 1.0f)
    ndcX *= aspect;
  else
    ndcY /= aspect;

  // mm単位に戻す
  float x_mm = ndcX * (maxDim * 0.5f);
  float y_mm = ndcY * (maxDim * 0.5f);

  if (x_mm < -w_mm / 2.0f || x_mm > w_mm / 2.0f || y_mm < -h_mm / 2.0f ||
      y_mm > h_mm / 2.0f)
    return QVector3D(qQNaN(), qQNaN(), qQNaN());

  double vx = 0.0, vy = 0.0, vz = 0.0;
  switch (m_viewOrientations[viewIndex]) {
  case DicomVolume::Orientation::Axial:
    vx = (x_mm + w_mm / 2.0f) / spacingX;
    vy = (h_mm / 2.0f - y_mm) / spacingY;
    vz = m_currentIndices[viewIndex];
    break;
  case DicomVolume::Orientation::Sagittal:
    vx = m_currentIndices[viewIndex];
    vy = (x_mm + w_mm / 2.0f) / spacingY;
    // SagittalビューではY方向の動きが患者座標系のZ軸に対応するが、
    // 元の実装では上下方向が反転していたため、ここで補正する。
    // 上方向の移動でZ値が増加するように、符号を反転させる。
    vz = (y_mm + h_mm / 2.0f) / spacingZ;
    break;
  case DicomVolume::Orientation::Coronal:
    vx = (x_mm + w_mm / 2.0f) / spacingX;
    vy = m_currentIndices[viewIndex];
    // Coronalビューでも同様に上下方向のZ軸を反転させる。
    vz = (y_mm + h_mm / 2.0f) / spacingZ;
    break;
  }

  return m_volume.voxelToPatient(vx, vy, vz);
}

QPointF
DicomViewer::planeCoordinateFromPatient(int viewIndex,
                                        const QVector3D &patient) const {
  QVector3D vox = m_volume.patientToVoxelContinuous(patient);
  double sx = m_volume.spacingX();
  double sy = m_volume.spacingY();
  double sz = m_volume.spacingZ();
  switch (m_viewOrientations[viewIndex]) {
  case DicomVolume::Orientation::Axial: {
    double w = m_volume.width() * sx;
    double h = m_volume.height() * sy;
    double x_mm = vox.x() * sx - w / 2.0;
    double y_mm = h / 2.0 - vox.y() * sy;
    return QPointF(x_mm, y_mm);
  }
  case DicomVolume::Orientation::Sagittal: {
    double w = m_volume.height() * sy;
    double h = m_volume.depth() * sz;
    double x_mm = vox.y() * sy - w / 2.0;
    double y_mm = vox.z() * sz - h / 2.0;
    return QPointF(x_mm, y_mm);
  }
  case DicomVolume::Orientation::Coronal: {
    double w = m_volume.width() * sx;
    double h = m_volume.depth() * sz;
    double x_mm = vox.x() * sx - w / 2.0;
    double y_mm = vox.z() * sz - h / 2.0;
    return QPointF(x_mm, y_mm);
  }
  }
  return QPointF();
}

float DicomViewer::sampleResampledDose(const QVector3D &voxel) const {
  auto sampled = sampleDoseValue(voxel);
  return sampled.has_value() ? static_cast<float>(sampled.value()) : 0.0f;
}

std::optional<double> DicomViewer::sampleCtValue(const QVector3D &voxel) const {
  const cv::Mat &vol = m_volume.data();
  if (vol.empty())
    return std::nullopt;
  const int w = m_volume.width();
  const int h = m_volume.height();
  const int d = m_volume.depth();
  const double x = voxel.x();
  const double y = voxel.y();
  const double z = voxel.z();
  if (x < 0.0 || y < 0.0 || z < 0.0 || x >= w - 1 || y >= h - 1 ||
      z >= d - 1)
    return std::nullopt;

  const int type = vol.type();
  if (type != CV_16SC1 && type != CV_16UC1 && type != CV_8UC1 &&
      type != CV_32FC1)
    return std::nullopt;

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int z0 = static_cast<int>(std::floor(z));
  const double xd = x - x0;
  const double yd = y - y0;
  const double zd = z - z0;

  auto valueAt = [&](int zi, int yi, int xi) -> double {
    switch (type) {
    case CV_16SC1:
      return static_cast<double>(vol.at<short>(zi, yi, xi));
    case CV_16UC1:
      return static_cast<double>(vol.at<unsigned short>(zi, yi, xi));
    case CV_8UC1:
      return static_cast<double>(vol.at<uchar>(zi, yi, xi));
    case CV_32FC1:
      return static_cast<double>(vol.at<float>(zi, yi, xi));
    default:
      return 0.0;
    }
  };

  const double c000 = valueAt(z0, y0, x0);
  const double c100 = valueAt(z0, y0, x0 + 1);
  const double c010 = valueAt(z0, y0 + 1, x0);
  const double c110 = valueAt(z0, y0 + 1, x0 + 1);
  const double c001 = valueAt(z0 + 1, y0, x0);
  const double c101 = valueAt(z0 + 1, y0, x0 + 1);
  const double c011 = valueAt(z0 + 1, y0 + 1, x0);
  const double c111 = valueAt(z0 + 1, y0 + 1, x0 + 1);

  const double c00 = c000 * (1.0 - xd) + c100 * xd;
  const double c01 = c001 * (1.0 - xd) + c101 * xd;
  const double c10 = c010 * (1.0 - xd) + c110 * xd;
  const double c11 = c011 * (1.0 - xd) + c111 * xd;
  const double c0 = c00 * (1.0 - yd) + c10 * yd;
  const double c1 = c01 * (1.0 - yd) + c11 * yd;
  return c0 * (1.0 - zd) + c1 * zd;
}

std::optional<double> DicomViewer::sampleDoseValue(const QVector3D &voxel) const {
  if (!m_resampledDose.isResampled())
    return std::nullopt;
  const int w = m_resampledDose.width();
  const int h = m_resampledDose.height();
  const int d = m_resampledDose.depth();
  const double x = voxel.x();
  const double y = voxel.y();
  const double z = voxel.z();
  if (x < 0.0 || y < 0.0 || z < 0.0 || x >= w - 1 || y >= h - 1 ||
      z >= d - 1)
    return std::nullopt;

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int z0 = static_cast<int>(std::floor(z));
  const double xd = x - x0;
  const double yd = y - y0;
  const double zd = z - z0;
  const cv::Mat &vol = m_resampledDose.data();

  auto valueAt = [&](int zi, int yi, int xi) -> double {
    return static_cast<double>(vol.at<float>(zi, yi, xi));
  };

  const double c000 = valueAt(z0, y0, x0);
  const double c100 = valueAt(z0, y0, x0 + 1);
  const double c010 = valueAt(z0, y0 + 1, x0);
  const double c110 = valueAt(z0, y0 + 1, x0 + 1);
  const double c001 = valueAt(z0 + 1, y0, x0);
  const double c101 = valueAt(z0 + 1, y0, x0 + 1);
  const double c011 = valueAt(z0 + 1, y0 + 1, x0);
  const double c111 = valueAt(z0 + 1, y0 + 1, x0 + 1);

  const double c00 = c000 * (1.0 - xd) + c100 * xd;
  const double c01 = c001 * (1.0 - xd) + c101 * xd;
  const double c10 = c010 * (1.0 - xd) + c110 * xd;
  const double c11 = c011 * (1.0 - xd) + c111 * xd;
  const double c0 = c00 * (1.0 - yd) + c10 * yd;
  const double c1 = c01 * (1.0 - yd) + c11 * yd;
  return c0 * (1.0 - zd) + c1 * zd;
}

void DicomViewer::updateCoordLabel(int viewIndex, const QPoint &pos) {
  if (!isVolumeLoaded() || viewIndex < 0 || viewIndex >= VIEW_COUNT ||
      m_isDVHView[viewIndex] || m_is3DView[viewIndex] ||
      m_isProfileView[viewIndex]) {
    if (viewIndex >= 0 && viewIndex < VIEW_COUNT) {
      if (m_coordLabels[viewIndex])
        m_coordLabels[viewIndex]->hide();
      if (m_cursorDoseLabels[viewIndex])
        m_cursorDoseLabels[viewIndex]->hide();
      m_imageWidgets[viewIndex]->clearCursorCross();
      // Clear stored coordinates
      m_lastPatientCoordinates[viewIndex] = QVector3D(qQNaN(), qQNaN(), qQNaN());
    }
    return;
  }
  QVector3D patient = patientCoordinateAt(viewIndex, pos);
  if (std::isnan(patient.x())) {
    m_coordLabels[viewIndex]->hide();
    if (m_cursorDoseLabels[viewIndex])
      m_cursorDoseLabels[viewIndex]->hide();
    m_imageWidgets[viewIndex]->clearCursorCross();
    // Clear stored coordinates
    m_lastPatientCoordinates[viewIndex] = QVector3D(qQNaN(), qQNaN(), qQNaN());
    return;
  }

  // Store the current patient coordinates
  m_lastPatientCoordinates[viewIndex] = patient;

  QString text = QString("X:%1 mm Y:%2 mm Z:%3 mm")
                     .arg(patient.x(), 0, 'f', 1)
                     .arg(patient.y(), 0, 'f', 1)
                     .arg(patient.z(), 0, 'f', 1);

  QVector3D voxel = m_volume.patientToVoxelContinuous(patient);
  int ix = static_cast<int>(std::round(voxel.x()));
  int iy = static_cast<int>(std::round(voxel.y()));
  int iz = static_cast<int>(std::round(voxel.z()));
  const bool voxelInBounds =
      ix >= 0 && ix < m_volume.width() && iy >= 0 && iy < m_volume.height() &&
      iz >= 0 && iz < m_volume.depth();

  bool ctShown = false;
  int ctValue = 0;
  if (voxelInBounds) {
    const cv::Mat &volumeData = m_volume.data();
    if (!volumeData.empty()) {
      switch (volumeData.type()) {
      case CV_16SC1:
        ctValue = static_cast<int>(volumeData.at<short>(iz, iy, ix));
        ctShown = true;
        break;
      case CV_16UC1:
        ctValue =
            static_cast<int>(volumeData.at<unsigned short>(iz, iy, ix));
        ctShown = true;
        break;
      case CV_8UC1:
        ctValue = static_cast<int>(volumeData.at<uchar>(iz, iy, ix));
        ctShown = true;
        break;
      case CV_32FC1:
        ctValue =
            static_cast<int>(std::lround(volumeData.at<float>(iz, iy, ix)));
        ctShown = true;
        break;
      default:
        break;
      }
    }
  }
  if (ctShown) {
    text += QString(" CT:%1 HU").arg(ctValue);
  }

  bool doseShown = false;
  float doseValue = std::numeric_limits<float>::quiet_NaN();
  if (voxelInBounds && m_doseLoaded && m_resampledDose.isResampled()) {
    float dose = m_resampledDose.voxelDose(ix, iy, iz);
    doseValue = dose;
    if (m_cursorDoseLabels[viewIndex]) {
      m_cursorDoseLabels[viewIndex]->setText(
          QString("%1 Gy").arg(dose, 0, 'f', 2));
      m_cursorDoseLabels[viewIndex]->adjustSize();
      QPoint lp =
          m_imageWidgets[viewIndex]->mapTo(m_viewContainers[viewIndex], pos);
      int x = lp.x() + 15;
      int y = lp.y() + 15;
      x = std::min(x, m_viewContainers[viewIndex]->width() -
                          m_cursorDoseLabels[viewIndex]->width());
      y = std::min(y, m_viewContainers[viewIndex]->height() -
                          m_cursorDoseLabels[viewIndex]->height());
      m_cursorDoseLabels[viewIndex]->move(x, y);
      m_cursorDoseLabels[viewIndex]->show();

      // カーソル位置のクロスマーク用座標を計算（mm単位）
      float w_mm = 0.0f, h_mm = 0.0f, cx_mm = 0.0f, cy_mm = 0.0f;
      switch (m_viewOrientations[viewIndex]) {
      case DicomVolume::Orientation::Axial:
        w_mm = m_volume.width() * m_volume.spacingX();
        h_mm = m_volume.height() * m_volume.spacingY();
        cx_mm = voxel.x() * m_volume.spacingX() - w_mm / 2.0f;
        cy_mm = h_mm / 2.0f - voxel.y() * m_volume.spacingY();
        break;
      case DicomVolume::Orientation::Sagittal:
        w_mm = m_volume.height() * m_volume.spacingY();
        h_mm = m_volume.depth() * m_volume.spacingZ();
        cx_mm = voxel.y() * m_volume.spacingY() - w_mm / 2.0f;
        // これまではZ軸方向が反転していたため、ボクセルのZ値から
        // 上下方向の座標を算出する際に符号を反転させる。
        cy_mm = voxel.z() * m_volume.spacingZ() - h_mm / 2.0f;
        break;
      case DicomVolume::Orientation::Coronal:
        w_mm = m_volume.width() * m_volume.spacingX();
        h_mm = m_volume.depth() * m_volume.spacingZ();
        cx_mm = voxel.x() * m_volume.spacingX() - w_mm / 2.0f;
        cy_mm = voxel.z() * m_volume.spacingZ() - h_mm / 2.0f;
        break;
      }
      m_imageWidgets[viewIndex]->setCursorCross(QPointF(cx_mm, cy_mm));
      doseShown = true;
    }
  }
  if (!std::isnan(doseValue)) {
    text += QString(" Dose:%1 Gy").arg(doseValue, 0, 'f', 2);
  }

  if (!doseShown) {
    if (m_cursorDoseLabels[viewIndex])
      m_cursorDoseLabels[viewIndex]->hide();
    m_imageWidgets[viewIndex]->clearCursorCross();
  }

  m_coordLabels[viewIndex]->setText(text);
  m_coordLabels[viewIndex]->adjustSize();
  m_coordLabels[viewIndex]->show();
  updateSliderPosition();
}

void DicomViewer::startWindowLevelDrag() {
  m_windowLevelDragActive = true;
  m_windowLevelButton->setChecked(true);
  setCursor(Qt::OpenHandCursor);

  // 自動終了タイマーを開始
  m_windowLevelTimer->start(WINDOW_LEVEL_TIMEOUT);

  qDebug() << "Window/Level drag mode started";
  updateOverlayInteractionStates();
}

void DicomViewer::stopWindowLevelDrag() {
  m_windowLevelDragActive = false;
  m_windowLevelButton->setChecked(false);
  setCursor(Qt::ArrowCursor);

  // タイマーを停止
  m_windowLevelTimer->stop();

  qDebug() << "Window/Level drag mode stopped";
  updateOverlayInteractionStates();
}

void DicomViewer::updateWindowLevelFromMouse(const QPoint &currentPos) {
  if (!m_windowLevelDragActive)
    return;

  QPoint delta = currentPos - m_dragStartPos;

  // 左右の動きでWindow調整
  double windowDelta = delta.x() * WINDOW_LEVEL_SENSITIVITY;
  double newWindow = qBound(1.0, m_dragStartWindow + windowDelta, 4096.0);

  // 上下の動きでLevel調整（上向きで増加）
  double levelDelta = -delta.y() * WINDOW_LEVEL_SENSITIVITY;
  double newLevel = qBound(-1024.0, m_dragStartLevel + levelDelta, 3072.0);

  // スライダーとスピンボックスを更新
  m_windowSlider->setValue(static_cast<int>(newWindow));
  m_levelSlider->setValue(static_cast<int>(newLevel));
  m_windowSpinBox->setValue(static_cast<int>(newWindow));
  m_levelSpinBox->setValue(static_cast<int>(newLevel));

  // 画像を更新
  setWindowLevel(newWindow, newLevel);

  // シグナルを送信
  emit windowLevelChanged(newWindow, newLevel);
}

void DicomViewer::onImageSliderChanged(int value) {
  QSlider *senderSlider = qobject_cast<QSlider *>(sender());
  int viewIndex = -1;
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_sliceSliders[i] == senderSlider) {
      viewIndex = i;
      break;
    }
  }
  int count = isVolumeLoaded()
                  ? sliceCountForOrientation(m_viewOrientations[viewIndex])
                  : m_dicomFiles.size();
  if (viewIndex >= 0 && value >= 0 && value < count &&
      value != m_currentIndices[viewIndex]) {
    m_activeViewIndex = viewIndex;
    loadSlice(viewIndex, value);
    updateSliceLabels();
  }
}

void DicomViewer::resizeEvent(QResizeEvent *event) {
  QWidget::resizeEvent(event);
  updateSliderPosition();
}

void DicomViewer::updateSliderPosition() {
  auto arrangeLeftButtons = [&](int viewIndex) {
    int x = 4;
    int maxHeight = 0;
    if (viewIndex >= 0 && viewIndex < VIEW_COUNT) {
      if (m_orientationButtons[viewIndex]) {
        m_orientationButtons[viewIndex]->move(x, 4);
        m_orientationButtons[viewIndex]->raise();
        m_orientationButtons[viewIndex]->show();
        maxHeight = std::max(maxHeight, m_orientationButtons[viewIndex]->height());
        x += m_orientationButtons[viewIndex]->width() + 4;
      }
      if (m_imageToggleButtons[viewIndex] &&
          m_imageToggleButtons[viewIndex]->isVisible()) {
        m_imageToggleButtons[viewIndex]->move(x, 4);
        m_imageToggleButtons[viewIndex]->raise();
        maxHeight = std::max(maxHeight, m_imageToggleButtons[viewIndex]->height());
        x += m_imageToggleButtons[viewIndex]->width() + 4;
      }
      if (m_exportButtons[viewIndex] &&
          m_exportButtons[viewIndex]->isVisible()) {
        m_exportButtons[viewIndex]->move(x, 4);
        m_exportButtons[viewIndex]->raise();
        maxHeight = std::max(maxHeight, m_exportButtons[viewIndex]->height());
        x += m_exportButtons[viewIndex]->width() + 4;
      }
      if (m_imageSeriesButtons[viewIndex] &&
          m_imageSeriesButtons[viewIndex]->isVisible()) {
        m_imageSeriesButtons[viewIndex]->move(x, 4);
        m_imageSeriesButtons[viewIndex]->raise();
        maxHeight = std::max(maxHeight, m_imageSeriesButtons[viewIndex]->height());
        x += m_imageSeriesButtons[viewIndex]->width() + 4;
      }

      // Second row: Line and Surface toggle buttons
      int secondRowY = 4 + maxHeight + 4; // Below first row with 4px spacing
      int x2 = 4;
      if (m_lineToggleButtons[viewIndex] &&
          m_lineToggleButtons[viewIndex]->isVisible()) {
        m_lineToggleButtons[viewIndex]->move(x2, secondRowY);
        m_lineToggleButtons[viewIndex]->raise();
        x2 += m_lineToggleButtons[viewIndex]->width() + 4;
      }
      if (m_surfaceToggleButtons[viewIndex] &&
          m_surfaceToggleButtons[viewIndex]->isVisible()) {
        m_surfaceToggleButtons[viewIndex]->move(x2, secondRowY);
        m_surfaceToggleButtons[viewIndex]->raise();
        x2 += m_surfaceToggleButtons[viewIndex]->width() + 4;
      }
    }
    return maxHeight;
  };

  auto arrangeRightButtons = [&](int viewIndex, int sliderWidth,
                                 int colorBarWidth) {
    int maxHeight = 0;
    if (viewIndex >= 0 && viewIndex < VIEW_COUNT) {
      constexpr int rightMargin = 8;
      int x = m_viewContainers[viewIndex]->width() - sliderWidth -
              colorBarWidth - rightMargin;
      auto placeButton = [&](QPushButton *button) {
        if (!button || !button->isVisible())
          return;
        button->adjustSize();
        x -= button->width();
        button->move(x, 4);
        button->raise();
        maxHeight = std::max(maxHeight, button->height());
        x -= 4;
      };
      placeButton(m_viewZoomButtons[viewIndex]);
      placeButton(m_viewPanButtons[viewIndex]);
      placeButton(m_viewWindowLevelButtons[viewIndex]);
    }
    return maxHeight;
  };

  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (!m_sliceIndexLabels[i])
      continue;
    updateInteractionButtonVisibility(i);
    if (m_isDVHView[i] || m_is3DView[i] || m_isProfileView[i]) {
      int sliderWidth =
          m_sliceSliders[i]->isVisible() ? m_sliceSliders[i]->width() : 0;
      arrangeLeftButtons(i);
      arrangeRightButtons(i, sliderWidth, 0);
      continue;
    }
    int sliderWidth =
        m_sliceSliders[i]->isVisible() ? m_sliceSliders[i]->width() : 0;
    int colorBarWidth = 0;
    if (m_colorBars[i] && m_colorBars[i]->isVisible()) {
      colorBarWidth = m_colorBars[i]->width();
      if (colorBarWidth == 0)
        colorBarWidth = m_colorBars[i]->preferredWidth();
    }
    QSize sz = m_sliceIndexLabels[i]->sizeHint();
    int x = m_viewContainers[i]->width() - sliderWidth - sz.width() - 4;
    int y = m_viewContainers[i]->height() - sz.height() - 4;
    m_sliceIndexLabels[i]->move(x, y);
    m_sliceIndexLabels[i]->raise();

    int leftHeight = arrangeLeftButtons(i);
    int rightHeight = arrangeRightButtons(i, sliderWidth, colorBarWidth);
    int topHeight = std::max(leftHeight, rightHeight);
    if (m_infoOverlays[i]) {
      int offsetY = 4 + topHeight;
      m_infoOverlays[i]->move(4, offsetY);
      m_infoOverlays[i]->raise();
    }
    if (m_doseShiftLabels[i]) {
      QSize sz = m_doseShiftLabels[i]->sizeHint();
      int x = (m_viewContainers[i]->width() - sz.width()) / 2;
      int y = 4 + topHeight; // top-center avoiding buttons
      m_doseShiftLabels[i]->move(x, y);
      m_doseShiftLabels[i]->raise();
    }
    if (m_coordLabels[i]) {
      int cy = m_viewContainers[i]->height() -
               (m_sliceIndexLabels[i]->isVisible()
                    ? m_sliceIndexLabels[i]->sizeHint().height()
                    : 0) -
               m_coordLabels[i]->sizeHint().height() - 8;
      m_coordLabels[i]->move(4, cy);
      m_coordLabels[i]->raise();
    }
  }

  // 注意: 再リサンプルは呼び出し側で一度だけ実行する（無限更新の回避）
}

void DicomViewer::updateInteractionButtonVisibility(int viewIndex) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return;
  if (!m_viewContainers[viewIndex])
    return;

  bool viewVisible = m_viewContainers[viewIndex]->isVisible();
  if (!viewVisible) {
    if (m_viewWindowLevelButtons[viewIndex])
      m_viewWindowLevelButtons[viewIndex]->hide();
    if (m_viewPanButtons[viewIndex])
      m_viewPanButtons[viewIndex]->hide();
    if (m_viewZoomButtons[viewIndex])
      m_viewZoomButtons[viewIndex]->hide();
    if (m_imageToggleButtons[viewIndex])
      m_imageToggleButtons[viewIndex]->hide();
    if (m_lineToggleButtons[viewIndex])
      m_lineToggleButtons[viewIndex]->hide();
    if (m_surfaceToggleButtons[viewIndex])
      m_surfaceToggleButtons[viewIndex]->hide();
    if (m_exportButtons[viewIndex])
      m_exportButtons[viewIndex]->hide();
    return;
  }

  bool hasImageContent = isVolumeLoaded() || !m_dicomFiles.isEmpty();
  bool show = hasImageContent && !m_isDVHView[viewIndex] &&
              !m_isProfileView[viewIndex];
  if (m_is3DView[viewIndex])
    show = show && isVolumeLoaded();

  if (m_viewWindowLevelButtons[viewIndex])
    m_viewWindowLevelButtons[viewIndex]->setVisible(show);
  if (m_viewPanButtons[viewIndex])
    m_viewPanButtons[viewIndex]->setVisible(show);
  if (m_viewZoomButtons[viewIndex])
    m_viewZoomButtons[viewIndex]->setVisible(show);

  // Show Image toggle button only in 3D view
  if (m_imageToggleButtons[viewIndex]) {
    bool showImageToggle = m_is3DView[viewIndex] && hasImageContent && isVolumeLoaded();
    m_imageToggleButtons[viewIndex]->setVisible(showImageToggle);
    if (showImageToggle) {
      m_imageToggleButtons[viewIndex]->adjustSize();
    }
  }

  // Show Line toggle button only in 3D view with RT Structure
  if (m_lineToggleButtons[viewIndex]) {
    bool showLineToggle = m_is3DView[viewIndex] && hasImageContent && isVolumeLoaded() && m_rtstructLoaded;
    m_lineToggleButtons[viewIndex]->setVisible(showLineToggle);
    if (showLineToggle) {
      m_lineToggleButtons[viewIndex]->adjustSize();
    }
  }

  // Show Surface toggle button only in 3D view with RT Structure
  if (m_surfaceToggleButtons[viewIndex]) {
    bool showSurfaceToggle = m_is3DView[viewIndex] && hasImageContent && isVolumeLoaded() && m_rtstructLoaded;
    m_surfaceToggleButtons[viewIndex]->setVisible(showSurfaceToggle);
    if (showSurfaceToggle) {
      m_surfaceToggleButtons[viewIndex]->adjustSize();
    }
  }

  // Show Export button only in 3D view
  if (m_exportButtons[viewIndex]) {
    bool showExport = m_is3DView[viewIndex] && hasImageContent && isVolumeLoaded();
    m_exportButtons[viewIndex]->setVisible(showExport);
    if (showExport) {
      m_exportButtons[viewIndex]->adjustSize();
    }
  }
}

void DicomViewer::updateOverlayInteractionStates() {
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_viewWindowLevelButtons[i]) {
      QSignalBlocker blocker(m_viewWindowLevelButtons[i]);
      m_viewWindowLevelButtons[i]->setChecked(m_windowLevelDragActive);
    }
    if (m_viewPanButtons[i]) {
      QSignalBlocker blocker(m_viewPanButtons[i]);
      m_viewPanButtons[i]->setChecked(m_panMode);
    }
    if (m_viewZoomButtons[i]) {
      QSignalBlocker blocker(m_viewZoomButtons[i]);
      m_viewZoomButtons[i]->setChecked(m_zoomMode);
    }
  }
}

void DicomViewer::onViewWindowLevelToggled(int viewIndex, bool checked) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return;
  m_activeViewIndex = viewIndex;
  if (m_windowLevelButton->isChecked() != checked) {
    m_windowLevelButton->setChecked(checked);
    onWindowLevelButtonClicked();
  } else {
    updateOverlayInteractionStates();
  }
}

void DicomViewer::onViewPanToggled(int viewIndex, bool checked) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return;
  m_activeViewIndex = viewIndex;
  if (m_panButton->isChecked() != checked) {
    m_panButton->setChecked(checked);
  } else {
    onPanModeToggled(checked);
  }
  updateOverlayInteractionStates();
}

void DicomViewer::onViewZoomToggled(int viewIndex, bool checked) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return;
  m_activeViewIndex = viewIndex;
  if (m_zoomButton->isChecked() != checked) {
    m_zoomButton->setChecked(checked);
  } else {
    onZoomModeToggled(checked);
  }
  updateOverlayInteractionStates();
}

void DicomViewer::onImageToggleClicked(int viewIndex) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return;
  if (!m_is3DView[viewIndex])
    return;

  // Toggle visibility
  m_show3DImages[viewIndex] = !m_show3DImages[viewIndex];
  m_show3DLines[viewIndex] = !m_show3DLines[viewIndex];
  m_show3DSurfaces[viewIndex] = !m_show3DSurfaces[viewIndex];

  // Update 3D widget
  if (m_3dWidgets[viewIndex]) {
    m_3dWidgets[viewIndex]->setShowImages(m_show3DImages[viewIndex]);
    m_3dWidgets[viewIndex]->setShowLines(m_show3DLines[viewIndex]);
    m_3dWidgets[viewIndex]->setShowSurfaces(m_show3DSurfaces[viewIndex]);
  }
}

void DicomViewer::onLineToggleClicked(int viewIndex) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return;
  if (!m_is3DView[viewIndex])
    return;

  // Toggle Line visibility only
  m_show3DLines[viewIndex] = !m_show3DLines[viewIndex];

  // Update 3D widget
  if (m_3dWidgets[viewIndex]) {
    m_3dWidgets[viewIndex]->setShowLines(m_show3DLines[viewIndex]);
  }
}

void DicomViewer::onSurfaceToggleClicked(int viewIndex) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return;
  if (!m_is3DView[viewIndex])
    return;

  // Toggle Surface visibility only
  m_show3DSurfaces[viewIndex] = !m_show3DSurfaces[viewIndex];

  // Update 3D widget
  if (m_3dWidgets[viewIndex]) {
    m_3dWidgets[viewIndex]->setShowSurfaces(m_show3DSurfaces[viewIndex]);
  }
}

void DicomViewer::updateSliceLabels() {
  int count = 1;
  if (m_viewMode == ViewMode::Dual)
    count = 2;
  else if (m_viewMode == ViewMode::Quad)
    count = 4;
  else if (m_viewMode == ViewMode::Five)
    count = VIEW_COUNT;
  for (int i = 0; i < count; ++i) {
    if (m_isDVHView[i] || m_is3DView[i] || m_isProfileView[i]) {
      m_sliceIndexLabels[i]->hide();
      continue;
    }
    if (m_fusionViewActive && i == 1) {
      m_sliceIndexLabels[i]->setText(tr("Fusion"));
      m_sliceIndexLabels[i]->adjustSize();
      m_sliceIndexLabels[i]->show();
      continue;
    }
    int total = isVolumeLoaded()
                    ? sliceCountForOrientation(m_viewOrientations[i])
                    : m_dicomFiles.size();
    QString text =
        (total == 0) ? QString("0/0")
                     : QString("%1/%2").arg(m_currentIndices[i] + 1).arg(total);
    m_sliceIndexLabels[i]->setText(text);
    m_sliceIndexLabels[i]->adjustSize();
    m_sliceIndexLabels[i]->show();
  }
  updateSliderPosition();
}

void DicomViewer::updateViewLayout() {
  if (!m_imageLayout)
    return;

  while (QLayoutItem *item = m_imageLayout->takeAt(0)) {
    // no need to delete widgets, they are owned elsewhere
  }

  for (int i = 0; i < VIEW_COUNT; ++i) {
    m_viewContainers[i]->setVisible(false);
    // ★新規追加: 方向ボタンの表示制御
    if (m_orientationButtons[i]) {
      m_orientationButtons[i]->setVisible(false);
    }
    if (m_imageSeriesButtons[i]) {
      m_imageSeriesButtons[i]->setVisible(false);
    }
    if (m_viewWindowLevelButtons[i])
      m_viewWindowLevelButtons[i]->setVisible(false);
    if (m_viewPanButtons[i])
      m_viewPanButtons[i]->setVisible(false);
    if (m_viewZoomButtons[i])
      m_viewZoomButtons[i]->setVisible(false);
  }

  switch (m_viewMode) {
  case ViewMode::Single:
    m_imageLayout->addWidget(m_viewContainers[0], 0, 0);
    m_viewContainers[0]->setVisible(true);
    if (m_orientationButtons[0]) {
      m_orientationButtons[0]->setVisible(isVolumeLoaded());
    }
    break;
  case ViewMode::Dual:
    m_imageLayout->addWidget(m_viewContainers[0], 0, 0);
    m_imageLayout->addWidget(m_viewContainers[1], 0, 1);
    m_viewContainers[0]->setVisible(true);
    m_viewContainers[1]->setVisible(true);
    if (m_orientationButtons[0] && m_orientationButtons[1]) {
      m_orientationButtons[0]->setVisible(isVolumeLoaded());
      m_orientationButtons[1]->setVisible(isVolumeLoaded());
    }
    break;
  case ViewMode::Quad:
    m_imageLayout->addWidget(m_viewContainers[0], 0, 0);
    m_imageLayout->addWidget(m_viewContainers[1], 0, 1);
    m_imageLayout->addWidget(m_viewContainers[2], 1, 0);
    m_imageLayout->addWidget(m_viewContainers[3], 1, 1);
    for (int i = 0; i < 4; ++i) {
      m_viewContainers[i]->setVisible(true);
      if (m_orientationButtons[i]) {
        m_orientationButtons[i]->setVisible(isVolumeLoaded());
      }
    }
    break;
  case ViewMode::Five:
    m_imageLayout->addWidget(m_viewContainers[0], 0, 0, 1,
                             2); // top-left spans two columns
    m_imageLayout->addWidget(m_viewContainers[1], 0, 2); // top-right
    m_imageLayout->addWidget(m_viewContainers[2], 1, 0); // bottom-left-left
    m_imageLayout->addWidget(m_viewContainers[3], 1, 1); // bottom-left-right
    m_imageLayout->addWidget(m_viewContainers[4], 1, 2); // bottom-right
    for (int i = 0; i < VIEW_COUNT; ++i) {
      m_viewContainers[i]->setVisible(true);
      if (m_orientationButtons[i]) {
        m_orientationButtons[i]->setVisible(isVolumeLoaded());
      }
    }
    break;
  }
  m_imageLayout->setRowStretch(0, 1);
  m_imageLayout->setRowStretch(
      1,
      (m_viewMode == ViewMode::Quad || m_viewMode == ViewMode::Five) ? 1 : 0);
  if (m_viewMode == ViewMode::Five) {
    // 行の比率: 上段と下段を完全に均等に
    m_imageLayout->setRowStretch(0, 1); // 上段 (Axial + 3D)
    m_imageLayout->setRowStretch(1, 1); // 下段 (Sag + Cor + DVH)

    // 列の比率: 左右を6:4に設定
    // 左側（列0+列1）：右側（列2）= 6:4
    m_imageLayout->setColumnStretch(0, 3); // 左列 (Axialの左半分, Sagittal)
    m_imageLayout->setColumnStretch(1, 3); // 中央列 (Axialの右半分, Coronal)
    m_imageLayout->setColumnStretch(2, 4); // 右列 (3D, DVH)
  } else {
    // 他のモードの設定
    m_imageLayout->setRowStretch(0, 1);
    m_imageLayout->setRowStretch(
        1,
        (m_viewMode == ViewMode::Quad || m_viewMode == ViewMode::Five) ? 1 : 0);
    m_imageLayout->setColumnStretch(0, 1);
    m_imageLayout->setColumnStretch(
        1,
        (m_viewMode == ViewMode::Dual || m_viewMode == ViewMode::Quad) ? 1 : 0);
    m_imageLayout->setColumnStretch(2, 0);
  }

  // ★新規追加: ボタン表示の更新
  updateImageSeriesButtons();
  updateOrientationButtonTexts();
  for (int i = 0; i < VIEW_COUNT; ++i)
    updateInteractionButtonVisibility(i);
  updateSliderPosition();
}

void DicomViewer::setWindowLevel(double window, double level) {
  if (m_activeImageSeriesIndex >= 0 &&
      m_activeImageSeriesIndex < m_seriesWindowValues.size())
    m_seriesWindowValues[m_activeImageSeriesIndex] = window;
  if (m_activeImageSeriesIndex >= 0 &&
      m_activeImageSeriesIndex < m_seriesLevelValues.size())
    m_seriesLevelValues[m_activeImageSeriesIndex] = level;
  if (m_activeImageSeriesIndex >= 0 &&
      m_activeImageSeriesIndex < m_seriesWindowLevelInitialized.size())
    m_seriesWindowLevelInitialized[m_activeImageSeriesIndex] = true;
  m_dicomReader->setWindowLevel(window, level);
  if (m_windowSlider) {
    m_windowSlider->blockSignals(true);
    m_levelSlider->blockSignals(true);
    m_windowSlider->setValue(static_cast<int>(window));
    m_levelSlider->setValue(static_cast<int>(level));
    m_windowSpinBox->setValue(static_cast<int>(window));
    m_levelSpinBox->setValue(static_cast<int>(level));
    m_windowSlider->blockSignals(false);
    m_levelSlider->blockSignals(false);
  }
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_isDVHView[i] || m_isProfileView[i])
      continue;
    if (isVolumeLoaded()) {
      int idx = m_currentIndices[i];
      QImage img = m_volume.getSlice(idx, m_viewOrientations[i], window, level);
      m_originalImages[i] = img;
      int oriIndex = static_cast<int>(m_viewOrientations[i]);
      m_orientationImages[oriIndex] = img;
      m_orientationIndices[oriIndex] = idx;
    } else if (m_currentIndices[i] >= 0 &&
               m_currentIndices[i] < m_dicomFiles.size()) {
      if (m_dicomReader->loadDicomFile(m_dicomFiles[m_currentIndices[i]])) {
        m_dicomReader->setWindowLevel(window, level);
        m_originalImages[i] = m_dicomReader->getImage();
      }
    }
    updateImage(i, false);
  }
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_is3DView[i]) {
      update3DView(i);
    }
  }
}

int DicomViewer::viewIndexFromGlobalPos(const QPoint &globalPos) const {
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (!m_viewContainers[i]->isVisible())
      continue;
    if (m_viewContainers[i]->rect().contains(
            m_viewContainers[i]->mapFromGlobal(globalPos))) {
      return i;
    }
  }
  return m_activeViewIndex;
}

void DicomViewer::loadVolumeSlice(int viewIndex, int sliceIndex) {
  if (!isVolumeLoaded() || m_isDVHView[viewIndex] || m_is3DView[viewIndex] ||
      m_isProfileView[viewIndex])
    return;
  int count = sliceCountForOrientation(m_viewOrientations[viewIndex]);
  if (sliceIndex < 0 || sliceIndex >= count)
    return;
  QImage img =
      m_volume.getSlice(sliceIndex, m_viewOrientations[viewIndex],
                        m_windowSlider->value(), m_levelSlider->value());
  m_originalImages[viewIndex] = img;
  int oriIndex = static_cast<int>(m_viewOrientations[viewIndex]);
  m_orientationImages[oriIndex] = img;
  m_orientationIndices[oriIndex] = sliceIndex;
  m_currentIndices[viewIndex] = sliceIndex;
  if (m_showSlicePosition)
    updateImage();
  else
    updateImage(viewIndex);

  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_is3DView[i]) {
      update3DView(i);
    }
  }

  // スライス変更時にカーソル情報をリセット
  if (viewIndex >= 0 && viewIndex < VIEW_COUNT) {
    if (m_cursorDoseLabels[viewIndex])
      m_cursorDoseLabels[viewIndex]->hide();
    m_imageWidgets[viewIndex]->clearCursorCross();
  }
}

int DicomViewer::sliceCountForOrientation(DicomVolume::Orientation ori) const {
  switch (ori) {
  case DicomVolume::Orientation::Axial:
    return m_volume.depth();
  case DicomVolume::Orientation::Sagittal:
    return m_volume.width();
  case DicomVolume::Orientation::Coronal:
    return m_volume.height();
  }
  return 0;
}

void DicomViewer::updateDoseAlignment() {
  // Validate geometry presence
  if (!isVolumeLoaded() || m_doseVolume.width() == 0 ||
      m_doseVolume.height() == 0 || m_doseVolume.depth() == 0) {
    m_doseVolume.setPatientShift(QVector3D(0, 0, 0));
    QMatrix4x4 id;
    id.setToIdentity();
    m_doseVolume.setCtToDoseTransform(id);
    m_doseShift = QVector3D(0, 0, 0);
    return;
  }

  // Always use identity rotation between CT patient and Dose patient
  QMatrix4x4 id;
  id.setToIdentity();
  m_doseVolume.setCtToDoseTransform(id);

  // Preserve RTDOSE native position on load: do not auto-shift.
  // Many datasets already share a common patient coordinate system; adding
  // additional origin/center alignment here can introduce large offsets.
  m_doseShift = QVector3D(0, 0, 0);
  m_doseVolume.setPatientShift(QVector3D(0, 0, 0));
  qDebug() << "[Dose Align] Preserving RTDOSE native position (no shift).";
  return;

  // Decide a simple, robust shift
  QString ctUID = m_volume.frameOfReferenceUID();
  QString doseUID = m_doseVolume.frameOfReferenceUID();
  bool foRMatch = (!ctUID.isEmpty() && !doseUID.isEmpty() && ctUID == doseUID);

  auto isZeroVec = [](const QVector3D &v) {
    return std::abs(v.x()) < 1e-6 && std::abs(v.y()) < 1e-6 &&
           std::abs(v.z()) < 1e-6;
  };

  QVector3D shift(0, 0, 0);
  QVector3D ctOrigin(m_volume.originX(), m_volume.originY(),
                     m_volume.originZ());
  QVector3D doseOrigin(m_doseVolume.originX(), m_doseVolume.originY(),
                       m_doseVolume.originZ());

  if (foRMatch) {
    // Prefer origin alignment if RTDose has valid IPP; otherwise fallback to
    // center alignment
    if (m_doseVolume.hasIPP()) {
      shift = ctOrigin - doseOrigin;
      qDebug() << QString("FoR match: origin-alignment shift: (%1, %2, %3)")
                      .arg(shift.x(), 0, 'f', 2)
                      .arg(shift.y(), 0, 'f', 2)
                      .arg(shift.z(), 0, 'f', 2);
    } else {
      QVector3D ctCenter = m_volume.voxelToPatient(m_volume.width() / 2.0,
                                                   m_volume.height() / 2.0,
                                                   m_volume.depth() / 2.0);
      m_doseVolume.setPatientShift(QVector3D(0, 0, 0));
      QVector3D doseCenter = m_doseVolume.voxelToPatient(
          m_doseVolume.width() / 2.0, m_doseVolume.height() / 2.0,
          m_doseVolume.depth() / 2.0);
      shift = ctCenter - doseCenter;
      qDebug() << QString("FoR match: center-alignment shift: (%1, %2, %3)")
                      .arg(shift.x(), 0, 'f', 2)
                      .arg(shift.y(), 0, 'f', 2)
                      .arg(shift.z(), 0, 'f', 2);
    }
  } else {
    // FoR mismatch: use center alignment as a simple fallback
    QVector3D ctCenter =
        m_volume.voxelToPatient(m_volume.width() / 2.0, m_volume.height() / 2.0,
                                m_volume.depth() / 2.0);
    m_doseVolume.setPatientShift(QVector3D(0, 0, 0));
    QVector3D doseCenter = m_doseVolume.voxelToPatient(
        m_doseVolume.width() / 2.0, m_doseVolume.height() / 2.0,
        m_doseVolume.depth() / 2.0);
    shift = ctCenter - doseCenter;
    qDebug() << QString("FoR mismatch: center-alignment shift: (%1, %2, %3)")
                    .arg(shift.x(), 0, 'f', 2)
                    .arg(shift.y(), 0, 'f', 2)
                    .arg(shift.z(), 0, 'f', 2);
  }

  m_doseShift = shift;
  m_doseVolume.setPatientShift(m_doseShift);

  // Quick verification
  QVector3D ctCenter = m_volume.voxelToPatient(
      m_volume.width() / 2, m_volume.height() / 2, m_volume.depth() / 2);
  QVector3D doseCenterVoxel = m_doseVolume.patientToVoxelContinuous(ctCenter);
  bool inBounds = (doseCenterVoxel.x() >= -0.5 &&
                   doseCenterVoxel.x() < m_doseVolume.width() - 0.5 &&
                   doseCenterVoxel.y() >= -0.5 &&
                   doseCenterVoxel.y() < m_doseVolume.height() - 0.5 &&
                   doseCenterVoxel.z() >= -0.5 &&
                   doseCenterVoxel.z() < m_doseVolume.depth() - 0.5);
  qDebug() << QString("[Dose Align] Final shift: (%1, %2, %3)")
                  .arg(m_doseShift.x(), 0, 'f', 2)
                  .arg(m_doseShift.y(), 0, 'f', 2)
                  .arg(m_doseShift.z(), 0, 'f', 2);
  qDebug()
      << QString(
             "[Dose Align] CT center -> Dose voxel (%1,%2,%3) | in-bounds: %4")
             .arg(doseCenterVoxel.x(), 0, 'f', 2)
             .arg(doseCenterVoxel.y(), 0, 'f', 2)
             .arg(doseCenterVoxel.z(), 0, 'f', 2)
             .arg(inBounds ? "YES" : "NO");

  // Extra debug: probe Y+10mm to verify Y mapping (sign/scale)
  {
    // X +10mm
    double dx = 10.0;
    QVector3D ctCenterPlusX = ctCenter + QVector3D(dx, 0, 0);
    QVector3D doseVoxX = m_doseVolume.patientToVoxelContinuous(ctCenterPlusX);
    qDebug()
        << QString(
               "[Dose Align Debug] CT center +10mmX -> Dose voxel (%1,%2,%3)")
               .arg(doseVoxX.x(), 0, 'f', 2)
               .arg(doseVoxX.y(), 0, 'f', 2)
               .arg(doseVoxX.z(), 0, 'f', 2);
    double dy = 10.0; // mm
    QVector3D ctCenterPlusY = ctCenter + QVector3D(0, dy, 0);
    QVector3D doseVoxY = m_doseVolume.patientToVoxelContinuous(ctCenterPlusY);
    qDebug()
        << QString(
               "[Dose Align Debug] CT center +10mmY -> Dose voxel (%1,%2,%3)")
               .arg(doseVoxY.x(), 0, 'f', 2)
               .arg(doseVoxY.y(), 0, 'f', 2)
               .arg(doseVoxY.z(), 0, 'f', 2);
    // Z +10mm
    double dz = 10.0; // mm
    QVector3D ctCenterPlusZ = ctCenter + QVector3D(0, 0, dz);
    QVector3D doseVoxZ = m_doseVolume.patientToVoxelContinuous(ctCenterPlusZ);
    qDebug()
        << QString(
               "[Dose Align Debug] CT center +10mmZ -> Dose voxel (%1,%2,%3)")
               .arg(doseVoxZ.x(), 0, 'f', 2)
               .arg(doseVoxZ.y(), 0, 'f', 2)
               .arg(doseVoxZ.z(), 0, 'f', 2);
  }
}
void DicomViewer::showOrientationMenu(int viewIndex, const QPoint &pos) {
  if ((m_viewMode != ViewMode::Quad && m_viewMode != ViewMode::Five) ||
      !isVolumeLoaded())
    return;

  QMenu menu;
  QAction *ax = menu.addAction("Axial");
  QAction *sag = menu.addAction("Sagittal");
  QAction *cor = menu.addAction("Coronal");
  QAction *sel = menu.exec(m_imageWidgets[viewIndex]->mapToGlobal(pos));
  if (!sel)
    return;
  if (sel == ax)
    m_viewOrientations[viewIndex] = DicomVolume::Orientation::Axial;
  else if (sel == sag)
    m_viewOrientations[viewIndex] = DicomVolume::Orientation::Sagittal;
  else if (sel == cor)
    m_viewOrientations[viewIndex] = DicomVolume::Orientation::Coronal;

  int count = sliceCountForOrientation(m_viewOrientations[viewIndex]);
  m_sliceSliders[viewIndex]->setRange(0, count > 0 ? count - 1 : 0);
  int mid = count > 0 ? count / 2 : 0;
  m_currentIndices[viewIndex] = mid;
  m_sliceSliders[viewIndex]->setValue(mid);
  loadVolumeSlice(viewIndex, mid);
  updateSliceLabels();
}

void DicomViewer::showJumpToMenu(int viewIndex, const QPoint &pos) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT || !isVolumeLoaded() ||
      m_isDVHView[viewIndex] || m_is3DView[viewIndex] ||
      m_isProfileView[viewIndex])
    return;

  QMenu menu;
  QAction *jump = menu.addAction("jump to");
  QAction *sel = menu.exec(m_imageWidgets[viewIndex]->mapToGlobal(pos));
  if (sel != jump)
    return;

  QVector3D patient = patientCoordinateAt(viewIndex, pos);
  if (std::isnan(patient.x()))
    return;
  QVector3D voxel = m_volume.patientToVoxelContinuous(patient);
  int vx = std::clamp(static_cast<int>(std::round(voxel.x())), 0,
                      m_volume.width() - 1);
  int vy = std::clamp(static_cast<int>(std::round(voxel.y())), 0,
                      m_volume.height() - 1);
  int vz = std::clamp(static_cast<int>(std::round(voxel.z())), 0,
                      m_volume.depth() - 1);

  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_isDVHView[i] || m_is3DView[i] || m_isProfileView[i])
      continue;
    int sliceIndex = 0;
    switch (m_viewOrientations[i]) {
    case DicomVolume::Orientation::Axial:
      sliceIndex = vz;
      break;
    case DicomVolume::Orientation::Sagittal:
      sliceIndex = vx;
      break;
    case DicomVolume::Orientation::Coronal:
      sliceIndex = vy;
      break;
    }
    int count = sliceCountForOrientation(m_viewOrientations[i]);
    sliceIndex = std::clamp(sliceIndex, 0, count > 0 ? count - 1 : 0);
    m_currentIndices[i] = sliceIndex;
    m_sliceSliders[i]->setValue(sliceIndex);
    loadSlice(i, sliceIndex);

    float spacingX = static_cast<float>(m_volume.spacingX());
    float spacingY = static_cast<float>(m_volume.spacingY());
    float spacingZ = static_cast<float>(m_volume.spacingZ());
    float w_mm = 0.0f, h_mm = 0.0f, cx_mm = 0.0f, cy_mm = 0.0f;
    switch (m_viewOrientations[i]) {
    case DicomVolume::Orientation::Axial:
      w_mm = m_volume.width() * spacingX;
      h_mm = m_volume.height() * spacingY;
      cx_mm = vx * spacingX - w_mm / 2.0f;
      cy_mm = h_mm / 2.0f - vy * spacingY;
      break;
    case DicomVolume::Orientation::Sagittal:
      w_mm = m_volume.height() * spacingY;
      h_mm = m_volume.depth() * spacingZ;
      cx_mm = vy * spacingY - w_mm / 2.0f;
      cy_mm = vz * spacingZ - h_mm / 2.0f;
      break;
    case DicomVolume::Orientation::Coronal:
      w_mm = m_volume.width() * spacingX;
      h_mm = m_volume.depth() * spacingZ;
      cx_mm = vx * spacingX - w_mm / 2.0f;
      cy_mm = vz * spacingZ - h_mm / 2.0f;
      break;
    }
    m_imageWidgets[i]->setCursorCross(QPointF(cx_mm, cy_mm));
  }
}

bool DicomViewer::eventFilter(QObject *obj, QEvent *event) {
  if (obj == m_scrollArea->viewport()) {
    switch (event->type()) {
    case QEvent::Wheel:
      wheelEvent(static_cast<QWheelEvent *>(event));
      return true;
    case QEvent::MouseButtonPress:
      mousePressEvent(static_cast<QMouseEvent *>(event));
      return true;
    case QEvent::MouseMove:
      mouseMoveEvent(static_cast<QMouseEvent *>(event));
      return true;
    case QEvent::MouseButtonRelease:
      mouseReleaseEvent(static_cast<QMouseEvent *>(event));
      return true;
    default:
      break;
    }
  }
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (obj == m_infoOverlays[i] &&
        event->type() == QEvent::MouseButtonDblClick) {
      m_privacyMode = !m_privacyMode;
      updateImageInfo();
      updateInfoOverlays();
      return true;
    }
    if (obj == m_imageWidgets[i]) {
      switch (event->type()) {
      case QEvent::Wheel:
        wheelEvent(static_cast<QWheelEvent *>(event));
        return true;
      case QEvent::MouseButtonPress:
        mousePressEvent(static_cast<QMouseEvent *>(event));
        return true;
      case QEvent::MouseMove:
        mouseMoveEvent(static_cast<QMouseEvent *>(event));
        return true;
      case QEvent::MouseButtonRelease:
        mouseReleaseEvent(static_cast<QMouseEvent *>(event));
        return true;
      default:
        break;
      }
    }
  }
  if (obj == m_infoTextBox) {
    if (event->type() == QEvent::MouseButtonDblClick) {
      m_privacyMode = !m_privacyMode;
      updateImageInfo();
      updateInfoOverlays();
      return true;
    } else if (event->type() == QEvent::MouseButtonPress) {
      // 単一クリック時にテキスト選択などが起こらないようにイベントを消費
      return true;
    }
  }
  return QWidget::eventFilter(obj, event);
}

void DicomViewer::onDoseDisplayModeChanged() {
  if (!m_doseColorMapCombo) {
    return;
  }

  switch (m_doseColorMapCombo->currentIndex()) {
  case 0:
    m_doseDisplayMode = DoseResampledVolume::DoseDisplayMode::Colorful;
    break;
  case 1:
    m_doseDisplayMode = DoseResampledVolume::DoseDisplayMode::Isodose;
    break;
  case 2:
    m_doseDisplayMode = DoseResampledVolume::DoseDisplayMode::IsodoseLines;
    break;
  case 3:
    m_doseDisplayMode = DoseResampledVolume::DoseDisplayMode::Simple;
    break;
  case 4:
    m_doseDisplayMode = DoseResampledVolume::DoseDisplayMode::Hot;
    break;
  default:
    m_doseDisplayMode = DoseResampledVolume::DoseDisplayMode::Colorful;
    break;
  }

  updateColorBars();
  updateImage(); // 表示を更新
}

void DicomViewer::onDoseRangeEditingFinished() {
  updateColorBars();
  updateImage();
}

void DicomViewer::onDoseOpacityChanged(int value) {
  m_doseOpacity = static_cast<double>(value) / 100.0;
  updateImage();
}

void DicomViewer::onDoseListContextMenu(const QPoint &pos) {
  if (!m_doseListWidget)
    return;
  QListWidgetItem *item = m_doseListWidget->itemAt(pos);
  if (!item)
    return;
  int row = m_doseListWidget->row(item);
  QMenu menu;
  QAction *copyAct = menu.addAction(tr("Copy"));
  QAction *delAct = menu.addAction(tr("Delete"));
  QAction *chosen = menu.exec(m_doseListWidget->mapToGlobal(pos));
  if (chosen == copyAct)
    copyDoseAt(row);
  else if (chosen == delAct)
    deleteDoseAt(row);
}

void DicomViewer::copyDoseAt(int index) {
  if (!m_doseListWidget || index < 0 ||
      index >= static_cast<int>(m_doseItems.size()))
    return;
  const auto &src = m_doseItems[index];
  QListWidgetItem *item = new QListWidgetItem();
  m_doseListWidget->insertItem(index + 1, item);
  QString label = src.widget ? src.widget->name() + tr(" (copy)") : tr("Copy");
  DoseItemWidget *widget = new DoseItemWidget(label, src.dose.maxDose());
  widget->setChecked(src.widget && src.widget->isChecked());
  widget->setDataFractions(src.widget ? src.widget->dataFractions() : 1.0);
  widget->setDisplayFractions(src.widget ? src.widget->displayFractions()
                                         : 1.0);
  widget->setShift(src.widget ? src.widget->shift() : QVector3D());
  item->setSizeHint(widget->sizeHint());
  m_doseListWidget->setItemWidget(item, widget);

  connect(widget, &DoseItemWidget::uiExpandedChanged, this,
          [this, item, widget]() {
            QTimer::singleShot(0, this, [this, item, widget]() {
              widget->adjustSize();
              item->setSizeHint(widget->sizeHint());
              m_doseListWidget->setItemWidget(item, widget);
              m_doseListWidget->doItemsLayout();
              if (m_doseListWidget->viewport())
                m_doseListWidget->viewport()->update();
            });
          });
  connect(widget, &DoseItemWidget::settingsChanged, this, [this]() {
    m_resampledDose.clear();
    m_doseLoaded = false;
    updateColorBars();
    updateImage();
  });
  connect(widget, &DoseItemWidget::visibilityChanged, this,
          [this](bool) {
            onDoseCalculateClicked();
            updateDoseShiftLabels();
          });

  // Connect save button
  connect(widget, &DoseItemWidget::saveRequested, this, [this, widget]() {
    onDoseSaveRequested(widget);
  });

  DoseItem newItem;
  newItem.volume = src.volume;
  newItem.widget = widget;
  newItem.dose = src.dose;
  newItem.isSaved = src.isSaved;
  newItem.savedFilePath = src.savedFilePath;
  if (widget) {
    widget->setSaved(src.isSaved);
  }
  m_doseItems.insert(m_doseItems.begin() + index + 1, newItem);

  onDoseCalculateClicked();
  updateDoseShiftLabels();
}

void DicomViewer::deleteDoseAt(int index) {
  if (!m_doseListWidget || index < 0 ||
      index >= static_cast<int>(m_doseItems.size()))
    return;
  QListWidgetItem *item = m_doseListWidget->takeItem(index);
  if (item)
    delete item;
  DoseItem removed = m_doseItems[index];
  if (removed.widget)
    delete removed.widget;
  m_doseItems.erase(m_doseItems.begin() + index);
  onDoseCalculateClicked();
  updateDoseShiftLabels();
}

void DicomViewer::onDoseSaveRequested(DoseItemWidget *widget) {
  if (!widget)
    return;

  // Find the corresponding DoseItem
  auto it = std::find_if(m_doseItems.begin(), m_doseItems.end(),
                        [widget](const DoseItem &item) {
                          return item.widget == widget;
                        });

  if (it == m_doseItems.end()) {
    qWarning() << "DoseItem not found for widget";
    return;
  }

  // Get patient info for file path
  DicomStudyInfo studyInfo = currentStudyInfo();
  QString patientKey;
  if (!studyInfo.patientName.isEmpty() && !studyInfo.patientID.isEmpty()) {
    patientKey = QString("%1_%2").arg(studyInfo.patientName).arg(studyInfo.patientID);
  } else {
    QMessageBox::warning(this, tr("Save RT-Dose"),
                        tr("Cannot save: No patient information available.\n"
                           "Please load a patient study first."));
    return;
  }

  // Build file path in patient folder
  QString filename;
  if (m_databaseManager && m_databaseManager->isOpen()) {
    QString dataRoot = QString::fromStdString(m_databaseManager->dataRoot());
    QString patientDir = QDir(dataRoot).filePath("Patients/" + patientKey);

    // Create patient directory if it doesn't exist
    QDir().mkpath(patientDir);

    // Create RTDOSE subdirectory for calculated doses
    QString rtdoseDir = QDir(patientDir).filePath("RTDOSE_Calculated");
    QDir().mkpath(rtdoseDir);

    // Generate filename with timestamp
    QString calculationType = widget->name().replace(" ", "_");
    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    QString baseName = QString("%1_%2.dcm").arg(calculationType).arg(timestamp);
    filename = QDir(rtdoseDir).filePath(baseName);

    qDebug() << "Auto-saving dose to:" << filename;
  } else {
    // Fallback: ask user for location if database not available
    QString defaultName = widget->name();
    if (!defaultName.endsWith(".dcm", Qt::CaseInsensitive)) {
      defaultName += ".dcm";
    }

    filename = QFileDialog::getSaveFileName(
        this, tr("Save RT-Dose"), defaultName,
        tr("DICOM RT-Dose Files (*.dcm);;All Files (*)"));

    if (filename.isEmpty())
      return; // User cancelled
  }

  // Save the dose volume
  bool success = it->dose.saveToFile(filename, [](int current, int total) {
    // Progress callback - could show a progress dialog if needed
    qDebug() << "Saving dose:" << current << "/" << total;
  });

  if (success) {
    // Update saved status
    it->isSaved = true;
    it->savedFilePath = filename;
    widget->setSaved(true);

    // Save to database
    qDebug() << "=== Database Save Debug ===";
    qDebug() << "m_databaseManager:" << (m_databaseManager ? "Valid" : "NULL");

    if (!m_databaseManager) {
      qWarning() << "DatabaseManager is NULL - cannot save to database";
    } else if (!m_databaseManager->isOpen()) {
      qWarning() << "DatabaseManager is not open - cannot save to database";
      qWarning() << "Database path:" << QString::fromStdString(m_databaseManager->databasePath());
    } else {
      qDebug() << "DatabaseManager is ready, proceeding with save";
      qDebug() << "Database path:" << QString::fromStdString(m_databaseManager->databasePath());

      // Helper function to escape SQL strings (replace ' with '')
      auto escapeSql = [](const QString &str) -> std::string {
        QString escaped = str;
        escaped.replace("'", "''");
        return escaped.toStdString();
      };

      // Get dose information
      const RTDoseVolume &dose = it->dose;
      QString calculationType = widget->name();

      // Get patient info if available
      DicomStudyInfo studyInfo = currentStudyInfo();
      QString patientKey;
      if (!studyInfo.patientName.isEmpty() && !studyInfo.patientID.isEmpty()) {
        patientKey = QString("%1_%2").arg(studyInfo.patientName).arg(studyInfo.patientID);
      }

      qDebug() << "Calculation type:" << calculationType;
      qDebug() << "Patient key:" << (patientKey.isEmpty() ? "NULL" : patientKey);
      qDebug() << "File path:" << filename;
      qDebug() << "Dose dimensions:" << dose.width() << "x" << dose.height() << "x" << dose.depth();
      qDebug() << "Max dose:" << dose.maxDose();

      // Build SQL INSERT statement
      std::ostringstream sql;
      sql << "INSERT INTO dose_volumes (";
      sql << "patient_key, calculation_type, file_path, format, ";
      sql << "width, height, depth, ";
      sql << "spacing_x, spacing_y, spacing_z, ";
      sql << "origin_x, origin_y, origin_z, ";
      sql << "max_dose, frame_uid, description";
      sql << ") VALUES (";

      // patient_key (can be NULL)
      if (patientKey.isEmpty()) {
        sql << "NULL, ";
      } else {
        sql << "'" << escapeSql(patientKey) << "', ";
      }

      // calculation_type
      sql << "'" << escapeSql(calculationType) << "', ";

      // file_path
      sql << "'" << escapeSql(filename) << "', ";

      // format
      sql << "'RTDOSE', ";

      // dimensions
      sql << dose.width() << ", " << dose.height() << ", " << dose.depth() << ", ";

      // spacing
      sql << dose.spacingX() << ", " << dose.spacingY() << ", " << dose.spacingZ() << ", ";

      // origin
      sql << dose.originX() << ", " << dose.originY() << ", " << dose.originZ() << ", ";

      // max_dose
      sql << dose.maxDose() << ", ";

      // frame_uid
      if (dose.frameOfReferenceUID().isEmpty()) {
        sql << "NULL, ";
      } else {
        sql << "'" << escapeSql(dose.frameOfReferenceUID()) << "', ";
      }

      // description
      QString description = QString("Dose calculation: %1").arg(calculationType);
      sql << "'" << escapeSql(description) << "'";

      sql << ");";

      // Log the SQL statement
      QString sqlStr = QString::fromStdString(sql.str());
      qDebug() << "Executing SQL:" << sqlStr;

      // First, check if study already exists, otherwise create it
      int studyId = -1;
      QFileInfo fileInfo(filename);
      QString relativePath = fileInfo.fileName();
      QString directoryPath = fileInfo.absolutePath();

      // Try to find existing study for ShioRIS3 calculated doses
      std::ostringstream findStudySql;
      findStudySql << "SELECT id FROM studies WHERE ";
      findStudySql << "patient_key='" << escapeSql(patientKey) << "' AND ";
      findStudySql << "modality='RTDOSE' AND ";
      findStudySql << "study_name='ShioRIS3 Calculated Dose' AND ";
      findStudySql << "path='" << escapeSql(directoryPath) << "'";

      if (!dose.frameOfReferenceUID().isEmpty()) {
        findStudySql << " AND frame_uid='" << escapeSql(dose.frameOfReferenceUID()) << "'";
      }

      findStudySql << " LIMIT 1;";

      bool foundExisting = m_databaseManager->query(findStudySql.str(),
        [&studyId](int argc, char** argv, char**) {
          if (argc > 0 && argv[0]) {
            studyId = std::atoi(argv[0]);
          }
        });

      // If not found, create new study
      if (studyId <= 0) {
        std::ostringstream studiesSql;
        studiesSql << "INSERT INTO studies (";
        studiesSql << "patient_key, modality, study_name, path, ";
        studiesSql << "frame_uid, series_uid, series_description";
        studiesSql << ") VALUES (";
        studiesSql << "'" << escapeSql(patientKey) << "', ";
        studiesSql << "'RTDOSE', ";
        studiesSql << "'ShioRIS3 Calculated Dose', ";
        studiesSql << "'" << escapeSql(directoryPath) << "', ";

        if (dose.frameOfReferenceUID().isEmpty()) {
          studiesSql << "NULL, NULL, ";
        } else {
          studiesSql << "'" << escapeSql(dose.frameOfReferenceUID()) << "', ";
          studiesSql << "'" << escapeSql(dose.frameOfReferenceUID()) << "', ";
        }

        studiesSql << "'" << escapeSql(calculationType) << "'";
        studiesSql << ");";

        if (m_databaseManager->exec(studiesSql.str())) {
          qDebug() << "New study created successfully";

          // Get the study_id of the inserted record
          std::string getIdSql = "SELECT last_insert_rowid();";
          m_databaseManager->query(getIdSql, [&studyId](int argc, char** argv, char**) {
            if (argc > 0 && argv[0]) {
              studyId = std::atoi(argv[0]);
            }
          });
          qDebug() << "New Study ID:" << studyId;
        } else {
          qWarning() << "Failed to create study:" << QString::fromStdString(m_databaseManager->lastError());
        }
      } else {
        qDebug() << "Reusing existing study ID:" << studyId;
      }

      // Now insert into dose_volumes and files if we have a valid study_id
      if (studyId > 0) {
        // Now insert into dose_volumes with study_id
        sql.str("");
        sql.clear();
        sql << "INSERT INTO dose_volumes (";
        sql << "patient_key, study_id, calculation_type, file_path, format, ";
        sql << "width, height, depth, ";
        sql << "spacing_x, spacing_y, spacing_z, ";
        sql << "origin_x, origin_y, origin_z, ";
        sql << "max_dose, frame_uid, description";
        sql << ") VALUES (";
        sql << "'" << escapeSql(patientKey) << "', ";
        sql << studyId << ", ";
        sql << "'" << escapeSql(calculationType) << "', ";
        sql << "'" << escapeSql(filename) << "', ";
        sql << "'RTDOSE', ";
        sql << dose.width() << ", " << dose.height() << ", " << dose.depth() << ", ";
        sql << dose.spacingX() << ", " << dose.spacingY() << ", " << dose.spacingZ() << ", ";
        sql << dose.originX() << ", " << dose.originY() << ", " << dose.originZ() << ", ";
        sql << dose.maxDose() << ", ";

        if (dose.frameOfReferenceUID().isEmpty()) {
          sql << "NULL, ";
        } else {
          sql << "'" << escapeSql(dose.frameOfReferenceUID()) << "', ";
        }

        sql << "'" << escapeSql(description) << "'";
        sql << ");";

        // Execute the SQL statement for dose_volumes
        if (m_databaseManager->exec(sql.str())) {
          qDebug() << "SUCCESS: Dose volume saved to database";

          // Also register in files table
          QFileInfo fileInfoForSize(filename);
          qint64 fileSize = fileInfoForSize.size();

          std::ostringstream filesSql;
          filesSql << "INSERT INTO files (";
          filesSql << "study_id, relative_path, size_bytes, file_type";
          filesSql << ") VALUES (";
          filesSql << studyId << ", ";
          filesSql << "'" << escapeSql(relativePath) << "', ";
          filesSql << fileSize << ", ";
          filesSql << "'DICOM'";
          filesSql << ");";

          if (m_databaseManager->exec(filesSql.str())) {
            qDebug() << "File registered in files table";
          } else {
            qWarning() << "Failed to register file:" << QString::fromStdString(m_databaseManager->lastError());
          }
        }

        // Verify the save by querying the database
        std::string countSql = "SELECT COUNT(*) FROM dose_volumes;";
        int count = 0;
        m_databaseManager->query(countSql, [&count](int argc, char** argv, char**) {
          if (argc > 0 && argv[0]) {
            count = std::atoi(argv[0]);
          }
        });
        qDebug() << "Total dose_volumes records in database:" << count;
      } else {
        QString errorMsg = QString::fromStdString(m_databaseManager->lastError());
        qWarning() << "FAILED: Could not save dose volume to database";
        qWarning() << "SQLite error:" << errorMsg;
        qWarning() << "SQL was:" << sqlStr;
      }
    }
    qDebug() << "=== End Database Save Debug ===";

    QMessageBox::information(this, tr("Save RT-Dose"),
                            tr("Dose distribution saved successfully to:\n%1")
                                .arg(filename));
  } else {
    QMessageBox::warning(this, tr("Save RT-Dose"),
                        tr("Failed to save dose distribution to:\n%1")
                            .arg(filename));
  }
}

void DicomViewer::onDoseCalculateClicked() {
  qDebug() << "onDoseCalculateClicked: m_doseItems.size =" << m_doseItems.size();
  if (m_doseItems.empty()) {
    m_doseLoaded = false;
    m_doseVisible = false;
    qDebug() << "  -> m_doseItems is empty, set m_doseVisible = false";
    m_resampledDose.clear();
  } else {
    // Fast path: single dose, no shifts, unity fractions -> reuse existing
    // resample
    if (m_doseItems.size() == 1) {
      const auto &it0 = m_doseItems.front();
      const bool unityFr =
          (it0.widget && std::abs(it0.widget->dataFractions() - 1.0) < 1e-6 &&
           std::abs(it0.widget->displayFractions() - 1.0) < 1e-6);
      const bool zeroShift =
          (it0.widget && std::abs(it0.widget->shiftX()) < 1e-6 &&
           std::abs(it0.widget->shiftY()) < 1e-6 &&
           std::abs(it0.widget->shiftZ()) < 1e-6);
      const bool baseZero =
          (std::abs(computeAlignmentShift(it0.dose).x()) < 1e-6 &&
           std::abs(computeAlignmentShift(it0.dose).y()) < 1e-6 &&
           std::abs(computeAlignmentShift(it0.dose).z()) < 1e-6);
      const bool visible = (it0.widget && it0.widget->isChecked());
      qDebug() << "  Fast path: visible=" << visible << "unityFr=" << unityFr
               << "zeroShift=" << zeroShift << "baseZero=" << baseZero
               << "isResampled=" << m_resampledDose.isResampled();
      if (visible && unityFr && zeroShift && baseZero &&
          m_resampledDose.isResampled()) {
        m_doseLoaded = true;
        m_doseVisible = true;
        qDebug() << "  -> Fast path taken, set m_doseVisible = true";
        resetDoseRange();
        updateColorBars();
        updateImage();
        updateImageInfo();
        return;
      }
    }
    const auto &base = m_doseItems.front().volume;
    int sizes[3] = {base.depth(), base.height(), base.width()};
    cv::Mat sum(3, sizes, CV_32F, cv::Scalar(0));

    bool any = false;
    for (const auto &it : m_doseItems) {
      bool isChecked = it.widget && it.widget->isChecked();
      qDebug() << "    dose item:" << (it.widget ? it.widget->name() : "no-widget")
               << "isChecked =" << isChecked;
      if (!isChecked)
        continue;
      any = true;
      // Apply combined shift: alignment shift + user shift
      QVector3D baseShift = computeAlignmentShift(it.dose);
      QVector3D userShift = it.widget->shift();
      RTDoseVolume shiftedDose = it.dose; // copy
      shiftedDose.setPatientShift(baseShift + userShift);

      DoseResampledVolume res;
      bool ok = res.resampleFromRTDose(m_volume, shiftedDose);
      if (!ok)
        continue;

      const cv::Mat &src = res.data();
      // Per-slice parallel accumulation to improve throughput
      const int W = base.width();
      const int H = base.height();
      const int D = base.depth();
      const double dataFr = it.widget->dataFractions();
      const double displayFr = it.widget->displayFractions();
      const double factor = it.widget->factor();
      QVector<int> zIndices(D);
      std::iota(zIndices.begin(), zIndices.end(), 0);
      QtConcurrent::blockingMap(zIndices, [&](int z) {
        const float *s = src.ptr<float>(z);
        float *d = sum.ptr<float>(z);
        for (int y = 0; y < H; ++y) {
          int row = y * W;
          for (int x = 0; x < W; ++x) {
            float raw = s[row + x];
            double val = transformDose(raw, dataFr, displayFr) * factor;
            d[row + x] += static_cast<float>(val);
          }
        }
      });
    }

    qDebug() << "  -> any =" << any << "after checking all dose items";
    if (any) {
      m_resampledDose.setFromMat(sum, base.spacingX(), base.spacingY(),
                                 base.spacingZ(), base.originX(),
                                 base.originY(), base.originZ());
      m_doseLoaded = true;
      m_doseVisible = true;
      qDebug() << "  -> Set m_doseVisible = true because any = true";
      resetDoseRange();
      if (m_doseRefSpinBox) {
        double maxDose = m_resampledDose.maxDose();
        m_doseRefSpinBox->setValue(maxDose);
        m_doseReference = maxDose;
      }
    } else {
      m_resampledDose.clear();
      m_doseLoaded = false;
      m_doseVisible = false;
      qDebug() << "  -> Set m_doseVisible = false because any = false (no checked items)";
      if (m_doseRefSpinBox) {
        m_doseRefSpinBox->setValue(0.0);
      }
      m_doseReference = 0.0;
      if (m_doseMinSpinBox && m_doseMaxSpinBox) {
        m_doseMinSpinBox->setValue(0.0);
        m_doseMaxSpinBox->setValue(0.0);
      }
      m_doseMinRange = 0.0;
      m_doseMaxRange = 0.0;
    }
  }

  m_dvhData.clear();
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_dvhWidgets[i]) {
      m_dvhWidgets[i]->setDVHData({});
      m_dvhWidgets[i]->setPatientInfo(QString());
    }
  }

  updateColorBars();
  updateImage();
  // Update info panel to reflect current RT Dose status
  updateImageInfo();
}

void DicomViewer::onRandomStudyClicked() {
  // Open a simple dialog placeholder for future DVH display
  auto *dlg = new RandomStudyDialog(this);
  dlg->setWindowFlag(Qt::WindowStaysOnTopHint, true);
  // Pass ROI names if RTSTRUCT is loaded
  if (m_rtstructLoaded) {
    QStringList roiNames;
    for (int r = 0; r < m_rtstruct.roiCount(); ++r) {
      roiNames << m_rtstruct.roiName(r);
    }
    dlg->setROINames(roiNames);
  }
  // Set dose axis suggestion
  if (m_doseLoaded && m_resampledDose.isResampled()) {
    dlg->setDoseAxisMax(m_resampledDose.maxDose());
  }

  // Start Monte Carlo simulation on request
  connect(
      dlg, &RandomStudyDialog::startCalculationRequested, this, [this, dlg]() {
        // Clear any previous cache to ensure fresh results
        dlg->clearCache();
        if (!isVolumeLoaded() || !m_doseLoaded || !m_rtstructLoaded) {
          QMessageBox::warning(dlg, tr("Random Study"),
                               tr("CT, RT Dose and RTSTRUCT must be loaded."));
          return;
        }
        int roiIndex = dlg->selectedROIIndex();
        if (roiIndex < 0 || roiIndex >= m_rtstruct.roiCount()) {
          QMessageBox::warning(dlg, tr("Random Study"),
                               tr("Please select a ROI."));
          return;
        }
        int nFr = std::max(1, dlg->fractionCount());
        int nIter = std::max(1, dlg->iterationCount());
        double sysMx, sysMy, sysMz, sysSx, sysSy, sysSz;
        dlg->systematicError(sysMx, sysMy, sysMz, sysSx, sysSy, sysSz);
        double randSx, randSy, randSz;
        dlg->randomError(randSx, randSy, randSz);

        // Cache ROI patient coordinates (voxel centers) within bounding box and
        // inside ROI
        QVector3D roiMin, roiMax;
        if (!m_rtstruct.roiBoundingBox(roiIndex, roiMin, roiMax)) {
          QMessageBox::warning(dlg, tr("Random Study"),
                               tr("ROI has no bounding box."));
          return;
        }
        QVector3D minVox = m_volume.patientToVoxelContinuous(roiMin);
        QVector3D maxVox = m_volume.patientToVoxelContinuous(roiMax);
        int x0 = std::clamp(
            static_cast<int>(std::floor(std::min(minVox.x(), maxVox.x()))), 0,
            m_volume.width() - 1);
        int x1 = std::clamp(
            static_cast<int>(std::ceil(std::max(minVox.x(), maxVox.x()))), 0,
            m_volume.width() - 1);
        int y0 = std::clamp(
            static_cast<int>(std::floor(std::min(minVox.y(), maxVox.y()))), 0,
            m_volume.height() - 1);
        int y1 = std::clamp(
            static_cast<int>(std::ceil(std::max(minVox.y(), maxVox.y()))), 0,
            m_volume.height() - 1);
        int z0 = std::clamp(static_cast<int>(static_cast<int>(
                                std::floor(std::min(minVox.z(), maxVox.z())))),
                            0, m_volume.depth() - 1);
        int z1 = std::clamp(
            static_cast<int>(std::ceil(std::max(minVox.z(), maxVox.z()))), 0,
            m_volume.depth() - 1);

        std::vector<QVector3D> roiPatientPoints;
        roiPatientPoints.reserve((x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1) /
                                 10);
        for (int z = z0; z <= z1; ++z) {
          for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
              QVector3D patient =
                  m_volume.voxelToPatient(x + 0.5, y + 0.5, z + 0.5);
              if (m_rtstruct.isPointInsideROI(patient, roiIndex)) {
                roiPatientPoints.push_back(patient);
              }
            }
          }
        }
        if (roiPatientPoints.empty()) {
          QMessageBox::warning(dlg, tr("Random Study"),
                               tr("No points found inside ROI."));
          return;
        }

        // Prepare dialog plot and baseline (depending on view)
        const auto vm = dlg->viewMode();
        const auto ht = dlg->histogramType();
        const double hparam = dlg->histogramParam();
        const int hbins = dlg->histogramBins();
        dlg->clearPlot();
        if (vm == RandomStudyDialog::ViewMode::DVH) {
          dlg->prepareDVHAxes();
          if (m_resampledDose.isResampled())
            dlg->setDoseAxisMax(m_resampledDose.maxDose());
        }

        // Baseline histogram (no error): always compute and store, and plot DVH
        // if in DVH view
        {
          const double voxelVolume =
              m_volume.spacingX() * m_volume.spacingY() * m_volume.spacingZ();
          const double maxDose = m_resampledDose.maxDose();
          const double binSize = std::max(0.001, maxDose / 200.0);
          const int binCount = std::min(
              5000, static_cast<int>(std::ceil((maxDose + binSize) / binSize)));
          std::vector<double> hist(binCount, 0.0);
          for (const auto &p : roiPatientPoints) {
            QVector3D v = m_volume.patientToVoxelContinuous(p);
            float dval = sampleResampledDose(v);
            int b = static_cast<int>(dval / binSize);
            if (b >= 0 && b < binCount)
              hist[b] += voxelVolume;
          }
          const double totalVol = roiPatientPoints.size() * voxelVolume;
          // store baseline histogram in dialog
          QVector<float> baseF;
          baseF.resize(binCount);
          for (int i = 0; i < binCount; ++i)
            baseF[i] = static_cast<float>(hist[i]);
          QMetaObject::invokeMethod(
              dlg,
              [dlg, baseF, binSize, totalVol]() {
                if (dlg)
                  dlg->setBaselineHistogram(baseF, binSize, totalVol);
              },
              Qt::QueuedConnection);
          if (vm == RandomStudyDialog::ViewMode::DVH) {
            double cumulative = 0.0;
            QVector<double> x, y;
            x.reserve(binCount);
            y.reserve(binCount);
            for (int i = binCount - 1; i >= 0; --i) {
              cumulative += hist[i];
              double volPct =
                  totalVol > 0.0 ? (cumulative / totalVol * 100.0) : 0.0;
              x.push_back(i * binSize);
              y.push_back(volPct);
            }
            std::reverse(x.begin(), x.end());
            std::reverse(y.begin(), y.end());
            // Baseline curve in opaque color
            QColor baseColor = QColor::fromHsv((roiIndex * 40) % 360, 220, 255);
            baseColor.setAlpha(255);
            dlg->addDVHCurve(x, y, baseColor, true, 2);
          }
        }

        // Prepare running state
        dlg->setRunning(true);
        auto cancelFlag = std::make_shared<std::atomic_bool>(false);
        connect(dlg, &RandomStudyDialog::cancelCalculationRequested, dlg,
                [cancelFlag]() {
                  if (cancelFlag)
                    cancelFlag->store(true);
                });

        // Run simulation asynchronously
        QPointer<RandomStudyDialog> dlgPtr(dlg);
        const bool fixSeed = dlg->isSeedFixed();
        const quint64 seedVal = dlg->seedValue();
        auto worker = [this, dlgPtr,
                       roiPatientPoints = std::move(roiPatientPoints), roiIndex,
                       nFr, nIter, sysMx, sysMy, sysMz, sysSx, sysSy, sysSz,
                       randSx, randSy, randSz, fixSeed, seedVal, cancelFlag, vm,
                       ht, hparam, hbins]() {
          // Random generators
          std::mt19937_64 gen;
          if (fixSeed) {
            gen.seed(seedVal);
          } else {
            std::random_device rd;
            gen.seed((static_cast<quint64>(rd()) << 32) ^ rd());
          }
          std::normal_distribution<double> sysX(sysMx, std::max(0.0, sysSx));
          std::normal_distribution<double> sysY(sysMy, std::max(0.0, sysSy));
          std::normal_distribution<double> sysZ(sysMz, std::max(0.0, sysSz));
          std::normal_distribution<double> rndX(0.0, std::max(0.0, randSx));
          std::normal_distribution<double> rndY(0.0, std::max(0.0, randSy));
          std::normal_distribution<double> rndZ(0.0, std::max(0.0, randSz));

          const double voxelVolume =
              m_volume.spacingX() * m_volume.spacingY() * m_volume.spacingZ();
          const double maxDose = m_resampledDose.maxDose();
          const double binSize = std::max(0.001, maxDose / 200.0);
          const int binCount = std::min(
              5000, static_cast<int>(std::ceil((maxDose + binSize) / binSize)));

          const int maxCurves = 100;
          const int stride = std::max(1, (nIter + maxCurves - 1) / maxCurves);
          int plotted = 0;
          QVector<double> metricVals;
          metricVals.reserve(nIter);

          // Cache all iteration histograms for instant recomputation later
          QVector<QVector<float>> cachedHists;
          cachedHists.reserve(nIter);
          auto updateHistogram = [&](const QVector<double> &vals) {
            if (!dlgPtr)
              return;
            QString xLabel = QObject::tr("Value");
            switch (ht) {
            case RandomStudyDialog::HistType::DxCc:
            case RandomStudyDialog::HistType::DxPct:
            case RandomStudyDialog::HistType::MinDose:
            case RandomStudyDialog::HistType::MaxDose:
            case RandomStudyDialog::HistType::MeanDose:
              xLabel = QObject::tr("Dose [Gy]");
              break;
            case RandomStudyDialog::HistType::VdCc:
              xLabel = QObject::tr("Volume [cc]");
              break;
            }
            // Compute mean and percentiles (5,10,25,75,90,95)
            double mean = 0.0;
            QVector<double> sorted = vals;
            if (!sorted.isEmpty()) {
              for (double v : sorted)
                mean += v;
              mean /= static_cast<double>(sorted.size());
              std::sort(sorted.begin(), sorted.end());
            }
            auto pctAt = [&](double p) -> double {
              if (sorted.isEmpty())
                return 0.0;
              double pos = p * (sorted.size() - 1) / 100.0;
              int idx = static_cast<int>(std::floor(pos));
              int idx2 = std::min(static_cast<int>(sorted.size() - 1), idx + 1);
              double t = pos - idx;
              return sorted[idx] * (1.0 - t) +
                     sorted[idx2] * t; // linear interp
            };
            QVector<QPair<double, QString>> marks;
            marks << qMakePair(pctAt(5.0), QObject::tr("P5"))
                  << qMakePair(pctAt(10.0), QObject::tr("P10"))
                  << qMakePair(pctAt(25.0), QObject::tr("P25"))
                  << qMakePair(pctAt(75.0), QObject::tr("P75"))
                  << qMakePair(pctAt(90.0), QObject::tr("P90"))
                  << qMakePair(pctAt(95.0), QObject::tr("P95"));

            QMetaObject::invokeMethod(
                dlgPtr,
                [dlgPtr, vals, xLabel, hbins, mean, marks]() {
                  if (!dlgPtr)
                    return;
                  dlgPtr->plotHistogram(vals, hbins, xLabel,
                                        QObject::tr("Count"));
                  dlgPtr->setHistogramMarkers(mean, marks);
                },
                Qt::QueuedConnection);
          };
          for (int it = 0; it < nIter; ++it) {
            if (!dlgPtr)
              return; // dialog deleted
            if (cancelFlag && cancelFlag->load())
              break;
            // sample systematic per iteration
            const QVector3D sys(sysX(gen), sysY(gen), sysZ(gen));
            // Pre-sample a single random setup shift for each fraction (global
            // per Fr)
            std::vector<QVector3D> fracDeltas;
            fracDeltas.reserve(nFr);
            for (int f = 0; f < nFr; ++f) {
              QVector3D rnd(rndX(gen), rndY(gen), rndZ(gen));
              fracDeltas.push_back(sys + rnd);
            }

            std::vector<double> doses;
            doses.reserve(roiPatientPoints.size());
            for (const auto &p : roiPatientPoints) {
              double total = 0.0;
              for (int f = 0; f < nFr; ++f) {
                const QVector3D &delta = fracDeltas[f];
                QVector3D shifted =
                    p - delta; // patient shift +delta => sample at p - delta
                QVector3D v = m_volume.patientToVoxelContinuous(shifted);
                float dval = sampleResampledDose(v);
                total += static_cast<double>(dval);
              }
              doses.push_back(total / static_cast<double>(nFr));
            }

            // Build histogram (non-cumulative volume per dose bin)
            std::vector<double> hist(binCount, 0.0);
            for (double d : doses) {
              int b = static_cast<int>(d / binSize);
              if (b >= 0 && b < binCount)
                hist[b] += voxelVolume;
            }
            // Save a float copy into cache (memory efficient)
            QVector<float> histF;
            histF.resize(binCount);
            for (int i = 0; i < binCount; ++i)
              histF[i] = static_cast<float>(hist[i]);
            cachedHists.push_back(std::move(histF));
            // cumulative from high to low
            double cumulative = 0.0;
            const double totalVol = doses.size() * voxelVolume;
            QVector<double> x, y;
            x.reserve(binCount);
            y.reserve(binCount);
            for (int i = binCount - 1; i >= 0; --i) {
              cumulative += hist[i];
              double volPct =
                  totalVol > 0.0 ? (cumulative / totalVol * 100.0) : 0.0;
              x.push_back(i * binSize);
              y.push_back(volPct);
            }
            std::reverse(x.begin(), x.end());
            std::reverse(y.begin(), y.end());

            // Emit plot update on GUI thread, but limit to ~100 curves
            if (!dlgPtr)
              return;
            if (vm == RandomStudyDialog::ViewMode::DVH) {
              if (it % stride == 0 && plotted < maxCurves) {
                ++plotted;
                QColor col = QColor::fromHsv((it * 37) % 360, 200, 255);
                col.setAlpha(70);
                QMetaObject::invokeMethod(
                    dlgPtr,
                    [dlgPtr, x, y, col]() {
                      if (dlgPtr)
                        dlgPtr->addDVHCurve(x, y, col, true);
                    },
                    Qt::QueuedConnection);
              }
            }

            // Histogram metric accumulation if needed
            if (vm == RandomStudyDialog::ViewMode::Histogram) {
              double metric = 0.0;
              switch (ht) {
              case RandomStudyDialog::HistType::DxCc: {
                double targetVol =
                    std::clamp(hparam, 0.0, doses.size() * voxelVolume);
                double cum = 0.0;
                double doseAt = 0.0;
                for (int i = binCount - 1; i >= 0; --i) {
                  cum += hist[i];
                  if (cum >= targetVol) {
                    doseAt = i * binSize;
                    break;
                  }
                }
                metric = doseAt;
                break;
              }
              case RandomStudyDialog::HistType::DxPct: {
                double pct = std::clamp(hparam, 0.0, 100.0);
                double targetVol = totalVol * (pct / 100.0);
                double cum = 0.0;
                double doseAt = 0.0;
                for (int i = binCount - 1; i >= 0; --i) {
                  cum += hist[i];
                  if (cum >= targetVol) {
                    doseAt = i * binSize;
                    break;
                  }
                }
                metric = doseAt;
                break;
              }
              case RandomStudyDialog::HistType::VdCc: {
                int thBin =
                    std::max(0, std::min(binCount - 1,
                                         static_cast<int>(hparam / binSize)));
                double volMm3 = 0.0;
                for (int i = thBin; i < binCount; ++i)
                  volMm3 += hist[i];
                metric = volMm3 / 1000.0; // convert mm^3 to cc
                break;
              }
              case RandomStudyDialog::HistType::MinDose: {
                double md = std::numeric_limits<double>::infinity();
                for (double d : doses)
                  md = std::min(md, d);
                if (!std::isfinite(md))
                  md = 0.0;
                metric = md;
                break;
              }
              case RandomStudyDialog::HistType::MaxDose: {
                double Md = 0.0;
                for (double d : doses)
                  Md = std::max(Md, d);
                metric = Md;
                break;
              }
              case RandomStudyDialog::HistType::MeanDose: {
                double sum = 0.0;
                for (double d : doses)
                  sum += d;
                metric = doses.empty()
                             ? 0.0
                             : (sum / static_cast<double>(doses.size()));
                break;
              }
              }
              metricVals.push_back(metric);
              // Update histogram occasionally
              if (it % stride == 0) {
                updateHistogram(metricVals);
              }
            }
            // progress
            QMetaObject::invokeMethod(
                dlgPtr,
                [dlgPtr, it, nIter]() {
                  if (dlgPtr)
                    dlgPtr->setCalculationProgress(it + 1, nIter);
                },
                Qt::QueuedConnection);
          }
          // Final histogram update
          if (vm == RandomStudyDialog::ViewMode::Histogram) {
            updateHistogram(metricVals);
          }
          // Store cache into dialog for instant reuse across view switches
          if (dlgPtr) {
            const double totalVol =
                static_cast<double>(roiPatientPoints.size()) * voxelVolume;
            QMetaObject::invokeMethod(
                dlgPtr,
                [dlgPtr, cachedHists, binSize, totalVol]() {
                  if (!dlgPtr)
                    return;
                  dlgPtr->setCachedHistograms(cachedHists, binSize, totalVol);
                },
                Qt::QueuedConnection);
          }
          // Re-enable controls when finished/cancelled
          if (dlgPtr) {
            QMetaObject::invokeMethod(
                dlgPtr,
                [dlgPtr]() {
                  if (dlgPtr)
                    dlgPtr->setRunning(false);
                },
                Qt::QueuedConnection);
          }
        };

        auto future = QtConcurrent::run(worker);
        Q_UNUSED(future);
      });

  dlg->show();
  dlg->raise();
  dlg->activateWindow();
}

void DicomViewer::onGammaAnalysisClicked() {
  if (!m_gammaAnalysisWindow) {
    m_gammaAnalysisWindow = new GammaAnalysisWindow(this);
    m_gammaAnalysisWindow->setWindowFlag(Qt::WindowStaysOnTopHint, true);
  }

  std::vector<GammaAnalysisWindow::DoseEntry> entries;
  entries.reserve(m_doseItems.size());
  for (const auto &item : m_doseItems) {
    if (!item.widget)
      continue;
    GammaAnalysisWindow::DoseEntry entry;
    entry.name = item.widget->name();
    entry.dose = item.dose;
    entry.dataFractions = item.widget->dataFractions();
    entry.displayFractions = item.widget->displayFractions();
    entry.factor = item.widget->factor();
    entry.shift = computeAlignmentShift(item.dose) + item.widget->shift();
    entries.push_back(entry);
  }

  if (entries.size() < 2) {
    QMessageBox::warning(
        this, tr("Gamma Analysis"),
        tr("Please load at least two dose distributions."));
  }

  m_gammaAnalysisWindow->setDoseEntries(
      entries, static_cast<int>(m_doseCalcMode), m_doseAlphaBeta);
  m_gammaAnalysisWindow->show();
  m_gammaAnalysisWindow->raise();
  m_gammaAnalysisWindow->activateWindow();
}

double DicomViewer::transformDose(double rawDose, double dataFr,
                                  double displayFr) const {
  double d = rawDose / std::max(1.0, dataFr);
  double physical = d * displayFr;
  if (m_doseCalcMode == DoseCalcMode::Physical) {
    return physical;
  }
  double bed = physical * (1.0 + d / m_doseAlphaBeta);
  if (m_doseCalcMode == DoseCalcMode::BED) {
    return bed;
  }
  return bed / (1.0 + 2.0 / m_doseAlphaBeta);
}

QVector3D DicomViewer::computeAlignmentShift(const RTDoseVolume &dose) const {
  Q_UNUSED(dose);
  // updateDoseAlignment() で確定したシフトをそのまま使用する
  // （センター合わせ等のフォールバックを含む）
  return m_doseShift;
}

void DicomViewer::updateColorBars() {
  bool hasDoseData = m_doseLoaded && isVolumeLoaded();
  bool hasDoseSources = !m_doseItems.empty();
  if (!hasDoseData && !hasDoseSources) {
    std::fill(std::begin(m_colorBarPersistentVisibility),
              std::end(m_colorBarPersistentVisibility), false);
  }

  bool showColorBars = hasDoseData && m_doseVisible;
  // Calculate default max regardless of visibility for proper dose range
  double defaultMax = hasDoseData ? m_resampledDose.maxDose() * 1.1 : 0.0;

  double minDose = 0.0;
  double maxDose = defaultMax;

  // Always read and use spinbox values if they exist
  if (m_doseMinSpinBox && m_doseMaxSpinBox) {
    // Initialize max spinbox with default if it's 0 and we have dose data
    if (m_doseMaxSpinBox->value() == 0.0 && defaultMax > 0.0) {
      QSignalBlocker blocker(m_doseMaxSpinBox);
      m_doseMaxSpinBox->setValue(defaultMax);
    }

    // Always read spinbox values
    minDose = m_doseMinSpinBox->value();
    maxDose = m_doseMaxSpinBox->value();

    // If max is still 0 or invalid, use default
    if (maxDose <= 0.0 && defaultMax > 0.0) {
      maxDose = defaultMax;
    }

    // Ensure max is greater than min
    if (maxDose <= minDose) {
      maxDose = minDose + 0.0001;
    }
  }

  m_doseMinRange = minDose;
  m_doseMaxRange = maxDose;

  // Debug output
  qDebug() << "updateColorBars: hasDoseData=" << hasDoseData
           << "m_doseVisible=" << m_doseVisible
           << "defaultMax=" << defaultMax
           << "minDose=" << minDose
           << "maxDose=" << maxDose;

  int count = 1;
  if (m_viewMode == ViewMode::Dual)
    count = 2;
  else if (m_viewMode == ViewMode::Quad)
    count = 4;
  else if (m_viewMode == ViewMode::Five)
    count = VIEW_COUNT;

  for (int i = 0; i < VIEW_COUNT; ++i) {
    bool hideForFiveMode = (m_viewMode == ViewMode::Five && (i == 2 || i == 3));
    if (!m_colorBars[i])
      continue;

    bool viewActive = (i < count) && !hideForFiveMode;
    if (!viewActive) {
      m_colorBars[i]->setVisible(false);
      continue;
    }

    if (showColorBars) {
      m_colorBars[i]->setDoseRange(minDose, maxDose);
      m_colorBars[i]->setDisplayMode(m_doseDisplayMode);
      m_colorBars[i]->setReferenceDose(m_doseReference);
      m_colorBars[i]->setVisible(true);
      m_colorBarPersistentVisibility[i] = true;
    } else if (m_colorBarPersistentVisibility[i]) {
      m_colorBars[i]->setVisible(true);
    } else {
      m_colorBars[i]->setVisible(false);
    }
  }
}

void DicomViewer::updateDoseShiftLabels() {
  // Determine if any user-defined dose shift is set (non-zero)
  bool hasUserShift = false;
  const double eps = 1e-6;
  for (const auto &it : m_doseItems) {
    if (!it.widget)
      continue;
    QVector3D s = it.widget->shift();
    if (std::fabs(s.x()) > eps || std::fabs(s.y()) > eps ||
        std::fabs(s.z()) > eps) {
      hasUserShift = true;
      break;
    }
  }

  // Determine if dose mode needs indicator
  bool hasModeInfo =
      m_doseCalcMode == DoseCalcMode::BED || m_doseCalcMode == DoseCalcMode::EqD2;

  // Show only when dose overlay is relevant
  bool shouldShow = (hasUserShift || hasModeInfo) && m_doseLoaded && m_doseVisible;

  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (!m_doseShiftLabels[i])
      continue;
    // Show only on image views (Ax/Sag/Cor), not DVH/3D/Profile
    bool isImageView = !(m_isDVHView[i] || m_is3DView[i] || m_isProfileView[i]);
    if (shouldShow && isImageView && m_viewContainers[i]->isVisible()) {
      QStringList parts;
      if (hasUserShift)
        parts << "Dose Shift";
      if (hasModeInfo) {
        QString modeStr =
            (m_doseCalcMode == DoseCalcMode::BED) ? "BED" : "EqD2";
        parts << QString("%1 α/β=%2").arg(modeStr).arg(m_doseAlphaBeta);
      }
      m_doseShiftLabels[i]->setText(parts.join(" "));
      m_doseShiftLabels[i]->adjustSize();
      m_doseShiftLabels[i]->show();
    } else {
      m_doseShiftLabels[i]->hide();
    }
  }

  // Reposition after visibility change
  updateSliderPosition();
}

void DicomViewer::resetDoseRange() {
  const double rawMaxDose = m_resampledDose.maxDose();
  const double maxDose = rawMaxDose * 1.1;
  const double minDose = maxDose > 0.0 ? maxDose * 0.05 : 0.0;

  if (m_doseMinSpinBox && m_doseMaxSpinBox) {
    QSignalBlocker minBlocker(m_doseMinSpinBox);
    QSignalBlocker maxBlocker(m_doseMaxSpinBox);
    m_doseMinSpinBox->setValue(minDose);
    m_doseMaxSpinBox->setValue(maxDose);
  }
  if (m_doseOpacitySlider) {
    m_doseOpacitySlider->setValue(80);
  }
  m_doseMinRange = minDose;
  m_doseMaxRange = maxDose;
}

// Colormap helper functions for dose display
QColor DicomViewer::getDoseColor(double doseGy, double maxDoseGy) const {
  if (maxDoseGy <= 0.0 || doseGy <= 0.0) {
    return QColor(0, 0, 0, 0);
  }

  float doseRatio = static_cast<float>(doseGy / maxDoseGy);
  QRgb rgb;

  switch (m_doseDisplayMode) {
    case DoseResampledVolume::DoseDisplayMode::Colorful:
      rgb = mapDoseToColorHSV(doseRatio);
      break;
    case DoseResampledVolume::DoseDisplayMode::Isodose:
      rgb = mapDoseToIsodose(doseRatio);
      break;
    case DoseResampledVolume::DoseDisplayMode::Hot:
      rgb = mapDoseToHot(doseRatio);
      break;
    case DoseResampledVolume::DoseDisplayMode::Simple:
      // Simple mode: use red color with intensity based on dose
      {
        int intensity = static_cast<int>(std::clamp(doseRatio * 255.0f, 0.0f, 255.0f));
        rgb = qRgba(intensity, 0, 0, 255);
      }
      break;
    default:
      rgb = mapDoseToColorHSV(doseRatio);
      break;
  }

  return QColor(qRed(rgb), qGreen(rgb), qBlue(rgb));
}

QRgb DicomViewer::mapDoseToColorHSV(float doseRatio) const {
  if (doseRatio <= 0.0f) {
    return qRgba(0, 0, 0, 0);
  }

  float hue = 0.0f;
  float saturation = 1.0f;
  float value = 1.0f;

  if (doseRatio <= 0.2f) {
    hue = 240.0f - (doseRatio / 0.2f) * 60.0f;
    saturation = 0.8f + (doseRatio / 0.2f) * 0.2f;
  } else if (doseRatio <= 0.4f) {
    float t = (doseRatio - 0.2f) / 0.2f;
    hue = 180.0f - t * 60.0f;
    saturation = 1.0f;
  } else if (doseRatio <= 0.6f) {
    float t = (doseRatio - 0.4f) / 0.2f;
    hue = 120.0f - t * 60.0f;
    saturation = 1.0f;
  } else if (doseRatio <= 0.8f) {
    float t = (doseRatio - 0.6f) / 0.2f;
    hue = 60.0f - t * 30.0f;
    saturation = 1.0f;
    value = 1.0f;
  } else if (doseRatio <= 1.0f) {
    float t = (doseRatio - 0.8f) / 0.2f;
    hue = 30.0f - t * 30.0f;
    saturation = 1.0f;
    value = 1.0f;
  } else {
    float t = std::min(1.0f, (doseRatio - 1.0f) / 0.5f);
    hue = 360.0f - t * 60.0f;
    saturation = 1.0f;
    value = 1.0f - t * 0.2f;
  }

  QColor color = QColor::fromHsvF(hue / 360.0f, saturation, value);
  return qRgba(color.red(), color.green(), color.blue(), 255);
}

QRgb DicomViewer::mapDoseToIsodose(float doseRatio) const {
  static const float isodoseLevels[] = {0.95f, 0.90f, 0.80f, 0.70f,
                                        0.50f, 0.30f, 0.10f};

  static const QRgb isodoseColors[] = {
      qRgba(255, 0, 0, 255),   // 赤
      qRgba(255, 128, 0, 255), // オレンジ
      qRgba(255, 255, 0, 255), // 黄
      qRgba(0, 255, 0, 255),   // 緑
      qRgba(0, 255, 255, 255), // シアン
      qRgba(0, 0, 255, 255),   // 青
      qRgba(128, 0, 255, 255)  // 紫
  };

  for (int i = 0; i < 7; ++i) {
    if (doseRatio >= isodoseLevels[i]) {
      return isodoseColors[i];
    }
  }

  return qRgba(0, 0, 0, 0);
}

QRgb DicomViewer::mapDoseToHot(float doseRatio) const {
  doseRatio = std::clamp(doseRatio, 0.0f, 1.0f);
  float r = std::min(1.0f, doseRatio * 3.0f);
  float g = std::clamp((doseRatio - 0.33f) * 3.0f, 0.0f, 1.0f);
  float b = std::clamp((doseRatio - 0.66f) * 3.0f, 0.0f, 1.0f);
  return qRgba(static_cast<int>(r * 255), static_cast<int>(g * 255),
               static_cast<int>(b * 255), 255);
}

void DicomViewer::onStructureVisibilityChanged(QListWidgetItem *item) {
  if (!item) {
    qWarning() << "onStructureVisibilityChanged: null item";
    return;
  }

  // インデックスの安全な取得
  int index = m_structureList->row(item);
  qDebug() << QString("Structure visibility changed: index=%1, item=%2")
                  .arg(index)
                  .arg(item->text());

  // インデックス範囲チェック
  if (index < 0) {
    qWarning() << QString("Invalid structure index: %1").arg(index);
    return;
  }

  // ROI数と範囲の確認
  int roiCount = m_rtstruct.roiCount();
  if (index >= roiCount) {
    qWarning() << QString("Structure index %1 out of range (max: %2)")
                      .arg(index)
                      .arg(roiCount - 1);
    return;
  }

  // チェック状態の取得
  bool visible = (item->checkState() == Qt::Checked);
  qDebug() << QString("Setting ROI %1 (%2) visibility to %3")
                  .arg(index)
                  .arg(item->text())
                  .arg(visible);

  try {
    // RTStructでの可視性設定（安全にチェック）
    if (index < roiCount) {
      m_rtstruct.setROIVisible(index, visible);
    }

    // 画像の更新（安全に実行）
    updateImage();

    qDebug() << QString("Structure visibility update completed for ROI %1")
                    .arg(index);

  } catch (const std::exception &e) {
    qCritical() << QString("Error updating structure visibility for ROI %1: %2")
                       .arg(index)
                       .arg(e.what());
  } catch (...) {
    qCritical() << QString(
                       "Unknown error updating structure visibility for ROI %1")
                       .arg(index);
  }
}

void DicomViewer::onShowAllStructures() {
  qDebug() << "Showing all structures";

  int listCount = m_structureList->count();
  int roiCount = m_rtstruct.roiCount();

  qDebug()
      << QString("List items: %1, ROI count: %2").arg(listCount).arg(roiCount);

  if (listCount == 0) {
    qDebug() << "No structures in list";
    return;
  }

  try {
    // シグナルを無効化して循環参照を防ぐ
    m_structureList->blockSignals(true);

    // 安全な範囲でループ
    int maxIndex = std::min(listCount, roiCount);
    for (int i = 0; i < maxIndex; ++i) {
      QListWidgetItem *item = m_structureList->item(i);
      if (item) {
        item->setCheckState(Qt::Checked);
        qDebug()
            << QString("Set item %1 (%2) to checked").arg(i).arg(item->text());
      }

      // RTStructの可視性を設定
      if (i < roiCount) {
        m_rtstruct.setROIVisible(i, true);
      }
    }

    // シグナルを再有効化
    m_structureList->blockSignals(false);

    // 画像を更新
    updateImage();

    qDebug() << "Show all structures completed";

  } catch (const std::exception &e) {
    qCritical() << "Error in onShowAllStructures:" << e.what();
    m_structureList->blockSignals(false); // エラー時も確実にシグナルを再有効化
  } catch (...) {
    qCritical() << "Unknown error in onShowAllStructures";
    m_structureList->blockSignals(false);
  }
}

void DicomViewer::onHideAllStructures() {
  qDebug() << "Hiding all structures";

  int listCount = m_structureList->count();
  int roiCount = m_rtstruct.roiCount();

  qDebug()
      << QString("List items: %1, ROI count: %2").arg(listCount).arg(roiCount);

  if (listCount == 0) {
    qDebug() << "No structures in list";
    return;
  }

  try {
    // シグナルを無効化して循環参照を防ぐ
    m_structureList->blockSignals(true);

    // 安全な範囲でループ
    int maxIndex = std::min(listCount, roiCount);
    for (int i = 0; i < maxIndex; ++i) {
      QListWidgetItem *item = m_structureList->item(i);
      if (item) {
        item->setCheckState(Qt::Unchecked);
        qDebug() << QString("Set item %1 (%2) to unchecked")
                        .arg(i)
                        .arg(item->text());
      }

      // RTStructの可視性を設定
      if (i < roiCount) {
        m_rtstruct.setROIVisible(i, false);
      }
    }

    // シグナルを再有効化
    m_structureList->blockSignals(false);

    // 画像を更新
    updateImage();

    qDebug() << "Hide all structures completed";

  } catch (const std::exception &e) {
    qCritical() << "Error in onHideAllStructures:" << e.what();
    m_structureList->blockSignals(false); // エラー時も確実にシグナルを再有効化
  } catch (...) {
    qCritical() << "Unknown error in onHideAllStructures";
    m_structureList->blockSignals(false);
  }
}

void DicomViewer::onStructureLineWidthChanged(int value) {
  m_structureLineWidth = value;
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_imageWidgets[i]) {
      m_imageWidgets[i]->setStructureLineWidth(value);
    }
    if (m_3dWidgets[i]) {
      m_3dWidgets[i]->setStructureLineWidth(value);
    }
  }
  updateImage();
}

void DicomViewer::onStructurePointsToggled(bool checked) {
  m_showStructurePoints = checked;
  updateImage();
}

void DicomViewer::updateOrientationButtonTexts() {
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (!m_orientationButtons[i])
      continue;

    QString text;
    if (m_isDVHView[i]) {
      text = "DVH";
    } else if (m_is3DView[i]) {
      text = "3D";
    } else if (m_isProfileView[i]) {
      text = "Line";
    } else {
      switch (m_viewOrientations[i]) {
      case DicomVolume::Orientation::Axial:
        text = "Ax";
        break;
      case DicomVolume::Orientation::Sagittal:
        text = "Sag";
        break;
      case DicomVolume::Orientation::Coronal:
        text = "Cor";
        break;
      }
    }
    m_orientationButtons[i]->setText(text);
    m_orientationButtons[i]->adjustSize();
  }
  updateSliderPosition();
}

// ★新規追加: 方向変更メニュー（全ビューで使用可能）
void DicomViewer::showOrientationMenuForView(int viewIndex) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT || !isVolumeLoaded())
    return;

  QMenu menu;
  QAction *ax = menu.addAction("Axial");
  QAction *sag = menu.addAction("Sagittal");
  QAction *cor = menu.addAction("Coronal");
  QAction *dvh = menu.addAction("DVH");
  QAction *threeD = menu.addAction("3D");
  QAction *profile = menu.addAction("Profile");

  // 現在の方向にチェックマークを付ける
  if (m_isDVHView[viewIndex]) {
    dvh->setChecked(true);
  } else if (m_is3DView[viewIndex]) {
    threeD->setChecked(true);
  } else if (m_isProfileView[viewIndex]) {
    profile->setChecked(true);
  } else {
    switch (m_viewOrientations[viewIndex]) {
    case DicomVolume::Orientation::Axial:
      ax->setChecked(true);
      break;
    case DicomVolume::Orientation::Sagittal:
      sag->setChecked(true);
      break;
    case DicomVolume::Orientation::Coronal:
      cor->setChecked(true);
      break;
    }
  }

  QAction *sel = menu.exec(m_orientationButtons[viewIndex]->mapToGlobal(
      QPoint(0, m_orientationButtons[viewIndex]->height())));
  if (!sel)
    return;

  if (sel == dvh) {
    if (!m_isDVHView[viewIndex]) {
      m_isDVHView[viewIndex] = true;
      m_is3DView[viewIndex] = false;
      m_isProfileView[viewIndex] = false;
      m_viewStacks[viewIndex]->setCurrentWidget(m_dvhWidgets[viewIndex]);
      if (!m_dvhData.empty()) {
        m_dvhWidgets[viewIndex]->setDVHData(m_dvhData);
        m_dvhWidgets[viewIndex]->setPatientInfo(patientInfoText());
      }
      m_sliceIndexLabels[viewIndex]->hide();
      if (m_infoOverlays[viewIndex])
        m_infoOverlays[viewIndex]->hide();
      if (m_coordLabels[viewIndex])
        m_coordLabels[viewIndex]->hide();
      if (m_cursorDoseLabels[viewIndex])
        m_cursorDoseLabels[viewIndex]->hide();
      m_imageWidgets[viewIndex]->clearCursorCross();
      updateOrientationButtonTexts();
      updateSliderPosition();
      updateImage();
      updateInteractionButtonVisibility(viewIndex);
    }
    return;
  }

  if (sel == threeD) {
    if (!m_is3DView[viewIndex]) {
      m_is3DView[viewIndex] = true;
      m_isDVHView[viewIndex] = false;
      m_isProfileView[viewIndex] = false;
      m_viewStacks[viewIndex]->setCurrentWidget(m_3dWidgets[viewIndex]);
      m_sliceIndexLabels[viewIndex]->hide();
      if (m_infoOverlays[viewIndex])
        m_infoOverlays[viewIndex]->hide();
      if (m_coordLabels[viewIndex])
        m_coordLabels[viewIndex]->hide();
      if (m_cursorDoseLabels[viewIndex])
        m_cursorDoseLabels[viewIndex]->hide();
      m_imageWidgets[viewIndex]->clearCursorCross();
      updateOrientationButtonTexts();
      updateSliderPosition();
      // update 3D view
      update3DView(viewIndex);
      updateInteractionButtonVisibility(viewIndex);
    }
    return;
  }

  if (sel == profile) {
    if (!m_isProfileView[viewIndex]) {
      m_isProfileView[viewIndex] = true;
      m_isDVHView[viewIndex] = false;
      m_is3DView[viewIndex] = false;
      m_viewStacks[viewIndex]->setCurrentWidget(m_profileWidgets[viewIndex]);
      m_sliceIndexLabels[viewIndex]->hide();
      if (m_infoOverlays[viewIndex])
        m_infoOverlays[viewIndex]->hide();
      if (m_coordLabels[viewIndex])
        m_coordLabels[viewIndex]->hide();
      if (m_cursorDoseLabels[viewIndex])
        m_cursorDoseLabels[viewIndex]->hide();
      m_imageWidgets[viewIndex]->clearCursorCross();
      updateOrientationButtonTexts();
      updateSliderPosition();
      updateInteractionButtonVisibility(viewIndex);
    }
    return;
  }

  if (m_isDVHView[viewIndex] || m_is3DView[viewIndex] ||
      m_isProfileView[viewIndex]) {
    m_isDVHView[viewIndex] = false;
    m_is3DView[viewIndex] = false;
    m_isProfileView[viewIndex] = false;
    m_viewStacks[viewIndex]->setCurrentWidget(m_imagePanels[viewIndex]);
    if (m_infoOverlays[viewIndex])
      m_infoOverlays[viewIndex]->show();
    if (m_coordLabels[viewIndex])
      m_coordLabels[viewIndex]->show();
    if (m_cursorDoseLabels[viewIndex])
      m_cursorDoseLabels[viewIndex]->hide();
    m_imageWidgets[viewIndex]->clearCursorCross();
    updateSliderPosition();
    updateInteractionButtonVisibility(viewIndex);
  }

  DicomVolume::Orientation newOri;
  if (sel == ax)
    newOri = DicomVolume::Orientation::Axial;
  else if (sel == sag)
    newOri = DicomVolume::Orientation::Sagittal;
  else if (sel == cor)
    newOri = DicomVolume::Orientation::Coronal;
  else
    return;

  if (newOri == m_viewOrientations[viewIndex]) {
    updateOrientationButtonTexts();
    return; // 変更なし
  }

  m_viewOrientations[viewIndex] = newOri;

  int count = sliceCountForOrientation(m_viewOrientations[viewIndex]);
  m_sliceSliders[viewIndex]->setRange(0, count > 0 ? count - 1 : 0);
  int mid = count > 0 ? count / 2 : 0;
  m_currentIndices[viewIndex] = mid;
  m_sliceSliders[viewIndex]->setValue(mid);
  loadVolumeSlice(viewIndex, mid);
  updateSliceLabels();
  updateOrientationButtonTexts();
}

void DicomViewer::updateImageSeriesButtons() {
  const int seriesCount = m_imageSeriesDirs.size();
  int activeIndex = m_activeImageSeriesIndex;
  if (seriesCount <= 0) {
    activeIndex = 0;
  } else {
    activeIndex = std::clamp(activeIndex, 0, seriesCount - 1);
  }

  QString label = tr("Image%1").arg(activeIndex + 1);
  QString modality;
  if (activeIndex >= 0 && activeIndex < m_imageSeriesModalities.size())
    modality = m_imageSeriesModalities.at(activeIndex);
  QString tooltip = modality.isEmpty() ? label
                                       : QStringLiteral("%1 (%2)").arg(label, modality);

  bool hasImages = isVolumeLoaded() || !m_dicomFiles.isEmpty();
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (!m_imageSeriesButtons[i])
      continue;
    if (m_fusionViewActive && i == 1) {
      m_imageSeriesButtons[i]->setVisible(true);
      m_imageSeriesButtons[i]->setEnabled(false);
      m_imageSeriesButtons[i]->setText(tr("Image2"));
      m_imageSeriesButtons[i]->setToolTip(tr("Fusionビュー (CT+MRI)"));
      m_imageSeriesButtons[i]->adjustSize();
      continue;
    }
    m_imageSeriesButtons[i]->setEnabled(true);
    bool visible = hasImages && seriesCount > 1;
    m_imageSeriesButtons[i]->setVisible(visible);
    m_imageSeriesButtons[i]->setText(label);
    m_imageSeriesButtons[i]->setToolTip(visible ? tooltip : QString());
    m_imageSeriesButtons[i]->adjustSize();
  }
}

void DicomViewer::showImageSeriesMenu(int viewIndex) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return;
  if (m_imageSeriesDirs.size() <= 1)
    return;

  QMenu menu;
  for (int i = 0; i < m_imageSeriesDirs.size(); ++i) {
    QString label = tr("Image%1").arg(i + 1);
    QString modality;
    if (i >= 0 && i < m_imageSeriesModalities.size())
      modality = m_imageSeriesModalities.at(i);
    QString text = modality.isEmpty()
                        ? label
                        : QStringLiteral("%1 (%2)").arg(label, modality);
    QAction *action = menu.addAction(text);
    action->setData(i);
    if (i == m_activeImageSeriesIndex) {
      QFont f = action->font();
      f.setBold(true);
      action->setFont(f);
    }
  }

  QAction *sel = menu.exec(m_imageSeriesButtons[viewIndex]->mapToGlobal(
      QPoint(0, m_imageSeriesButtons[viewIndex]->height())));
  if (!sel)
    return;

  bool ok = false;
  int index = sel->data().toInt(&ok);
  if (!ok)
    return;
  switchToImageSeries(index);
}

bool DicomViewer::switchToImageSeries(int index) {
  if (index < 0 || index >= m_imageSeriesDirs.size())
    return false;
  if (index == m_activeImageSeriesIndex)
    return true;
  if (m_activeImageSeriesIndex >= 0 &&
      m_activeImageSeriesIndex <
          static_cast<int>(m_seriesVolumeCache.size()) &&
      isVolumeLoaded()) {
    m_seriesVolumeCache[m_activeImageSeriesIndex].volume = m_volume;
    m_seriesVolumeCache[m_activeImageSeriesIndex].prepared = true;
  }

  if (!ensureImageSeriesVolume(index))
    return false;

  if (index < 0 || index >= static_cast<int>(m_seriesVolumeCache.size()))
    return false;

  m_volume = m_seriesVolumeCache[index].volume;
  invalidateStructureSurfaceCache();
  m_activeImageSeriesIndex = index;

  double window = m_seriesWindowValues.value(
      index, m_windowSlider ? m_windowSlider->value() : 256.0);
  double level = m_seriesLevelValues.value(
      index, m_levelSlider ? m_levelSlider->value() : 128.0);
  setWindowLevel(window, level);

  if (index >= 0 && index < m_imageSeriesDirs.size()) {
    QFileInfo info(m_imageSeriesDirs.at(index));
    m_ctFilename = info.fileName();
  }

  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_isDVHView[i] || m_is3DView[i] || m_isProfileView[i])
      continue;
    int count = sliceCountForOrientation(m_viewOrientations[i]);
    if (count <= 0) {
      m_sliceSliders[i]->blockSignals(true);
      m_sliceSliders[i]->setRange(0, 0);
      m_sliceSliders[i]->setValue(0);
      m_sliceSliders[i]->setEnabled(false);
      m_sliceSliders[i]->blockSignals(false);
      continue;
    }
    int clamped = std::clamp(m_currentIndices[i], 0, count - 1);
    m_sliceSliders[i]->blockSignals(true);
    m_sliceSliders[i]->setRange(0, count - 1);
    m_sliceSliders[i]->setValue(clamped);
    m_sliceSliders[i]->setEnabled(true);
    m_sliceSliders[i]->blockSignals(false);
    loadVolumeSlice(i, clamped);
  }

  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_is3DView[i])
      update3DView(i);
  }

  updateSliceLabels();
  updateImageInfo();
  updateInfoOverlays();
  updateImageSeriesButtons();
  updateColorBars();
  updateDoseShiftLabels();
  updateSliderPosition();

  return true;
}

void DicomViewer::initializeSeriesWindowLevel(int index,
                                              const QString &directory) {
  if (index < 0 || index >= m_seriesWindowLevelInitialized.size())
    return;
  if (m_seriesWindowLevelInitialized[index])
    return;

  double window = m_seriesWindowValues.value(index, 256.0);
  double level = m_seriesLevelValues.value(index, 128.0);

  if (!directory.isEmpty()) {
    QDir seriesDir(directory);
    QFileInfoList dicomFiles =
        seriesDir.entryInfoList(QDir::Files, QDir::Name | QDir::IgnoreCase);
    DicomReader reader;
    for (const QFileInfo &fi : dicomFiles) {
      if (reader.loadDicomFile(fi.absoluteFilePath())) {
        reader.getWindowLevel(window, level);
        break;
      }
    }
  }

  if (index >= 0 && index < m_seriesWindowValues.size())
    m_seriesWindowValues[index] = window;
  if (index >= 0 && index < m_seriesLevelValues.size())
    m_seriesLevelValues[index] = level;
  m_seriesWindowLevelInitialized[index] = true;
}

bool DicomViewer::ensureImageSeriesVolume(int index) {
  if (m_imageSeriesDirs.isEmpty())
    return false;
  if (index < 0 || index >= m_imageSeriesDirs.size())
    return false;

  if (m_seriesVolumeCache.size() != m_imageSeriesDirs.size())
    m_seriesVolumeCache.resize(m_imageSeriesDirs.size());

  if (m_primaryImageSeriesIndex < 0 ||
      m_primaryImageSeriesIndex >=
          static_cast<int>(m_seriesVolumeCache.size())) {
    if (!m_seriesVolumeCache.empty())
      m_primaryImageSeriesIndex = 0;
  }

  ImageSeriesCacheEntry &entry = m_seriesVolumeCache[index];
  if (entry.prepared)
    return true;

  if (index == m_activeImageSeriesIndex && isVolumeLoaded()) {
    entry.volume = m_volume;
    entry.prepared = true;
    return true;
  }

  const QString dir = m_imageSeriesDirs.value(index);
  if (dir.isEmpty())
    return false;

  DicomVolume loadedVolume;
  bool loadedFromFusion = false;
  FusionSeriesMetadata meta;
  if (!loadedVolume.loadFromDirectory(dir)) {
    if (!loadFusionVolumeFromDirectory(dir, loadedVolume, &meta))
      return false;
    loadedFromFusion = true;
  }

  if (loadedFromFusion) {
    if (index >= 0 && index < m_seriesWindowValues.size() &&
        std::isfinite(meta.window))
      m_seriesWindowValues[index] = meta.window;
    if (index >= 0 && index < m_seriesLevelValues.size() &&
        std::isfinite(meta.level))
      m_seriesLevelValues[index] = meta.level;
    if (index >= 0 && index < m_seriesWindowLevelInitialized.size())
      m_seriesWindowLevelInitialized[index] = true;
  } else {
    initializeSeriesWindowLevel(index, dir);
  }

  if (index == m_primaryImageSeriesIndex || m_imageSeriesDirs.size() <= 1) {
    entry.volume = std::move(loadedVolume);
    entry.prepared = true;
    return true;
  }

  if (m_primaryImageSeriesIndex < 0 ||
      m_primaryImageSeriesIndex >= static_cast<int>(m_seriesVolumeCache.size()))
    return false;

  if (loadedFromFusion) {
    entry.volume = std::move(loadedVolume);
    entry.prepared = true;
    return true;
  }

  if (!ensureImageSeriesVolume(m_primaryImageSeriesIndex))
    return false;

  const ImageSeriesCacheEntry &referenceEntry =
      m_seriesVolumeCache[m_primaryImageSeriesIndex];
  if (!referenceEntry.prepared)
    return false;

  const int sliceCount = referenceEntry.volume.depth();
  std::unique_ptr<QProgressDialog> progressDialog;
  if (sliceCount > 0) {
    auto dialog = std::make_unique<QProgressDialog>(
        tr("Image%1 を座標合わせ中...").arg(index + 1), QString(), 0,
        sliceCount, this);
    dialog->setWindowModality(Qt::ApplicationModal);
    dialog->setCancelButton(nullptr);
    dialog->setMinimumDuration(0);
    dialog->setValue(0);
    progressDialog = std::move(dialog);
  }
  QProgressDialog *progressPtr = progressDialog.get();
  std::function<void(int, int)> progressCallback;
  if (progressPtr) {
    progressCallback = [progressPtr](int done, int total) {
      QMetaObject::invokeMethod(
          progressPtr,
          [progressPtr, done, total]() {
            if (!progressPtr)
              return;
            if (progressPtr->maximum() != total)
              progressPtr->setMaximum(total);
            progressPtr->setValue(done);
          },
          Qt::QueuedConnection);
    };
  }

  cv::Mat resampled = resampleVolumeToReference(referenceEntry.volume,
                                                loadedVolume, progressCallback);
  if (progressPtr) {
    progressPtr->setValue(progressPtr->maximum());
    progressPtr->close();
    QCoreApplication::processEvents();
  }
  if (resampled.empty())
    return false;

  DicomVolume prepared;
  if (!prepared.createFromReference(referenceEntry.volume, resampled))
    return false;

  entry.volume = std::move(prepared);
  entry.prepared = true;
  return true;
}

void DicomViewer::setViewToImage(int viewIndex) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return;
  m_isDVHView[viewIndex] = false;
  m_is3DView[viewIndex] = false;
  m_isProfileView[viewIndex] = false;
  m_viewStacks[viewIndex]->setCurrentWidget(m_imagePanels[viewIndex]);
  if (m_sliceIndexLabels[viewIndex])
    m_sliceIndexLabels[viewIndex]->show();
  if (m_infoOverlays[viewIndex])
    m_infoOverlays[viewIndex]->show();
  if (m_coordLabels[viewIndex])
    m_coordLabels[viewIndex]->show();
  if (m_cursorDoseLabels[viewIndex])
    m_cursorDoseLabels[viewIndex]->hide();
  m_imageWidgets[viewIndex]->clearCursorCross();
  updateDoseShiftLabels();
  updateInteractionButtonVisibility(viewIndex);
}

void DicomViewer::setViewTo3D(int viewIndex) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return;
  m_is3DView[viewIndex] = true;
  m_isDVHView[viewIndex] = false;
  m_isProfileView[viewIndex] = false;
  m_viewStacks[viewIndex]->setCurrentWidget(m_3dWidgets[viewIndex]);
  if (m_sliceIndexLabels[viewIndex])
    m_sliceIndexLabels[viewIndex]->hide();
  if (m_infoOverlays[viewIndex])
    m_infoOverlays[viewIndex]->hide();
  if (m_coordLabels[viewIndex])
    m_coordLabels[viewIndex]->hide();
  if (m_cursorDoseLabels[viewIndex])
    m_cursorDoseLabels[viewIndex]->hide();
  m_imageWidgets[viewIndex]->clearCursorCross();
  // ボリューム未ロード時に空データを渡さない
  if (isVolumeLoaded()) {
    update3DView(viewIndex);
  }
  // Set initial visibility state for images and lines in 3D view
  if (m_3dWidgets[viewIndex]) {
    m_3dWidgets[viewIndex]->setShowImages(m_show3DImages[viewIndex]);
    m_3dWidgets[viewIndex]->setShowLines(m_show3DLines[viewIndex]);
    m_3dWidgets[viewIndex]->setShowSurfaces(m_show3DSurfaces[viewIndex]);
  }
  updateDoseShiftLabels();
  updateInteractionButtonVisibility(viewIndex);
}

void DicomViewer::setViewToDVH(int viewIndex) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return;
  m_isDVHView[viewIndex] = true;
  m_is3DView[viewIndex] = false;
  m_isProfileView[viewIndex] = false;
  m_viewStacks[viewIndex]->setCurrentWidget(m_dvhWidgets[viewIndex]);
  if (!m_dvhData.empty()) {
    m_dvhWidgets[viewIndex]->setDVHData(m_dvhData);
    m_dvhWidgets[viewIndex]->setPatientInfo(patientInfoText());
    m_dvhWidgets[viewIndex]->setPrescriptionDose(m_doseReference);
  }
  if (m_sliceIndexLabels[viewIndex])
    m_sliceIndexLabels[viewIndex]->hide();
  if (m_infoOverlays[viewIndex])
    m_infoOverlays[viewIndex]->hide();
  if (m_coordLabels[viewIndex])
    m_coordLabels[viewIndex]->hide();
  if (m_cursorDoseLabels[viewIndex])
    m_cursorDoseLabels[viewIndex]->hide();
  m_imageWidgets[viewIndex]->clearCursorCross();
  updateDoseShiftLabels();
  updateInteractionButtonVisibility(viewIndex);
}

void DicomViewer::setViewToProfile(int viewIndex) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return;
  m_isProfileView[viewIndex] = true;
  m_isDVHView[viewIndex] = false;
  m_is3DView[viewIndex] = false;
  m_viewStacks[viewIndex]->setCurrentWidget(m_profileWidgets[viewIndex]);
  if (m_sliceIndexLabels[viewIndex])
    m_sliceIndexLabels[viewIndex]->hide();
  if (m_infoOverlays[viewIndex])
    m_infoOverlays[viewIndex]->hide();
  if (m_coordLabels[viewIndex])
    m_coordLabels[viewIndex]->hide();
  if (m_cursorDoseLabels[viewIndex])
    m_cursorDoseLabels[viewIndex]->hide();
  m_imageWidgets[viewIndex]->clearCursorCross();
  updateDoseShiftLabels();
  updateInteractionButtonVisibility(viewIndex);
}

bool DicomViewer::switchViewContentFromString(int viewIndex,
                                              const QString &mode) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return false;
  if (!isViewIndexVisible(viewIndex))
    return false;
  const QString key = mode.trimmed().toLower();
  if (key.isEmpty())
    return false;

  auto finalizeUpdates = [this]() {
    updateOrientationButtonTexts();
    updateSliceLabels();
    updateSliderPosition();
    updateImage();
    updateColorBars();
  };

  if (key == QStringLiteral("dvh")) {
    setViewToDVH(viewIndex);
    finalizeUpdates();
    return true;
  }
  if (key == QStringLiteral("3d") || key == QStringLiteral("volume")) {
    setViewTo3D(viewIndex);
    finalizeUpdates();
    return true;
  }
  if (key == QStringLiteral("profile") || key == QStringLiteral("dpsd")) {
    setViewToProfile(viewIndex);
    finalizeUpdates();
    return true;
  }

  if (m_isDVHView[viewIndex] || m_is3DView[viewIndex] ||
      m_isProfileView[viewIndex]) {
    setViewToImage(viewIndex);
  }

  DicomVolume::Orientation desired = m_viewOrientations[viewIndex];
  if (key == QStringLiteral("axial") || key == QStringLiteral("ax")) {
    desired = DicomVolume::Orientation::Axial;
  } else if (key == QStringLiteral("sagittal") || key == QStringLiteral("sag")) {
    desired = DicomVolume::Orientation::Sagittal;
  } else if (key == QStringLiteral("coronal") || key == QStringLiteral("cor")) {
    desired = DicomVolume::Orientation::Coronal;
  } else {
    return false;
  }

  if (m_viewOrientations[viewIndex] != desired) {
    m_viewOrientations[viewIndex] = desired;
    int count = sliceCountForOrientation(desired);
    m_sliceSliders[viewIndex]->setRange(0, count > 0 ? count - 1 : 0);
    int mid = count > 0 ? count / 2 : 0;
    m_currentIndices[viewIndex] = mid;
    m_sliceSliders[viewIndex]->setValue(mid);
    loadVolumeSlice(viewIndex, mid);
  } else {
    loadVolumeSlice(viewIndex, m_currentIndices[viewIndex]);
  }

  finalizeUpdates();
  return true;
}

int DicomViewer::visibleViewCount() const {
  switch (m_viewMode) {
  case ViewMode::Single:
    return 1;
  case ViewMode::Dual:
    return 2;
  case ViewMode::Quad:
    return 4;
  case ViewMode::Five:
  default:
    return VIEW_COUNT;
  }
}

bool DicomViewer::isViewIndexVisible(int index) const {
  return index >= 0 && index < visibleViewCount();
}

int DicomViewer::clampToVisibleViewIndex(int index) const {
  const int count = visibleViewCount();
  if (count <= 0)
    return 0;
  return std::clamp(index, 0, count - 1);
}

int DicomViewer::findVisibleDvhView() const {
  const int count = visibleViewCount();
  for (int i = 0; i < count; ++i) {
    if (m_isDVHView[i])
      return i;
  }
  return -1;
}

int DicomViewer::activeOrDefaultViewIndex(int fallback) const {
  if (isViewIndexVisible(m_activeViewIndex))
    return m_activeViewIndex;
  int candidate = fallback;
  if (!isViewIndexVisible(candidate))
    candidate = clampToVisibleViewIndex(candidate);
  if (!isViewIndexVisible(candidate))
    candidate = 0;
  return candidate;
}

void DicomViewer::updateSyncedScale() {
  if (!m_syncScale)
    return;

  // アクティブビューのズーム係数を他のビューに適用
  syncZoomToAllViews(m_zoomFactor);
}

void DicomViewer::syncZoomToAllViews(double zoomFactor) {
  if (!m_syncScale)
    return;

  int count = 1;
  if (m_viewMode == ViewMode::Dual)
    count = 2;
  else if (m_viewMode == ViewMode::Quad)
    count = 4;
  else if (m_viewMode == ViewMode::Five)
    count = VIEW_COUNT;

  for (int i = 0; i < count; ++i) {
    if (m_is3DView[i])
      m_3dWidgets[i]->setZoom(qMin(zoomFactor * ZOOM_3D_RATIO, MAX_ZOOM_3D));
    else
      m_imageWidgets[i]->setZoom(qMin(zoomFactor, MAX_ZOOM));
  }
}

void DicomViewer::onShowDVH() {
  qDebug() << "=== DVH Show Request ===";

  int index = m_activeViewIndex;
  qDebug() << QString("Active view index: %1").arg(index);

  if (!isViewIndexVisible(index)) {
    int fallback = findVisibleDvhView();
    if (fallback < 0)
      fallback = clampToVisibleViewIndex(0);
    if (!isViewIndexVisible(fallback)) {
      qDebug() << "No visible view available for DVH";
      return;
    }
    index = fallback;
    m_activeViewIndex = index;
    qDebug() << QString("Adjusted DVH view index to %1 due to layout")
                    .arg(index);
  }

  if (index < 0 || index >= VIEW_COUNT) {
    qDebug() << "Invalid view index";
    return;
  }

  // DVHデータが空の場合でもここでは計算せず、
  // チェックボックス操作時に計算を行う
  if (m_dvhData.empty()) {
    qDebug() << "DVH data will be calculated when a ROI is selected";
  }

  // DVHビューに切り替え
  qDebug() << QString("Switching view %1 to DVH mode").arg(index);

  m_isDVHView[index] = true;
  m_viewStacks[index]->setCurrentWidget(m_dvhWidgets[index]);

  // UI要素を非表示
  if (m_sliceIndexLabels[index])
    m_sliceIndexLabels[index]->hide();
  if (m_infoOverlays[index])
    m_infoOverlays[index]->hide();
  if (m_coordLabels[index])
    m_coordLabels[index]->hide();
  if (m_cursorDoseLabels[index])
    m_cursorDoseLabels[index]->hide();
  m_imageWidgets[index]->clearCursorCross();

  // DVHデータを設定
  qDebug() << QString("Setting DVH data to widget %1 (%2 ROIs)")
                  .arg(index)
                  .arg(m_dvhData.size());
  m_dvhWidgets[index]->setDVHData(m_dvhData);
  m_dvhWidgets[index]->setPatientInfo(patientInfoText());
  m_dvhWidgets[index]->setPrescriptionDose(m_doseReference);

  // ボタンテキストを更新
  updateOrientationButtonTexts();
  updateSliderPosition();
  updateImage();

  qDebug() << QString("DVH view %1 setup complete").arg(index);
}

void DicomViewer::onDvhCalculationRequested(const QString &roiName) {
  qDebug() << "=== DVH single ROI calculation ===";
  if (!isVolumeLoaded() || !m_doseLoaded || !m_rtstructLoaded ||
      !m_resampledDose.isResampled()) {
    QMessageBox::warning(this, tr("DVH"),
                         tr("DVH計算に必要なデータが不足しています"));
    return;
  }

  int roiIndex = -1;
  for (int i = 0; i < m_rtstruct.roiCount(); ++i) {
    if (m_rtstruct.roiName(i) == roiName) {
      roiIndex = i;
      break;
    }
  }
  if (roiIndex < 0) {
    qWarning() << "ROI not found:" << roiName;
    return;
  }

  if (m_dvhData.size() < static_cast<size_t>(m_rtstruct.roiCount()))
    m_dvhData.resize(m_rtstruct.roiCount());

  // 既に計算済みなら可視化のみ更新
  if (!m_dvhData[roiIndex].points.empty()) {
    m_dvhData[roiIndex].isVisible = true;
    for (int i = 0; i < VIEW_COUNT; ++i) {
      if (m_dvhWidgets[i]) {
        m_dvhWidgets[i]->setDVHData(m_dvhData);
        m_dvhWidgets[i]->setPatientInfo(patientInfoText());
        m_dvhWidgets[i]->setPrescriptionDose(m_doseReference);
      }
    }
    return;
  }

  if (m_dvhWatchers.contains(roiIndex)) {
    qDebug() << "DVH calculation already in progress for" << roiName;
    return;
  }

  // ROIのサイズ推定と確認ダイアログ
  QVector3D bbMin, bbMax;
  int voxelEstimate = 0;
  if (m_rtstruct.roiBoundingBox(roiIndex, bbMin, bbMax)) {
    voxelEstimate = DVHCalculator::estimateVoxelCount(m_volume, bbMin, bbMax);
  }
  const int LARGE_ROI_THRESHOLD = 2000000; // おおよその目安
  if (voxelEstimate > LARGE_ROI_THRESHOLD) {
    QMessageBox::StandardButton res = QMessageBox::question(
        this, tr("DVH"),
        tr("ROI '%1' の計算には時間がかかる可能性があります (推定 %2 "
           "ボクセル)。\n計算を続行しますか?")
            .arg(roiName)
            .arg(voxelEstimate),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
    if (res != QMessageBox::Yes) {
      for (int i = 0; i < VIEW_COUNT; ++i) {
        if (m_dvhWidgets[i])
          m_dvhWidgets[i]->setROIChecked(roiName, false);
      }
      return;
    }
  }

  double binSize = m_resampledDose.maxDose() / 200.0;
  auto watcher = new QFutureWatcher<DVHCalculator::DVHData>(this);
  m_dvhWatchers.insert(roiIndex, watcher);

  auto progressCallback = [this](int processed, int total) {
    for (int i = 0; i < VIEW_COUNT; ++i) {
      if (m_dvhWidgets[i]) {
        DVHWindow *w = m_dvhWidgets[i];
        QMetaObject::invokeMethod(
            w,
            [w, processed, total]() {
              w->setCalculationProgress(processed, total);
            },
            Qt::QueuedConnection);
      }
    }
  };

  connect(watcher, &QFutureWatcher<DVHCalculator::DVHData>::finished, this,
          [this, watcher, roiIndex]() {
            watcher->deleteLater();
            auto data = watcher->result();
            data.isVisible = true;
            m_dvhData[roiIndex] = std::move(data);
            m_dvhWatchers.remove(roiIndex);
            for (int i = 0; i < VIEW_COUNT; ++i) {
              if (m_dvhWidgets[i]) {
                m_dvhWidgets[i]->setDVHData(m_dvhData);
                m_dvhWidgets[i]->setPatientInfo(patientInfoText());
                m_dvhWidgets[i]->setPrescriptionDose(m_doseReference);
                m_dvhWidgets[i]->setCalculationProgress(0, 0);
              }
            }
          });

  QFuture<DVHCalculator::DVHData> future =
      QtConcurrent::run([this, roiIndex, binSize, progressCallback]() {
        return DVHCalculator::calculateSingleROI(m_volume, m_resampledDose,
                                                 m_rtstruct, roiIndex, binSize,
                                                 nullptr, progressCallback);
      });
  watcher->setFuture(future);
}

// CalcMax（Gy）変更: 現在計算済みの全ROIを再計算（201 bins）
void DicomViewer::onDvhCalcMaxChanged(double calcMaxGy) {
  if (!isVolumeLoaded() || !m_doseLoaded || !m_rtstructLoaded ||
      !m_resampledDose.isResampled()) {
    return;
  }

  // Auto mode: use resampled dose max as cap
  double capGy =
      (calcMaxGy > 0.0) ? calcMaxGy : std::max(1e-6, m_resampledDose.maxDose());
  double binSize = capGy / 200.0; // 201 bins (0..200)

  // スレッドで各ROIを再計算（既にデータがあるもののみ）
  for (int r = 0; r < m_rtstruct.roiCount(); ++r) {
    if (r < 0 || static_cast<size_t>(r) >= m_dvhData.size())
      continue;
    if (m_dvhData[r].points.empty())
      continue; // 未計算はスキップ
    if (m_dvhWatchers.contains(r))
      continue; // 進行中はスキップ

    auto watcher = new QFutureWatcher<DVHCalculator::DVHData>(this);
    m_dvhWatchers.insert(r, watcher);

    connect(watcher, &QFutureWatcher<DVHCalculator::DVHData>::finished, this,
            [this, watcher, r]() {
              watcher->deleteLater();
              auto data = watcher->result();
              // 既存の可視性を維持
              data.isVisible = m_dvhData[r].isVisible;
              m_dvhData[r] = std::move(data);
              m_dvhWatchers.remove(r);
              for (int i = 0; i < VIEW_COUNT; ++i) {
                if (m_dvhWidgets[i]) {
                  m_dvhWidgets[i]->setDVHData(m_dvhData);
                  m_dvhWidgets[i]->setPatientInfo(patientInfoText());
                  m_dvhWidgets[i]->setPrescriptionDose(m_doseReference);
                  m_dvhWidgets[i]->setCalculationProgress(0, 0);
                }
              }
            });

    QFuture<DVHCalculator::DVHData> future =
        QtConcurrent::run([this, r, binSize, capGy]() {
          return DVHCalculator::calculateSingleROI(m_volume, m_resampledDose,
                                                   m_rtstruct, r, binSize,
                                                   nullptr, {}, capGy);
        });
    watcher->setFuture(future);
  }
}

void DicomViewer::onDvhVisibilityChanged(int roiIndex, bool visible) {
  qDebug()
      << QString("DicomViewer::onDvhVisibilityChanged: roiIndex=%1, visible=%2")
             .arg(roiIndex)
             .arg(visible);

  // 範囲チェック
  if (roiIndex < 0) {
    qWarning() << QString("Invalid ROI index: %1").arg(roiIndex);
    return;
  }

  // DVHデータの範囲チェック
  if (static_cast<size_t>(roiIndex) >= m_dvhData.size()) {
    qWarning() << QString("ROI index %1 out of range for DVH data (max: %2)")
                      .arg(roiIndex)
                      .arg(static_cast<int>(m_dvhData.size()) - 1);
    return;
  }

  try {
    // DVHデータの可視性を更新
    m_dvhData[roiIndex].isVisible = visible;

    // 変更を他のDVHウィンドウにも反映
    if (auto source = qobject_cast<DVHWindow *>(sender())) {
      for (int i = 0; i < VIEW_COUNT; ++i) {
        if (m_dvhWidgets[i] && m_dvhWidgets[i] != source) {
          QSignalBlocker blocker(m_dvhWidgets[i]);
          m_dvhWidgets[i]->updateVisibility(roiIndex, visible);
        }
      }
    }

    qDebug() << QString("DVH visibility updated for ROI %1").arg(roiIndex);

  } catch (const std::exception &e) {
    qCritical() << QString("Error in onDvhVisibilityChanged for ROI %1: %2")
                       .arg(roiIndex)
                       .arg(e.what());
  } catch (...) {
    qCritical() << QString("Unknown error in onDvhVisibilityChanged for ROI %1")
                       .arg(roiIndex);
  }
}

void DicomViewer::onProfileLineSelection(int viewIndex) {
  m_profileRequester = viewIndex;
  m_selectingProfileLine = true;
  m_profileLineHasStart = false;
  if (m_profileLineVisible) {
    m_profileLineVisible = false;
    updateImage(m_profileLineView);
  }
}

void DicomViewer::onProfileLineSaveRequested(int viewIndex, int slotIndex) {
  if (slotIndex < 0 || slotIndex >= 3)
    return;
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return;
  auto hasValidPatientPoint = [](const QVector3D &p) {
    return !std::isnan(p.x()) && !std::isnan(p.y()) && !std::isnan(p.z());
  };

  const bool hasLineGeometry =
      m_profileLine.points.size() >= 2 &&
      (m_profileLine.points[0] != m_profileLine.points[1]);

  if (!hasLineGeometry || !hasValidPatientPoint(m_profileStartPatient) ||
      !hasValidPatientPoint(m_profileEndPatient)) {
    QMessageBox::information(this, tr("Dose Profile"),
                             tr("No dose profile line is active to save."));
    return;
  }

  SavedProfileLine &slot = m_savedProfileLines[viewIndex][slotIndex];
  slot.valid = true;
  slot.startPatient = m_profileStartPatient;
  slot.endPatient = m_profileEndPatient;

  writeProfileLineSlot(viewIndex, slotIndex);
  updateProfileSlotButtons(viewIndex);
}

void DicomViewer::onProfileLineLoadRequested(int viewIndex, int slotIndex) {
  if (slotIndex < 0 || slotIndex >= 3)
    return;
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return;
  const bool wasVisible = m_profileLineVisible;
  const int previousView = m_profileLineView;
  const SavedProfileLine &slot = m_savedProfileLines[viewIndex][slotIndex];
  if (!slot.valid) {
    QMessageBox::information(this, tr("Dose Profile"),
                             tr("No saved line in the selected slot."));
    return;
  }

  QPointF startPlane = planeCoordinateFromPatient(viewIndex, slot.startPatient);
  QPointF endPlane = planeCoordinateFromPatient(viewIndex, slot.endPatient);
  if (std::isnan(startPlane.x()) || std::isnan(startPlane.y()) ||
      std::isnan(endPlane.x()) || std::isnan(endPlane.y())) {
    QMessageBox::warning(this, tr("Dose Profile"),
                         tr("Saved line cannot be displayed in this view."));
    return;
  }

  m_profileLineVisible = true;
  m_profileLineView = viewIndex;
  m_profileLine.color = Qt::yellow;
  m_profileLine.points.resize(2);
  m_profileLine.points[0] = startPlane;
  m_profileLine.points[1] = endPlane;
  m_profileStartPatient = slot.startPatient;
  m_profileEndPatient = slot.endPatient;
  m_selectingProfileLine = false;
  m_profileLineHasStart = false;
  m_dragProfileStart = false;
  m_dragProfileEnd = false;
  m_profileRequester = viewIndex;

  computeDoseProfile();
  if (wasVisible && previousView != viewIndex && previousView >= 0 &&
      previousView < VIEW_COUNT)
    updateImage(previousView);
  updateImage(viewIndex);
}

void DicomViewer::computeDoseProfile() {
  if (!m_resampledDose.isResampled() || m_profileRequester < 0)
    return;

  double length = (m_profileEndPatient - m_profileStartPatient).length();
  if (length <= 0.0)
    return;

  int samples = std::max(2, static_cast<int>(length));
  QVector<double> positions;
  QVector<double> doses;
  positions.reserve(samples);
  doses.reserve(samples);

  struct RoiTrack {
    QString name;
    bool inside{false};
    double segStart{0.0};
    QVector<QPair<double, double>> segments; // start,end
    double maxDose{-std::numeric_limits<double>::infinity()};
    double minDose{std::numeric_limits<double>::infinity()};
    QColor color;
    int index{0};
  };
  QVector<RoiTrack> roiInfos;
  if (m_rtstructLoaded) {
    int roiCount = m_rtstruct.roiCount();
    for (int r = 0; r < roiCount; ++r) {
      if (!m_rtstruct.isROIVisible(r))
        continue;
      RoiTrack info;
      info.name = m_rtstruct.roiName(r);
      info.color = QColor::fromHsv((r * 40) % 360, 255, 255, 255);
      info.index = r;
      roiInfos.append(info);
    }
  }

  for (int i = 0; i < samples; ++i) {
    double t = static_cast<double>(i) / (samples - 1);
    QVector3D p = m_profileStartPatient +
                  t * (m_profileEndPatient - m_profileStartPatient);
    QVector3D vox = m_volume.patientToVoxelContinuous(p);
    float dose = sampleResampledDose(vox);
    double pos = t * length;
    positions.append(pos);
    doses.append(dose);

    for (auto &info : roiInfos) {
      bool inside = m_rtstruct.isPointInsideROI(p, info.index);
      if (inside) {
        if (!info.inside) {
          info.inside = true;
          info.segStart = pos;
        }
        info.maxDose = std::max(info.maxDose, static_cast<double>(dose));
        info.minDose = std::min(info.minDose, static_cast<double>(dose));
      } else if (info.inside) {
        info.segments.append(qMakePair(info.segStart, pos));
        info.inside = false;
      }
    }
  }

  for (auto &info : roiInfos) {
    if (info.inside) {
      info.segments.append(qMakePair(info.segStart, length));
      info.inside = false;
    }
  }

  double minDose = *std::min_element(doses.begin(), doses.end());
  double maxDose = *std::max_element(doses.begin(), doses.end());

  QVector<DoseProfileWindow::Segment> segs;
  for (const auto &info : roiInfos) {
    if (info.segments.isEmpty())
      continue;
    for (const auto &se : info.segments) {
      DoseProfileWindow::Segment s;
      s.startMm = se.first;
      s.endMm = se.second;
      s.maxDoseGy = info.maxDose;
      s.minDoseGy = info.minDose;
      s.color = info.color;
      s.name = info.name;
      segs.append(s);
    }
  }

  QVector<DoseProfileWindow::SamplePoint> samplePoints;
  if (length > 0.0) {
    const QVector3D direction =
        (m_profileEndPatient - m_profileStartPatient) / length;
    auto appendSample = [&](double posMm) {
      DoseProfileWindow::SamplePoint sample;
      sample.positionMm = posMm;
      const QVector3D patientPoint =
          m_profileStartPatient + direction * posMm;
      const QVector3D voxelPoint =
          m_volume.patientToVoxelContinuous(patientPoint);
      if (auto ct = sampleCtValue(voxelPoint))
        sample.ctHu = ct;
      if (auto dose = sampleDoseValue(voxelPoint))
        sample.doseGy = dose;
      samplePoints.append(sample);
    };
    const double epsilon = 1e-3;
    const int mmSteps = static_cast<int>(std::floor(length + epsilon));
    for (int mm = 0; mm <= mmSteps; ++mm)
      appendSample(static_cast<double>(mm));
    const double fractional = length - static_cast<double>(mmSteps);
    if (fractional > epsilon)
      appendSample(length);
  }

  if (m_profileRequester >= 0 && m_profileRequester < VIEW_COUNT &&
      m_profileWidgets[m_profileRequester]) {
    m_profileWidgets[m_profileRequester]->setStats(length, minDose, maxDose,
                                                   segs, samplePoints);
    m_profileWidgets[m_profileRequester]->setProfile(positions, doses, segs);
  }
}

void DicomViewer::loadProfileLinePresets() {
  for (auto &viewSlots : m_savedProfileLines) {
    for (auto &slot : viewSlots)
      slot = SavedProfileLine{};
  }

  QSettings settings(QStringLiteral("ShioRIS3"), QStringLiteral("ShioRIS3"));
  for (int view = 0; view < VIEW_COUNT; ++view) {
    for (int slot = 0; slot < 3; ++slot) {
      const QString base =
          QStringLiteral("doseProfile/view%1/slot%2").arg(view).arg(slot);
      const bool valid =
          settings.value(base + QStringLiteral("/valid"), false).toBool();
      if (!valid)
        continue;
      const bool hasStart = settings.contains(base + QStringLiteral("/startX")) &&
                            settings.contains(base + QStringLiteral("/startY")) &&
                            settings.contains(base + QStringLiteral("/startZ"));
      const bool hasEnd = settings.contains(base + QStringLiteral("/endX")) &&
                          settings.contains(base + QStringLiteral("/endY")) &&
                          settings.contains(base + QStringLiteral("/endZ"));
      if (!hasStart || !hasEnd)
        continue;

      SavedProfileLine entry;
      entry.valid = true;
      entry.startPatient = QVector3D(
          settings.value(base + QStringLiteral("/startX")).toDouble(),
          settings.value(base + QStringLiteral("/startY")).toDouble(),
          settings.value(base + QStringLiteral("/startZ")).toDouble());
      entry.endPatient = QVector3D(
          settings.value(base + QStringLiteral("/endX")).toDouble(),
          settings.value(base + QStringLiteral("/endY")).toDouble(),
          settings.value(base + QStringLiteral("/endZ")).toDouble());
      m_savedProfileLines[view][slot] = entry;
    }
    updateProfileSlotButtons(view);
  }
}

void DicomViewer::writeProfileLineSlot(int viewIndex, int slotIndex) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return;
  if (slotIndex < 0 || slotIndex >= 3)
    return;

  QSettings settings(QStringLiteral("ShioRIS3"), QStringLiteral("ShioRIS3"));
  const QString base = QStringLiteral("doseProfile/view%1/slot%2")
                           .arg(viewIndex)
                           .arg(slotIndex);
  const SavedProfileLine &slot = m_savedProfileLines[viewIndex][slotIndex];
  if (!slot.valid) {
    settings.remove(base);
    return;
  }

  settings.setValue(base + QStringLiteral("/valid"), true);
  settings.setValue(base + QStringLiteral("/startX"), slot.startPatient.x());
  settings.setValue(base + QStringLiteral("/startY"), slot.startPatient.y());
  settings.setValue(base + QStringLiteral("/startZ"), slot.startPatient.z());
  settings.setValue(base + QStringLiteral("/endX"), slot.endPatient.x());
  settings.setValue(base + QStringLiteral("/endY"), slot.endPatient.y());
  settings.setValue(base + QStringLiteral("/endZ"), slot.endPatient.z());
}

void DicomViewer::updateProfileSlotButtons(int viewIndex) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT)
    return;
  if (!m_profileWidgets[viewIndex])
    return;
  for (int slot = 0; slot < 3; ++slot) {
    m_profileWidgets[viewIndex]->setLineSlotAvailable(
        slot, m_savedProfileLines[viewIndex][slot].valid);
  }
}

void DicomViewer::onImageDoubleClicked(int viewIndex) {
  qDebug() << QString("Double-clicked on view %1, current mode: %2")
                  .arg(viewIndex)
                  .arg(static_cast<int>(m_viewMode));

  // viewIndexの範囲チェック
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT) {
    qWarning() << "Invalid viewIndex in onImageDoubleClicked:" << viewIndex;
    return;
  }

  if (m_viewMode == ViewMode::Single) {
    // シングルビュー時は元のマルチビューに戻る
    qDebug() << QString("Returning to previous view mode: %1")
                    .arg(static_cast<int>(m_previousViewMode));

    if (m_fullScreenViewIndex >= 0 && m_fullScreenViewIndex < VIEW_COUNT) {
      // DVHビューの場合は安全にデータを同期
      if (m_isDVHView[m_fullScreenViewIndex]) {
        try {
          // DVHデータの安全な取得と設定
          if (m_dvhWidgets[0] && !m_dvhWidgets[0]->dvhData().empty()) {
            auto safeData = m_dvhWidgets[0]->dvhData();
            int currentIdx = m_dvhWidgets[0]->currentRoiIndex();
            m_dvhData = safeData;
            if (m_dvhWidgets[m_fullScreenViewIndex]) {
              // シグナルを一時的に無効にしてデータを設定
              m_dvhWidgets[m_fullScreenViewIndex]->blockSignals(true);
              m_dvhWidgets[m_fullScreenViewIndex]->setDVHData(safeData);
              m_dvhWidgets[m_fullScreenViewIndex]->setPatientInfo(
                  patientInfoText());
              m_dvhWidgets[m_fullScreenViewIndex]->setPrescriptionDose(
                  m_dvhWidgets[0]->prescriptionDose());
              m_dvhWidgets[m_fullScreenViewIndex]->setAxisUnits(
                  m_dvhWidgets[0]->isXAxisPercent(),
                  m_dvhWidgets[0]->isYAxisCc());
              m_dvhWidgets[m_fullScreenViewIndex]->setCurrentRoiIndex(
                  currentIdx);
              // Sync CalcMax without triggering full recompute
              if (m_dvhWidgets[0]->isCalcMaxAuto()) {
                m_dvhWidgets[m_fullScreenViewIndex]->setCalcMaxAuto();
              } else {
                m_dvhWidgets[m_fullScreenViewIndex]->setCalcMaxGyNoRecalc(
                    m_dvhWidgets[0]->calcMaxGy());
              }
              m_dvhWidgets[m_fullScreenViewIndex]->blockSignals(false);
            }
          }
        } catch (const std::exception &e) {
          qCritical() << "Error in DVH data synchronization:" << e.what();
        } catch (...) {
          qCritical() << "Unknown error in DVH data synchronization";
        }
      } else {
        // 通常の画像ビューの場合
        if (m_fullScreenViewIndex < VIEW_COUNT) {
          m_originalImages[m_fullScreenViewIndex] = m_originalImages[0];
          m_currentIndices[m_fullScreenViewIndex] = m_currentIndices[0];
          m_viewOrientations[m_fullScreenViewIndex] = m_viewOrientations[0];
          m_panOffsets[m_fullScreenViewIndex] = m_panOffsets[0];

          // スライダーの安全な更新
          if (m_sliceSliders[m_fullScreenViewIndex] && m_sliceSliders[0]) {
            m_sliceSliders[m_fullScreenViewIndex]->blockSignals(true);
            m_sliceSliders[m_fullScreenViewIndex]->setRange(
                m_sliceSliders[0]->minimum(), m_sliceSliders[0]->maximum());
            m_sliceSliders[m_fullScreenViewIndex]->setValue(
                m_sliceSliders[0]->value());
            m_sliceSliders[m_fullScreenViewIndex]->setEnabled(
                m_sliceSliders[0]->isEnabled());
            m_sliceSliders[m_fullScreenViewIndex]->blockSignals(false);
          }
        }
      }
    }

    // 保存されたビュー0の状態を復元
    if (m_hasSavedView0) {
      try {
        m_originalImages[0] = m_savedImage0;
        m_currentIndices[0] = m_savedIndex0;
        m_viewOrientations[0] = m_savedOrientation0;
        m_panOffsets[0] = m_savedPanOffset0;
        m_isDVHView[0] = m_savedIsDVH0;
        m_is3DView[0] = m_savedIs3D0;

        // スライダーの安全な復元
        if (m_sliceSliders[0]) {
          m_sliceSliders[0]->blockSignals(true);
          m_sliceSliders[0]->setRange(m_savedSliderMin0, m_savedSliderMax0);
          m_sliceSliders[0]->setValue(m_savedSliderValue0);
          m_sliceSliders[0]->setEnabled(m_savedSliderEnabled0);
          m_sliceSliders[0]->blockSignals(false);
        }

        // DVHビューまたは画像ビューの復元
        if (m_savedIsDVH0) {
          if (m_viewStacks[0] && m_dvhWidgets[0]) {
            m_viewStacks[0]->setCurrentWidget(m_dvhWidgets[0]);

            // DVHデータの安全な設定
            if (!m_savedDVHData0.empty()) {
              m_dvhWidgets[0]->blockSignals(true);
              m_dvhWidgets[0]->setDVHData(m_savedDVHData0);
              m_dvhWidgets[0]->setPatientInfo(patientInfoText());
              m_dvhWidgets[0]->setPrescriptionDose(m_savedPrescriptionDose0);
              m_dvhWidgets[0]->setAxisUnits(m_savedXAxisPercent0,
                                            m_savedYAxisCc0);
              m_dvhWidgets[0]->setCurrentRoiIndex(m_savedCurrentRoiIndex0);
              m_dvhWidgets[0]->blockSignals(false);
            }

            // UI要素を隠す
            if (m_sliceIndexLabels[0])
              m_sliceIndexLabels[0]->hide();
            if (m_infoOverlays[0])
              m_infoOverlays[0]->hide();
            if (m_coordLabels[0])
              m_coordLabels[0]->hide();
            if (m_cursorDoseLabels[0])
              m_cursorDoseLabels[0]->hide();
            m_imageWidgets[0]->clearCursorCross();
          }
        } else if (m_savedIs3D0) {
          if (m_viewStacks[0] && m_3dWidgets[0]) {
            m_viewStacks[0]->setCurrentWidget(m_3dWidgets[0]);
            if (m_sliceIndexLabels[0])
              m_sliceIndexLabels[0]->hide();
            if (m_infoOverlays[0])
              m_infoOverlays[0]->hide();
            if (m_coordLabels[0])
              m_coordLabels[0]->hide();
            if (m_cursorDoseLabels[0])
              m_cursorDoseLabels[0]->hide();
            m_imageWidgets[0]->clearCursorCross();
          }
        } else {
          if (m_viewStacks[0] && m_imagePanels[0]) {
            m_viewStacks[0]->setCurrentWidget(m_imagePanels[0]);

            // UI要素を表示
            if (m_sliceIndexLabels[0])
              m_sliceIndexLabels[0]->show();
            if (m_infoOverlays[0])
              m_infoOverlays[0]->show();
            if (m_coordLabels[0])
              m_coordLabels[0]->show();
            if (m_cursorDoseLabels[0])
              m_cursorDoseLabels[0]->hide();
            m_imageWidgets[0]->clearCursorCross();
          }
        }
        m_hasSavedView0 = false;
      } catch (const std::exception &e) {
        qCritical() << "Error restoring saved view state:" << e.what();
      } catch (...) {
        qCritical() << "Unknown error restoring saved view state";
      }
    }

    // ビューモードを安全に復元
    setViewMode(m_previousViewMode);
    updateSliceLabels();
    updateOrientationButtonTexts();
    m_fullScreenViewIndex = -1;

  } else {
    // マルチビュー時はそのビューを全画面表示
    qDebug() << QString("Switching to full screen for view %1").arg(viewIndex);

    try {
      // 現在のモードを保存
      m_previousViewMode = m_viewMode;
      m_fullScreenViewIndex = viewIndex;

      // ダブルクリックされたビューの内容を1番目のビューにコピー
      if (viewIndex != 0) {
        // 現在のビュー0の状態を安全に退避
        m_savedImage0 = m_originalImages[0];
        m_savedIndex0 = m_currentIndices[0];
        m_savedOrientation0 = m_viewOrientations[0];
        m_savedPanOffset0 = m_panOffsets[0];
        m_savedIsDVH0 = m_isDVHView[0];
        m_savedIs3D0 = m_is3DView[0];

        // スライダー状態の保存
        if (m_sliceSliders[0]) {
          m_savedSliderMin0 = m_sliceSliders[0]->minimum();
          m_savedSliderMax0 = m_sliceSliders[0]->maximum();
          m_savedSliderValue0 = m_sliceSliders[0]->value();
          m_savedSliderEnabled0 = m_sliceSliders[0]->isEnabled();
        }

        // DVHデータと表示状態の保存
        if (m_isDVHView[0] && m_dvhWidgets[0]) {
          m_savedDVHData0 = m_dvhWidgets[0]->dvhData();
          m_savedPrescriptionDose0 = m_dvhWidgets[0]->prescriptionDose();
          m_savedXAxisPercent0 = m_dvhWidgets[0]->isXAxisPercent();
          m_savedYAxisCc0 = m_dvhWidgets[0]->isYAxisCc();
          m_savedCurrentRoiIndex0 = m_dvhWidgets[0]->currentRoiIndex();
        }

        m_hasSavedView0 = true;

        // ダブルクリックされたビューの内容をビュー0にコピー
        m_originalImages[0] = m_originalImages[viewIndex];
        m_currentIndices[0] = m_currentIndices[viewIndex];
        m_viewOrientations[0] = m_viewOrientations[viewIndex];
        m_panOffsets[0] = m_panOffsets[viewIndex];
        m_isDVHView[0] = m_isDVHView[viewIndex];
        m_is3DView[0] = m_is3DView[viewIndex];

        // スライダーの安全なコピー
        if (m_sliceSliders[0] && m_sliceSliders[viewIndex]) {
          m_sliceSliders[0]->blockSignals(true);
          m_sliceSliders[0]->setRange(m_sliceSliders[viewIndex]->minimum(),
                                      m_sliceSliders[viewIndex]->maximum());
          m_sliceSliders[0]->setValue(m_sliceSliders[viewIndex]->value());
          m_sliceSliders[0]->setEnabled(m_sliceSliders[viewIndex]->isEnabled());
          m_sliceSliders[0]->blockSignals(false);
        }

        // DVHビューまたは画像ビューの設定
        if (m_isDVHView[viewIndex]) {
          if (m_viewStacks[0] && m_dvhWidgets[0] && m_dvhWidgets[viewIndex]) {
            m_viewStacks[0]->setCurrentWidget(m_dvhWidgets[0]);

            // DVHデータの安全なコピーと表示設定の同期
            auto source = m_dvhWidgets[viewIndex];
            auto sourceData = source->dvhData();
            if (!sourceData.empty()) {
              m_dvhWidgets[0]->blockSignals(true);
              m_dvhWidgets[0]->setDVHData(sourceData);
              m_dvhWidgets[0]->setPatientInfo(patientInfoText());
              m_dvhWidgets[0]->setPrescriptionDose(source->prescriptionDose());
              m_dvhWidgets[0]->setAxisUnits(source->isXAxisPercent(),
                                            source->isYAxisCc());
              m_dvhWidgets[0]->setCurrentRoiIndex(source->currentRoiIndex());
              // Sync CalcMax without triggering full recompute
              if (source->isCalcMaxAuto()) {
                m_dvhWidgets[0]->setCalcMaxAuto();
              } else {
                m_dvhWidgets[0]->setCalcMaxGyNoRecalc(source->calcMaxGy());
              }
              m_dvhWidgets[0]->blockSignals(false);
            }

            // UI要素を隠す
            if (m_sliceIndexLabels[0])
              m_sliceIndexLabels[0]->hide();
            if (m_infoOverlays[0])
              m_infoOverlays[0]->hide();
            if (m_coordLabels[0])
              m_coordLabels[0]->hide();
            if (m_cursorDoseLabels[0])
              m_cursorDoseLabels[0]->hide();
            m_imageWidgets[0]->clearCursorCross();
          }
        } else if (m_is3DView[viewIndex]) {
          if (m_viewStacks[0] && m_3dWidgets[0]) {
            m_viewStacks[0]->setCurrentWidget(m_3dWidgets[0]);
            if (m_sliceIndexLabels[0])
              m_sliceIndexLabels[0]->hide();
            if (m_infoOverlays[0])
              m_infoOverlays[0]->hide();
            if (m_coordLabels[0])
              m_coordLabels[0]->hide();
            if (m_cursorDoseLabels[0])
              m_cursorDoseLabels[0]->hide();
            m_imageWidgets[0]->clearCursorCross();
          }
        } else {
          if (m_viewStacks[0] && m_imagePanels[0]) {
            m_viewStacks[0]->setCurrentWidget(m_imagePanels[0]);

            // UI要素を表示
            if (m_sliceIndexLabels[0])
              m_sliceIndexLabels[0]->show();
            if (m_infoOverlays[0])
              m_infoOverlays[0]->show();
            if (m_coordLabels[0])
              m_coordLabels[0]->show();
            if (m_cursorDoseLabels[0])
              m_cursorDoseLabels[0]->hide();
            m_imageWidgets[0]->clearCursorCross();
          }
        }
      }

      // シングルビューモードに設定
      setViewMode(ViewMode::Single);
      updateSliceLabels();
      updateOrientationButtonTexts();

    } catch (const std::exception &e) {
      qCritical() << "Error in full screen mode switch:" << e.what();
      // エラーが発生した場合は状態をリセット
      m_fullScreenViewIndex = -1;
      m_hasSavedView0 = false;
    } catch (...) {
      qCritical() << "Unknown error in full screen mode switch";
      // エラーが発生した場合は状態をリセット
      m_fullScreenViewIndex = -1;
      m_hasSavedView0 = false;
    }
  }
}

void DicomViewer::onSlicePositionToggled(bool checked) {
  m_showSlicePosition = checked;
  updateImage();
}

//=============================================================================
// ファイル: src/visualization/dicom_viewer.cpp - getCurrentVolume メソッド修正
// 修正内容: 実際のクラス構造に合わせた実装
//=============================================================================

// 6518行目周辺の getCurrentVolume() メソッドを以下に置き換え:

cv::Mat DicomViewer::getCurrentVolume() const {
    // DicomVolumeが読み込まれている場合
    if (isVolumeLoaded()) {
        qDebug() << "Getting volume from m_volume (3D volume loaded)";
        
        try {
            int width = m_volume.width();
            int height = m_volume.height(); 
            int depth = m_volume.depth();
            
            if (width <= 0 || height <= 0 || depth <= 0) {
                qWarning() << "Invalid volume dimensions:" << width << "x" << height << "x" << depth;
                return cv::Mat();
            }
            
            qDebug() << "Volume dimensions:" << width << "x" << height << "x" << depth;
            
            // OpenCVの3Dボリューム作成
            int sizes[] = {depth, height, width};
            cv::Mat volume(3, sizes, CV_16SC1);
            
            // DicomVolumeからデータをコピー
            // 注意: DicomVolumeの内部実装に依存するため、適宜調整が必要
            for (int z = 0; z < depth; ++z) {
                // 各スライスを取得してコピー
                QImage slice = m_volume.getSlice(z, DicomVolume::Orientation::Axial, 
                                               m_windowSlider ? m_windowSlider->value() : 256, 
                                               m_levelSlider ? m_levelSlider->value() : 128);
                
                if (slice.isNull()) {
                    qWarning() << "Failed to get slice" << z;
                    continue;
                }
                
                // QImageからOpenCV Matに変換（簡易実装）
                if (slice.format() != QImage::Format_Grayscale8) {
                    slice = slice.convertToFormat(QImage::Format_Grayscale8);
                }
                
                for (int y = 0; y < height && y < slice.height(); ++y) {
                    for (int x = 0; x < width && x < slice.width(); ++x) {
                        // グレースケール値を16bitに変換（CT値範囲に調整）
                        uchar gray = slice.pixelColor(x, y).red();
                        int16_t ctValue = static_cast<int16_t>(gray - 128) * 16; // 簡易的なCT値変換
                        volume.at<int16_t>(z, y, x) = ctValue;
                    }
                }
            }
            
            qDebug() << "Volume created successfully from DicomVolume";
            return volume;
            
        } catch (const std::exception &e) {
            qCritical() << "Error creating volume from DicomVolume:" << e.what();
            return cv::Mat();
        }
    }
    
    // 単一ファイルまたは複数ファイルが読み込まれている場合
    if (!m_dicomFiles.isEmpty()) {
        qDebug() << "Getting volume from m_dicomFiles:" << m_dicomFiles.size() << "files";
        
        if (m_dicomFiles.size() == 1) {
            // 単一スライスの場合
            return getSingleSliceAsMatrix();
        } else {
            // 複数スライスから3Dボリューム構築
            return buildVolumeFromFiles();
        }
    }
    
    qWarning() << "No volume or files available";
    return cv::Mat();
}

cv::Mat DicomViewer::getSingleSliceAsMatrix() const {
  if (m_dicomFiles.isEmpty())
    return cv::Mat();

  DicomReader reader;
  const QString path = m_dicomFiles.first();
  if (!reader.loadDicomFile(path))
    return cv::Mat();

  if (m_windowSlider && m_levelSlider)
    reader.setWindowLevel(m_windowSlider->value(), m_levelSlider->value());

  QImage image = reader.getImage();
  if (image.isNull())
    return cv::Mat();

  QImage gray = image.convertToFormat(QImage::Format_Grayscale8);
  cv::Mat slice(gray.height(), gray.width(), CV_16SC1);
  const int stride = gray.bytesPerLine();
  const uchar *data = gray.constBits();
  for (int y = 0; y < gray.height(); ++y) {
    const uchar *row = data + y * stride;
    for (int x = 0; x < gray.width(); ++x) {
      int16_t ctValue =
          static_cast<int16_t>(static_cast<int>(row[x]) - 128) * 16;
      slice.at<int16_t>(y, x) = ctValue;
    }
  }
  return slice;
}

cv::Mat DicomViewer::buildVolumeFromFiles() const {
  if (m_dicomFiles.isEmpty())
    return cv::Mat();

  DicomReader reader;
  if (!reader.loadDicomFile(m_dicomFiles.first()))
    return cv::Mat();

  if (m_windowSlider && m_levelSlider)
    reader.setWindowLevel(m_windowSlider->value(), m_levelSlider->value());

  QImage firstImage = reader.getImage();
  if (firstImage.isNull())
    return cv::Mat();

  QImage firstGray = firstImage.convertToFormat(QImage::Format_Grayscale8);
  const int width = firstGray.width();
  const int height = firstGray.height();
  const int depth = m_dicomFiles.size();
  if (width <= 0 || height <= 0 || depth <= 0)
    return cv::Mat();

  int sizes[] = {depth, height, width};
  cv::Mat volume(3, sizes, CV_16SC1, cv::Scalar(0));

  for (int z = 0; z < depth; ++z) {
    DicomReader sliceReader;
    if (!sliceReader.loadDicomFile(m_dicomFiles.at(z)))
      return cv::Mat();
    if (m_windowSlider && m_levelSlider)
      sliceReader.setWindowLevel(m_windowSlider->value(), m_levelSlider->value());
    QImage image = sliceReader.getImage();
    if (image.isNull())
      return cv::Mat();
    QImage gray = image.convertToFormat(QImage::Format_Grayscale8);
    if (gray.width() != width || gray.height() != height)
      return cv::Mat();
    const int stride = gray.bytesPerLine();
    const uchar *data = gray.constBits();
    for (int y = 0; y < height; ++y) {
      const uchar *row = data + y * stride;
      for (int x = 0; x < width; ++x) {
        int16_t ctValue =
            static_cast<int16_t>(static_cast<int>(row[x]) - 128) * 16;
        volume.at<int16_t>(z, y, x) = ctValue;
      }
    }
  }

  return volume;
}


void DicomViewer::onDoseIsosurfaceClicked() {
  // Check if dose is loaded
  if (!m_doseLoaded || m_doseItems.empty()) {
    QMessageBox::warning(this, tr("3D Isosurface"),
                        tr("Please load and calculate dose first."));
    return;
  }

  // Get active dose volume
  RTDoseVolume *activeDose = nullptr;
  for (const auto &item : m_doseItems) {
    if (item.widget && item.widget->isChecked()) {
      activeDose = const_cast<RTDoseVolume*>(&item.dose);
      break;
    }
  }

  if (!activeDose || activeDose->data().empty()) {
    QMessageBox::warning(this, tr("3D Isosurface"),
                        tr("No active dose volume found."));
    return;
  }

  // Create dialog for isodose level selection
  QDialog dialog(this);
  dialog.setWindowTitle(tr("Generate 3D Isosurface"));
  QVBoxLayout *layout = new QVBoxLayout(&dialog);

  // Check if reference dose is set
  if (m_doseReference <= 0.0) {
    QMessageBox::warning(this, tr("3D Isosurface"),
                        tr("Please set the reference dose (100% =) in the left panel first."));
    return;
  }

  // Use reference dose instead of max dose
  double referenceDoseGy = m_doseReference;
  QString colormapMode;
  switch (m_doseDisplayMode) {
    case DoseResampledVolume::DoseDisplayMode::Colorful:
      colormapMode = tr("Colorful (HSV)");
      break;
    case DoseResampledVolume::DoseDisplayMode::Isodose:
      colormapMode = tr("Isodose");
      break;
    case DoseResampledVolume::DoseDisplayMode::Hot:
      colormapMode = tr("Hot");
      break;
    case DoseResampledVolume::DoseDisplayMode::Simple:
      colormapMode = tr("Simple (Red)");
      break;
    default:
      colormapMode = tr("Unknown");
      break;
  }

  QLabel *infoLabel = new QLabel(tr("Reference Dose (100%%): %1 Gy | Colormap: %2")
                                  .arg(referenceDoseGy, 0, 'f', 2)
                                  .arg(colormapMode));
  layout->addWidget(infoLabel);

  // Isodose level input (percentage-based)
  QGroupBox *levelGroup = new QGroupBox(tr("Isodose Levels (%)"));
  QVBoxLayout *levelLayout = new QVBoxLayout(levelGroup);

  QLineEdit *levelsEdit = new QLineEdit("10, 30, 50, 70, 90, 100");
  levelsEdit->setPlaceholderText(tr("Enter percentage levels separated by commas"));
  levelLayout->addWidget(new QLabel(tr("Dose values (% of reference dose):")));
  levelLayout->addWidget(levelsEdit);
  layout->addWidget(levelGroup);

  // Opacity control
  QHBoxLayout *opacityLayout = new QHBoxLayout();
  opacityLayout->addWidget(new QLabel(tr("Maximum Opacity (高線量):")));
  QSlider *opacitySlider = new QSlider(Qt::Horizontal);
  opacitySlider->setRange(10, 100);
  opacitySlider->setValue(50);
  QLabel *opacityValueLabel = new QLabel("50%");
  connect(opacitySlider, &QSlider::valueChanged, [opacityValueLabel](int value) {
    opacityValueLabel->setText(QString("%1%").arg(value));
  });
  opacityLayout->addWidget(opacitySlider);
  opacityLayout->addWidget(opacityValueLabel);
  layout->addLayout(opacityLayout);

  // Add explanation label for automatic opacity scaling
  QLabel *explanationLabel = new QLabel(tr("※ 高線量ほど自動的に不透明度が高くなります（最小20%～最大上記設定値）"));
  explanationLabel->setWordWrap(true);
  explanationLabel->setStyleSheet("color: #666; font-size: 10pt;");
  layout->addWidget(explanationLabel);

  // Buttons
  QDialogButtonBox *buttonBox = new QDialogButtonBox(
      QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  connect(buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
  connect(buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
  layout->addWidget(buttonBox);

  if (dialog.exec() != QDialog::Accepted)
    return;

  // Parse isodose levels (percentage values)
  QStringList levelStrings = levelsEdit->text().split(',', Qt::SkipEmptyParts);
  QVector<double> levels;  // Dose levels in Gy
  for (const QString &str : levelStrings) {
    bool ok;
    double percentValue = str.trimmed().toDouble(&ok);
    if (ok && percentValue > 0.0 && percentValue <= 200.0) {  // Allow up to 200%
      // Convert percentage to Gy using reference dose
      double doseGy = (percentValue / 100.0) * referenceDoseGy;
      levels.append(doseGy);
    }
  }

  if (levels.isEmpty()) {
    QMessageBox::warning(this, tr("3D Isosurface"),
                        tr("No valid isodose levels entered.\nPlease enter percentage values (e.g., 10, 30, 50, 70, 90, 100)."));
    return;
  }

  float opacity = opacitySlider->value() / 100.0f;

  // Generate isosurfaces
  m_doseIsosurfaces.clear();
  QProgressDialog progress(tr("Generating isosurfaces..."), tr("Cancel"), 0, levels.size(), this);
  progress.setWindowModality(Qt::WindowModal);

  // Determine min and max dose levels for automatic opacity scaling
  QVector<double> sortedLevels = levels;
  std::sort(sortedLevels.begin(), sortedLevels.end());
  double minLevel = sortedLevels.first();
  double maxLevel = sortedLevels.last();

  for (int i = 0; i < levels.size(); ++i) {
    if (progress.wasCanceled())
      break;

    progress.setValue(i);
    QApplication::processEvents();

    DoseIsosurface surface;

    // Use colormap-based color instead of predefined palette
    QColor doseColor = getDoseColor(levels[i], referenceDoseGy);
    surface.setColor(doseColor);

    // Automatic opacity scaling: higher dose = higher opacity
    float calculatedOpacity;
    if (maxLevel > minLevel) {
      // Linear interpolation based on dose level
      // Minimum opacity: 20%, Maximum opacity: user-defined value
      float ratio = (levels[i] - minLevel) / (maxLevel - minLevel);
      calculatedOpacity = 0.2f + ratio * (opacity - 0.2f);
    } else {
      // If only one dose level, use user-defined opacity
      calculatedOpacity = opacity;
    }
    surface.setOpacity(calculatedOpacity);

    // Get direction cosines (use identity if not available)
    double rowDir[3] = {1.0, 0.0, 0.0};
    double colDir[3] = {0.0, 1.0, 0.0};
    double sliceDir[3] = {0.0, 0.0, 1.0};

    // Generate mesh using Marching Cubes
    surface.generateIsosurface(
        activeDose->data(),
        levels[i],
        activeDose->originX(),
        activeDose->originY(),
        activeDose->originZ(),
        activeDose->spacingX(),
        activeDose->spacingY(),
        activeDose->spacingZ(),
        rowDir,
        colDir,
        sliceDir
    );

    // 患者座標系から3Dウィジェット座標系に変換
    if (!surface.isEmpty()) {
      surface.transformTo3DWidgetSpace(m_volume);
      m_doseIsosurfaces.append(surface);
    }
  }

  progress.setValue(levels.size());

  // Update 3D view
  for (int i = 0; i < VIEW_COUNT; ++i) {
    if (m_3dWidgets[i]) {
      m_3dWidgets[i]->setDoseIsosurfaces(m_doseIsosurfaces);
    }
  }

  QMessageBox::information(this, tr("3D Isosurface"),
                          tr("Generated %1 isosurface(s) with %2 total triangles.")
                          .arg(m_doseIsosurfaces.size())
                          .arg(std::accumulate(m_doseIsosurfaces.begin(), m_doseIsosurfaces.end(), 0,
                               [](int sum, const DoseIsosurface &s) { return sum + s.triangleCount(); })));
}

void DicomViewer::onExportButtonClicked(int viewIndex) {
  if (viewIndex < 0 || viewIndex >= VIEW_COUNT) {
    qWarning() << "Invalid view index for export:" << viewIndex;
    return;
  }

  if (!m_is3DView[viewIndex]) {
    qWarning() << "Export button clicked but view is not 3D:" << viewIndex;
    return;
  }

  if (!m_3dWidgets[viewIndex]) {
    qWarning() << "3D widget not available for export:" << viewIndex;
    return;
  }

  // Get 3D data from OpenGL widget
  const StructureLine3DList& structureLines = m_3dWidgets[viewIndex]->getStructureLines();
  const QVector<DoseIsosurface>& isosurfaces = m_3dWidgets[viewIndex]->getDoseIsosurfaces();

  // Check if there's any data to export
  if (structureLines.isEmpty() && isosurfaces.isEmpty()) {
    QMessageBox::information(this, tr("Export USDZ"),
                            tr("No 3D data available to export.\n\n"
                               "Please load RT Structure contours or generate dose isosurfaces first."));
    return;
  }

  // Show file save dialog
  QString defaultFileName = "ShioRIS3_Export.usdz";
  if (!m_ctFilename.isEmpty()) {
    QFileInfo ctInfo(m_ctFilename);
    defaultFileName = ctInfo.completeBaseName() + "_3D.usdz";
  }

  QString filename = QFileDialog::getSaveFileName(
      this,
      tr("Export 3D Data to USDZ for Vision Pro"),
      defaultFileName,
      tr("USDZ Files (*.usdz);;All Files (*)")
  );

  if (filename.isEmpty()) {
    return; // User cancelled
  }

  // Ensure .usdz extension
  if (!filename.endsWith(".usdz", Qt::CaseInsensitive)) {
    filename += ".usdz";
  }

  // Get CT volume data if available
  const DicomVolume* ctVolumePtr = isVolumeLoaded() ? &m_volume : nullptr;

  // Get window/level settings for CT display from UI controls
  double ctWindow = 400.0;
  double ctLevel = 40.0;
  if (m_windowSpinBox && m_levelSpinBox) {
    ctWindow = m_windowSpinBox->value();
    ctLevel = m_levelSpinBox->value();
  }

  // Get current slice indices for Axial, Sagittal, Coronal
  // m_orientationIndices[0] = Axial, [1] = Sagittal, [2] = Coronal
  int axialIndex = m_orientationIndices[0];
  int sagittalIndex = m_orientationIndices[1];
  int coronalIndex = m_orientationIndices[2];

  // Export using USDZ exporter
  USDZExporter exporter;
  bool success = exporter.exportToUSDZ(filename, structureLines, isosurfaces,
                                       ctVolumePtr, ctWindow, ctLevel,
                                       axialIndex, sagittalIndex, coronalIndex);

  if (success) {
    QString exportedData;
    if (ctVolumePtr) {
      // Count how many slices are actually exported
      int numSlices = 0;
      if (axialIndex >= 0 && axialIndex < ctVolumePtr->depth()) numSlices++;
      if (sagittalIndex >= 0 && sagittalIndex < ctVolumePtr->width()) numSlices++;
      if (coronalIndex >= 0 && coronalIndex < ctVolumePtr->height()) numSlices++;

      exportedData = tr("Exported data includes:\n"
                       "- %1 CT image slice(s) (Axial, Sagittal, Coronal)\n"
                       "- %2 RT Structure contour(s)\n"
                       "- %3 Dose isosurface(s)")
                       .arg(numSlices)
                       .arg(structureLines.size())
                       .arg(isosurfaces.size());
    } else {
      exportedData = tr("Exported data includes:\n"
                       "- %1 RT Structure contour(s)\n"
                       "- %2 Dose isosurface(s)")
                       .arg(structureLines.size())
                       .arg(isosurfaces.size());
    }

    QMessageBox::information(this, tr("Export Successful"),
                            tr("3D data exported successfully to:\n%1\n\n"
                               "You can now view this file on Vision Pro or other AR/VR devices.\n\n"
                               "%2")
                            .arg(filename)
                            .arg(exportedData));
  } else {
    QMessageBox::warning(this, tr("Export Failed"),
                        tr("Failed to export 3D data to USDZ file.\n\n"
                           "Please check:\n"
                           "- Write permission to the output directory\n"
                           "- Disk space availability\n"
                           "- System 'zip' command is available (for best compatibility)"));
  }
}

// moc ファイルのインクルード（cpp内にQ_OBJECTがあるため）
#include "dicom_viewer.moc"
