#include "visualization/fusion_dialog.h"

#include <QApplication>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileInfo>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QTextStream>
#include <QPainter>
#include <QPaintEvent>
#include "theme_manager.h"
#include <QCursor>
#include <QMouseEvent>
#include <QProgressBar>
#include <QPushButton>
#include <QSet>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSlider>
#include <QStringList>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QTimer>
#include <QVariant>
#include <QVBoxLayout>
#include <QtMath>
#include <QUuid>
#include <QtConcurrent/QtConcurrentRun>
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
#include <QStringConverter>
#endif

#include "database/database_manager.h"
#include "data/file_structure_manager.h"
#include "visualization/dicom_viewer.h"
#include "dicom/dicom_reader.h"

#include <algorithm>
#include <array>
#include <tuple>
#include <utility>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <initializer_list>
#include <limits>
#include <atomic>
#include <random>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#if defined(__has_include)
#if __has_include(<opencv2/core/parallel/parallel.hpp>)
#include <opencv2/core/parallel/parallel.hpp>
#define SHIORIS3_HAS_CV_PARALLEL 1
#elif __has_include(<opencv2/core/parallel.hpp>)
#include <opencv2/core/parallel.hpp>
#define SHIORIS3_HAS_CV_PARALLEL 1
#else
#define SHIORIS3_HAS_CV_PARALLEL 0
#endif
#else
#include <opencv2/core/parallel.hpp>
#define SHIORIS3_HAS_CV_PARALLEL 1
#endif

#ifndef SHIORIS3_HAS_CV_PARALLEL
#define SHIORIS3_HAS_CV_PARALLEL 0
#endif

namespace {
#if SHIORIS3_HAS_CV_PARALLEL
template <typename Func> void runParallelFor(const cv::Range &range, const Func &func) {
  cv::parallel_for_(range, func);
}
#else
template <typename Func> void runParallelFor(const cv::Range &range, const Func &func) {
  func(range);
}
#endif

template <typename T>
double trilinearSample(const cv::Mat &volume, int width, int height, int x0,
                       int x1, int y0, int y1, int z0, int z1, double fx,
                       double fy, double fz) {
  const T *planeZ0 = volume.ptr<T>(z0);
  const T *planeZ1 = volume.ptr<T>(z1);
  auto fetch = [&](const T *plane, int yi, int xi) -> double {
    return static_cast<double>(plane[yi * width + xi]);
  };

  const double c000 = fetch(planeZ0, y0, x0);
  const double c100 = fetch(planeZ0, y0, x1);
  const double c010 = fetch(planeZ0, y1, x0);
  const double c110 = fetch(planeZ0, y1, x1);
  const double c001 = fetch(planeZ1, y0, x0);
  const double c101 = fetch(planeZ1, y0, x1);
  const double c011 = fetch(planeZ1, y1, x0);
  const double c111 = fetch(planeZ1, y1, x1);

  const double c00 = c000 * (1.0 - fx) + c100 * fx;
  const double c01 = c001 * (1.0 - fx) + c101 * fx;
  const double c10 = c010 * (1.0 - fx) + c110 * fx;
  const double c11 = c011 * (1.0 - fx) + c111 * fx;
  const double c0 = c00 * (1.0 - fy) + c10 * fy;
  const double c1 = c01 * (1.0 - fy) + c11 * fy;
  return c0 * (1.0 - fz) + c1 * fz;
}

struct TransferJobInput {
  DicomVolume primaryVolume;
  DicomVolume secondaryVolume;
  bool primaryLoaded{false};
  bool secondaryLoaded{false};
  QVector3D centerShift{0.0, 0.0, 0.0};
  QVector3D manualTranslation{0.0, 0.0, 0.0};
  QQuaternion rotationQuat;
  std::atomic<int> *progressCounter{nullptr};
  int progressTotal{0};
};

std::string sqlEscape(const QString &value) {
  const QByteArray bytes = value.toUtf8();
  std::string escaped;
  escaped.reserve(bytes.size() + 8);
  for (char c : bytes) {
    if (c == '\'')
      escaped += "''";
    else
      escaped.push_back(c);
  }
  return escaped;
}

QVector3D computeVolumeCenterImpl(const DicomVolume &volume) {
  if (volume.width() <= 0 || volume.height() <= 0 || volume.depth() <= 0)
    return QVector3D();
  const double cx = (volume.width() - 1) / 2.0;
  const double cy = (volume.height() - 1) / 2.0;
  const double cz = (volume.depth() - 1) / 2.0;
  return volume.voxelToPatient(cx, cy, cz);
}

QVector3D transformToSecondaryPatientImpl(
    const DicomVolume &secondaryVolume, bool secondaryLoaded,
    const QVector3D &centerShift, const QVector3D &manualTranslation,
    const QQuaternion &rotationQuat, const QVector3D &primaryPoint) {
  if (!secondaryLoaded)
    return primaryPoint - centerShift;

  const QVector3D secondaryCenter = computeVolumeCenterImpl(secondaryVolume);
  const QVector3D translation = centerShift + manualTranslation;
  const QVector3D relative = primaryPoint - secondaryCenter - translation;
  const QQuaternion inv = rotationQuat.conjugated();
  const QVector3D rotated = inv.rotatedVector(relative);
  return rotated + secondaryCenter;
}

double sampleVolumeValueImpl(const cv::Mat &volume, double x, double y,
                             double z) {
  if (volume.empty())
    return 0.0;

  const int depth = volume.size[0];
  const int height = volume.size[1];
  const int width = volume.size[2];

  if (x < 0.0 || y < 0.0 || z < 0.0 || x > width - 1 || y > height - 1 ||
      z > depth - 1)
    return 0.0;

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int z0 = static_cast<int>(std::floor(z));
  const int x1 = std::min(x0 + 1, width - 1);
  const int y1 = std::min(y0 + 1, height - 1);
  const int z1 = std::min(z0 + 1, depth - 1);

  const double fx = x - x0;
  const double fy = y - y0;
  const double fz = z - z0;

  switch (volume.type()) {
  case CV_16SC1:
    return trilinearSample<short>(volume, width, height, x0, x1, y0, y1, z0,
                                  z1, fx, fy, fz);
  case CV_32FC1:
    return trilinearSample<float>(volume, width, height, x0, x1, y0, y1, z0,
                                  z1, fx, fy, fz);
  case CV_8UC1:
    return trilinearSample<uchar>(volume, width, height, x0, x1, y0, y1, z0,
                                  z1, fx, fy, fz);
  default:
    break;
  }
  return 0.0;
}

cv::Mat resampleSecondaryVolumeToPrimaryImpl(const TransferJobInput &input) {
  if (!input.primaryLoaded || !input.secondaryLoaded)
    return cv::Mat();

  const int width = input.primaryVolume.width();
  const int height = input.primaryVolume.height();
  const int depth = input.primaryVolume.depth();
  if (width <= 0 || height <= 0 || depth <= 0)
    return cv::Mat();

  const cv::Mat &secondaryData = input.secondaryVolume.data();
  if (secondaryData.empty() || secondaryData.dims != 3)
    return cv::Mat();

  int sizes[3] = {depth, height, width};
  int targetType = secondaryData.type();
  if (targetType != CV_16SC1 && targetType != CV_32FC1 &&
      targetType != CV_8UC1) {
    targetType = CV_32FC1;
  }

  cv::Mat resampled(3, sizes, targetType);
  resampled.setTo(0);

  auto sampleAt = [&](double vx, double vy, double vz) -> double {
    const QVector3D primaryPoint =
        input.primaryVolume.voxelToPatient(vx, vy, vz);
    const QVector3D secondaryPoint = transformToSecondaryPatientImpl(
        input.secondaryVolume, input.secondaryLoaded, input.centerShift,
        input.manualTranslation, input.rotationQuat, primaryPoint);
    const QVector3D voxelCoord = input.secondaryVolume.patientToVoxelContinuous(
        secondaryPoint);
    return sampleVolumeValueImpl(secondaryData, voxelCoord.x(),
                                 voxelCoord.y(), voxelCoord.z());
  };

  if (targetType == CV_16SC1) {
    const long minShort = static_cast<long>(std::numeric_limits<short>::min());
    const long maxShort = static_cast<long>(std::numeric_limits<short>::max());
    runParallelFor(cv::Range(0, depth), [&](const cv::Range &range) {
      for (int z = range.start; z < range.end; ++z) {
        short *dstPlane = resampled.ptr<short>(z);
        const double vz = z + 0.5;
        for (int y = 0; y < height; ++y) {
          short *dstRow = dstPlane + y * width;
          const double vy = y + 0.5;
          for (int x = 0; x < width; ++x) {
            const double vx = x + 0.5;
            const double sampled = sampleAt(vx, vy, vz);
            const long rounded = std::lround(sampled);
            dstRow[x] = static_cast<short>(
                std::clamp(rounded, minShort, maxShort));
          }
        }
        if (input.progressCounter)
          input.progressCounter->fetch_add(1, std::memory_order_relaxed);
      }
    });
  } else if (targetType == CV_32FC1) {
    runParallelFor(cv::Range(0, depth), [&](const cv::Range &range) {
      for (int z = range.start; z < range.end; ++z) {
        float *dstPlane = resampled.ptr<float>(z);
        const double vz = z + 0.5;
        for (int y = 0; y < height; ++y) {
          float *dstRow = dstPlane + y * width;
          const double vy = y + 0.5;
          for (int x = 0; x < width; ++x) {
            const double vx = x + 0.5;
            const double sampled = sampleAt(vx, vy, vz);
            dstRow[x] = static_cast<float>(sampled);
          }
        }
        if (input.progressCounter)
          input.progressCounter->fetch_add(1, std::memory_order_relaxed);
      }
    });
  } else {
    const long minByte = 0;
    const long maxByte = 255;
    runParallelFor(cv::Range(0, depth), [&](const cv::Range &range) {
      for (int z = range.start; z < range.end; ++z) {
        uchar *dstPlane = resampled.ptr<uchar>(z);
        const double vz = z + 0.5;
        for (int y = 0; y < height; ++y) {
          uchar *dstRow = dstPlane + y * width;
          const double vy = y + 0.5;
          for (int x = 0; x < width; ++x) {
            const double vx = x + 0.5;
            const double sampled = sampleAt(vx, vy, vz);
            const long rounded = std::lround(sampled);
            dstRow[x] = static_cast<uchar>(
                std::clamp(rounded, minByte, maxByte));
          }
        }
        if (input.progressCounter)
          input.progressCounter->fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  return resampled;
}

FusionTransferJobResult runTransferJob(const TransferJobInput &input) {
  FusionTransferJobResult result;
  cv::Mat transformedData = resampleSecondaryVolumeToPrimaryImpl(input);
  if (transformedData.empty()) {
    if (input.progressCounter && input.progressTotal > 0)
      input.progressCounter->store(input.progressTotal,
                                   std::memory_order_relaxed);
    result.error = FusionTransferJobResult::Error::ResampleFailed;
    return result;
  }

  if (!result.volume.createFromReference(input.primaryVolume, transformedData)) {
    if (input.progressCounter && input.progressTotal > 0)
      input.progressCounter->store(input.progressTotal,
                                   std::memory_order_relaxed);
    result.error = FusionTransferJobResult::Error::VolumeCreationFailed;
    return result;
  }

  if (input.progressCounter && input.progressTotal > 0)
    input.progressCounter->store(input.progressTotal,
                                 std::memory_order_relaxed);
  result.error = FusionTransferJobResult::Error::None;
  return result;
}

QString canonicalModality(const QString &raw) {
  QString mod = raw.trimmed().toUpper();
  if (mod.isEmpty())
    return QStringLiteral("OTHERS");

  if (mod == "MR")
    return QStringLiteral("MRI");
  if (mod == "PT")
    return QStringLiteral("PET");
  if (mod == "CT" || mod == "MRI" || mod == "PET" || mod == "OTHERS")
    return mod;

  if (mod.startsWith("FUSION")) {
    QString suffix = mod.mid(QStringLiteral("FUSION").size());
    suffix = suffix.trimmed();
    if (suffix.startsWith(QLatin1Char('/')))
      suffix.remove(0, 1);
    if (suffix.isEmpty())
      suffix = QStringLiteral("OTHERS");
    return QStringLiteral("Fusion/%1").arg(suffix);
  }

  QString normalized = mod;
  normalized.replace('-', ' ');
  normalized.replace('_', ' ');
  normalized = normalized.simplified();
  const QStringList tokens =
      normalized.split(' ', Qt::SkipEmptyParts);

  auto hasToken = [&](std::initializer_list<const char *> words) {
    for (const QString &token : tokens) {
      for (const char *word : words) {
        if (token == QLatin1String(word))
          return true;
      }
    }
    return false;
  };

  if (hasToken({"FUSION"})) {
    QString target;
    if (hasToken({"CT"}))
      target = QStringLiteral("CT");
    else if (hasToken({"MRI", "MR"}))
      target = QStringLiteral("MRI");
    else if (hasToken({"PET"}))
      target = QStringLiteral("PET");
    else
      target = QStringLiteral("OTHERS");
    return QStringLiteral("Fusion/%1").arg(target);
  }

  if (hasToken({"RTSTRUCT", "RTSS", "STRUCTURE", "STRUCTURES", "STRUCTS"}))
    return QStringLiteral("RTSTRUCT");
  if (hasToken({"RTDOSE", "DOSE", "DOSES"}))
    return QStringLiteral("RTDOSE");
  if (hasToken({"RTPLAN", "PLAN", "PLANS"}))
    return QStringLiteral("RTPLAN");
  if (hasToken({"RTIMAGE"}) ||
      (hasToken({"RT"}) && hasToken({"IMAGE", "IMAGES"})))
    return QStringLiteral("RTIMAGE");
  if (hasToken({"RTRECORD"}) ||
      (hasToken({"RT"}) && hasToken({"RECORD", "RECORDS"})))
    return QStringLiteral("RTRECORD");
  if (hasToken({"RTANALYSIS", "ANALYSIS"}))
    return QStringLiteral("RTANALYSIS");
  if (mod.startsWith("RT"))
    return mod;
  return QStringLiteral("OTHERS");
}

bool isImagingModality(const QString &modality) {
  return modality == "CT" || modality == "MRI" || modality == "PET" ||
         modality == "OTHERS" || modality.startsWith(QLatin1String("Fusion/"));
}

QString formatStudyLabel(const FusionStudyRecord &rec) {
  QStringList parts;
  parts << (rec.patientName.isEmpty() ? QObject::tr("患者名不明") : rec.patientName);
  parts << rec.modality;

  const QString folderName = [&]() -> QString {
    if (rec.path.isEmpty())
      return QString();
    QFileInfo info(rec.path);
    if (info.fileName().isEmpty())
      return info.dir().dirName();
    return info.fileName();
  }();

  QString studyName = rec.studyName.trimmed();
  QString studyDate = rec.studyDate.trimmed();

  QString detailSegment;
  if (!folderName.isEmpty())
    detailSegment = folderName;

  if (!studyName.isEmpty() &&
      QString::compare(studyName, detailSegment, Qt::CaseInsensitive) != 0) {
    if (!detailSegment.isEmpty()) {
      detailSegment += QObject::tr(" (%1)").arg(studyName);
    } else {
      detailSegment = studyName;
    }
  }

  if (!studyDate.isEmpty() &&
      (detailSegment.isEmpty() ||
       !detailSegment.contains(studyDate, Qt::CaseInsensitive))) {
    if (!detailSegment.isEmpty()) {
      detailSegment += QObject::tr(" [%1]").arg(studyDate);
    } else {
      detailSegment = studyDate;
    }
  }

  if (detailSegment.isEmpty())
    detailSegment = QObject::tr("-");

  parts << detailSegment;
  return parts.join(QStringLiteral(" | "));
}
} // namespace

FusionSliceView::FusionSliceView(QWidget *parent) : QWidget(parent) {
  setMinimumSize(320, 320);
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  setAutoFillBackground(true);
  setMouseTracking(true);
}

void FusionSliceView::setBaseImage(const QImage &image) {
  m_baseImage = image;
  update();
}

void FusionSliceView::setOverlayImage(const QImage &image) {
  m_overlayImage = image;
  update();
}

void FusionSliceView::setMessage(const QString &text) {
  m_message = text;
  update();
}

void FusionSliceView::setOrientationLabel(const QString &text) {
  m_orientationLabel = text;
  update();
}

void FusionSliceView::setOrientation(DicomVolume::Orientation orientation) {
  if (m_orientation == orientation)
    return;
  m_orientation = orientation;
  update();
}

void FusionSliceView::setSliceIndex(int index) {
  if (m_sliceIndex == index)
    return;
  m_sliceIndex = index;
  update();
}

void FusionSliceView::setVolumeDimensions(int width, int height, int depth) {
  if (m_volumeWidth == width && m_volumeHeight == height &&
      m_volumeDepth == depth)
    return;
  m_volumeWidth = width;
  m_volumeHeight = height;
  m_volumeDepth = depth;
  update();
}

void FusionSliceView::setROI(const QVector3D &minEdges,
                             const QVector3D &maxEdges) {
  m_roiMinEdges = minEdges;
  m_roiMaxEdges = maxEdges;
  update();
}

void FusionSliceView::setROIEnabled(bool enabled) {
  if (m_roiEnabled == enabled)
    return;
  m_roiEnabled = enabled;
  if (!m_roiEnabled)
    m_dragMode = DragMode::None;
  update();
}

void FusionSliceView::setPixelSpacing(double horizontal, double vertical) {
  const double safeHorizontal = horizontal > 0.0 ? horizontal : 1.0;
  const double safeVertical = vertical > 0.0 ? vertical : 1.0;
  if (qFuzzyCompare(1.0 + m_horizontalSpacing, 1.0 + safeHorizontal) &&
      qFuzzyCompare(1.0 + m_verticalSpacing, 1.0 + safeVertical))
    return;
  m_horizontalSpacing = safeHorizontal;
  m_verticalSpacing = safeVertical;
  update();
}

void FusionSliceView::setZoomFactor(double factor) {
  const double clamped = std::clamp(factor, 0.1, 10.0);
  if (qFuzzyCompare(1.0 + m_zoomFactor, 1.0 + clamped))
    return;
  m_zoomFactor = clamped;
  update();
}

void FusionSliceView::paintEvent(QPaintEvent *event) {
  Q_UNUSED(event);
  QPainter painter(this);
  painter.fillRect(rect(), Qt::black);
  const QColor textColor = ThemeManager::instance().textColor();

  QRectF imageRect;
  if (!m_baseImage.isNull()) {
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
    imageRect = computeTargetRect();
    if (!imageRect.isNull()) {
      painter.drawImage(imageRect, m_baseImage);
      if (!m_overlayImage.isNull()) {
        painter.drawImage(imageRect, m_overlayImage);
      }
    }
  } else {
    painter.setPen(textColor);
    painter.drawText(rect(), Qt::AlignCenter,
                     m_message.isEmpty()
                         ? tr("データベースからボリュームを選択してください")
                         : m_message);
  }

  if (m_roiEnabled && !imageRect.isNull()) {
    const QRectF roiRect = roiRectInWidget(imageRect);
    if (!roiRect.isNull()) {
      QColor borderColor(0, 220, 120);
      QColor fillColor = borderColor;
      fillColor.setAlpha(40);
      QPen pen(borderColor);
      pen.setWidthF(2.0);
      pen.setStyle(Qt::DashLine);
      painter.setPen(pen);
      painter.setBrush(fillColor);
      painter.drawRect(roiRect);

      painter.setPen(Qt::NoPen);
      painter.setBrush(borderColor);
      const double handleHalf = 4.0;
      const double handleSize = handleHalf * 2.0;
      const QPointF topLeft = roiRect.topLeft();
      const QPointF topRight = roiRect.topRight();
      const QPointF bottomLeft = roiRect.bottomLeft();
      const QPointF bottomRight = roiRect.bottomRight();
      const QPointF topCenter = QPointF((topLeft.x() + topRight.x()) / 2.0,
                                        topLeft.y());
      const QPointF bottomCenter =
          QPointF((bottomLeft.x() + bottomRight.x()) / 2.0, bottomLeft.y());
      const QPointF leftCenter = QPointF(topLeft.x(),
                                         (topLeft.y() + bottomLeft.y()) / 2.0);
      const QPointF rightCenter = QPointF(topRight.x(),
                                          (topRight.y() + bottomRight.y()) /
                                              2.0);

      const std::array<QPointF, 8> handles = {
          topLeft,      topRight,    bottomLeft, bottomRight,
          topCenter,    bottomCenter, leftCenter, rightCenter};
      for (const QPointF &pt : handles) {
        painter.drawRect(QRectF(pt.x() - handleHalf, pt.y() - handleHalf,
                                handleSize, handleSize));
      }
    }
  }

  if (!m_orientationLabel.isEmpty()) {
    const QFontMetrics fm(painter.font());
    const QRect textRect = fm.boundingRect(m_orientationLabel)
                               .adjusted(-8, -4, 8, 4);
    QRect box = textRect;
    box.moveTo(12, 12);
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(0, 0, 0, 160));
    painter.drawRoundedRect(box, 6, 6);
    painter.setPen(textColor);
    painter.drawText(box.adjusted(8, 4, -8, -4), Qt::AlignLeft | Qt::AlignVCenter,
                     m_orientationLabel);
  }
}

FusionSliceView::AxisMapping FusionSliceView::mappingForOrientation() const {
  switch (m_orientation) {
  case DicomVolume::Orientation::Axial:
    return {0, 1, 2, false, false};
  case DicomVolume::Orientation::Sagittal:
    return {1, 2, 0, false, true};
  case DicomVolume::Orientation::Coronal:
  default:
    return {0, 2, 1, false, true};
  }
}

QRectF FusionSliceView::computeTargetRect() const {
  if (m_baseImage.isNull())
    return QRectF();

  QRectF targetRect(rect());
  const double physicalWidth =
      static_cast<double>(m_baseImage.width()) * m_horizontalSpacing;
  const double physicalHeight =
      static_cast<double>(m_baseImage.height()) * m_verticalSpacing;
  if (physicalWidth <= 0.0 || physicalHeight <= 0.0)
    return QRectF();

  const double baseScale =
      std::min(static_cast<double>(width()) / physicalWidth,
               static_cast<double>(height()) / physicalHeight);
  const double scale = baseScale * m_zoomFactor;
  const double drawWidth = physicalWidth * scale;
  const double drawHeight = physicalHeight * scale;
  const double left = (static_cast<double>(width()) - drawWidth) / 2.0;
  const double top = (static_cast<double>(height()) - drawHeight) / 2.0;
  targetRect = QRectF(left, top, drawWidth, drawHeight);
  return targetRect;
}

double FusionSliceView::axisLength(int axis) const {
  switch (axis) {
  case 0:
    return static_cast<double>(m_volumeWidth);
  case 1:
    return static_cast<double>(m_volumeHeight);
  case 2:
    return static_cast<double>(m_volumeDepth);
  default:
    break;
  }
  return 0.0;
}

QRectF FusionSliceView::roiRectInWidget(const QRectF &imageRect) const {
  if (!m_roiEnabled || imageRect.isNull())
    return QRectF();

  const AxisMapping map = mappingForOrientation();
  const double perpLen = axisLength(map.axisPerp);
  if (perpLen <= 0.0)
    return QRectF();

  const double sliceCenter = static_cast<double>(m_sliceIndex) + 0.5;
  const double minPerp = static_cast<double>(m_roiMinEdges[map.axisPerp]);
  const double maxPerp = static_cast<double>(m_roiMaxEdges[map.axisPerp]);
  if (sliceCenter < minPerp || sliceCenter > maxPerp)
    return QRectF();

  const double lenH = axisLength(map.axisH);
  const double lenV = axisLength(map.axisV);
  if (lenH <= 0.0 || lenV <= 0.0)
    return QRectF();

  auto project = [&](double edge, double len, bool invert, bool horizontal) {
    if (len <= 0.0)
      return horizontal ? imageRect.left() : imageRect.top();
    double norm = edge / len;
    norm = std::clamp(norm, 0.0, 1.0);
    if (invert)
      norm = 1.0 - norm;
    if (horizontal)
      return imageRect.left() + norm * imageRect.width();
    return imageRect.top() + norm * imageRect.height();
  };

  const double left =
      project(static_cast<double>(m_roiMinEdges[map.axisH]), lenH, map.invertH,
              true);
  const double right =
      project(static_cast<double>(m_roiMaxEdges[map.axisH]), lenH, map.invertH,
              true);
  const double top =
      project(static_cast<double>(m_roiMinEdges[map.axisV]), lenV, map.invertV,
              false);
  const double bottom =
      project(static_cast<double>(m_roiMaxEdges[map.axisV]), lenV, map.invertV,
              false);

  QRectF roi(QPointF(left, top), QPointF(right, bottom));
  roi = roi.normalized();
  return roi;
}

bool FusionSliceView::mapPointToAxes(const QPointF &pt, double &outH,
                                     double &outV) const {
  const QRectF imageRect = computeTargetRect();
  if (imageRect.isNull() || imageRect.width() <= 0.0 || imageRect.height() <= 0.0)
    return false;

  const AxisMapping map = mappingForOrientation();
  const double lenH = axisLength(map.axisH);
  const double lenV = axisLength(map.axisV);
  if (lenH <= 0.0 || lenV <= 0.0)
    return false;

  double normH = (pt.x() - imageRect.left()) / imageRect.width();
  double normV = (pt.y() - imageRect.top()) / imageRect.height();
  normH = std::clamp(normH, 0.0, 1.0);
  normV = std::clamp(normV, 0.0, 1.0);
  if (map.invertH)
    normH = 1.0 - normH;
  if (map.invertV)
    normV = 1.0 - normV;
  outH = normH * lenH;
  outV = normV * lenV;
  return true;
}

double FusionSliceView::horizontalDeltaForPixels(double pixels) const {
  const QRectF imageRect = computeTargetRect();
  if (imageRect.width() <= 0.0)
    return 0.0;
  const AxisMapping map = mappingForOrientation();
  const double lenH = axisLength(map.axisH);
  if (lenH <= 0.0)
    return 0.0;
  double norm = pixels / imageRect.width();
  if (map.invertH)
    norm = -norm;
  return norm * lenH;
}

double FusionSliceView::verticalDeltaForPixels(double pixels) const {
  const QRectF imageRect = computeTargetRect();
  if (imageRect.height() <= 0.0)
    return 0.0;
  const AxisMapping map = mappingForOrientation();
  const double lenV = axisLength(map.axisV);
  if (lenV <= 0.0)
    return 0.0;
  double norm = pixels / imageRect.height();
  if (map.invertV)
    norm = -norm;
  return norm * lenV;
}

FusionSliceView::DragMode
FusionSliceView::hitTestHandles(const QPointF &pos, const QRectF &roiRect) const {
  if (roiRect.isNull())
    return DragMode::None;

  const double threshold = 8.0;
  const bool nearLeft = std::abs(pos.x() - roiRect.left()) <= threshold;
  const bool nearRight = std::abs(pos.x() - roiRect.right()) <= threshold;
  const bool nearTop = std::abs(pos.y() - roiRect.top()) <= threshold;
  const bool nearBottom = std::abs(pos.y() - roiRect.bottom()) <= threshold;

  if (nearLeft && nearTop)
    return DragMode::ResizeTopLeft;
  if (nearLeft && nearBottom)
    return DragMode::ResizeBottomLeft;
  if (nearRight && nearTop)
    return DragMode::ResizeTopRight;
  if (nearRight && nearBottom)
    return DragMode::ResizeBottomRight;
  if (nearLeft)
    return DragMode::ResizeLeft;
  if (nearRight)
    return DragMode::ResizeRight;
  if (nearTop)
    return DragMode::ResizeTop;
  if (nearBottom)
    return DragMode::ResizeBottom;
  if (roiRect.contains(pos))
    return DragMode::Move;
  return DragMode::None;
}

void FusionSliceView::updateHoverCursor(const QPointF &pos) {
  if (!m_roiEnabled || m_baseImage.isNull()) {
    unsetCursor();
    return;
  }
  const QRectF imageRect = computeTargetRect();
  const QRectF roiRect = roiRectInWidget(imageRect);
  if (roiRect.isNull()) {
    unsetCursor();
    return;
  }
  const DragMode mode = hitTestHandles(pos, roiRect);
  switch (mode) {
  case DragMode::Move:
    setCursor(Qt::SizeAllCursor);
    break;
  case DragMode::ResizeLeft:
  case DragMode::ResizeRight:
    setCursor(Qt::SizeHorCursor);
    break;
  case DragMode::ResizeTop:
  case DragMode::ResizeBottom:
    setCursor(Qt::SizeVerCursor);
    break;
  case DragMode::ResizeTopLeft:
  case DragMode::ResizeBottomRight:
    setCursor(Qt::SizeFDiagCursor);
    break;
  case DragMode::ResizeTopRight:
  case DragMode::ResizeBottomLeft:
    setCursor(Qt::SizeBDiagCursor);
    break;
  default:
    unsetCursor();
    break;
  }
}

void FusionSliceView::mousePressEvent(QMouseEvent *event) {
  if (!m_roiEnabled || m_baseImage.isNull()) {
    QWidget::mousePressEvent(event);
    return;
  }
  const QRectF imageRect = computeTargetRect();
  const QRectF roiRect = roiRectInWidget(imageRect);
  const DragMode mode = hitTestHandles(event->pos(), roiRect);
  if (mode == DragMode::None) {
    QWidget::mousePressEvent(event);
    return;
  }
  m_dragMode = mode;
  m_dragStartPos = event->pos();
  m_dragStartMinEdges = m_roiMinEdges;
  m_dragStartMaxEdges = m_roiMaxEdges;
  event->accept();
}

void FusionSliceView::mouseMoveEvent(QMouseEvent *event) {
  if (m_dragMode == DragMode::None) {
    updateHoverCursor(event->pos());
    QWidget::mouseMoveEvent(event);
    return;
  }

  QVector3D newMin = m_dragStartMinEdges;
  QVector3D newMax = m_dragStartMaxEdges;
  const AxisMapping map = mappingForOrientation();

  auto clampToRange = [&](double value, int axis) {
    const double len = axisLength(axis);
    if (len <= 0.0)
      return 0.0;
    return std::clamp(value, 0.0, len);
  };

  auto setHorizontal = [&](bool minEdge, double value) {
    value = clampToRange(value, map.axisH);
    if (minEdge)
      newMin[map.axisH] = static_cast<float>(value);
    else
      newMax[map.axisH] = static_cast<float>(value);
  };

  auto setVertical = [&](bool minEdge, double value) {
    value = clampToRange(value, map.axisV);
    if (minEdge)
      newMin[map.axisV] = static_cast<float>(value);
    else
      newMax[map.axisV] = static_cast<float>(value);
  };

  switch (m_dragMode) {
  case DragMode::Move: {
    const double deltaH =
        horizontalDeltaForPixels(event->pos().x() - m_dragStartPos.x());
    const double deltaV =
        verticalDeltaForPixels(event->pos().y() - m_dragStartPos.y());

    const double startMinH = static_cast<double>(m_dragStartMinEdges[map.axisH]);
    const double startMaxH = static_cast<double>(m_dragStartMaxEdges[map.axisH]);
    const double lenH = axisLength(map.axisH);
    const double minDeltaH = -startMinH;
    const double maxDeltaH = lenH - startMaxH;
    const double appliedH = std::clamp(deltaH, minDeltaH, maxDeltaH);

    const double startMinV = static_cast<double>(m_dragStartMinEdges[map.axisV]);
    const double startMaxV = static_cast<double>(m_dragStartMaxEdges[map.axisV]);
    const double lenV = axisLength(map.axisV);
    const double minDeltaV = -startMinV;
    const double maxDeltaV = lenV - startMaxV;
    const double appliedV = std::clamp(deltaV, minDeltaV, maxDeltaV);

    newMin[map.axisH] = static_cast<float>(startMinH + appliedH);
    newMax[map.axisH] = static_cast<float>(startMaxH + appliedH);
    newMin[map.axisV] = static_cast<float>(startMinV + appliedV);
    newMax[map.axisV] = static_cast<float>(startMaxV + appliedV);
    break;
  }
  case DragMode::ResizeLeft:
  case DragMode::ResizeRight:
  case DragMode::ResizeTop:
  case DragMode::ResizeBottom:
  case DragMode::ResizeTopLeft:
  case DragMode::ResizeTopRight:
  case DragMode::ResizeBottomLeft:
  case DragMode::ResizeBottomRight: {
    double axisH = 0.0;
    double axisV = 0.0;
    mapPointToAxes(event->pos(), axisH, axisV);
    switch (m_dragMode) {
    case DragMode::ResizeLeft:
      setHorizontal(true, axisH);
      break;
    case DragMode::ResizeRight:
      setHorizontal(false, axisH);
      break;
    case DragMode::ResizeTop:
      setVertical(!map.invertV, axisV);
      break;
    case DragMode::ResizeBottom:
      setVertical(map.invertV, axisV);
      break;
    case DragMode::ResizeTopLeft:
      setHorizontal(true, axisH);
      setVertical(!map.invertV, axisV);
      break;
    case DragMode::ResizeTopRight:
      setHorizontal(false, axisH);
      setVertical(!map.invertV, axisV);
      break;
    case DragMode::ResizeBottomLeft:
      setHorizontal(true, axisH);
      setVertical(map.invertV, axisV);
      break;
    case DragMode::ResizeBottomRight:
      setHorizontal(false, axisH);
      setVertical(map.invertV, axisV);
      break;
    default:
      break;
    }
    break;
  }
  default:
    break;
  }

  auto nearlyEqual = [](double a, double b) {
    return std::abs(a - b) < 1e-3;
  };
  if (nearlyEqual(newMin.x(), m_roiMinEdges.x()) &&
      nearlyEqual(newMin.y(), m_roiMinEdges.y()) &&
      nearlyEqual(newMin.z(), m_roiMinEdges.z()) &&
      nearlyEqual(newMax.x(), m_roiMaxEdges.x()) &&
      nearlyEqual(newMax.y(), m_roiMaxEdges.y()) &&
      nearlyEqual(newMax.z(), m_roiMaxEdges.z())) {
    event->accept();
    return;
  }

  emit roiChanging(newMin, newMax);
  event->accept();
}

void FusionSliceView::mouseReleaseEvent(QMouseEvent *event) {
  if (m_dragMode == DragMode::None) {
    QWidget::mouseReleaseEvent(event);
    return;
  }
  m_dragMode = DragMode::None;
  emit roiChanged(m_roiMinEdges, m_roiMaxEdges);
  updateHoverCursor(event->pos());
  event->accept();
}

void FusionSliceView::leaveEvent(QEvent *event) {
  if (m_dragMode == DragMode::None)
    unsetCursor();
  QWidget::leaveEvent(event);
}

bool FusionSliceView::hasHeightForWidth() const { return true; }

int FusionSliceView::heightForWidth(int w) const { return w; }

QSize FusionSliceView::sizeHint() const { return QSize(360, 360); }

QSize FusionSliceView::minimumSizeHint() const { return QSize(240, 240); }

FusionDialog::FusionDialog(DatabaseManager &db, DicomViewer *viewer,
                           QWidget *parent)
    : QDialog(parent), m_db(db), m_viewer(viewer) {
  setWindowTitle(tr("Image Fusion"));
  setWindowFlag(Qt::WindowStaysOnTopHint, true);
  resize(1200, 900);
  m_rotationQuat = QQuaternion::fromEulerAngles(0.0f, 0.0f, 0.0f);
  connect(&m_transferWatcher, &QFutureWatcher<TransferJobResult>::finished, this,
          &FusionDialog::onTransferComputationFinished);
  setupUi();
  loadStudyList();
  setPrimaryFromViewer(m_viewer);
}

FusionDialog::~FusionDialog() {
  if (m_transferWatcher.isRunning()) {
    m_transferWatcher.cancel();
    m_transferWatcher.waitForFinished();
  }
  if (m_transferInProgress)
    QApplication::restoreOverrideCursor();
  m_transferInProgress = false;
  stopTransferProgress();
  if (m_viewer)
    m_viewer->clearFusionPreviewImage();
}

void FusionDialog::setPrimaryFromViewer(DicomViewer *viewer) {
  m_viewer = viewer;
  m_currentPrimaryStudy = -1;
  if (!viewer || !viewer->isVolumeLoaded()) {
    m_primaryLoaded = false;
    m_primaryVolume = DicomVolume();
    m_primaryPatientKey.clear();
    m_primaryPatientId.clear();
    m_primaryPatientName.clear();
    m_primaryStudyPath.clear();
    m_primaryStudyDescription.clear();
    m_primaryStudyDate.clear();
    m_primaryModality.clear();
    m_primaryFrameUid.clear();
    updatePrimaryInfoDisplay();
    updateSliceRanges();
    updateCenterShift();
    updateStatusLabels();
    disableROI();
    populateStudyCombos();
    updateImages();
    updateTransferButtonState();
    return;
  }

  m_primaryVolume = viewer->getVolume();
  m_primaryLoaded = m_primaryVolume.depth() > 0;

  const DicomStudyInfo info = viewer->currentStudyInfo();
  m_primaryPatientId = info.patientID.trimmed();
  m_primaryPatientName = info.patientName.trimmed();
  m_primaryStudyPath = info.seriesDirectory.trimmed();
  m_primaryStudyDescription = info.studyDescription.trimmed();
  m_primaryStudyDate = info.studyDate.trimmed();
  m_primaryModality = info.modality.trimmed();
  m_primaryFrameUid = info.frameOfReferenceUID.trimmed();

  updatePrimaryPatientKey();

  updatePrimaryInfoDisplay();
  updateSliceRanges();
  updateCenterShift();
  updateStatusLabels();
  initializeDefaultROI();
  onResetTransform();
  populateStudyCombos();
  selectSecondaryForPrimaryPatient();
  updateTransferButtonState();
}

void FusionDialog::setupUi() {
  auto *mainLayout = new QVBoxLayout(this);

  // Study selection area
  auto *selectionGroup = new QGroupBox(tr("ボリューム選択"), this);
  auto *selectionLayout = new QVBoxLayout(selectionGroup);

  auto *selectionRow = new QHBoxLayout();
  auto *primaryBox = new QGroupBox(tr("Primary"), selectionGroup);
  auto *primaryLayout = new QVBoxLayout(primaryBox);
  m_primaryStudyCombo = new QComboBox(primaryBox);
  m_primaryStudyCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
  primaryLayout->addWidget(m_primaryStudyCombo);
  m_primaryInfoLabel = new QLabel(tr("患者: -\nスタディ: -\nパス: -"), primaryBox);
  m_primaryInfoLabel->setWordWrap(true);
  m_primaryInfoLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
  primaryLayout->addWidget(m_primaryInfoLabel);
  m_primaryUidLabel = new QLabel(tr("Frame UID: -"), primaryBox);
  primaryLayout->addWidget(m_primaryUidLabel);
  selectionRow->addWidget(primaryBox, 1);

  auto *secondaryBox = new QGroupBox(tr("Secondary"), selectionGroup);
  auto *secondaryLayout = new QVBoxLayout(secondaryBox);
  m_secondaryStudyCombo = new QComboBox(secondaryBox);
  m_secondaryStudyCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
  secondaryLayout->addWidget(m_secondaryStudyCombo);
  m_secondaryInfoLabel =
      new QLabel(tr("患者: -\nスタディ: -\nパス: -"), secondaryBox);
  m_secondaryInfoLabel->setWordWrap(true);
  m_secondaryInfoLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
  secondaryLayout->addWidget(m_secondaryInfoLabel);
  m_secondaryUidLabel = new QLabel(tr("Frame UID: -"), secondaryBox);
  secondaryLayout->addWidget(m_secondaryUidLabel);
  selectionRow->addWidget(secondaryBox, 1);

  auto *refreshLayout = new QVBoxLayout();
  m_refreshButton = new QPushButton(tr("更新"), selectionGroup);
  refreshLayout->addWidget(m_refreshButton);
  refreshLayout->addStretch();
  selectionRow->addLayout(refreshLayout);

  selectionLayout->addLayout(selectionRow);
  mainLayout->addWidget(selectionGroup);

  // Transform and display controls
  auto *controlRow = new QHBoxLayout();
  controlRow->setSpacing(12);

  auto *transformGroup = new QGroupBox(tr("Secondary ボリューム調整"), this);
  setupTransformControls(transformGroup);
  controlRow->addWidget(transformGroup, 1);

  auto *displayGroup = new QGroupBox(tr("表示調整"), this);
  auto *displayLayout = new QHBoxLayout(displayGroup);
  displayLayout->setSpacing(20);

  auto setupDisplayControl = [](QWidget *parent, const QString &title,
                                QSlider *slider, QLabel *valueLabel) {
    auto *blockLayout = new QVBoxLayout();
    blockLayout->setSpacing(4);

    auto *labelRow = new QHBoxLayout();
    auto *titleLabel = new QLabel(title, parent);
    labelRow->addWidget(titleLabel);
    labelRow->addStretch();
    labelRow->addWidget(valueLabel);
    blockLayout->addLayout(labelRow);

    slider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    blockLayout->addWidget(slider);

    return blockLayout;
  };

  auto *ctGroup = new QGroupBox(tr("CT"), displayGroup);
  auto *ctLayout = new QVBoxLayout(ctGroup);
  m_primaryDisplayModeCombo = new QComboBox(ctGroup);
  m_primaryDisplayModeCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
  m_primaryDisplayModeCombo->addItem(
      tr("標準 (グレースケール)"),
      static_cast<int>(PrimaryDisplayMode::Grayscale));
  m_primaryDisplayModeCombo->addItem(
      tr("反転 (ネガポジ)"), static_cast<int>(PrimaryDisplayMode::Inverted));
  m_primaryDisplayModeCombo->addItem(
      tr("骨強調 (擬似カラー)"),
      static_cast<int>(PrimaryDisplayMode::BoneHighlight));
  m_primaryDisplayModeCombo->setCurrentIndex(0);
  ctLayout->addWidget(m_primaryDisplayModeCombo);
  m_primaryWindowSlider = new QSlider(Qt::Horizontal, ctGroup);
  m_primaryWindowSlider->setRange(1, 4096);
  m_primaryWindowSlider->setValue(static_cast<int>(m_primaryWindow));
  m_primaryWindowValueLabel =
      new QLabel(QString::number(m_primaryWindow), ctGroup);
  ctLayout->addLayout(setupDisplayControl(ctGroup, tr("Window"),
                                          m_primaryWindowSlider,
                                          m_primaryWindowValueLabel));

  m_primaryLevelSlider = new QSlider(Qt::Horizontal, ctGroup);
  m_primaryLevelSlider->setRange(-1024, 3072);
  m_primaryLevelSlider->setValue(static_cast<int>(m_primaryLevel));
  m_primaryLevelValueLabel =
      new QLabel(QString::number(m_primaryLevel), ctGroup);
  ctLayout->addLayout(setupDisplayControl(ctGroup, tr("Level"),
                                          m_primaryLevelSlider,
                                          m_primaryLevelValueLabel));
  ctLayout->addStretch();

  auto *mriGroup = new QGroupBox(tr("MRI"), displayGroup);
  auto *mriLayout = new QVBoxLayout(mriGroup);
  m_secondaryDisplayModeCombo = new QComboBox(mriGroup);
  m_secondaryDisplayModeCombo->setSizeAdjustPolicy(
      QComboBox::AdjustToContents);
  m_secondaryDisplayModeCombo->addItem(
      tr("赤色オーバーレイ"),
      static_cast<int>(SecondaryDisplayMode::RedOverlay));
  m_secondaryDisplayModeCombo->addItem(
      tr("シアンオーバーレイ"),
      static_cast<int>(SecondaryDisplayMode::CyanOverlay));
  m_secondaryDisplayModeCombo->addItem(
      tr("輪郭強調"),
      static_cast<int>(SecondaryDisplayMode::EdgeHighlight));
  m_secondaryDisplayModeCombo->setCurrentIndex(0);
  mriLayout->addWidget(m_secondaryDisplayModeCombo);

  m_secondaryWindowSlider = new QSlider(Qt::Horizontal, mriGroup);
  m_secondaryWindowSlider->setRange(1, 4096);
  m_secondaryWindowSlider->setValue(static_cast<int>(m_secondaryWindow));
  m_secondaryWindowValueLabel =
      new QLabel(QString::number(m_secondaryWindow), mriGroup);
  mriLayout->addLayout(setupDisplayControl(mriGroup, tr("Window"),
                                           m_secondaryWindowSlider,
                                           m_secondaryWindowValueLabel));

  m_secondaryLevelSlider = new QSlider(Qt::Horizontal, mriGroup);
  m_secondaryLevelSlider->setRange(-1024, 3072);
  m_secondaryLevelSlider->setValue(static_cast<int>(m_secondaryLevel));
  m_secondaryLevelValueLabel =
      new QLabel(QString::number(m_secondaryLevel), mriGroup);
  mriLayout->addLayout(setupDisplayControl(mriGroup, tr("Level"),
                                           m_secondaryLevelSlider,
                                           m_secondaryLevelValueLabel));

  m_opacitySlider = new QSlider(Qt::Horizontal, mriGroup);
  m_opacitySlider->setRange(0, 100);
  m_opacitySlider->setValue(static_cast<int>(m_overlayOpacity * 100.0));
  m_opacityValueLabel = new QLabel(
      tr("Opacity: %1%").arg(static_cast<int>(m_overlayOpacity * 100.0)),
      mriGroup);
  mriLayout->addLayout(setupDisplayControl(mriGroup, tr("Overlay"),
                                           m_opacitySlider, m_opacityValueLabel));
  m_transferSecondaryButton = new QPushButton(tr("Image2 に転送"), mriGroup);
  m_transferSecondaryButton->setToolTip(
      tr("MRIボリュームをメインウィンドウへ送信します"));
  m_transferSecondaryButton->setEnabled(false);
  m_transferButtonDefaultText = m_transferSecondaryButton->text();
  mriLayout->addWidget(m_transferSecondaryButton);
  m_transferProgressBar = new QProgressBar(mriGroup);
  m_transferProgressBar->setRange(0, 100);
  m_transferProgressBar->setValue(0);
  m_transferProgressBar->setVisible(false);
  mriLayout->addWidget(m_transferProgressBar);
  mriLayout->addStretch();

  displayLayout->addWidget(ctGroup, 1);
  displayLayout->addWidget(mriGroup, 1);
  displayLayout->setStretch(0, 1);
  displayLayout->setStretch(1, 1);

  controlRow->addWidget(displayGroup, 1);
  controlRow->setStretch(0, 1);
  controlRow->setStretch(1, 1);
  mainLayout->addLayout(controlRow);

  // Slice views layout
  auto *viewsRow = new QHBoxLayout();
  viewsRow->setSpacing(12);
  const std::array<DicomVolume::Orientation, 3> orientations = {
      DicomVolume::Orientation::Axial, DicomVolume::Orientation::Sagittal,
      DicomVolume::Orientation::Coronal};
  const std::array<QString, 3> orientationShortLabels = {tr("Ax"), tr("Sag"),
                                                         tr("Cor")};

  for (int i = 0; i < 3; ++i) {
    auto *viewColumn = new QVBoxLayout();
    viewColumn->setSpacing(8);

    m_sliceViews[i] = new FusionSliceView(this);
    m_sliceViews[i]->setOrientation(orientations[i]);
    m_sliceViews[i]->setOrientationLabel(
        orientationDisplayName(orientations[i]));
    viewColumn->addWidget(m_sliceViews[i], 1);

    auto *sliceRow = new QHBoxLayout();
    sliceRow->setSpacing(6);
    auto *sliceLabel = new QLabel(orientationShortLabels[i], this);
    sliceLabel->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    sliceRow->addWidget(sliceLabel);

    m_sliceSliders[i] = new QSlider(Qt::Horizontal, this);
    m_sliceSliders[i]->setRange(0, 0);
    m_sliceSliders[i]->setEnabled(false);
    m_sliceSliders[i]->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    sliceRow->addWidget(m_sliceSliders[i], 1);

    m_sliceValueLabels[i] = new QLabel(tr("-"), this);
    m_sliceValueLabels[i]->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    m_sliceValueLabels[i]->setMinimumWidth(80);
    sliceRow->addWidget(m_sliceValueLabels[i]);
    viewColumn->addLayout(sliceRow);

    auto *zoomRow = new QHBoxLayout();
    zoomRow->setSpacing(6);
    auto *zoomLabel = new QLabel(tr("ズーム"), this);
    zoomLabel->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    zoomRow->addWidget(zoomLabel);

    m_zoomSliders[i] = new QSlider(Qt::Horizontal, this);
    m_zoomSliders[i]->setRange(25, 400);
    m_zoomSliders[i]->setValue(static_cast<int>(m_zoomFactors[i] * 100.0));
    m_zoomSliders[i]->setEnabled(false);
    m_zoomSliders[i]->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    zoomRow->addWidget(m_zoomSliders[i], 1);

    m_zoomValueLabels[i] = new QLabel(QStringLiteral("100%"), this);
    m_zoomValueLabels[i]->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    m_zoomValueLabels[i]->setMinimumWidth(60);
    zoomRow->addWidget(m_zoomValueLabels[i]);
    viewColumn->addLayout(zoomRow);

    viewsRow->addLayout(viewColumn, 1);

    connect(m_sliceViews[i], &FusionSliceView::roiChanging, this,
            [this, i](const QVector3D &minEdges, const QVector3D &maxEdges) {
              onSliceViewROIChanging(i, minEdges, maxEdges);
            });
    connect(m_sliceViews[i], &FusionSliceView::roiChanged, this,
            [this, i](const QVector3D &minEdges, const QVector3D &maxEdges) {
              onSliceViewROIChanged(i, minEdges, maxEdges);
            });

    connect(m_sliceSliders[i], &QSlider::valueChanged, this,
            [this, i](int value) { onSliceSliderChanged(i, value); });
    connect(m_zoomSliders[i], &QSlider::valueChanged, this,
            [this, i](int value) { onZoomSliderChanged(i, value); });
    onZoomSliderChanged(i, m_zoomSliders[i]->value());
  }
  mainLayout->addLayout(viewsRow, 1);

  connect(m_primaryDisplayModeCombo,
          qOverload<int>(&QComboBox::currentIndexChanged), this,
          &FusionDialog::onPrimaryDisplayModeChanged);
  connect(m_secondaryDisplayModeCombo,
          qOverload<int>(&QComboBox::currentIndexChanged), this,
          &FusionDialog::onSecondaryDisplayModeChanged);
  if (m_transferSecondaryButton) {
    connect(m_transferSecondaryButton, &QPushButton::clicked, this,
            &FusionDialog::onSendSecondaryToViewer);
  }
  connect(m_secondaryStudyCombo,
          qOverload<int>(&QComboBox::currentIndexChanged), this,
          &FusionDialog::onSecondaryStudyChanged);
  connect(m_refreshButton, &QPushButton::clicked, this,
          &FusionDialog::onRefreshStudies);

  connect(m_primaryWindowSlider, &QSlider::valueChanged, this,
          &FusionDialog::onPrimaryWindowChanged);
  connect(m_primaryLevelSlider, &QSlider::valueChanged, this,
          &FusionDialog::onPrimaryLevelChanged);
  connect(m_secondaryWindowSlider, &QSlider::valueChanged, this,
          &FusionDialog::onSecondaryWindowChanged);
  connect(m_secondaryLevelSlider, &QSlider::valueChanged, this,
          &FusionDialog::onSecondaryLevelChanged);
  connect(m_opacitySlider, &QSlider::valueChanged, this,
          &FusionDialog::onOpacityChanged);

  updateTransferButtonState();
}

void FusionDialog::setupTransformControls(QGroupBox *group) {
  auto *outerLayout = new QHBoxLayout(group);
  outerLayout->setSpacing(20);

  auto *controlColumn = new QVBoxLayout();
  controlColumn->setSpacing(10);
  auto *grid = new QGridLayout();
  grid->setHorizontalSpacing(10);
  grid->setVerticalSpacing(6);
  const QString axes[3] = {QStringLiteral("X"), QStringLiteral("Y"),
                           QStringLiteral("Z")};
  for (int i = 0; i < 3; ++i) {
    auto *transLabel =
        new QLabel(tr("Δ%1 (mm)").arg(axes[i]), group);
    m_translationSpins[i] = new QDoubleSpinBox(group);
    m_translationSpins[i]->setRange(-200.0, 200.0);
    m_translationSpins[i]->setDecimals(2);
    m_translationSpins[i]->setSingleStep(1.0);
    m_translationSpins[i]->setAlignment(Qt::AlignRight);
    m_translationSpins[i]->setFixedWidth(76);
    grid->addWidget(transLabel, i, 0);
    grid->addWidget(m_translationSpins[i], i, 1);
    connect(m_translationSpins[i],
            qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            [this, i](double value) { onTranslationChanged(i, value); });

    auto *rotLabel = new QLabel(tr("R%1 (°)").arg(axes[i]), group);
    m_rotationSpins[i] = new QDoubleSpinBox(group);
    m_rotationSpins[i]->setRange(-180.0, 180.0);
    m_rotationSpins[i]->setDecimals(2);
    m_rotationSpins[i]->setSingleStep(1.0);
    m_rotationSpins[i]->setAlignment(Qt::AlignRight);
    m_rotationSpins[i]->setFixedWidth(76);
    grid->addWidget(rotLabel, i, 2);
    grid->addWidget(m_rotationSpins[i], i, 3);
    connect(m_rotationSpins[i], qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, [this, i](double value) { onRotationChanged(i, value); });
  }
  grid->setColumnStretch(1, 0);
  grid->setColumnStretch(3, 0);
  controlColumn->addLayout(grid);

  auto *buttonRow = new QHBoxLayout();
  buttonRow->setSpacing(8);
  auto *resetButton = new QPushButton(tr("リセット"), group);
  resetButton->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
  connect(resetButton, &QPushButton::clicked, this,
          &FusionDialog::onResetTransform);
  buttonRow->addWidget(resetButton);

  m_autoAlignButton = new QPushButton(tr("自動位置合わせ"), group);
  m_autoAlignButton->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
  buttonRow->addWidget(m_autoAlignButton);
  buttonRow->addStretch();
  controlColumn->addLayout(buttonRow);
  controlColumn->addStretch();
  outerLayout->addLayout(controlColumn, 1);
  outerLayout->addStretch();

  if (m_autoAlignButton) {
    connect(m_autoAlignButton, &QPushButton::clicked, this,
            &FusionDialog::onAutoAlign);
  }

  m_transferProgressTimer = new QTimer(this);
  m_transferProgressTimer->setInterval(100);
  connect(m_transferProgressTimer, &QTimer::timeout, this,
          &FusionDialog::updateTransferProgressBar);
  updateTransferProgressBar();
}

void FusionDialog::loadStudyList() {
  m_studyRecords.clear();
  if (!m_db.isOpen())
    return;

  const std::string sql =
      "SELECT studies.id, patients.patient_key, patients.name, studies.modality, "
      "studies.study_name, studies.study_date, studies.path, studies.frame_uid "
      "FROM studies INNER JOIN patients ON patients.patient_key = studies.patient_key "
      "WHERE studies.path IS NOT NULL AND studies.path != '' "
      "ORDER BY patients.name COLLATE NOCASE, studies.id;";

  m_db.query(sql, [&](int argc, char **argv, char **) {
    if (argc < 8)
      return;

    FusionStudyRecord rec;
    rec.id = std::atoi(argv[0] ? argv[0] : "0");
    rec.patientKey = argv[1] ? QString::fromUtf8(argv[1]) : QString();
    rec.patientName = argv[2] ? QString::fromUtf8(argv[2]).trimmed()
                              : QString();
    rec.modality = canonicalModality(argv[3] ? QString::fromUtf8(argv[3])
                                             : QString());
    if (!isImagingModality(rec.modality))
      return;
    rec.studyName = argv[4] ? QString::fromUtf8(argv[4]).trimmed() : QString();
    rec.studyDate = argv[5] ? QString::fromUtf8(argv[5]).trimmed() : QString();
    QString rawPath = argv[6] ? QString::fromUtf8(argv[6]).trimmed() : QString();
    if (rawPath.isEmpty())
      return;
    const QString nativeFree = QDir::fromNativeSeparators(rawPath);
    QDir pathDir(nativeFree);
    QString absolutePath =
        pathDir.isAbsolute() ? nativeFree : pathDir.absolutePath();
    QString cleanedPath = QDir::cleanPath(absolutePath);
    if (cleanedPath.isEmpty())
      return;
    rec.path = cleanedPath;
    if (!studyRecordLooksValid(rec))
      return;
    rec.normalizedPath = normalizedStudyPath(rec.path);
    QString rawFrame =
        argv[7] ? QString::fromUtf8(argv[7]).trimmed() : QString();
    rec.frameUid = rawFrame;
    rec.normalizedFrameUid = normalizedFrameUid(rec.frameUid);
    m_studyRecords.push_back(rec);
  });

  updatePrimaryPatientKey();
}

void FusionDialog::populateStudyCombos() {
  if (m_primaryStudyCombo) {
    QSignalBlocker blockPrimary(m_primaryStudyCombo);
    m_primaryStudyCombo->clear();
    if (m_primaryLoaded) {
      m_primaryStudyCombo->addItem(tr("メインウィンドウの画像"));
    } else {
      m_primaryStudyCombo->addItem(tr("-- Primary 未設定 --"));
    }
    m_primaryStudyCombo->setCurrentIndex(0);
    m_primaryStudyCombo->setEnabled(false);
  }

  if (!m_secondaryStudyCombo)
    return;

  const QString primaryIdKey = normalizedPatientKey(m_primaryPatientId);
  const QString primaryNameKey = normalizedPatientKey(m_primaryPatientName);
  const QString primaryKeyKey = normalizedPatientKey(m_primaryPatientKey);
  const QString primaryPathKey = normalizedStudyPath(m_primaryStudyPath);
  const QString primaryFrameKey = normalizedFrameUid(m_primaryFrameUid);
  QSet<QString> seenStudyKeys;
  if (!primaryPathKey.isEmpty()) {
    seenStudyKeys.insert(primaryPathKey);
    if (!primaryFrameKey.isEmpty())
      seenStudyKeys.insert(primaryPathKey + QStringLiteral("|") +
                           primaryFrameKey);
  }
  int matchCount = 0;
  {
    QSignalBlocker blockSecondary(m_secondaryStudyCombo);
    m_secondaryStudyCombo->clear();
    m_secondaryStudyCombo->addItem(tr("-- スタディを選択 --"), -1);
    for (int i = 0; i < static_cast<int>(m_studyRecords.size()); ++i) {
      const auto &rec = m_studyRecords[i];
      if (!recordMatchesPrimaryPatient(rec, primaryIdKey, primaryNameKey,
                                       primaryKeyKey))
        continue;
      if (recordIsPrimaryStudy(rec, primaryPathKey, primaryFrameKey))
        continue;
      const QString recordKey = studyRecordKey(rec);
      const QString recordPathKey =
          rec.normalizedPath.isEmpty() ? normalizedStudyPath(rec.path)
                                       : rec.normalizedPath;
      if (!recordKey.isEmpty() && seenStudyKeys.contains(recordKey))
        continue;
      if (!recordPathKey.isEmpty() && seenStudyKeys.contains(recordPathKey))
        continue;
      const QString label = formatStudyLabel(rec);
      m_secondaryStudyCombo->addItem(label, i);
      if (!recordKey.isEmpty())
        seenStudyKeys.insert(recordKey);
      if (!recordPathKey.isEmpty())
        seenStudyKeys.insert(recordPathKey);
      ++matchCount;
    }
    m_secondaryStudyCombo->setCurrentIndex(0);
  }

  m_secondaryStudyCombo->setEnabled(matchCount > 0);
  m_currentSecondaryStudy = -1;
  m_secondaryLoaded = false;
  m_secondaryVolume = DicomVolume();
  m_secondaryStudyPath.clear();
  m_secondaryModality.clear();
  if (m_secondaryInfoLabel)
    m_secondaryInfoLabel->setText(tr("患者: -\nスタディ: -\nパス: -"));
  updateCenterShift();
  updateTransferButtonState();
}

void FusionDialog::updatePrimaryInfoDisplay() {
  if (!m_primaryInfoLabel)
    return;

  QStringList lines;
  QString patientLine;
  if (m_primaryPatientName.isEmpty()) {
    patientLine = tr("患者: -");
  } else {
    patientLine = tr("患者: %1").arg(m_primaryPatientName);
  }
  if (!m_primaryPatientId.isEmpty()) {
    patientLine += tr(" (ID: %1)").arg(m_primaryPatientId);
  }
  lines << patientLine;

  QStringList studyParts;
  if (!m_primaryModality.isEmpty())
    studyParts << m_primaryModality;
  if (!m_primaryStudyDescription.isEmpty())
    studyParts << m_primaryStudyDescription;
  if (!m_primaryStudyDate.isEmpty())
    studyParts << m_primaryStudyDate;
  lines << tr("スタディ: %1")
               .arg(studyParts.isEmpty() ? tr("-")
                                        : studyParts.join(QStringLiteral(" / ")));

  lines << tr("パス: %1")
               .arg(m_primaryStudyPath.isEmpty() ? tr("-") : m_primaryStudyPath);

  m_primaryInfoLabel->setText(lines.join(QLatin1Char('\n')));
}

void FusionDialog::updatePrimaryPatientKey() {
  m_primaryPatientKey.clear();

  const QString primaryPathKey = normalizedStudyPath(m_primaryStudyPath);
  const QString primaryFrameKey = normalizedFrameUid(m_primaryFrameUid);

  auto matchesRecord = [&](const FusionStudyRecord &rec) {
    const QString recPathKey = rec.normalizedPath.isEmpty()
                                   ? normalizedStudyPath(rec.path)
                                   : rec.normalizedPath;
    if (!primaryPathKey.isEmpty() && !recPathKey.isEmpty() &&
        recPathKey == primaryPathKey) {
      m_primaryPatientKey = rec.patientKey;
      return true;
    }
    const QString recFrameKey = rec.normalizedFrameUid.isEmpty()
                                     ? normalizedFrameUid(rec.frameUid)
                                     : rec.normalizedFrameUid;
    if (!primaryFrameKey.isEmpty() && !recFrameKey.isEmpty() &&
        recFrameKey == primaryFrameKey) {
      m_primaryPatientKey = rec.patientKey;
      return true;
    }
    return false;
  };

  for (const auto &rec : m_studyRecords) {
    if (matchesRecord(rec))
      return;
  }

  if (!m_db.isOpen())
    return;

  if (!m_primaryStudyPath.trimmed().isEmpty()) {
    QString nativeFree = QDir::fromNativeSeparators(m_primaryStudyPath.trimmed());
    QDir dir(nativeFree);
    const QString absolutePath =
        dir.isAbsolute() ? nativeFree : dir.absolutePath();
    std::stringstream queryByPath;
    queryByPath << "SELECT patient_key FROM studies WHERE path='"
                << sqlEscape(absolutePath) << "' LIMIT 1;";
    m_db.query(queryByPath.str(), [&](int argc, char **argv, char **) {
      if (argc >= 1 && argv[0] && m_primaryPatientKey.isEmpty())
        m_primaryPatientKey = QString::fromUtf8(argv[0]).trimmed();
    });
    if (!m_primaryPatientKey.isEmpty())
      return;
  }

  if (!m_primaryFrameUid.trimmed().isEmpty()) {
    std::stringstream queryByFrame;
    queryByFrame << "SELECT patient_key FROM studies WHERE frame_uid='"
                 << sqlEscape(m_primaryFrameUid) << "' LIMIT 1;";
    m_db.query(queryByFrame.str(), [&](int argc, char **argv, char **) {
      if (argc >= 1 && argv[0] && m_primaryPatientKey.isEmpty())
        m_primaryPatientKey = QString::fromUtf8(argv[0]).trimmed();
    });
  }
}

QString FusionDialog::normalizedPatientKey(const QString &value) const {
  QString normalized;
  const QString trimmed = value.trimmed();
  normalized.reserve(trimmed.size());
  for (QChar ch : trimmed) {
    if (ch.isLetterOrNumber())
      normalized.append(ch.toUpper());
  }
  return normalized;
}

QString FusionDialog::normalizedStudyPath(const QString &path) const {
  const QString trimmed = path.trimmed();
  if (trimmed.isEmpty())
    return QString();

  const QString nativeFree = QDir::fromNativeSeparators(trimmed);
  QDir dir(nativeFree);
  QString absolute = dir.isAbsolute() ? nativeFree : dir.absolutePath();
  QString cleaned = QDir::cleanPath(absolute);
#if defined(Q_OS_WIN) || defined(Q_OS_DARWIN)
  return cleaned.toCaseFolded();
#else
  return cleaned;
#endif
}

QString FusionDialog::normalizedFrameUid(const QString &uid) const {
  QString trimmed = uid.trimmed();
  if (trimmed.isEmpty())
    return QString();
  return trimmed.toCaseFolded();
}

QString FusionDialog::studyRecordKey(const FusionStudyRecord &rec) const {
  const QString pathKey = rec.normalizedPath.isEmpty()
                              ? normalizedStudyPath(rec.path)
                              : rec.normalizedPath;
  const QString frameKey = rec.normalizedFrameUid.isEmpty()
                               ? normalizedFrameUid(rec.frameUid)
                               : rec.normalizedFrameUid;
  if (!pathKey.isEmpty() && !frameKey.isEmpty())
    return pathKey + QStringLiteral("|") + frameKey;
  if (!pathKey.isEmpty())
    return pathKey;
  return frameKey;
}

bool FusionDialog::studyDirectoryHasDicomFiles(const QString &path) const {
  QDir dir(path);
  if (!dir.exists())
    return false;

  const QFileInfoList fileInfos = dir.entryInfoList(
      QDir::Files | QDir::NoDotAndDotDot, QDir::Name | QDir::IgnoreCase);
  for (const QFileInfo &info : fileInfos) {
    DicomReader reader;
    if (reader.loadDicomFile(info.absoluteFilePath()))
      return true;
  }

  return false;
}

bool FusionDialog::studyRecordLooksValid(const FusionStudyRecord &rec) const {
  if (!rec.modality.startsWith(QLatin1String("Fusion/")))
    return studyDirectoryHasDicomFiles(rec.path);

  QDir dir(rec.path);
  if (!dir.exists())
    return false;

  const QString metaPath = dir.filePath(QStringLiteral("fusion_meta.json"));
  if (!QFile::exists(metaPath))
    return false;

  QFile metaFile(metaPath);
  if (!metaFile.open(QIODevice::ReadOnly))
    return false;

  const QByteArray metaData = metaFile.readAll();
  QJsonParseError parseError{};
  const QJsonDocument doc = QJsonDocument::fromJson(metaData, &parseError);
  if (parseError.error != QJsonParseError::NoError || !doc.isObject())
    return false;

  QString volumeFile =
      doc.object().value(QStringLiteral("volume_file")).toString().trimmed();
  if (volumeFile.isEmpty())
    volumeFile = QStringLiteral("fusion_volume.bin");
  const QString volumePath = dir.filePath(volumeFile);
  return QFile::exists(volumePath);
}

bool FusionDialog::recordMatchesPrimaryPatient(const FusionStudyRecord &rec,
                                               const QString &primaryIdKey,
                                               const QString &primaryNameKey,
                                               const QString &primaryKeyKey) const {
  if (primaryIdKey.isEmpty() && primaryNameKey.isEmpty() &&
      primaryKeyKey.isEmpty())
    return true;

  const QString recIdKey = normalizedPatientKey(rec.patientKey);
  const QString recNameKey = normalizedPatientKey(rec.patientName);

  if (!primaryKeyKey.isEmpty() && !recIdKey.isEmpty() &&
      recIdKey == primaryKeyKey)
    return true;

  if (!primaryIdKey.isEmpty()) {
    if (!recIdKey.isEmpty() && recIdKey == primaryIdKey)
      return true;
    if (!recNameKey.isEmpty() && recNameKey == primaryIdKey)
      return true;
  }

  if (!primaryNameKey.isEmpty()) {
    if (!recIdKey.isEmpty() && recIdKey == primaryNameKey)
      return true;
    if (!recNameKey.isEmpty() && recNameKey == primaryNameKey)
      return true;
  }

  return false;
}

bool FusionDialog::recordIsPrimaryStudy(const FusionStudyRecord &rec,
                                        const QString &primaryPathKey,
                                        const QString &primaryFrameUid) const {
  if (primaryPathKey.isEmpty() && primaryFrameUid.isEmpty())
    return false;

  const QString recPathKey =
      rec.normalizedPath.isEmpty() ? normalizedStudyPath(rec.path)
                                   : rec.normalizedPath;
  if (!primaryPathKey.isEmpty() && !recPathKey.isEmpty() &&
      recPathKey == primaryPathKey)
    return true;

  const QString recFrameUid = rec.normalizedFrameUid.isEmpty()
                                  ? normalizedFrameUid(rec.frameUid)
                                  : rec.normalizedFrameUid;
  if (!primaryFrameUid.isEmpty() && !recFrameUid.isEmpty() &&
      recFrameUid == primaryFrameUid)
    return true;

  return false;
}

void FusionDialog::selectSecondaryForPrimaryPatient() {
  if (!m_secondaryStudyCombo)
    return;

  const QString primaryIdKey = normalizedPatientKey(m_primaryPatientId);
  const QString primaryNameKey = normalizedPatientKey(m_primaryPatientName);
  const QString primaryKeyKey = normalizedPatientKey(m_primaryPatientKey);

  int matchedComboIndex = -1;
  for (int comboIndex = 1; comboIndex < m_secondaryStudyCombo->count();
       ++comboIndex) {
    const QVariant comboData = m_secondaryStudyCombo->itemData(comboIndex);
    if (!comboData.isValid())
      continue;
    const int recordIndex = comboData.toInt();
    if (recordIndex < 0 ||
        recordIndex >= static_cast<int>(m_studyRecords.size()))
      continue;
    const auto &rec = m_studyRecords[recordIndex];
    if (recordMatchesPrimaryPatient(rec, primaryIdKey, primaryNameKey,
                                    primaryKeyKey)) {
      matchedComboIndex = comboIndex;
      break;
    }
  }

  if (matchedComboIndex < 0) {
    matchedComboIndex = m_secondaryStudyCombo->count() > 1 ? 1 : 0;
  }

  if (matchedComboIndex >= 0 &&
      matchedComboIndex < m_secondaryStudyCombo->count()) {
    m_secondaryStudyCombo->setCurrentIndex(matchedComboIndex);
  }
}

bool FusionDialog::loadVolumeFromStudy(int studyIndex, DicomVolume &volume,
                                       bool &loadedFlag,
                                       int &currentStudyIndex,
                                       QLabel *infoLabel) {
  if (studyIndex < 0 || studyIndex >= static_cast<int>(m_studyRecords.size()))
    return false;

  const FusionStudyRecord &rec = m_studyRecords[studyIndex];
  QDir dir(rec.path);
  if (!dir.exists()) {
    QMessageBox::warning(this, tr("読み込みエラー"),
                         tr("ディレクトリが存在しません: %1").arg(rec.path));
    return false;
  }

  QApplication::setOverrideCursor(Qt::WaitCursor);
  double storedWindow = std::numeric_limits<double>::quiet_NaN();
  double storedLevel = std::numeric_limits<double>::quiet_NaN();
  QStringList fusionInfoLines;
  bool ok = false;
  if (rec.modality.startsWith(QLatin1String("Fusion/"))) {
    ok = loadFusionVolume(rec, volume, &storedWindow, &storedLevel,
                          &fusionInfoLines);
  } else {
    ok = volume.loadFromDirectory(rec.path);
  }
  QApplication::restoreOverrideCursor();

  const bool showGenericError =
      !rec.modality.startsWith(QLatin1String("Fusion/"));

  if (!ok) {
    if (showGenericError) {
      QMessageBox::warning(this, tr("読み込みエラー"),
                           tr("ボリュームの読み込みに失敗しました: %1").arg(rec.path));
    }
    loadedFlag = false;
    return false;
  }

  loadedFlag = true;
  currentStudyIndex = studyIndex;

  if (!fusionInfoLines.isEmpty()) {
    infoLabel->setText(fusionInfoLines.join(QLatin1Char('\n')));
  } else {
    QStringList lines;
    lines << tr("患者: %1")
                 .arg(rec.patientName.isEmpty() ? tr("(不明)") : rec.patientName);
    QString studyLine = rec.modality;
    if (!rec.studyName.isEmpty())
      studyLine += QStringLiteral(" / %1").arg(rec.studyName);
    if (!rec.studyDate.isEmpty())
      studyLine += QStringLiteral(" (%1)").arg(rec.studyDate);
    lines << tr("スタディ: %1").arg(studyLine);
    lines << tr("パス: %1").arg(rec.path);
    infoLabel->setText(lines.join(QLatin1Char('\n')));
  }

  auto applyWindowLevel = [&](double window, double level) {
    if (!std::isfinite(window) || !std::isfinite(level))
      return;
    if (infoLabel == m_primaryInfoLabel) {
      m_primaryWindow = window;
      m_primaryLevel = level;
    } else if (infoLabel == m_secondaryInfoLabel) {
      m_secondaryWindow = window;
      m_secondaryLevel = level;
    }
  };
  applyWindowLevel(storedWindow, storedLevel);
  return true;
}

void FusionDialog::updateSliceRanges() {
  if (!m_primaryLoaded || m_primaryVolume.depth() <= 0 ||
      m_primaryVolume.width() <= 0 || m_primaryVolume.height() <= 0) {
    for (int i = 0; i < 3; ++i) {
      if (m_sliceSliders[i]) {
        m_sliceSliders[i]->setEnabled(false);
        m_sliceSliders[i]->setRange(0, 0);
        m_sliceSliders[i]->setValue(0);
      }
      if (m_sliceValueLabels[i])
        m_sliceValueLabels[i]->setText(tr("-"));
      if (m_zoomSliders[i])
        m_zoomSliders[i]->setEnabled(false);
    }
    return;
  }

  const int axialMax = m_primaryVolume.depth();
  const int sagittalMax = m_primaryVolume.width();
  const int coronalMax = m_primaryVolume.height();

  struct RangeInfo {
    int sliderIndex;
    int maxValue;
  } ranges[] = {{0, axialMax}, {1, sagittalMax}, {2, coronalMax}};

  for (const auto &range : ranges) {
    QSlider *slider = m_sliceSliders[range.sliderIndex];
    QLabel *label = m_sliceValueLabels[range.sliderIndex];
    if (!slider || !label)
      continue;
    if (range.maxValue <= 0) {
      slider->setEnabled(false);
      slider->setRange(0, 0);
      slider->setValue(0);
      label->setText(tr("-"));
      if (m_zoomSliders[range.sliderIndex])
        m_zoomSliders[range.sliderIndex]->setEnabled(false);
    } else {
      slider->setEnabled(true);
      slider->setRange(0, range.maxValue - 1);
      slider->setValue(range.maxValue / 2);
      label->setText(tr("%1 / %2").arg(slider->value()).arg(range.maxValue - 1));
      if (m_zoomSliders[range.sliderIndex])
        m_zoomSliders[range.sliderIndex]->setEnabled(true);
    }
  }
}

void FusionDialog::updateCenterShift() {
  if (!(m_primaryLoaded && m_secondaryLoaded) ||
      m_primaryVolume.depth() <= 0 || m_secondaryVolume.depth() <= 0) {
    m_centerShift = QVector3D(0.0, 0.0, 0.0);
    return;
  }

  const QVector3D primaryCenter = computeVolumeCenter(m_primaryVolume);
  const QVector3D secondaryCenter = computeVolumeCenter(m_secondaryVolume);
  m_centerShift = primaryCenter - secondaryCenter;
}

void FusionDialog::updateImages() {
  const std::array<DicomVolume::Orientation, 3> orientations = {
      DicomVolume::Orientation::Axial, DicomVolume::Orientation::Sagittal,
      DicomVolume::Orientation::Coronal};

  if (m_viewer)
    m_viewer->clearFusionPreviewImage();

  if (!m_primaryLoaded || m_primaryVolume.depth() <= 0) {
    for (int i = 0; i < 3; ++i) {
      FusionSliceView *view = m_sliceViews[i];
      if (!view)
        continue;
      view->setZoomFactor(m_zoomFactors[i]);
      view->setPixelSpacing(1.0, 1.0);
      view->setBaseImage(QImage());
      view->setOverlayImage(QImage());
      view->setMessage(tr("メインウィンドウでボリュームを開いてください"));
    }
    return;
  }

  for (int i = 0; i < 3; ++i) {
    const auto orientation = orientations[i];
    const int sliceIndex = m_sliceSliders[i] ? m_sliceSliders[i]->value() : 0;
    cv::Mat primarySlice = extractPrimarySlice(sliceIndex, orientation);
    QImage baseImage = createBaseImage(primarySlice);
    if (m_sliceViews[i]) {
      m_sliceViews[i]->setOrientation(orientation);
      if (m_primaryLoaded) {
        m_sliceViews[i]->setVolumeDimensions(m_primaryVolume.width(),
                                             m_primaryVolume.height(),
                                             m_primaryVolume.depth());
      } else {
        m_sliceViews[i]->setVolumeDimensions(0, 0, 0);
      }
      m_sliceViews[i]->setSliceIndex(sliceIndex);
      if (roiEnabled()) {
        m_sliceViews[i]->setROIEnabled(true);
        m_sliceViews[i]->setROI(m_roiMinEdges, m_roiMaxEdges);
      } else {
        m_sliceViews[i]->setROIEnabled(false);
      }
      const auto spacing = viewPixelSpacing(orientation);
      m_sliceViews[i]->setZoomFactor(m_zoomFactors[i]);
      m_sliceViews[i]->setPixelSpacing(spacing[0], spacing[1]);
      m_sliceViews[i]->setBaseImage(baseImage);
    }

    QImage overlayImage;
    if (m_secondaryLoaded && m_secondaryVolume.depth() > 0) {
      cv::Mat secondarySlice =
          resampleSecondarySlice(sliceIndex, orientation);
      if (m_sliceViews[i]) {
        overlayImage = createOverlayImage(secondarySlice);
        m_sliceViews[i]->setOverlayImage(overlayImage);
        m_sliceViews[i]->setMessage(QString());
      }
    } else if (m_sliceViews[i]) {
      m_sliceViews[i]->setOverlayImage(QImage());
      m_sliceViews[i]->setMessage(tr("Secondaryボリューム未選択"));
    }

  }
}

void FusionDialog::updateStatusLabels() {
  const QString primaryUid =
      m_primaryLoaded ? m_primaryVolume.frameOfReferenceUID() : QString();
  const QString secondaryUid =
      m_secondaryLoaded ? m_secondaryVolume.frameOfReferenceUID() : QString();
  if (m_autoAlignButton)
    m_autoAlignButton->setEnabled(m_primaryLoaded && m_secondaryLoaded);
  if (m_primaryUidLabel) {
    m_primaryUidLabel->setText(
        tr("Frame UID: %1")
            .arg(primaryUid.isEmpty() ? QStringLiteral("-") : primaryUid));
  }
  if (m_secondaryUidLabel) {
    m_secondaryUidLabel->setText(
        tr("Frame UID: %1")
            .arg(secondaryUid.isEmpty() ? QStringLiteral("-") : secondaryUid));
  }

  updateTransferButtonState();
}

void FusionDialog::updateTransferButtonState() {
  if (!m_transferSecondaryButton)
    return;

  const bool enabled =
      m_viewer && m_primaryLoaded && m_primaryVolume.depth() > 0 &&
      m_secondaryLoaded && m_secondaryVolume.depth() > 0 &&
      !m_secondaryStudyPath.trimmed().isEmpty() && !m_transferInProgress;
  m_transferSecondaryButton->setEnabled(enabled);
}

void FusionDialog::startTransferProgress(int totalSteps) {
  m_transferProgressTotal = std::max(0, totalSteps);
  m_transferProgressCounter.store(0, std::memory_order_relaxed);
  if (m_transferProgressBar) {
    if (m_transferProgressTotal > 0) {
      m_transferProgressBar->setRange(0, m_transferProgressTotal);
      m_transferProgressBar->setValue(0);
    } else {
      m_transferProgressBar->setRange(0, 0);
    }
    m_transferProgressBar->setVisible(true);
  }
  updateTransferProgressBar();
  if (m_transferProgressTimer)
    m_transferProgressTimer->start();
}

void FusionDialog::stopTransferProgress() {
  if (m_transferProgressTimer && m_transferProgressTimer->isActive())
    m_transferProgressTimer->stop();
  if (m_transferProgressBar) {
    if (m_transferProgressTotal > 0) {
      m_transferProgressBar->setRange(0, m_transferProgressTotal);
      m_transferProgressBar->setValue(m_transferProgressTotal);
    } else {
      m_transferProgressBar->setRange(0, 0);
    }
    m_transferProgressBar->setVisible(false);
  }
  m_transferProgressCounter.store(0, std::memory_order_relaxed);
  m_transferProgressTotal = 0;
}

void FusionDialog::updateTransferProgressBar() {
  if (!m_transferProgressBar)
    return;

  const int total = m_transferProgressTotal;
  const int current =
      m_transferProgressCounter.load(std::memory_order_relaxed);

  if (total > 0) {
    const int clamped = std::clamp(current, 0, total);
    m_transferProgressBar->setRange(0, total);
    m_transferProgressBar->setValue(clamped);
  } else {
    m_transferProgressBar->setRange(0, 0);
  }
}

bool FusionDialog::roiEnabled() const {
  return m_roiActive && m_primaryLoaded && m_primaryVolume.width() > 0 &&
         m_primaryVolume.height() > 0 && m_primaryVolume.depth() > 0;
}

bool FusionDialog::sanitizeROIEdges(QVector3D &minEdges,
                                    QVector3D &maxEdges) const {
  if (!m_primaryLoaded || m_primaryVolume.width() <= 0 ||
      m_primaryVolume.height() <= 0 || m_primaryVolume.depth() <= 0)
    return false;

  const double dims[3] = {static_cast<double>(m_primaryVolume.width()),
                          static_cast<double>(m_primaryVolume.height()),
                          static_cast<double>(m_primaryVolume.depth())};
  const double minThickness = 2.0;

  for (int axis = 0; axis < 3; ++axis) {
    double minVal = static_cast<double>(minEdges[axis]);
    double maxVal = static_cast<double>(maxEdges[axis]);
    if (minVal > maxVal)
      std::swap(minVal, maxVal);
    minVal = std::clamp(minVal, 0.0, dims[axis]);
    maxVal = std::clamp(maxVal, 0.0, dims[axis]);
    if (maxVal - minVal < minThickness) {
      const double desired = std::min(minThickness, dims[axis]);
      double center = (minVal + maxVal) * 0.5;
      minVal = center - desired * 0.5;
      maxVal = center + desired * 0.5;
      if (minVal < 0.0) {
        maxVal -= minVal;
        minVal = 0.0;
      }
      if (maxVal > dims[axis]) {
        const double diff = maxVal - dims[axis];
        maxVal = dims[axis];
        minVal = std::max(0.0, minVal - diff);
      }
      if (maxVal - minVal < desired * 0.5) {
        minVal = 0.0;
        maxVal = dims[axis];
      }
    }
    minEdges[axis] = static_cast<float>(minVal);
    maxEdges[axis] = static_cast<float>(maxVal);
  }
  return true;
}

void FusionDialog::propagateROIToViews() {
  const bool enabled = roiEnabled();
  for (int i = 0; i < 3; ++i) {
    if (!m_sliceViews[i])
      continue;
    m_sliceViews[i]->setROIEnabled(enabled);
    if (enabled)
      m_sliceViews[i]->setROI(m_roiMinEdges, m_roiMaxEdges);
  }
}

void FusionDialog::applyROIEdges(const QVector3D &minEdges,
                                 const QVector3D &maxEdges, bool finalChange) {
  QVector3D sanitizedMin = minEdges;
  QVector3D sanitizedMax = maxEdges;
  if (!sanitizeROIEdges(sanitizedMin, sanitizedMax)) {
    disableROI();
    return;
  }

  auto nearlyEqual = [](double a, double b) {
    return std::abs(a - b) < 1e-3;
  };

  if (roiEnabled() && nearlyEqual(m_roiMinEdges.x(), sanitizedMin.x()) &&
      nearlyEqual(m_roiMinEdges.y(), sanitizedMin.y()) &&
      nearlyEqual(m_roiMinEdges.z(), sanitizedMin.z()) &&
      nearlyEqual(m_roiMaxEdges.x(), sanitizedMax.x()) &&
      nearlyEqual(m_roiMaxEdges.y(), sanitizedMax.y()) &&
      nearlyEqual(m_roiMaxEdges.z(), sanitizedMax.z())) {
    return;
  }

  m_roiMinEdges = sanitizedMin;
  m_roiMaxEdges = sanitizedMax;
  m_roiActive = true;
  propagateROIToViews();

  if (finalChange)
    updateImages();
}

void FusionDialog::initializeDefaultROI() {
  if (!m_primaryLoaded || m_primaryVolume.width() <= 0 ||
      m_primaryVolume.height() <= 0 || m_primaryVolume.depth() <= 0) {
    disableROI();
    return;
  }

  const double width = static_cast<double>(m_primaryVolume.width());
  const double height = static_cast<double>(m_primaryVolume.height());
  const double depth = static_cast<double>(m_primaryVolume.depth());
  const double marginRatio = 0.2;

  QVector3D minEdges(static_cast<float>(width * marginRatio),
                     static_cast<float>(height * marginRatio),
                     static_cast<float>(depth * marginRatio));
  QVector3D maxEdges(static_cast<float>(width * (1.0 - marginRatio)),
                     static_cast<float>(height * (1.0 - marginRatio)),
                     static_cast<float>(depth * (1.0 - marginRatio)));

  applyROIEdges(minEdges, maxEdges, false);
  propagateROIToViews();
}

void FusionDialog::disableROI() {
  m_roiActive = false;
  m_roiMinEdges = QVector3D(0.0f, 0.0f, 0.0f);
  m_roiMaxEdges = QVector3D(0.0f, 0.0f, 0.0f);
  for (int i = 0; i < 3; ++i) {
    if (m_sliceViews[i])
      m_sliceViews[i]->setROIEnabled(false);
  }
}

QImage FusionDialog::createBaseImage(const cv::Mat &slice) const {
  if (slice.empty())
    return QImage();

  const double minVal = m_primaryLevel - m_primaryWindow / 2.0;
  const double maxVal = m_primaryLevel + m_primaryWindow / 2.0;
  const double scale = (maxVal - minVal) != 0.0 ? 255.0 / (maxVal - minVal)
                                               : 0.0;
  const double shift = -minVal * scale;

  cv::Mat normalized8U;
  if (scale != 0.0) {
    slice.convertTo(normalized8U, CV_8U, scale, shift);
  } else {
    normalized8U =
        cv::Mat(slice.size(), CV_8U, cv::Scalar(static_cast<uchar>(128)));
  }

  switch (m_primaryDisplayMode) {
  case PrimaryDisplayMode::Grayscale: {
    return QImage(normalized8U.data, normalized8U.cols, normalized8U.rows,
                  normalized8U.step, QImage::Format_Grayscale8)
        .copy();
  }
  case PrimaryDisplayMode::Inverted: {
    cv::Mat inverted;
    cv::subtract(cv::Scalar::all(255), normalized8U, inverted);
    return QImage(inverted.data, inverted.cols, inverted.rows, inverted.step,
                  QImage::Format_Grayscale8)
        .copy();
  }
  case PrimaryDisplayMode::BoneHighlight: {
    cv::Mat colored;
    cv::applyColorMap(normalized8U, colored, cv::COLORMAP_BONE);
    cv::Mat coloredRgb;
    cv::cvtColor(colored, coloredRgb, cv::COLOR_BGR2RGB);
    return QImage(coloredRgb.data, coloredRgb.cols, coloredRgb.rows,
                  coloredRgb.step, QImage::Format_RGB888)
        .copy();
  }
  }
  return QImage();
}

QImage FusionDialog::createOverlayImage(const cv::Mat &slice) const {
  if (slice.empty())
    return QImage();

  QImage overlay(slice.cols, slice.rows, QImage::Format_ARGB32);
  overlay.fill(Qt::transparent);
  const double minVal = m_secondaryLevel - m_secondaryWindow / 2.0;
  const double maxVal = m_secondaryLevel + m_secondaryWindow / 2.0;
  const double invRange = (maxVal - minVal) != 0.0 ? 1.0 / (maxVal - minVal)
                                                   : 0.0;

  cv::Mat floatSlice;
  if (slice.depth() == CV_32F) {
    floatSlice = slice;
  } else {
    slice.convertTo(floatSlice, CV_32F);
  }

  if (m_secondaryDisplayMode == SecondaryDisplayMode::EdgeHighlight) {
    cv::Mat gradX;
    cv::Mat gradY;
    cv::Sobel(floatSlice, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(floatSlice, gradY, CV_32F, 0, 1, 3);
    cv::Mat magnitude;
    cv::magnitude(gradX, gradY, magnitude);
    double maxMag = 0.0;
    cv::minMaxLoc(magnitude, nullptr, &maxMag);
    const double invMag = maxMag > 0.0 ? 1.0 / maxMag : 0.0;

    for (int y = 0; y < magnitude.rows; ++y) {
      const float *magPtr = magnitude.ptr<float>(y);
      QRgb *dst = reinterpret_cast<QRgb *>(overlay.scanLine(y));
      for (int x = 0; x < magnitude.cols; ++x) {
        double norm = magPtr[x] * invMag;
        norm = std::clamp(norm, 0.0, 1.0);
        norm = std::sqrt(norm);
        const int alpha =
            static_cast<int>(norm * m_overlayOpacity * 255.0);
        const int green = static_cast<int>(std::clamp(norm * 255.0, 0.0, 255.0));
        dst[x] = qRgba(255, green, 0, alpha);
      }
    }
    return overlay;
  }

  for (int y = 0; y < floatSlice.rows; ++y) {
    const float *rowPtr = floatSlice.ptr<float>(y);
    QRgb *dst = reinterpret_cast<QRgb *>(overlay.scanLine(y));
    for (int x = 0; x < floatSlice.cols; ++x) {
      double norm = (rowPtr[x] - minVal) * invRange;
      norm = std::clamp(norm, 0.0, 1.0);
      const int alpha = static_cast<int>(norm * m_overlayOpacity * 255.0);
      const int intensity = static_cast<int>(norm * 255.0);
      switch (m_secondaryDisplayMode) {
      case SecondaryDisplayMode::RedOverlay:
        dst[x] = qRgba(intensity, 0, 0, alpha);
        break;
      case SecondaryDisplayMode::CyanOverlay:
        dst[x] = qRgba(0, intensity, intensity, alpha);
        break;
      case SecondaryDisplayMode::EdgeHighlight:
        dst[x] = qRgba(intensity, 0, 0, alpha);
        break;
      }
    }
  }
  return overlay;
}

cv::Mat FusionDialog::extractPrimarySlice(
    int sliceIndex, DicomVolume::Orientation orientation) const {
  switch (orientation) {
  case DicomVolume::Orientation::Axial:
    return getSliceAxial(m_primaryVolume.data(), sliceIndex);
  case DicomVolume::Orientation::Sagittal:
    return getSliceSagittal(m_primaryVolume.data(), sliceIndex);
  case DicomVolume::Orientation::Coronal:
    return getSliceCoronal(m_primaryVolume.data(), sliceIndex);
  }
  return cv::Mat();
}

cv::Mat FusionDialog::resampleSecondarySlice(
    int sliceIndex, DicomVolume::Orientation orientation) const {
  if (!m_secondaryLoaded)
    return cv::Mat();

  const int primaryWidth = m_primaryVolume.width();
  const int primaryHeight = m_primaryVolume.height();
  const int primaryDepth = m_primaryVolume.depth();

  int outCols = 0;
  int outRows = 0;

  switch (orientation) {
  case DicomVolume::Orientation::Axial:
    outCols = primaryWidth;
    outRows = primaryHeight;
    break;
  case DicomVolume::Orientation::Sagittal:
    outCols = primaryHeight;
    outRows = primaryDepth;
    break;
  case DicomVolume::Orientation::Coronal:
    outCols = primaryWidth;
    outRows = primaryDepth;
    break;
  }

  if (outCols <= 0 || outRows <= 0)
    return cv::Mat();

  cv::Mat resampled(outRows, outCols, CV_32FC1, cv::Scalar(0));

  const cv::Mat &secondaryVolume = m_secondaryVolume.data();
  if (secondaryVolume.empty())
    return resampled;

  const int secondaryWidth = m_secondaryVolume.width();
  const int secondaryHeight = m_secondaryVolume.height();
  const int secondaryDepth = m_secondaryVolume.depth();
  const int volumeType = secondaryVolume.type();

  const QVector3D secondaryCenter = computeVolumeCenter(m_secondaryVolume);
  const QVector3D translation = m_centerShift + m_manualTranslation;
  const QQuaternion invRotation = m_rotationQuat.conjugated();
  const double sliceCoord = sliceIndex + 0.5;
  const int depthLast = primaryDepth - 1;

  auto transformPoint = [&](const QVector3D &primaryPoint) {
    const QVector3D relative = primaryPoint - secondaryCenter - translation;
    const QVector3D rotated = invRotation.rotatedVector(relative);
    return rotated + secondaryCenter;
  };

  auto sampleSecondary = [&](double x, double y, double z) -> double {
    if (x < 0.0 || y < 0.0 || z < 0.0 || x > secondaryWidth - 1 ||
        y > secondaryHeight - 1 || z > secondaryDepth - 1)
      return 0.0;

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int z0 = static_cast<int>(std::floor(z));
    const int x1 = std::min(x0 + 1, secondaryWidth - 1);
    const int y1 = std::min(y0 + 1, secondaryHeight - 1);
    const int z1 = std::min(z0 + 1, secondaryDepth - 1);

    const double fx = x - x0;
    const double fy = y - y0;
    const double fz = z - z0;

    switch (volumeType) {
    case CV_16SC1:
      return trilinearSample<short>(secondaryVolume, secondaryWidth,
                                    secondaryHeight, x0, x1, y0, y1, z0, z1,
                                    fx, fy, fz);
    case CV_32FC1:
      return trilinearSample<float>(secondaryVolume, secondaryWidth,
                                    secondaryHeight, x0, x1, y0, y1, z0, z1,
                                    fx, fy, fz);
    case CV_8UC1:
      return trilinearSample<uchar>(secondaryVolume, secondaryWidth,
                                    secondaryHeight, x0, x1, y0, y1, z0, z1,
                                    fx, fy, fz);
    default:
      break;
    }
    return 0.0;
  };

  runParallelFor(cv::Range(0, outRows), [&](const cv::Range &range) {
    for (int y = range.start; y < range.end; ++y) {
      float *dstRow = resampled.ptr<float>(y);
      for (int x = 0; x < outCols; ++x) {
        double vx = 0.0;
        double vy = 0.0;
        double vz = 0.0;

        switch (orientation) {
        case DicomVolume::Orientation::Axial:
          vx = x + 0.5;
          vy = y + 0.5;
          vz = sliceCoord;
          break;
        case DicomVolume::Orientation::Sagittal:
          vx = sliceCoord;
          vy = x + 0.5;
          vz = (depthLast - y) + 0.5;
          break;
        case DicomVolume::Orientation::Coronal:
          vx = x + 0.5;
          vy = sliceCoord;
          vz = (depthLast - y) + 0.5;
          break;
        }

        const QVector3D primaryPoint =
            m_primaryVolume.voxelToPatient(vx, vy, vz);
        const QVector3D secondaryPoint = transformPoint(primaryPoint);
        const QVector3D voxelCoord =
            m_secondaryVolume.patientToVoxelContinuous(secondaryPoint);
        dstRow[x] = static_cast<float>(
            sampleSecondary(voxelCoord.x(), voxelCoord.y(), voxelCoord.z()));
      }
    }
  });

  return resampled;
}

cv::Mat FusionDialog::resampleSecondaryVolumeToPrimary() const {
  TransferJobInput input;
  input.primaryVolume = m_primaryVolume;
  input.secondaryVolume = m_secondaryVolume;
  input.primaryLoaded = m_primaryLoaded;
  input.secondaryLoaded = m_secondaryLoaded;
  input.centerShift = m_centerShift;
  input.manualTranslation = m_manualTranslation;
  input.rotationQuat = m_rotationQuat;
  return resampleSecondaryVolumeToPrimaryImpl(input);
}

double FusionDialog::sampleVolumeValue(const cv::Mat &volume, double x,
                                       double y, double z) const {
  return sampleVolumeValueImpl(volume, x, y, z);
}

QVector3D FusionDialog::sampleVolumeGradient(const cv::Mat &volume, double x,
                                             double y, double z) const {
  if (volume.empty())
    return QVector3D(0.0f, 0.0f, 0.0f);

  const double gx = sampleVolumeValue(volume, x + 1.0, y, z) -
                    sampleVolumeValue(volume, x - 1.0, y, z);
  const double gy = sampleVolumeValue(volume, x, y + 1.0, z) -
                    sampleVolumeValue(volume, x, y - 1.0, z);
  const double gz = sampleVolumeValue(volume, x, y, z + 1.0) -
                    sampleVolumeValue(volume, x, y, z - 1.0);

  return QVector3D(static_cast<float>(gx), static_cast<float>(gy),
                   static_cast<float>(gz));
}

std::vector<FusionDialog::SamplePoint>
FusionDialog::generatePrimarySamples(int maxSamples) const {
  std::vector<SamplePoint> samples;
  if (!m_primaryLoaded)
    return samples;

  const int width = m_primaryVolume.width();
  const int height = m_primaryVolume.height();
  const int depth = m_primaryVolume.depth();
  if (width <= 2 || height <= 2 || depth <= 2)
    return samples;

  auto computeIndexRange = [](double minEdge, double maxEdge, int size) {
    int minIdx = std::max(1, static_cast<int>(std::ceil(minEdge - 0.5)));
    int maxIdx = std::min(size - 2, static_cast<int>(std::floor(maxEdge - 0.5)));
    if (minIdx > maxIdx) {
      minIdx = 1;
      maxIdx = size - 2;
    }
    return std::pair<int, int>(minIdx, maxIdx);
  };

  int minX = 1;
  int maxX = width - 2;
  int minY = 1;
  int maxY = height - 2;
  int minZ = 1;
  int maxZ = depth - 2;
  if (roiEnabled()) {
    std::tie(minX, maxX) =
        computeIndexRange(m_roiMinEdges.x(), m_roiMaxEdges.x(), width);
    std::tie(minY, maxY) =
        computeIndexRange(m_roiMinEdges.y(), m_roiMaxEdges.y(), height);
    std::tie(minZ, maxZ) =
        computeIndexRange(m_roiMinEdges.z(), m_roiMaxEdges.z(), depth);
  }

  const int target = std::max(400, std::min(maxSamples, 8000));
  samples.reserve(target);

  std::mt19937 rng(0xC0FFEEu);
  std::uniform_int_distribution<int> distX(minX, maxX);
  std::uniform_int_distribution<int> distY(minY, maxY);
  std::uniform_int_distribution<int> distZ(minZ, maxZ);
  std::uniform_real_distribution<double> dist01(0.0, 1.0);

  const cv::Mat &volume = m_primaryVolume.data();
  const int maxAttempts = std::max(target * 20, target * 4);
  int attempts = 0;
  const double gradientThreshold = 25.0;

  while (samples.size() < static_cast<size_t>(target) &&
         attempts < maxAttempts) {
    ++attempts;
    const int x = distX(rng);
    const int y = distY(rng);
    const int z = distZ(rng);

    const double value = sampleVolumeValue(volume, x, y, z);
    const double gx = sampleVolumeValue(volume, x + 1, y, z) -
                      sampleVolumeValue(volume, x - 1, y, z);
    const double gy = sampleVolumeValue(volume, x, y + 1, z) -
                      sampleVolumeValue(volume, x, y - 1, z);
    const double gz = sampleVolumeValue(volume, x, y, z + 1) -
                      sampleVolumeValue(volume, x, y, z - 1);
    const double gradMag = std::sqrt(gx * gx + gy * gy + gz * gz);
    if (gradMag < gradientThreshold && dist01(rng) > 0.15)
      continue;

    SamplePoint point;
    point.voxel = QVector3D(static_cast<float>(x) + 0.5f,
                            static_cast<float>(y) + 0.5f,
                            static_cast<float>(z) + 0.5f);
    point.patient =
        m_primaryVolume.voxelToPatient(x + 0.5, y + 0.5, z + 0.5);
    point.primaryValue = value;
    point.primaryGradient =
        QVector3D(static_cast<float>(gx), static_cast<float>(gy),
                   static_cast<float>(gz));
    point.gradientMagnitude = gradMag;
    samples.push_back(point);
  }

  if (samples.size() < 200) {
    samples.clear();
    const int stepX = std::max(1, width / 16);
    const int stepY = std::max(1, height / 16);
    const int stepZ = std::max(1, depth / 16);
    for (int z = minZ;
         z <= maxZ && samples.size() < static_cast<size_t>(target);
         z += stepZ) {
      for (int y = minY;
           y <= maxY && samples.size() < static_cast<size_t>(target);
           y += stepY) {
        for (int x = minX;
             x <= maxX && samples.size() < static_cast<size_t>(target);
             x += stepX) {
          SamplePoint point;
          point.voxel = QVector3D(static_cast<float>(x) + 0.5f,
                                  static_cast<float>(y) + 0.5f,
                                  static_cast<float>(z) + 0.5f);
          point.patient =
              m_primaryVolume.voxelToPatient(x + 0.5, y + 0.5, z + 0.5);
          const double primaryValue = sampleVolumeValue(volume, x, y, z);
          const double gx = sampleVolumeValue(volume, x + 1, y, z) -
                            sampleVolumeValue(volume, x - 1, y, z);
          const double gy = sampleVolumeValue(volume, x, y + 1, z) -
                            sampleVolumeValue(volume, x, y - 1, z);
          const double gz = sampleVolumeValue(volume, x, y, z + 1) -
                            sampleVolumeValue(volume, x, y, z - 1);
          point.primaryValue = primaryValue;
          point.primaryGradient =
              QVector3D(static_cast<float>(gx), static_cast<float>(gy),
                        static_cast<float>(gz));
          point.gradientMagnitude = std::sqrt(gx * gx + gy * gy + gz * gz);
          samples.push_back(point);
        }
      }
    }
  }

  return samples;
}

double FusionDialog::evaluateAlignmentCost(
    const QVector3D &manualTranslation, const QQuaternion &rotation,
    const std::vector<SamplePoint> &samples) const {
  if (!m_secondaryLoaded || samples.empty())
    return std::numeric_limits<double>::max();

  const int width = m_secondaryVolume.width();
  const int height = m_secondaryVolume.height();
  const int depth = m_secondaryVolume.depth();
  if (width <= 0 || height <= 0 || depth <= 0)
    return std::numeric_limits<double>::max();

  const cv::Mat &volume = m_secondaryVolume.data();
  const QVector3D secondaryCenter = computeVolumeCenter(m_secondaryVolume);
  const QVector3D translation = m_centerShift + manualTranslation;
  const QQuaternion inv = rotation.conjugated();

  struct WeightedPair {
    double primary{0.0};
    double secondary{0.0};
    double weight{1.0};
  };

  std::vector<WeightedPair> pairs;
  pairs.reserve(samples.size());

  double sumGradDot = 0.0;
  double sumGradWeight = 0.0;
  int gradCount = 0;

  for (const SamplePoint &point : samples) {
    if (roiEnabled()) {
      const QVector3D voxelCenter = point.voxel;
      if (voxelCenter.x() < m_roiMinEdges.x() ||
          voxelCenter.x() > m_roiMaxEdges.x() ||
          voxelCenter.y() < m_roiMinEdges.y() ||
          voxelCenter.y() > m_roiMaxEdges.y() ||
          voxelCenter.z() < m_roiMinEdges.z() ||
          voxelCenter.z() > m_roiMaxEdges.z()) {
        continue;
      }
    }

    const QVector3D relative = point.patient - secondaryCenter - translation;
    const QVector3D rotated = inv.rotatedVector(relative);
    const QVector3D secondaryPatient = rotated + secondaryCenter;
    QVector3D voxelCoord =
        m_secondaryVolume.patientToVoxelContinuous(secondaryPatient);

    if (voxelCoord.x() < 0.0 || voxelCoord.y() < 0.0 || voxelCoord.z() < 0.0 ||
        voxelCoord.x() > width - 1 || voxelCoord.y() > height - 1 ||
        voxelCoord.z() > depth - 1) {
      continue;
    }

    const double secondaryValue =
        sampleVolumeValue(volume, voxelCoord.x(), voxelCoord.y(),
                          voxelCoord.z());

    double pairWeight = 0.35;
    const double primaryGradMag = point.gradientMagnitude;
    if (primaryGradMag > 1e-4) {
      pairWeight = std::clamp(primaryGradMag / 60.0, 0.35, 2.5);
      QVector3D secondaryGrad =
          sampleVolumeGradient(volume, voxelCoord.x(), voxelCoord.y(),
                               voxelCoord.z());
      const double secondaryGradMag = secondaryGrad.length();
      if (secondaryGradMag > 1e-4) {
        const QVector3D primaryDir =
            point.primaryGradient / static_cast<float>(primaryGradMag);
        const QVector3D secondaryDir =
            secondaryGrad / static_cast<float>(secondaryGradMag);
        double dot = QVector3D::dotProduct(primaryDir, secondaryDir);
        dot = std::clamp(dot, -1.0, 1.0);
        const double primaryWeight =
            std::clamp(primaryGradMag / 60.0, 0.35, 2.5);
        const double secondaryWeight =
            std::clamp(secondaryGradMag / 60.0, 0.35, 2.5);
        const double combinedWeight =
            std::clamp((primaryWeight + secondaryWeight) * 0.5, 0.2, 4.0);
        sumGradDot += dot * combinedWeight;
        sumGradWeight += combinedWeight;
        ++gradCount;
        pairWeight = combinedWeight;
      }
    }

    pairWeight = std::clamp(pairWeight, 0.1, 4.0);

    pairs.push_back(WeightedPair{point.primaryValue, secondaryValue, pairWeight});
  }

  if (pairs.size() < std::max<size_t>(120, samples.size() / 6))
    return std::numeric_limits<double>::max();

  double totalWeight = 0.0;
  double minPrimary = pairs.front().primary;
  double maxPrimary = pairs.front().primary;
  double minSecondary = pairs.front().secondary;
  double maxSecondary = pairs.front().secondary;

  for (const WeightedPair &pair : pairs) {
    totalWeight += pair.weight;
    minPrimary = std::min(minPrimary, pair.primary);
    maxPrimary = std::max(maxPrimary, pair.primary);
    minSecondary = std::min(minSecondary, pair.secondary);
    maxSecondary = std::max(maxSecondary, pair.secondary);
  }

  if (totalWeight <= 0.0)
    return std::numeric_limits<double>::max();

  const double rangePrimary = maxPrimary - minPrimary;
  const double rangeSecondary = maxSecondary - minSecondary;
  if (rangePrimary < 1e-6 || rangeSecondary < 1e-6)
    return std::numeric_limits<double>::max();

  constexpr int bins = 64;
  std::array<double, bins> histPrimary{};
  std::array<double, bins> histSecondary{};
  std::array<double, bins * bins> jointHist{};

  const double invPrimaryRange = (bins - 1) / rangePrimary;
  const double invSecondaryRange = (bins - 1) / rangeSecondary;

  auto toBin = [](double value, double minValue, double invRange, int binCount) {
    const double scaled = (value - minValue) * invRange;
    int idx = static_cast<int>(std::floor(scaled + 1e-6));
    if (idx < 0)
      idx = 0;
    if (idx >= binCount)
      idx = binCount - 1;
    return idx;
  };

  for (const WeightedPair &pair : pairs) {
    const int binPrimary =
        toBin(pair.primary, minPrimary, invPrimaryRange, bins);
    const int binSecondary =
        toBin(pair.secondary, minSecondary, invSecondaryRange, bins);
    const double w = pair.weight;
    histPrimary[binPrimary] += w;
    histSecondary[binSecondary] += w;
    jointHist[binPrimary * bins + binSecondary] += w;
  }

  auto computeEntropy = [](const auto &hist, double total) {
    double entropy = 0.0;
    for (double count : hist) {
      if (count <= 0.0)
        continue;
      const double p = count / total;
      entropy -= p * std::log(std::max(p, 1e-12));
    }
    return entropy;
  };

  const double entropyPrimary = computeEntropy(histPrimary, totalWeight);
  const double entropySecondary = computeEntropy(histSecondary, totalWeight);
  const double entropyJoint = computeEntropy(jointHist, totalWeight);

  if (entropyPrimary <= 1e-8 || entropySecondary <= 1e-8)
    return std::numeric_limits<double>::max();

  double mutualInformation =
      entropyPrimary + entropySecondary - entropyJoint;
  if (!std::isfinite(mutualInformation) || mutualInformation < 0.0)
    mutualInformation = 0.0;

  const double entropySum = entropyPrimary + entropySecondary;
  if (entropySum <= 1e-8)
    return std::numeric_limits<double>::max();

  double normalizedMI = (2.0 * mutualInformation) / entropySum;
  normalizedMI = std::clamp(normalizedMI, 0.0, 1.0);
  double miCost = 1.0 - normalizedMI;

  const double maxJointEntropy =
      std::log(static_cast<double>(bins * bins));
  double entropyCost = 1.0;
  if (maxJointEntropy > 1e-8) {
    const double normalizedJoint =
        std::clamp(entropyJoint / maxJointEntropy, 0.0, 1.0);
    entropyCost = normalizedJoint;
  }

  double gradientCost = 0.5;
  bool reliableGradient =
      gradCount > std::max<int>(80, samples.size() / 10) &&
      sumGradWeight > 1e-6;
  if (reliableGradient) {
    const double gradientScore =
        std::clamp(sumGradDot / sumGradWeight, -1.0, 1.0);
    gradientCost = 0.5 * (1.0 - gradientScore);
  }

  const double reliabilityPenalty = reliableGradient ? 0.0 : 0.12;
  const double combinedCost =
      0.6 * miCost + 0.25 * entropyCost + 0.15 * gradientCost +
      reliabilityPenalty;
  return combinedCost;
}

QVector3D FusionDialog::computeVolumeCenter(const DicomVolume &volume) const {
  return computeVolumeCenterImpl(volume);
}

QVector3D FusionDialog::transformToSecondaryPatient(
    const QVector3D &primaryPoint) const {
  return transformToSecondaryPatientImpl(m_secondaryVolume, m_secondaryLoaded,
                                         m_centerShift, m_manualTranslation,
                                         m_rotationQuat, primaryPoint);
}

int FusionDialog::orientationIndex(DicomVolume::Orientation orientation) const {
  switch (orientation) {
  case DicomVolume::Orientation::Axial:
    return 0;
  case DicomVolume::Orientation::Sagittal:
    return 1;
  case DicomVolume::Orientation::Coronal:
    return 2;
  }
  return 0;
}

DicomVolume::Orientation FusionDialog::orientationFromIndex(int idx) const {
  switch (idx) {
  case 0:
    return DicomVolume::Orientation::Axial;
  case 1:
    return DicomVolume::Orientation::Sagittal;
  case 2:
  default:
    return DicomVolume::Orientation::Coronal;
  }
}

QString FusionDialog::orientationDisplayName(
    DicomVolume::Orientation orientation) const {
  switch (orientation) {
  case DicomVolume::Orientation::Axial:
    return tr("Axial (Ax)");
  case DicomVolume::Orientation::Sagittal:
    return tr("Sagittal (Sag)");
  case DicomVolume::Orientation::Coronal:
    return tr("Coronal (Cor)");
  }
  return QString();
}

std::array<double, 2>
FusionDialog::viewPixelSpacing(DicomVolume::Orientation orientation) const {
  auto safeSpacing = [](double value) {
    return value > 0.0 ? value : 1.0;
  };

  if (!m_primaryLoaded)
    return {1.0, 1.0};

  const double spacingX = safeSpacing(m_primaryVolume.spacingX());
  const double spacingY = safeSpacing(m_primaryVolume.spacingY());
  const double spacingZ = safeSpacing(m_primaryVolume.spacingZ());

  switch (orientation) {
  case DicomVolume::Orientation::Axial:
    return {spacingX, spacingY};
  case DicomVolume::Orientation::Sagittal:
    return {spacingY, spacingZ};
  case DicomVolume::Orientation::Coronal:
  default:
    return {spacingX, spacingZ};
  }
}

bool FusionDialog::loadFusionVolume(const FusionStudyRecord &rec,
                                    DicomVolume &volume, double *outWindow,
                                    double *outLevel, QStringList *infoLines) {
  QDir dir(rec.path);
  if (!dir.exists()) {
    QMessageBox::warning(this, tr("読み込みエラー"),
                         tr("Fusionデータのディレクトリが存在しません: %1")
                             .arg(rec.path));
    return false;
  }

  const QString metaPath = dir.filePath(QStringLiteral("fusion_meta.json"));
  if (!QFile::exists(metaPath)) {
    QMessageBox::warning(this, tr("読み込みエラー"),
                         tr("Fusionメタデータが見つかりません: %1")
                             .arg(metaPath));
    return false;
  }

  QFile metaFile(metaPath);
  if (!metaFile.open(QIODevice::ReadOnly)) {
    QMessageBox::warning(this, tr("読み込みエラー"),
                         tr("Fusionメタデータを開けません: %1").arg(metaPath));
    return false;
  }

  const QByteArray metaData = metaFile.readAll();
  QJsonParseError parseError{};
  const QJsonDocument doc = QJsonDocument::fromJson(metaData, &parseError);
  if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
    QMessageBox::warning(this, tr("読み込みエラー"),
                         tr("Fusionメタデータの解析に失敗しました: %1")
                             .arg(metaPath));
    return false;
  }

  const QJsonObject metaObj = doc.object();
  QString volumeFileName =
      metaObj.value(QStringLiteral("volume_file")).toString().trimmed();
  if (volumeFileName.isEmpty())
    volumeFileName = QStringLiteral("fusion_volume.bin");
  const QString volumePath = dir.filePath(volumeFileName);
  if (!QFile::exists(volumePath)) {
    QMessageBox::warning(this, tr("読み込みエラー"),
                         tr("Fusionボリュームが見つかりません: %1")
                             .arg(volumePath));
    return false;
  }

  if (!volume.loadFromFile(volumePath)) {
    QMessageBox::warning(this, tr("読み込みエラー"),
                         tr("Fusionボリュームの読み込みに失敗しました: %1")
                             .arg(volumePath));
    return false;
  }

  if (outWindow) {
    const QJsonValue winVal = metaObj.value(QStringLiteral("window"));
    if (winVal.isDouble())
      *outWindow = winVal.toDouble();
  }
  if (outLevel) {
    const QJsonValue lvlVal = metaObj.value(QStringLiteral("level"));
    if (lvlVal.isDouble())
      *outLevel = lvlVal.toDouble();
  }

  if (infoLines) {
    QStringList lines;
    lines << tr("患者: %1")
                 .arg(rec.patientName.isEmpty() ? tr("(不明)") : rec.patientName);

    QStringList studyParts;
    studyParts << rec.modality;
    const QString storedName =
        metaObj.value(QStringLiteral("study_name")).toString().trimmed();
    if (!storedName.isEmpty())
      studyParts << storedName;
    const QString createdAt =
        metaObj.value(QStringLiteral("created_at")).toString().trimmed();
    if (!createdAt.isEmpty())
      studyParts << createdAt;
    lines << tr("スタディ: %1")
                 .arg(studyParts.isEmpty() ? rec.modality
                                           : studyParts.join(QStringLiteral(" / ")));

    lines << tr("パス: %1").arg(rec.path);

    const QString primaryPath =
        metaObj.value(QStringLiteral("primary_path")).toString().trimmed();
    if (!primaryPath.isEmpty())
      lines << tr("Primary: %1").arg(primaryPath);

    const QString secondaryPath =
        metaObj.value(QStringLiteral("secondary_path")).toString().trimmed();
    if (!secondaryPath.isEmpty())
      lines << tr("Secondary: %1").arg(secondaryPath);

    *infoLines = lines;
  }

  return true;
}

bool FusionDialog::saveFusionResult(const DicomVolume &volume,
                                    QString &outDirectory,
                                    QString &outModality) {
  if (!m_db.isOpen())
    return false;

  const QString dataRoot = QString::fromStdString(m_db.dataRoot());
  if (dataRoot.isEmpty())
    return false;

  QString patientKey = m_primaryPatientKey.trimmed();
  QString patientName = m_primaryPatientName.trimmed();
  QString patientId = m_primaryPatientId.trimmed();

  QString secondaryMod = canonicalModality(m_secondaryModality);
  if (secondaryMod.isEmpty() || secondaryMod == QStringLiteral("OTHERS")) {
    const QString pendingMod = canonicalModality(m_pendingTransferModality);
    if (!pendingMod.isEmpty())
      secondaryMod = pendingMod;
  }
  if (secondaryMod.isEmpty() || secondaryMod == QStringLiteral("OTHERS"))
    secondaryMod = QStringLiteral("MRI");

  QString fusionCategory = secondaryMod;
  if (fusionCategory.startsWith(QLatin1String("Fusion/")))
    fusionCategory = fusionCategory.mid(QStringLiteral("Fusion/").size());
  if (fusionCategory.isEmpty())
    fusionCategory = QStringLiteral("MRI");

  const QString fusionModality =
      QStringLiteral("Fusion/%1").arg(fusionCategory);

  QString patientDir;
  if (!patientKey.isEmpty()) {
    patientDir =
        QDir(dataRoot).filePath(QStringLiteral("Patients/%1").arg(patientKey));
  }

  if (patientDir.isEmpty() || !QDir(patientDir).exists()) {
    QString resolvedName = patientName;
    if (resolvedName.isEmpty())
      resolvedName = patientKey.isEmpty() ? QStringLiteral("Unknown")
                                          : patientKey;
    QString resolvedId = patientId;
    if (resolvedId.isEmpty())
      resolvedId = normalizedPatientKey(resolvedName);

    FileStructureManager fsm(m_db);
    const std::string ensured = fsm.ensurePatientFolderFor(
        resolvedName.toStdString(), resolvedId.toStdString());
    if (ensured.empty())
      return false;
    patientDir = QString::fromStdString(ensured);
    patientKey = QFileInfo(patientDir).fileName();
    if (patientName.isEmpty())
      patientName = resolvedName;
    if (patientId.isEmpty())
      patientId = resolvedId;
  }

  if (patientDir.isEmpty())
    return false;

  if (!QDir().mkpath(patientDir))
    return false;

  const QString fusionRoot =
      QDir(patientDir).filePath(QStringLiteral("Images/Fusion"));
  if (!QDir().mkpath(fusionRoot))
    return false;

  const QString categoryDir =
      QDir(fusionRoot).filePath(fusionCategory.isEmpty() ? QStringLiteral("OTHERS")
                                                         : fusionCategory);
  if (!QDir().mkpath(categoryDir))
    return false;

  const QString patientInfoPath =
      QDir(patientDir).filePath(QStringLiteral("patient_info.txt"));
  if (!QFile::exists(patientInfoPath)) {
    QFile infoFile(patientInfoPath);
    if (infoFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
      QTextStream ts(&infoFile);
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
      ts.setEncoding(QStringConverter::Utf8);
#else
      ts.setCodec("UTF-8");
#endif
      ts << "# Patient Info\n";
      ts << "Name: "
         << (patientName.isEmpty() ? patientKey : patientName) << "\n";
      if (!patientId.isEmpty())
        ts << "ID: " << patientId << "\n";
      ts << "Created: (auto)\n";
    }
  }

  const QString timestamp =
      QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmmsszzz"));
  QString folderName = QStringLiteral("%1_Fusion").arg(timestamp);
  QString finalDir = QDir(categoryDir).filePath(folderName);
  int attempt = 1;
  while (QDir(finalDir).exists()) {
    folderName = QStringLiteral("%1_Fusion_%2").arg(timestamp).arg(++attempt);
    finalDir = QDir(categoryDir).filePath(folderName);
  }
  if (!QDir().mkpath(finalDir))
    return false;

  const QString volumeFileName = QStringLiteral("fusion_volume.bin");
  const QString volumePath = QDir(finalDir).filePath(volumeFileName);
  if (!volume.saveToFile(volumePath))
    return false;

  const QString seriesUid = QUuid::createUuid().toString(QUuid::WithoutBraces);
  m_primaryPatientKey = patientKey;
  if (!writeFusionMetadata(finalDir, fusionModality, volumeFileName, seriesUid))
    return false;

  const QString metaFileName = QStringLiteral("fusion_meta.json");
  int studyId = 0;
  if (!recordFusionStudyInDatabase(patientKey,
                                   patientName.isEmpty() ? patientKey
                                                         : patientName,
                                   patientInfoPath, fusionModality, finalDir,
                                   metaFileName, volumeFileName, seriesUid,
                                   studyId))
    return false;

  outDirectory = QDir::cleanPath(QDir(finalDir).absolutePath());
  outModality = fusionModality;
  return true;
}

bool FusionDialog::writeFusionMetadata(const QString &directory,
                                       const QString &modality,
                                       const QString &volumeFileName,
                                       const QString &seriesUid) const {
  QJsonObject obj;
  obj.insert(QStringLiteral("version"), 1);
  obj.insert(QStringLiteral("modality"), modality);
  obj.insert(QStringLiteral("series_uid"), seriesUid);
  obj.insert(QStringLiteral("volume_file"), volumeFileName);
  obj.insert(QStringLiteral("patient_key"), m_primaryPatientKey);
  const QString storedPatientName =
      m_primaryPatientName.isEmpty() ? m_primaryPatientKey
                                     : m_primaryPatientName;
  obj.insert(QStringLiteral("patient_name"), storedPatientName);
  if (!m_primaryPatientId.trimmed().isEmpty())
    obj.insert(QStringLiteral("dicom_patient_id"), m_primaryPatientId.trimmed());
  obj.insert(QStringLiteral("primary_path"), m_primaryStudyPath);
  obj.insert(QStringLiteral("primary_modality"), m_primaryModality);
  obj.insert(QStringLiteral("primary_frame_uid"), m_primaryFrameUid);
  obj.insert(QStringLiteral("secondary_path"), m_secondaryStudyPath);
  obj.insert(QStringLiteral("secondary_modality"), m_secondaryModality);
  obj.insert(QStringLiteral("secondary_frame_uid"),
             m_secondaryVolume.frameOfReferenceUID());
  obj.insert(QStringLiteral("study_name"), m_primaryStudyDescription);
  obj.insert(QStringLiteral("created_at"),
             QDateTime::currentDateTimeUtc().toString(Qt::ISODate));
  obj.insert(QStringLiteral("window"), m_pendingTransferWindow);
  obj.insert(QStringLiteral("level"), m_pendingTransferLevel);

  auto vectorToArray = [](const QVector3D &vec) {
    QJsonArray arr;
    arr.append(vec.x());
    arr.append(vec.y());
    arr.append(vec.z());
    return arr;
  };

  QJsonArray rotationArray;
  rotationArray.append(m_rotationQuat.scalar());
  rotationArray.append(m_rotationQuat.x());
  rotationArray.append(m_rotationQuat.y());
  rotationArray.append(m_rotationQuat.z());

  obj.insert(QStringLiteral("center_shift"), vectorToArray(m_centerShift));
  obj.insert(QStringLiteral("manual_translation"),
             vectorToArray(m_manualTranslation));
  obj.insert(QStringLiteral("manual_rotation_deg"),
             vectorToArray(m_manualRotation));
  obj.insert(QStringLiteral("rotation_quaternion"), rotationArray);

  const QString metaPath = QDir(directory).filePath(QStringLiteral("fusion_meta.json"));
  QFile file(metaPath);
  if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    return false;

  QJsonDocument doc(obj);
  file.write(doc.toJson(QJsonDocument::Indented));
  return true;
}

bool FusionDialog::recordFusionStudyInDatabase(const QString &patientKey,
                                               const QString &patientName,
                                               const QString &patientInfoPath,
                                               const QString &modality,
                                               const QString &directory,
                                               const QString &metaFileName,
                                               const QString &volumeFileName,
                                               const QString &seriesUid,
                                               int &outStudyId) {
  outStudyId = 0;
  if (!m_db.isOpen())
    return false;

  const QString trimmedPatientKey = patientKey.trimmed();
  if (trimmedPatientKey.isEmpty())
    return false;

  const QString cleanedDir =
      QDir::cleanPath(QDir(directory).absolutePath());
  const QString studyDate =
      QDateTime::currentDateTimeUtc().date().toString(QStringLiteral("yyyyMMdd"));

  QString studyName = m_primaryStudyDescription.trimmed();
  if (studyName.isEmpty()) {
    QFileInfo info(m_primaryStudyPath);
    studyName = info.fileName();
  }
  if (studyName.isEmpty())
    studyName = tr("Fusion Study");

  const QString secondaryLabel =
      canonicalModality(m_secondaryModality).isEmpty()
          ? QStringLiteral("MRI")
          : canonicalModality(m_secondaryModality);

  QString seriesDescription =
      tr("Fusion of %1 with %2")
          .arg(m_primaryModality.isEmpty() ? tr("Primary") : m_primaryModality,
               secondaryLabel);

  const bool startedTx = m_db.beginTransaction();

  const QString trimmedPatientName =
      patientName.trimmed().isEmpty() ? trimmedPatientKey : patientName.trimmed();
  QString cleanedInfoPath = patientInfoPath.trimmed();
  if (!cleanedInfoPath.isEmpty()) {
    QString nativeFree = QDir::fromNativeSeparators(cleanedInfoPath);
    QDir infoDir(nativeFree);
    cleanedInfoPath =
        QDir::cleanPath(infoDir.isAbsolute() ? nativeFree : infoDir.absolutePath());
  }

  std::stringstream insertPatient;
  insertPatient << "INSERT INTO patients(patient_key, name, created_at, info_path)\n"
                << "VALUES('" << sqlEscape(trimmedPatientKey) << "','"
                << sqlEscape(trimmedPatientName)
                << "', strftime('%s','now'), '"
                << sqlEscape(cleanedInfoPath) << "')\n"
                << "ON CONFLICT(patient_key) DO UPDATE SET name=excluded.name, info_path=excluded.info_path;";

  if (!m_db.exec(insertPatient.str())) {
    if (startedTx)
      m_db.rollback();
    return false;
  }

  std::stringstream insertStudy;
  insertStudy << "INSERT INTO studies(patient_key, modality, study_date, study_name, path, "
                 "frame_uid, series_uid, series_description)\n"
              << "VALUES('" << sqlEscape(trimmedPatientKey) << "','"
              << sqlEscape(modality) << "','" << sqlEscape(studyDate) << "','"
              << sqlEscape(studyName) << "','" << sqlEscape(cleanedDir)
              << "','" << sqlEscape(m_primaryFrameUid)
              << "','" << sqlEscape(seriesUid) << "','"
              << sqlEscape(seriesDescription) << "')\n"
              << "ON CONFLICT(patient_key, modality, path, series_uid) DO UPDATE SET\n"
              << "  study_date=excluded.study_date, study_name=excluded.study_name,\n"
              << "  frame_uid=excluded.frame_uid, series_description=excluded.series_description;";

  if (!m_db.exec(insertStudy.str())) {
    if (startedTx)
      m_db.rollback();
    return false;
  }

  std::stringstream queryId;
  queryId << "SELECT id FROM studies WHERE patient_key='"
          << sqlEscape(trimmedPatientKey)
          << "' AND modality='" << sqlEscape(modality) << "' AND path='"
          << sqlEscape(cleanedDir) << "' AND series_uid='"
          << sqlEscape(seriesUid) << "' LIMIT 1;";

  bool queryOk = m_db.query(queryId.str(), [&](int argc, char **argv, char **) {
    if (argc >= 1 && argv[0])
      outStudyId = std::atoi(argv[0]);
  });
  if (!queryOk || outStudyId <= 0) {
    if (startedTx)
      m_db.rollback();
    return false;
  }

  auto insertFileRecord = [&](const QString &fileName, const QString &type) {
    const QString absPath = QDir(directory).filePath(fileName);
    QFileInfo info(absPath);
    const long long size = info.size();
    const long long mtime = info.lastModified().toSecsSinceEpoch();
    const QString relative = QDir(directory).relativeFilePath(absPath);

    std::stringstream ss;
    ss << "INSERT INTO files(study_id, relative_path, size_bytes, mtime, file_type)\n"
       << "VALUES(" << outStudyId << ", '" << sqlEscape(relative) << "', "
       << size << ", " << mtime << ", '" << sqlEscape(type) << "')\n"
       << "ON CONFLICT(study_id, relative_path) DO UPDATE SET size_bytes=excluded.size_bytes,\n"
       << "  mtime=excluded.mtime, file_type=excluded.file_type;";
    return m_db.exec(ss.str());
  };

  if (!insertFileRecord(metaFileName, QStringLiteral("fusion_meta")) ||
      !insertFileRecord(volumeFileName, QStringLiteral("fusion_volume"))) {
    if (startedTx)
      m_db.rollback();
    return false;
  }

  if (startedTx && !m_db.commit())
    return false;

  return true;
}

void FusionDialog::onPrimaryStudyChanged(int index) {
  if (!m_primaryStudyCombo)
    return;
  const int studyIndex = m_primaryStudyCombo->itemData(index).toInt();
  if (studyIndex == m_currentPrimaryStudy)
    return;
  if (studyIndex < 0) {
    m_primaryLoaded = false;
    m_primaryVolume = DicomVolume();
    m_currentPrimaryStudy = -1;
    m_primaryPatientKey.clear();
    m_primaryPatientId.clear();
    m_primaryPatientName.clear();
    m_primaryStudyPath.clear();
    m_primaryStudyDescription.clear();
    m_primaryStudyDate.clear();
    m_primaryModality.clear();
    m_primaryFrameUid.clear();
    m_primaryInfoLabel->setText(tr("患者: -\nスタディ: -\nパス: -"));
    updateSliceRanges();
    updateCenterShift();
    updateStatusLabels();
    disableROI();
    updateImages();
    updatePrimaryInfoDisplay();
    return;
  }

  const FusionStudyRecord &rec = m_studyRecords[studyIndex];
  if (loadVolumeFromStudy(studyIndex, m_primaryVolume, m_primaryLoaded,
                          m_currentPrimaryStudy, m_primaryInfoLabel)) {
    m_primaryPatientKey = rec.patientKey;
    m_primaryPatientId = rec.patientKey;
    m_primaryPatientName = rec.patientName;
    m_primaryStudyPath = rec.path;
    m_primaryStudyDescription = rec.studyName;
    m_primaryStudyDate = rec.studyDate;
    m_primaryModality = rec.modality;
    m_primaryFrameUid = rec.frameUid;
    updatePrimaryInfoDisplay();
    updateSliceRanges();
    updateCenterShift();
    updateStatusLabels();
    initializeDefaultROI();
    updateImages();
    if (m_primaryLoaded && m_secondaryLoaded)
      onResetTransform();
  } else {
    disableROI();
    updateImages();
  }
}

void FusionDialog::onSecondaryStudyChanged(int index) {
  if (!m_secondaryStudyCombo)
    return;
  const int studyIndex = m_secondaryStudyCombo->itemData(index).toInt();
  if (studyIndex == m_currentSecondaryStudy)
    return;
  if (studyIndex < 0) {
    m_secondaryLoaded = false;
    m_secondaryVolume = DicomVolume();
    m_currentSecondaryStudy = -1;
    m_secondaryStudyPath.clear();
    m_secondaryModality.clear();
    m_secondaryInfoLabel->setText(tr("患者: -\nスタディ: -\nパス: -"));
    updateCenterShift();
    updateStatusLabels();
    updateImages();
    updateTransferButtonState();
    return;
  }

  const FusionStudyRecord &rec = m_studyRecords[studyIndex];
  if (loadVolumeFromStudy(studyIndex, m_secondaryVolume, m_secondaryLoaded,
                          m_currentSecondaryStudy, m_secondaryInfoLabel)) {
    m_secondaryStudyPath = rec.path;
    m_secondaryModality = rec.modality;
    updateCenterShift();
    updateStatusLabels();
    updateImages();
    if (m_primaryLoaded && m_secondaryLoaded)
      onResetTransform();
    updateTransferButtonState();
  }
}

void FusionDialog::onSliceSliderChanged(int orientationIndex, int value) {
  if (orientationIndex < 0 || orientationIndex >= 3)
    return;
  if (m_sliceValueLabels[orientationIndex] && m_sliceSliders[orientationIndex]) {
    m_sliceValueLabels[orientationIndex]->setText(
        tr("%1 / %2")
            .arg(value)
            .arg(m_sliceSliders[orientationIndex]->maximum()));
  }
  updateImages();
}

void FusionDialog::onZoomSliderChanged(int orientationIndex, int value) {
  if (orientationIndex < 0 || orientationIndex >= 3)
    return;
  const double factor = static_cast<double>(value) / 100.0;
  m_zoomFactors[orientationIndex] = factor;
  if (m_zoomValueLabels[orientationIndex]) {
    const double percent = factor * 100.0;
    m_zoomValueLabels[orientationIndex]->setText(
        tr("%1%").arg(QString::number(percent, 'f', 0)));
  }
  if (m_sliceViews[orientationIndex])
    m_sliceViews[orientationIndex]->setZoomFactor(factor);
}

void FusionDialog::onSliceViewROIChanging(int viewIndex,
                                          const QVector3D &minEdges,
                                          const QVector3D &maxEdges) {
  Q_UNUSED(viewIndex);
  applyROIEdges(minEdges, maxEdges, false);
}

void FusionDialog::onSliceViewROIChanged(int viewIndex,
                                         const QVector3D &minEdges,
                                         const QVector3D &maxEdges) {
  Q_UNUSED(viewIndex);
  applyROIEdges(minEdges, maxEdges, true);
}

void FusionDialog::onPrimaryWindowChanged(int value) {
  m_primaryWindow = static_cast<double>(value);
  if (m_primaryWindowValueLabel)
    m_primaryWindowValueLabel->setText(QString::number(value));
  updateImages();
}

void FusionDialog::onPrimaryLevelChanged(int value) {
  m_primaryLevel = static_cast<double>(value);
  if (m_primaryLevelValueLabel)
    m_primaryLevelValueLabel->setText(QString::number(value));
  updateImages();
}

void FusionDialog::onSecondaryWindowChanged(int value) {
  m_secondaryWindow = static_cast<double>(value);
  if (m_secondaryWindowValueLabel)
    m_secondaryWindowValueLabel->setText(QString::number(value));
  updateImages();
}

void FusionDialog::onSecondaryLevelChanged(int value) {
  m_secondaryLevel = static_cast<double>(value);
  if (m_secondaryLevelValueLabel)
    m_secondaryLevelValueLabel->setText(QString::number(value));
  updateImages();
}

void FusionDialog::onOpacityChanged(int value) {
  m_overlayOpacity = static_cast<double>(value) / 100.0;
  if (m_opacityValueLabel)
    m_opacityValueLabel->setText(tr("Opacity: %1%").arg(value));
  updateImages();
}

void FusionDialog::onPrimaryDisplayModeChanged(int index) {
  if (!m_primaryDisplayModeCombo)
    return;
  const QVariant modeData = m_primaryDisplayModeCombo->itemData(index);
  if (!modeData.isValid())
    return;
  const auto mode =
      static_cast<PrimaryDisplayMode>(modeData.toInt());
  if (mode == m_primaryDisplayMode)
    return;
  m_primaryDisplayMode = mode;
  updateImages();
}

void FusionDialog::onSecondaryDisplayModeChanged(int index) {
  if (!m_secondaryDisplayModeCombo)
    return;
  const QVariant modeData = m_secondaryDisplayModeCombo->itemData(index);
  if (!modeData.isValid())
    return;
  const auto mode =
      static_cast<SecondaryDisplayMode>(modeData.toInt());
  if (mode == m_secondaryDisplayMode)
    return;
  m_secondaryDisplayMode = mode;
  updateImages();
}

void FusionDialog::onRefreshStudies() {
  loadStudyList();
  populateStudyCombos();
  selectSecondaryForPrimaryPatient();
  updateCenterShift();
  updateStatusLabels();
  updateImages();
}

void FusionDialog::onSendSecondaryToViewer() {
  if (m_transferInProgress || m_transferWatcher.isRunning())
    return;
  if (!m_viewer) {
    QMessageBox::warning(this, tr("転送できません"),
                         tr("メインウィンドウのビューアが利用できません。"));
    return;
  }
  if (!m_primaryLoaded || m_primaryVolume.depth() <= 0) {
    QMessageBox::warning(this, tr("転送できません"),
                         tr("CTボリュームが読み込まれていません。"));
    return;
  }
  if (!m_secondaryLoaded || m_secondaryVolume.depth() <= 0) {
    QMessageBox::warning(this, tr("転送できません"),
                         tr("MRIボリュームが読み込まれていません。"));
    return;
  }

  QString path = m_secondaryStudyPath.trimmed();
  if (path.isEmpty() && m_currentSecondaryStudy >= 0 &&
      m_currentSecondaryStudy < static_cast<int>(m_studyRecords.size())) {
    path = m_studyRecords[m_currentSecondaryStudy].path.trimmed();
  }
  if (path.isEmpty()) {
    QMessageBox::warning(this, tr("転送できません"),
                         tr("MRIスタディの保存場所が不明です。"));
    return;
  }

  QString modality = m_secondaryModality.trimmed();
  if (modality.isEmpty() && m_currentSecondaryStudy >= 0 &&
      m_currentSecondaryStudy < static_cast<int>(m_studyRecords.size())) {
    modality = m_studyRecords[m_currentSecondaryStudy].modality.trimmed();
  }

  const double window = m_secondaryWindow > 0.0 ? m_secondaryWindow : 400.0;
  const double level = m_secondaryLevel;

  m_pendingTransferPath = path;
  m_pendingTransferModality = modality;
  m_pendingTransferWindow = window;
  m_pendingTransferLevel = level;

  TransferJobInput input;
  input.primaryVolume = m_primaryVolume;
  input.secondaryVolume = m_secondaryVolume;
  input.primaryLoaded = m_primaryLoaded;
  input.secondaryLoaded = m_secondaryLoaded;
  input.centerShift = m_centerShift;
  input.manualTranslation = m_manualTranslation;
  input.rotationQuat = m_rotationQuat;
  input.progressCounter = &m_transferProgressCounter;
  input.progressTotal = m_primaryVolume.depth();

  startTransferProgress(input.progressTotal);
  m_transferInProgress = true;
  if (m_transferSecondaryButton) {
    if (m_transferButtonDefaultText.isEmpty())
      m_transferButtonDefaultText = m_transferSecondaryButton->text();
    m_transferSecondaryButton->setText(tr("転送中..."));
  }
  updateTransferButtonState();
  QApplication::setOverrideCursor(Qt::BusyCursor);

  auto future = QtConcurrent::run([input]() { return runTransferJob(input); });
  m_transferWatcher.setFuture(future);
}

void FusionDialog::onTransferComputationFinished() {
  QApplication::restoreOverrideCursor();
  m_transferInProgress = false;
  if (m_transferSecondaryButton)
    m_transferSecondaryButton->setText(
        m_transferButtonDefaultText.isEmpty()
            ? tr("Image2 に転送")
            : m_transferButtonDefaultText);
  updateTransferButtonState();

  const TransferJobResult result = m_transferWatcher.future().result();
  stopTransferProgress();
  if (result.error != TransferJobResult::Error::None) {
    QString message = tr("Fusion変換済みMRIボリュームの生成に失敗しました。");
    QMessageBox::warning(this, tr("転送できません"), message);
    m_pendingTransferPath.clear();
    m_pendingTransferModality.clear();
    return;
  }

  if (!m_viewer) {
    QMessageBox::warning(this, tr("転送できません"),
                         tr("メインウィンドウのビューアが利用できません。"));
    m_pendingTransferPath.clear();
    m_pendingTransferModality.clear();
    return;
  }

  DicomVolume transformedVolume = result.volume;

  const QString originalTransferPath = m_pendingTransferPath;
  const QString originalTransferModality = m_pendingTransferModality;
  QString fusionDirectory;
  QString fusionModality;
  if (!saveFusionResult(transformedVolume, fusionDirectory, fusionModality)) {
    fusionDirectory = originalTransferPath;
    fusionModality = originalTransferModality;
  } else {
    m_pendingTransferPath = fusionDirectory;
    m_pendingTransferModality = fusionModality;
  }

  if (!m_viewer->showExternalImageSeries(m_pendingTransferPath,
                                         m_pendingTransferModality,
                                         transformedVolume, m_pendingTransferWindow,
                                         m_pendingTransferLevel)) {
    QMessageBox::warning(this, tr("転送失敗"),
                         tr("MRIボリュームをImage2として表示できませんでした。"));
  }

  m_pendingTransferPath.clear();
  m_pendingTransferModality.clear();
}

void FusionDialog::onResetTransform() {
  {
    QSignalBlocker bx(m_translationSpins[0]);
    QSignalBlocker by(m_translationSpins[1]);
    QSignalBlocker bz(m_translationSpins[2]);
    if (m_translationSpins[0])
      m_translationSpins[0]->setValue(0.0);
    if (m_translationSpins[1])
      m_translationSpins[1]->setValue(0.0);
    if (m_translationSpins[2])
      m_translationSpins[2]->setValue(0.0);
  }
  {
    QSignalBlocker bx(m_rotationSpins[0]);
    QSignalBlocker by(m_rotationSpins[1]);
    QSignalBlocker bz(m_rotationSpins[2]);
    if (m_rotationSpins[0])
      m_rotationSpins[0]->setValue(0.0);
    if (m_rotationSpins[1])
      m_rotationSpins[1]->setValue(0.0);
    if (m_rotationSpins[2])
      m_rotationSpins[2]->setValue(0.0);
  }
  m_manualTranslation = QVector3D(0.0, 0.0, 0.0);
  m_manualRotation = QVector3D(0.0, 0.0, 0.0);
  m_rotationQuat = QQuaternion::fromEulerAngles(0.0f, 0.0f, 0.0f);
  updateImages();
}

void FusionDialog::onTranslationChanged(int axis, double value) {
  double current = 0.0;
  switch (axis) {
  case 0:
    current = m_manualTranslation.x();
    break;
  case 1:
    current = m_manualTranslation.y();
    break;
  case 2:
    current = m_manualTranslation.z();
    break;
  }
  if (qFuzzyCompare(1.0 + current, 1.0 + value))
    return;
  switch (axis) {
  case 0:
    m_manualTranslation.setX(value);
    break;
  case 1:
    m_manualTranslation.setY(value);
    break;
  case 2:
    m_manualTranslation.setZ(value);
    break;
  }
  updateImages();
}

void FusionDialog::onRotationChanged(int axis, double value) {
  double current = 0.0;
  switch (axis) {
  case 0:
    current = m_manualRotation.x();
    break;
  case 1:
    current = m_manualRotation.y();
    break;
  case 2:
    current = m_manualRotation.z();
    break;
  }
  if (qFuzzyCompare(1.0 + current, 1.0 + value))
    return;
  switch (axis) {
  case 0:
    m_manualRotation.setX(value);
    break;
  case 1:
    m_manualRotation.setY(value);
    break;
  case 2:
    m_manualRotation.setZ(value);
    break;
  }
  m_rotationQuat = QQuaternion::fromEulerAngles(m_manualRotation.x(),
                                                m_manualRotation.y(),
                                                m_manualRotation.z());
  updateImages();
}

void FusionDialog::onAutoAlign() {
  if (!m_primaryLoaded || !m_secondaryLoaded) {
    QMessageBox::warning(this, tr("自動位置合わせ"),
                         tr("Primary と Secondary の両方のボリュームを読み込んでください。"));
    return;
  }

  if (m_autoAlignButton)
    m_autoAlignButton->setEnabled(false);
  QApplication::setOverrideCursor(Qt::BusyCursor);

  bool success = false;
  do {
    const std::vector<SamplePoint> samples = generatePrimarySamples(6000);
    if (samples.size() < 120) {
      QMessageBox::warning(this, tr("自動位置合わせ"),
                           tr("十分な特徴点を抽出できませんでした。"));
      break;
    }

    QVector3D bestTranslation = m_manualTranslation;
    QVector3D bestRotationDeg = m_manualRotation;
    QQuaternion bestQuat =
        QQuaternion::fromEulerAngles(bestRotationDeg.x(), bestRotationDeg.y(),
                                     bestRotationDeg.z());
    double bestCost =
        evaluateAlignmentCost(bestTranslation, bestQuat, samples);

    const double translationLimit = 60.0;
    const double rotationLimit = 20.0;
    double translationStep = 12.0;
    double rotationStep = 4.0;
    const double minTranslationStep = 0.5;
    const double minRotationStep = 0.5;
    const int maxIterations = 160;
    int iteration = 0;

    auto componentAt = [](const QVector3D &v, int axis) -> double {
      switch (axis) {
      case 0:
        return v.x();
      case 1:
        return v.y();
      default:
        return v.z();
      }
    };
    auto setComponent = [](QVector3D &v, int axis, double value) {
      switch (axis) {
      case 0:
        v.setX(static_cast<float>(value));
        break;
      case 1:
        v.setY(static_cast<float>(value));
        break;
      default:
        v.setZ(static_cast<float>(value));
        break;
      }
    };

    auto evaluateCandidate = [&](const QVector3D &translation,
                                 const QVector3D &rotationDeg) {
      QQuaternion quat = QQuaternion::fromEulerAngles(rotationDeg.x(),
                                                     rotationDeg.y(),
                                                     rotationDeg.z());
      const double cost =
          evaluateAlignmentCost(translation, quat, samples);
      if (!std::isfinite(cost))
        return false;
      if (std::isfinite(bestCost) && cost >= bestCost - 1e-6)
        return false;
      bestCost = cost;
      bestTranslation = translation;
      bestRotationDeg = rotationDeg;
      bestQuat = quat;
      return true;
    };

    while ((translationStep >= minTranslationStep ||
            rotationStep >= minRotationStep) &&
           iteration < maxIterations) {
      ++iteration;
      bool improved = false;

      if (translationStep >= minTranslationStep) {
        for (int axis = 0; axis < 3; ++axis) {
          for (int dir : {-1, 1}) {
            const double current = componentAt(bestTranslation, axis);
            double candidate =
                current + static_cast<double>(dir) * translationStep;
            candidate =
                std::clamp(candidate, -translationLimit, translationLimit);
            if (qFuzzyCompare(1.0 + current, 1.0 + candidate))
              continue;
            QVector3D candidateTranslation = bestTranslation;
            setComponent(candidateTranslation, axis, candidate);
            if (evaluateCandidate(candidateTranslation, bestRotationDeg))
              improved = true;
          }
        }
      }

      if (rotationStep >= minRotationStep) {
        for (int axis = 0; axis < 3; ++axis) {
          for (int dir : {-1, 1}) {
            const double current = componentAt(bestRotationDeg, axis);
            double candidate =
                current + static_cast<double>(dir) * rotationStep;
            candidate =
                std::clamp(candidate, -rotationLimit, rotationLimit);
            if (qFuzzyCompare(1.0 + current, 1.0 + candidate))
              continue;
            QVector3D candidateRotation = bestRotationDeg;
            setComponent(candidateRotation, axis, candidate);
            if (evaluateCandidate(bestTranslation, candidateRotation))
              improved = true;
          }
        }
      }

      if (!improved) {
        if (translationStep >= minTranslationStep)
          translationStep *= 0.5;
        if (rotationStep >= minRotationStep)
          rotationStep *= 0.5;
      }

      if (translationStep < minTranslationStep &&
          rotationStep < minRotationStep)
        break;
    }

    if (!std::isfinite(bestCost)) {
      QMessageBox::warning(this, tr("自動位置合わせ"),
                           tr("適切なオーバーラップを評価できませんでした。"));
      break;
    }

    m_manualTranslation = bestTranslation;
    m_manualRotation = bestRotationDeg;
    m_rotationQuat = bestQuat;

    {
      QSignalBlocker bx(m_translationSpins[0]);
      QSignalBlocker by(m_translationSpins[1]);
      QSignalBlocker bz(m_translationSpins[2]);
      if (m_translationSpins[0])
        m_translationSpins[0]->setValue(m_manualTranslation.x());
      if (m_translationSpins[1])
        m_translationSpins[1]->setValue(m_manualTranslation.y());
      if (m_translationSpins[2])
        m_translationSpins[2]->setValue(m_manualTranslation.z());
    }
    {
      QSignalBlocker rx(m_rotationSpins[0]);
      QSignalBlocker ry(m_rotationSpins[1]);
      QSignalBlocker rz(m_rotationSpins[2]);
      if (m_rotationSpins[0])
        m_rotationSpins[0]->setValue(m_manualRotation.x());
      if (m_rotationSpins[1])
        m_rotationSpins[1]->setValue(m_manualRotation.y());
      if (m_rotationSpins[2])
        m_rotationSpins[2]->setValue(m_manualRotation.z());
    }

    success = true;
  } while (false);

  QApplication::restoreOverrideCursor();
  if (m_autoAlignButton)
    m_autoAlignButton->setEnabled(m_primaryLoaded && m_secondaryLoaded);

  if (!success)
    return;

  updateImages();
}
