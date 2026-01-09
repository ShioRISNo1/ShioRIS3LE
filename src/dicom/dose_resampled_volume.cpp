#include "dicom/dose_resampled_volume.h"
#include <QColor>
#include <QDebug>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QVector>
#include <array>
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <QtConcurrent>
#include <atomic>
#include <numeric>
#include <vector>

namespace {

constexpr std::array<float, 7> kIsodoseLevels = {0.95f, 0.90f, 0.80f,
                                                 0.70f, 0.50f, 0.30f,
                                                 0.10f};

constexpr std::array<QRgb, 7> kIsodoseColors = {
    qRgba(255, 0, 0, 200),   qRgba(255, 128, 0, 180), qRgba(255, 255, 0, 160),
    qRgba(0, 255, 0, 140),   qRgba(0, 255, 255, 120), qRgba(0, 0, 255, 100),
    qRgba(128, 0, 255, 80)};

struct LevelContours {
  QColor color;
  std::vector<std::vector<cv::Point>> paths;
};

std::vector<LevelContours> extractIsodoseContours(
    const cv::Mat &slice, double minDose, double maxDose, double referenceDose,
    double fallbackMaxDose,
    const std::function<double(double)> &transform) {
  std::vector<LevelContours> result;
  if (slice.empty())
    return result;

  double refDose = referenceDose > 0.0 ? referenceDose : fallbackMaxDose;
  if (refDose <= 0.0)
    return result;

  cv::Mat processed(slice.rows, slice.cols, CV_32F);
  for (int y = 0; y < slice.rows; ++y) {
    const float *src = slice.ptr<float>(y);
    float *dst = processed.ptr<float>(y);
    for (int x = 0; x < slice.cols; ++x) {
      double v = src[x];
      if (transform)
        v = transform(v);
      // マイナス値も許容するように変更（minDose未満のみ0にする）
      if (maxDose > minDose && v < minDose)
        v = 0.0;
      dst[x] = static_cast<float>(v);
    }
  }

  for (std::size_t i = 0; i < kIsodoseLevels.size(); ++i) {
    double threshold = static_cast<double>(kIsodoseLevels[i]) * refDose;
    if (maxDose > minDose && threshold < minDose)
      continue;

    cv::Mat mask;
    cv::threshold(processed, mask, threshold, 255.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U);
    if (cv::countNonZero(mask) == 0)
      continue;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_LIST,
                     cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty())
      continue;

    LevelContours level;
    QColor color(kIsodoseColors[i]);
    int alpha = std::max(150, qAlpha(kIsodoseColors[i]));
    color.setAlpha(alpha);
    level.color = color;

    for (const auto &contour : contours) {
      if (contour.size() < 2)
        continue;
      double area = std::fabs(cv::contourArea(contour));
      if (area < 1.0)
        continue;
      level.paths.push_back(contour);
    }

    if (!level.paths.empty())
      result.push_back(std::move(level));
  }

  return result;
}

QImage createIsodoseLineOverlay(const cv::Mat &slice, double minDose,
                                double maxDose, double referenceDose,
                                double fallbackMaxDose,
                                const std::function<double(double)> &transform) {
  if (slice.empty())
    return QImage();

  QImage img(slice.cols, slice.rows, QImage::Format_ARGB32);
  img.fill(Qt::transparent);

  auto contours = extractIsodoseContours(slice, minDose, maxDose, referenceDose,
                                         fallbackMaxDose, transform);
  if (contours.empty())
    return img;

  QPainter painter(&img);
  painter.setRenderHint(QPainter::Antialiasing);
  painter.setBrush(Qt::NoBrush);

  for (const auto &level : contours) {
    QPen pen(level.color);
    pen.setWidthF(1.5f);
    pen.setJoinStyle(Qt::RoundJoin);
    pen.setCapStyle(Qt::RoundCap);
    painter.setPen(pen);

    for (const auto &contour : level.paths) {
      if (contour.size() < 2)
        continue;
      QPainterPath path;
      const cv::Point &first = contour.front();
      path.moveTo(first.x + 0.5, first.y + 0.5);
      for (std::size_t idx = 1; idx < contour.size(); ++idx) {
        const cv::Point &pt = contour[idx];
        path.lineTo(pt.x + 0.5, pt.y + 0.5);
      }
      if (contour.size() > 2)
        path.closeSubpath();
      painter.drawPath(path);
    }
  }

  return img;
}

} // namespace

static cv::Mat getSliceAxialFloat(const cv::Mat &vol, int index) {
  int depth = vol.size[0];
  int height = vol.size[1];
  int width = vol.size[2];
  if (index < 0 || index >= depth)
    return cv::Mat();
  return cv::Mat(height, width, CV_32F,
                 const_cast<float *>(vol.ptr<float>(index)))
      .clone();
}

static cv::Mat getSliceSagittalFloat(const cv::Mat &vol, int index) {
  int depth = vol.size[0];
  int height = vol.size[1];
  int width = vol.size[2];
  if (index < 0 || index >= width)
    return cv::Mat();
  cv::Mat slice(height, depth, CV_32F);
  for (int z = 0; z < depth; ++z) {
    const float *srcRow = vol.ptr<float>(z);
    for (int y = 0; y < height; ++y) {
      slice.at<float>(y, z) = srcRow[y * width + index];
    }
  }
  // Transpose and flip vertically so the head appears at the top.
  cv::transpose(slice, slice);
  cv::flip(slice, slice, 0);
  return slice;
}

static cv::Mat getSliceCoronalFloat(const cv::Mat &vol, int index) {
  int depth = vol.size[0];
  int height = vol.size[1];
  int width = vol.size[2];
  if (index < 0 || index >= height)
    return cv::Mat();
  cv::Mat slice(width, depth, CV_32F);
  for (int z = 0; z < depth; ++z) {
    const float *srcRow = vol.ptr<float>(z);
    for (int x = 0; x < width; ++x) {
      slice.at<float>(x, z) = srcRow[index * width + x];
    }
  }
  // After transposing, vertical axis is patient Z. Flip vertically so the
  // superior direction (head) is displayed at the top.
  cv::transpose(slice, slice);
  cv::flip(slice, slice, 0);
  return slice;
}

// 線量値を美しいカラーマップに変換
QRgb mapDoseToColor(float doseRatio, float alpha = 1.0f) {
  // マイナス線量の場合は寒色系（紫～青）で表示
  if (doseRatio < 0.0f) {
    float negRatio = std::max(-1.0f, doseRatio); // -1.0以下は-1.0として扱う
    float absRatio = std::abs(negRatio);

    // 紫から濃い青へ (270° → 240°)
    float hue = 270.0f - absRatio * 30.0f;
    float saturation = 0.7f + absRatio * 0.3f;
    float value = 0.5f + absRatio * 0.5f;

    QColor color = QColor::fromHsvF(hue / 360.0f, saturation, value);
    int alphaValue = static_cast<int>(255 * alpha * (0.4f + absRatio * 0.6f));
    alphaValue = std::clamp(alphaValue, 40, 255);

    return qRgba(color.red(), color.green(), color.blue(), alphaValue);
  }

  if (doseRatio == 0.0f) {
    return qRgba(0, 0, 0, 0); // 透明
  }

  // HSVカラーマップを使用（青→緑→黄→オレンジ→赤→マゼンタ）
  // 線量が高いほど「熱い」色になる
  float hue = 0.0f;
  float saturation = 1.0f;
  float value = 1.0f;

  if (doseRatio <= 0.2f) {
    // 低線量域：青→シアン (240°→180°)
    hue = 240.0f - (doseRatio / 0.2f) * 60.0f;
    saturation = 0.8f + (doseRatio / 0.2f) * 0.2f;
  } else if (doseRatio <= 0.4f) {
    // 中低線量域：シアン→緑 (180°→120°)
    float t = (doseRatio - 0.2f) / 0.2f;
    hue = 180.0f - t * 60.0f;
    saturation = 1.0f;
  } else if (doseRatio <= 0.6f) {
    // 中線量域：緑→黄 (120°→60°)
    float t = (doseRatio - 0.4f) / 0.2f;
    hue = 120.0f - t * 60.0f;
    saturation = 1.0f;
  } else if (doseRatio <= 0.8f) {
    // 中高線量域：黄→オレンジ (60°→30°)
    float t = (doseRatio - 0.6f) / 0.2f;
    hue = 60.0f - t * 30.0f;
    saturation = 1.0f;
    value = 1.0f;
  } else if (doseRatio <= 1.0f) {
    // 高線量域：オレンジ→赤 (30°→0°)
    float t = (doseRatio - 0.8f) / 0.2f;
    hue = 30.0f - t * 30.0f;
    saturation = 1.0f;
    value = 1.0f;
  } else {
    // 超高線量域：赤→マゼンタ (0°→300°)
    float t = std::min(1.0f, (doseRatio - 1.0f) / 0.5f);
    hue = 360.0f - t * 60.0f; // 赤からマゼンタへ
    saturation = 1.0f;
    value = 1.0f - t * 0.2f; // 少し暗くする
  }

  // HSVからRGBに変換
  QColor color = QColor::fromHsvF(hue / 360.0f, saturation, value);

  // 透明度を計算（線量レベルに応じて調整）
  int alphaValue = static_cast<int>(255 * alpha);

  // 低線量域はより透明に、高線量域はより不透明に
  if (doseRatio < 0.1f) {
    alphaValue =
        static_cast<int>(alphaValue * (0.3f + 0.7f * doseRatio / 0.1f));
  } else if (doseRatio > 0.9f) {
    alphaValue = std::min(255, static_cast<int>(alphaValue * 1.2f));
  }

  alphaValue = std::clamp(alphaValue, 20, 255);

  return qRgba(color.red(), color.green(), color.blue(), alphaValue);
}

// 等線量曲線風の表示を生成
QRgb mapDoseToIsodose(float doseRatio, float alpha = 1.0f) {
  // マイナス線量の場合は暗い青で表示
  if (doseRatio < 0.0f) {
    float absRatio = std::min(1.0f, std::abs(doseRatio));
    int blue = static_cast<int>(100 + absRatio * 155);
    int alphaValue = static_cast<int>(150 * alpha * absRatio);
    return qRgba(50, 50, blue, alphaValue);
  }

  if (doseRatio == 0.0f) {
    return qRgba(0, 0, 0, 0);
  }

  // 適切な等線量レベルを選択
  for (int i = 0; i < static_cast<int>(kIsodoseLevels.size()); ++i) {
    if (doseRatio >= kIsodoseLevels[i]) {
      QColor color(kIsodoseColors[i]);
      int adjustedAlpha = static_cast<int>(qAlpha(kIsodoseColors[i]) * alpha);
      return qRgba(color.red(), color.green(), color.blue(), adjustedAlpha);
    }
  }

  return qRgba(0, 0, 0, 0);
}

// Hotカラーマップの表示を生成
QRgb mapDoseToHot(float doseRatio, float alpha = 1.0f) {
  // マイナス線量の場合は暗い青から黒へ
  if (doseRatio < 0.0f) {
    float absRatio = std::min(1.0f, std::abs(doseRatio));
    int blue = static_cast<int>(absRatio * 200);
    int alphaValue = static_cast<int>(255 * alpha * absRatio);
    return qRgba(0, 0, blue, alphaValue);
  }

  doseRatio = std::clamp(doseRatio, 0.0f, 1.0f);

  float r = std::min(1.0f, doseRatio * 3.0f);
  float g = std::clamp((doseRatio - 0.33f) * 3.0f, 0.0f, 1.0f);
  float b = std::clamp((doseRatio - 0.66f) * 3.0f, 0.0f, 1.0f);

  int alphaValue = static_cast<int>(255 * alpha);
  return qRgba(static_cast<int>(r * 255), static_cast<int>(g * 255),
               static_cast<int>(b * 255), alphaValue);
}

DoseResampledVolume::DoseResampledVolume() = default;

DoseResampledVolume::~DoseResampledVolume() = default;

void DoseResampledVolume::clear() {
  m_resampledVolume.release();
  m_width = m_height = m_depth = 0;
  m_spacingX = m_spacingY = m_spacingZ = 1.0;
  m_originX = m_originY = m_originZ = 0.0;
  m_maxDose = 0.0;
  m_isResampled = false;
}

bool DoseResampledVolume::resampleFromRTDose(
    const DicomVolume &ctVol, const RTDoseVolume &rtDose,
    std::function<void(int, int)> progressCallback,
    bool useNativeDoseGeometry) {
  clear();

  if (rtDose.width() == 0 || rtDose.height() == 0 || rtDose.depth() == 0)
    return false;

  // デバッグ: resample開始時のパラメータ
  qDebug() << "=== RESAMPLE FROM RTDOSE DEBUG ===";
  qDebug() << QString("  CT size: %1 x %2 x %3")
      .arg(ctVol.width()).arg(ctVol.height()).arg(ctVol.depth());
  qDebug() << QString("  CT spacing: %1 x %2 x %3")
      .arg(ctVol.spacingX()).arg(ctVol.spacingY()).arg(ctVol.spacingZ());
  qDebug() << QString("  CT origin: (%1, %2, %3)")
      .arg(ctVol.originX()).arg(ctVol.originY()).arg(ctVol.originZ());
  qDebug() << QString("  CT voxelToPatient(0,0,0): (%1, %2, %3)")
      .arg(ctVol.voxelToPatient(0,0,0).x())
      .arg(ctVol.voxelToPatient(0,0,0).y())
      .arg(ctVol.voxelToPatient(0,0,0).z());
  qDebug() << QString("  RTDose size: %1 x %2 x %3")
      .arg(rtDose.width()).arg(rtDose.height()).arg(rtDose.depth());
  qDebug() << QString("  RTDose spacing: %1 x %2 x %3")
      .arg(rtDose.spacingX()).arg(rtDose.spacingY()).arg(rtDose.spacingZ());
  qDebug() << QString("  RTDose origin: (%1, %2, %3)")
      .arg(rtDose.originX()).arg(rtDose.originY()).arg(rtDose.originZ());
  qDebug() << QString("  RTDose voxelToPatient(0,0,0): (%1, %2, %3)")
      .arg(rtDose.voxelToPatient(0,0,0).x())
      .arg(rtDose.voxelToPatient(0,0,0).y())
      .arg(rtDose.voxelToPatient(0,0,0).z());
  qDebug() << QString("  Resampled will use CT spacing and origin");

  m_width = ctVol.width();
  m_height = ctVol.height();
  m_depth = ctVol.depth();
  m_spacingX = ctVol.spacingX();
  m_spacingY = ctVol.spacingY();
  m_spacingZ = ctVol.spacingZ();
  m_originX = ctVol.originX();
  m_originY = ctVol.originY();
  m_originZ = ctVol.originZ();
  m_maxDose = 0.0;

  int sizes[3] = {m_depth, m_height, m_width};
  try {
    m_resampledVolume.create(3, sizes, CV_32F);
    // Initialize with zeros so we can safely skip non-overlap regions
    m_resampledVolume.setTo(cv::Scalar(0));
  } catch (const cv::Exception &e) {
    return false;
  }

  if (progressCallback)
    progressCallback(0, m_depth);

  QVector<float> sliceMax(m_depth, 0.0f);
  QVector<int> sliceValid(m_depth, 0);
  QVector<int> zIndices(m_depth);
  std::iota(zIndices.begin(), zIndices.end(), 0);
  std::atomic<int> done{0};

  // Pre-compute native extents when requested and derive CT voxel bounding box
  double nMinX=0, nMaxX=0, nMinY=0, nMaxY=0, nMinZ=0, nMaxZ=0;
  int bbX0 = 0, bbX1 = m_width - 1;
  int bbY0 = 0, bbY1 = m_height - 1;
  int bbZ0 = 0, bbZ1 = m_depth - 1;
  bool hasOverlapBounds = false;
  // When using native dose geometry, still honor any patientShift set on rtDose
  // by offsetting the native extents in patient space. Sampling will also use
  // the shifted coordinates.
  QVector3D doseShift = rtDose.patientShift();
  if (useNativeDoseGeometry && rtDose.nativeExtents(nMinX, nMaxX, nMinY, nMaxY, nMinZ, nMaxZ)) {
    // Transform dose native patient-space AABB corners to CT voxel space and clamp
    // Apply patient shift to dose-native bounds to obtain actual patient-space bounds
    const double sX = static_cast<double>(doseShift.x());
    const double sY = static_cast<double>(doseShift.y());
    const double sZ = static_cast<double>(doseShift.z());
    const double sMinX = nMinX + sX, sMaxX = nMaxX + sX;
    const double sMinY = nMinY + sY, sMaxY = nMaxY + sY;
    const double sMinZ = nMinZ + sZ, sMaxZ = nMaxZ + sZ;
    QVector3D corners[8] = {
      QVector3D(sMinX, sMinY, sMinZ), QVector3D(sMaxX, sMinY, sMinZ),
      QVector3D(sMinX, sMaxY, sMinZ), QVector3D(sMaxX, sMaxY, sMinZ),
      QVector3D(sMinX, sMinY, sMaxZ), QVector3D(sMaxX, sMinY, sMaxZ),
      QVector3D(sMinX, sMaxY, sMaxZ), QVector3D(sMaxX, sMaxY, sMaxZ)
    };
    double vxMin = std::numeric_limits<double>::infinity();
    double vyMin = std::numeric_limits<double>::infinity();
    double vzMin = std::numeric_limits<double>::infinity();
    double vxMax = -std::numeric_limits<double>::infinity();
    double vyMax = -std::numeric_limits<double>::infinity();
    double vzMax = -std::numeric_limits<double>::infinity();
    for (const auto &p : corners) {
      QVector3D v = ctVol.patientToVoxelContinuous(p);
      vxMin = std::min(vxMin, static_cast<double>(v.x()));
      vyMin = std::min(vyMin, static_cast<double>(v.y()));
      vzMin = std::min(vzMin, static_cast<double>(v.z()));
      vxMax = std::max(vxMax, static_cast<double>(v.x()));
      vyMax = std::max(vyMax, static_cast<double>(v.y()));
      vzMax = std::max(vzMax, static_cast<double>(v.z()));
    }
    // Expand slightly to be safe and convert to integer voxel ranges
    bbX0 = std::clamp(static_cast<int>(std::floor(vxMin)) - 1, 0, m_width - 1);
    bbY0 = std::clamp(static_cast<int>(std::floor(vyMin)) - 1, 0, m_height - 1);
    bbZ0 = std::clamp(static_cast<int>(std::floor(vzMin)) - 1, 0, m_depth - 1);
    bbX1 = std::clamp(static_cast<int>(std::ceil(vxMax)) + 1, 0, m_width - 1);
    bbY1 = std::clamp(static_cast<int>(std::ceil(vyMax)) + 1, 0, m_height - 1);
    bbZ1 = std::clamp(static_cast<int>(std::ceil(vzMax)) + 1, 0, m_depth - 1);
    hasOverlapBounds = (bbX0 <= bbX1 && bbY0 <= bbY1 && bbZ0 <= bbZ1);
  }

  QtConcurrent::blockingMap(zIndices, [&](int z) {
    // Skip slices completely outside overlap bounds
    if (hasOverlapBounds && (z < bbZ0 || z > bbZ1)) {
      sliceMax[z] = 0.0f;
      sliceValid[z] = 0;
      int current = done.fetch_add(1) + 1;
      if (progressCallback) progressCallback(current, m_depth);
      return;
    }
    float *slice = m_resampledVolume.ptr<float>(z);
    float localMax = 0.0f;
    int localValid = 0;
    const int yStart = hasOverlapBounds ? bbY0 : 0;
    const int yEnd   = hasOverlapBounds ? bbY1 : (m_height - 1);
    const int xStart = hasOverlapBounds ? bbX0 : 0;
    const int xEnd   = hasOverlapBounds ? bbX1 : (m_width - 1);
    // Precompute shifted bounds for quick patient-space inclusion test
    const double sMinX = nMinX + static_cast<double>(doseShift.x());
    const double sMaxX = nMaxX + static_cast<double>(doseShift.x());
    const double sMinY = nMinY + static_cast<double>(doseShift.y());
    const double sMaxY = nMaxY + static_cast<double>(doseShift.y());
    const double sMinZ = nMinZ + static_cast<double>(doseShift.z());
    const double sMaxZ = nMaxZ + static_cast<double>(doseShift.z());
    for (int y = yStart; y <= yEnd; ++y) {
      for (int x = xStart; x <= xEnd; ++x) {
        // Sample at CT voxel centers to align with ROI/DVH computations
        QVector3D ctPatient = ctVol.voxelToPatient(x + 0.5, y + 0.5, z + 0.5);
        bool inside = false;
        float dose = 0.0f;
        if (useNativeDoseGeometry) {
          // Extra clamp to native extents in patient space to avoid any accidental sampling
          if (!hasOverlapBounds || (ctPatient.x() >= sMinX && ctPatient.x() <= sMaxX &&
                                    ctPatient.y() >= sMinY && ctPatient.y() <= sMaxY &&
                                    ctPatient.z() >= sMinZ && ctPatient.z() <= sMaxZ)) {
            // In native mode, apply the dose shift by sampling the dose field
            // at ctPatient shifted into dose-native space.
            QVector3D pNative = ctPatient - doseShift;
            dose = rtDose.doseAtPatientNative(pNative, &inside);
          } else {
            inside = false;
          }
        } else {
          dose = rtDose.doseAtPatient(ctPatient, &inside);
        }
        if (inside) {
          slice[y * m_width + x] = dose;
          if (dose > localMax) localMax = dose;
          if (dose > 0.001f) localValid++;
        }
      }
    }
    sliceMax[z] = localMax;
    sliceValid[z] = localValid;
    int current = done.fetch_add(1) + 1;
    if (progressCallback)
      progressCallback(current, m_depth);
  });

  m_maxDose = 0.0f;
  int validVoxels = 0;
  for (int z = 0; z < m_depth; ++z) {
    if (sliceMax[z] > m_maxDose)
      m_maxDose = sliceMax[z];
    validVoxels += sliceValid[z];
  }

  m_isResampled = true;
  qDebug() << QString("Dose resampling: %1 valid voxels, max dose: %2")
                  .arg(validVoxels)
                  .arg(m_maxDose, 0, 'f', 6);

  qDebug() << "=== RESAMPLE FINAL ===";
  qDebug() << QString("  Size: %1 x %2 x %3").arg(m_width).arg(m_height).arg(m_depth);
  qDebug() << QString("  Spacing: %1 x %2 x %3").arg(m_spacingX).arg(m_spacingY).arg(m_spacingZ);
  qDebug() << QString("  Origin: (%1, %2, %3)").arg(m_originX).arg(m_originY).arg(m_originZ);

  return true;
}

QImage DoseResampledVolume::getSlice(
    int index, DicomVolume::Orientation ori, double minDose, double maxDose,
    DoseDisplayMode mode, double referenceDose,
    std::function<double(double)> transform) const {
  if (!m_isResampled) {
    qDebug() << "ERROR: Volume not resampled";
    return QImage();
  }

  cv::Mat slice;
  switch (ori) {
  case DicomVolume::Orientation::Axial:
    slice = getSliceAxialFloat(m_resampledVolume, index);
    break;
  case DicomVolume::Orientation::Sagittal:
    slice = getSliceSagittalFloat(m_resampledVolume, index);
    break;
  case DicomVolume::Orientation::Coronal:
    slice = getSliceCoronalFloat(m_resampledVolume, index);
    break;
  }

  if (slice.empty()) {
    return QImage();
  }

  if (mode == DoseDisplayMode::IsodoseLines) {
    return QImage();
  }

  QImage img(slice.cols, slice.rows, QImage::Format_ARGB32);
  img.fill(Qt::transparent);

  // 画像生成（改良されたカラーマッピング）
  for (int y = 0; y < slice.rows; ++y) {
    const float *src = slice.ptr<float>(y);
    QRgb *dst = reinterpret_cast<QRgb *>(img.scanLine(y));
    for (int x = 0; x < slice.cols; ++x) {
      double v = src[x];
      if (transform)
        v = transform(v);

      // マイナス値も表示できるように条件を変更
      bool shouldDisplay = (maxDose > minDose) ?
                          (v >= minDose || v < 0.0) : // minDose以上、またはマイナス値
                          (std::abs(v) > 1e-6);       // ゼロでない値

      if (shouldDisplay) {
        double ratio = 0.0;
        if (mode == DoseDisplayMode::Isodose && referenceDose > 0.0) {
          ratio = v / referenceDose;
        } else if (maxDose > minDose) {
          // マイナス値の場合は負の比率、正の値の場合は0～1の範囲
          if (v < 0.0) {
            ratio = v / std::abs(minDose); // マイナス値は-1.0を下限とする
          } else {
            double vClamped = std::min(v, maxDose);
            ratio = (vClamped - minDose) / (maxDose - minDose);
          }
        } else {
          ratio = v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0);
        }

        QRgb color;
        switch (mode) {
        case DoseDisplayMode::Colorful:
          color = mapDoseToColor(static_cast<float>(ratio), 0.8f);
          break;
        case DoseDisplayMode::Isodose:
          color = mapDoseToIsodose(static_cast<float>(ratio), 0.9f);
          break;
        case DoseDisplayMode::Hot:
          color = mapDoseToHot(static_cast<float>(ratio), 0.8f);
          break;
        case DoseDisplayMode::Simple:
        default: {
          if (ratio >= 0.0) {
            ratio = std::clamp(ratio, 0.0, 1.0);
            int alpha = std::max(40, std::min(255, static_cast<int>(255 * ratio)));
            color = qRgba(255, 0, 0, alpha);
          } else {
            // マイナス値は青で表示
            double absRatio = std::min(1.0, std::abs(ratio));
            int alpha = std::max(40, std::min(255, static_cast<int>(255 * absRatio)));
            color = qRgba(0, 0, 255, alpha);
          }
          break;
        }
        }
        dst[x] = color;
      } else {
        dst[x] = qRgba(0, 0, 0, 0);
      }
    }
  }

  return img;
}

DoseResampledVolume::IsodoseLineList
DoseResampledVolume::getIsodoseLines(
    int index, DicomVolume::Orientation ori, double minDose, double maxDose,
    double referenceDose, std::function<double(double)> transform) const {
  IsodoseLineList lines;
  if (!m_isResampled)
    return lines;

  cv::Mat slice;
  switch (ori) {
  case DicomVolume::Orientation::Axial:
    slice = getSliceAxialFloat(m_resampledVolume, index);
    break;
  case DicomVolume::Orientation::Sagittal:
    slice = getSliceSagittalFloat(m_resampledVolume, index);
    break;
  case DicomVolume::Orientation::Coronal:
    slice = getSliceCoronalFloat(m_resampledVolume, index);
    break;
  }

  if (slice.empty())
    return lines;

  auto contours = extractIsodoseContours(slice, minDose, maxDose, referenceDose,
                                         m_maxDose, transform);
  if (contours.empty())
    return lines;

  auto pixelToMM = [&](const cv::Point &pt) -> QPointF {
    double px = static_cast<double>(pt.x) + 0.5;
    double py = static_cast<double>(pt.y) + 0.5;
    switch (ori) {
    case DicomVolume::Orientation::Axial: {
      double w_mm = static_cast<double>(m_width) * m_spacingX;
      double h_mm = static_cast<double>(m_height) * m_spacingY;
      double x_mm = px * m_spacingX - w_mm / 2.0;
      double y_mm = h_mm / 2.0 - py * m_spacingY;
      return QPointF(x_mm, y_mm);
    }
    case DicomVolume::Orientation::Sagittal: {
      double w_mm = static_cast<double>(m_height) * m_spacingY;
      double h_mm = static_cast<double>(m_depth) * m_spacingZ;
      double x_mm = px * m_spacingY - w_mm / 2.0;
      double zIndex = static_cast<double>(slice.rows) - py;
      double y_mm = zIndex * m_spacingZ - h_mm / 2.0;
      return QPointF(x_mm, y_mm);
    }
    case DicomVolume::Orientation::Coronal: {
      double w_mm = static_cast<double>(m_width) * m_spacingX;
      double h_mm = static_cast<double>(m_depth) * m_spacingZ;
      double x_mm = px * m_spacingX - w_mm / 2.0;
      double zIndex = static_cast<double>(slice.rows) - py;
      double y_mm = zIndex * m_spacingZ - h_mm / 2.0;
      return QPointF(x_mm, y_mm);
    }
    }
    return QPointF();
  };

  for (const auto &level : contours) {
    QColor color = level.color;
    color.setAlpha(255);
    for (const auto &contour : level.paths) {
      if (contour.size() < 2)
        continue;
      QVector<QPointF> pts;
      pts.reserve(static_cast<int>(contour.size()) + 1);
      for (const auto &pt : contour) {
        pts.append(pixelToMM(pt));
      }
      if (!pts.isEmpty() && pts.first() != pts.last())
        pts.append(pts.first());
      if (pts.size() >= 2)
        lines.append({pts, color});
    }
  }

  return lines;
}

size_t DoseResampledVolume::getMemoryUsage() const {
  return static_cast<size_t>(m_resampledVolume.total() *
                             m_resampledVolume.elemSize());
}

float DoseResampledVolume::voxelDose(int x, int y, int z) const {
  if (!m_isResampled)
    return 0.0f;
  if (x < 0 || x >= m_width || y < 0 || y >= m_height || z < 0 || z >= m_depth)
    return 0.0f;
  const float *slice = m_resampledVolume.ptr<float>(z);
  return slice[y * m_width + x];
}

void DoseResampledVolume::setFromMat(const cv::Mat &vol, double spacingX,
                                     double spacingY, double spacingZ,
                                     double originX, double originY,
                                     double originZ) {
  m_resampledVolume = vol.clone();
  m_depth = vol.size[0];
  m_height = vol.size[1];
  m_width = vol.size[2];
  m_spacingX = spacingX;
  m_spacingY = spacingY;
  m_spacingZ = spacingZ;
  m_originX = originX;
  m_originY = originY;
  m_originZ = originZ;
  m_isResampled = true;

  qDebug() << "=== SETFROMMAT FINAL ===";
  qDebug() << QString("  Size: %1 x %2 x %3").arg(m_width).arg(m_height).arg(m_depth);
  qDebug() << QString("  Spacing: %1 x %2 x %3").arg(m_spacingX).arg(m_spacingY).arg(m_spacingZ);
  qDebug() << QString("  Origin: (%1, %2, %3)").arg(m_originX).arg(m_originY).arg(m_originZ);

  updateMaxDose();
}

void DoseResampledVolume::updateMaxDose() {
  m_maxDose = 0.0;
  if (!m_resampledVolume.empty()) {
    const float *ptr = m_resampledVolume.ptr<float>();
    size_t total = static_cast<size_t>(m_resampledVolume.total());
    for (size_t i = 0; i < total; ++i) {
      if (ptr[i] > m_maxDose)
        m_maxDose = ptr[i];
    }
  }
}
