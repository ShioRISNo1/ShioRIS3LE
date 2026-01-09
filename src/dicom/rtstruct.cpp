#include "dicom/rtstruct.h"
#include <QDebug>
#include <QPainter>
#include <algorithm>
#include <cmath>
#include <dcmtk/dcmdata/dctk.h>
#include <map>
#include <QVector>
#include <QtConcurrent>
#include <QFutureWatcher>

bool RTStructureSet::loadFromFile(const QString &filename,
                                  std::function<void(int, int)> progress) {
  m_rois.clear();
  DcmFileFormat file;
  if (file.loadFile(filename.toLocal8Bit().data()).bad()) {
    qWarning() << "Failed to load RTSTRUCT" << filename;
    return false;
  }
  DcmDataset *ds = file.getDataset();

  // Map ROI number to name
  std::map<int, QString> roiNames;
  DcmSequenceOfItems *roiSeq = nullptr;
  if (ds->findAndGetSequence(DCM_StructureSetROISequence, roiSeq).good() &&
      roiSeq) {
    for (unsigned long i = 0; i < roiSeq->card(); ++i) {
      DcmItem *item = roiSeq->getItem(i);
      Sint32 num = 0;
      OFString name;
      item->findAndGetSint32(DCM_ROINumber, num);
      item->findAndGetOFString(DCM_ROIName, name);
      roiNames[num] = QString::fromLatin1(name.c_str()).trimmed();
    }
  }

  DcmSequenceOfItems *roiContourSeq = nullptr;
  if (ds->findAndGetSequence(DCM_ROIContourSequence, roiContourSeq).good() &&
      roiContourSeq) {
    unsigned long total = roiContourSeq->card();
    if (progress)
      progress(0, static_cast<int>(total));
    for (unsigned long i = 0; i < total; ++i) {
      DcmItem *roiItem = roiContourSeq->getItem(i);
      Sint32 refNum = 0;
      roiItem->findAndGetSint32(DCM_ReferencedROINumber, refNum);
      ROI roi;
      roi.name = roiNames.count(refNum) ? roiNames[refNum]
                                        : QString("ROI%1").arg(refNum);
      roi.visible = true;

      DcmSequenceOfItems *contourSeq = nullptr;
      if (roiItem->findAndGetSequence(DCM_ContourSequence, contourSeq).good() &&
          contourSeq) {
        for (unsigned long j = 0; j < contourSeq->card(); ++j) {
          DcmItem *contItem = contourSeq->getItem(j);
          OFString data;
          if (contItem->findAndGetOFStringArray(DCM_ContourData, data).good()) {
            QStringList nums = QString::fromLatin1(data.c_str())
                                   .split('\\', Qt::SkipEmptyParts);
            Contour c;
            for (int k = 0; k + 2 < nums.size(); k += 3) {
              bool okX, okY, okZ;
              double x = nums[k].toDouble(&okX);
              double y = nums[k + 1].toDouble(&okY);
              double z = nums[k + 2].toDouble(&okZ);
              if (okX && okY && okZ) {
                QVector3D p(x, y, z);
                c.points.push_back(p);
                if (!roi.hasBoundingBox) {
                  roi.minPoint = roi.maxPoint = p;
                  roi.hasBoundingBox = true;
                } else {
                  roi.minPoint.setX(std::min(roi.minPoint.x(), p.x()));
                  roi.minPoint.setY(std::min(roi.minPoint.y(), p.y()));
                  roi.minPoint.setZ(std::min(roi.minPoint.z(), p.z()));
                  roi.maxPoint.setX(std::max(roi.maxPoint.x(), p.x()));
                  roi.maxPoint.setY(std::max(roi.maxPoint.y(), p.y()));
                  roi.maxPoint.setZ(std::max(roi.maxPoint.z(), p.z()));
                }
              }
            }
            if (!c.points.empty()) {
              roi.contours.push_back(std::move(c));
            }
          }
        }
      }
      if (!roi.contours.empty()) {
        for (size_t idx = 0; idx < roi.contours.size(); ++idx) {
          const Contour &c = roi.contours[idx];
          if (c.points.empty())
            continue;
          int zKey = static_cast<int>(std::lround(c.points.front().z()));
          roi.contoursBySlice[zKey].push_back(idx);
        }
        m_rois.push_back(std::move(roi));
      }
      if (progress)
        progress(static_cast<int>(i + 1), static_cast<int>(total));
    }
  }

  qDebug() << "Loaded" << m_rois.size() << "ROIs from" << filename;
  for (size_t i = 0; i < m_rois.size(); ++i) {
    qDebug() << QString("ROI %1: %2 (%3 contours)")
                    .arg(i)
                    .arg(m_rois[i].name)
                    .arg(m_rois[i].contours.size());
  }

  return !m_rois.empty();
}

QImage RTStructureSet::axialOverlay(const DicomVolume &volume,
                                    int sliceIndex) const {
  QImage img(volume.width(), volume.height(), QImage::Format_ARGB32);
  img.fill(Qt::transparent);

  if (m_rois.empty()) {
    return img;
  }

  QPainter p(&img);
  p.setRenderHint(QPainter::Antialiasing);

  // Axial表示での患者座標のZ値を取得
  QVector3D slicePosition = volume.voxelToPatient(
      volume.width() / 2, volume.height() / 2, sliceIndex);
  double targetZ = slicePosition.z();

  qDebug() << QString("Axial slice %1: target Z = %2 mm")
                  .arg(sliceIndex)
                  .arg(targetZ, 0, 'f', 1);

  for (size_t r = 0; r < m_rois.size(); ++r) {
    if (!m_rois[r].visible)
      continue;

    QColor color = QColor::fromHsv((r * 40) % 360, 255, 255, 200);
    QPen pen(color, 1);
    p.setPen(pen);
    p.setBrush(Qt::NoBrush); // 塗りつぶしを完全に無効化

    int contoursDrawn = 0;
    for (const Contour &c : m_rois[r].contours) {
      if (c.points.empty())
        continue;

      // この輪郭がこのスライスに近いかチェック
      double contourZ = c.points[0].z();
      if (std::abs(contourZ - targetZ) > 1.0) { // 1mm の許容範囲
        continue;
      }

      // 輪郭を閉じた線として描画（塗りつぶしなし）
      std::vector<QPointF> contourPoints;

      for (const QVector3D &pt : c.points) {
        QVector3D vox = volume.patientToVoxelContinuous(pt);
        if (vox.x() < 0 || vox.x() >= volume.width() || vox.y() < 0 ||
            vox.y() >= volume.height()) {
          // 画像範囲外の点は境界にクリップ
          float clippedX = std::max(
              0.0f, std::min(static_cast<float>(volume.width() - 1), vox.x()));
          float clippedY = std::max(
              0.0f, std::min(static_cast<float>(volume.height() - 1), vox.y()));
          contourPoints.push_back(QPointF(clippedX, clippedY));
        } else {
          contourPoints.push_back(QPointF(vox.x(), vox.y()));
        }
      }

      // 輪郭を線分として描画
      if (contourPoints.size() >= 2) {
        for (size_t i = 0; i < contourPoints.size(); ++i) {
          size_t nextI = (i + 1) % contourPoints.size();
          p.drawLine(contourPoints[i], contourPoints[nextI]);
        }
        contoursDrawn++;
      }
    }

    if (contoursDrawn > 0) {
      qDebug() << QString("ROI %1: drew %2 contours on axial slice %3")
                      .arg(m_rois[r].name)
                      .arg(contoursDrawn)
                      .arg(sliceIndex);
    }
  }

  p.end();
  // Axial CT slices are already oriented correctly, so return the overlay
  // without additional mirroring to keep the structure positions aligned.
  return img;
}

QImage RTStructureSet::sagittalOverlay(const DicomVolume &volume,
                                       int sliceIndex, int stride) const {
  QImage img(volume.height(), volume.depth(), QImage::Format_ARGB32);
  img.fill(Qt::transparent);

  if (m_rois.empty()) {
    return img;
  }

  int cols = volume.height();
  int rows = volume.depth();

  QPainter p(&img);
  p.setRenderHint(QPainter::Antialiasing, false);
  p.setBrush(Qt::NoBrush);

  // 現在のスライス位置（患者座標系）
  QVector3D slicePos = volume.voxelToPatient(sliceIndex, 0, 0);

  for (size_t r = 0; r < m_rois.size(); ++r) {
    if (!m_rois[r].visible)
      continue;

    QVector3D minPt, maxPt;
    if (!roiBoundingBox(static_cast<int>(r), minPt, maxPt))
      continue;

    // ROI がこのスライスと交差しない場合はスキップ
    if (slicePos.x() < minPt.x() || slicePos.x() > maxPt.x())
      continue;

    // ROI の範囲をボクセル座標系に変換してループを制限
    QVector3D minVox = volume.patientToVoxelContinuous(
        QVector3D(slicePos.x(), minPt.y(), minPt.z()));
    QVector3D maxVox = volume.patientToVoxelContinuous(
        QVector3D(slicePos.x(), maxPt.y(), maxPt.z()));
    int yStart = std::clamp(
        static_cast<int>(std::floor(std::min(minVox.y(), maxVox.y()))), 0,
        cols - 1);
    int yEnd = std::clamp(
        static_cast<int>(std::ceil(std::max(minVox.y(), maxVox.y()))), 0,
        cols - 1);
    int zStart = std::clamp(
        static_cast<int>(std::floor(std::min(minVox.z(), maxVox.z()))), 0,
        rows - 1);
    int zEnd = std::clamp(
        static_cast<int>(std::ceil(std::max(minVox.z(), maxVox.z()))), 0,
        rows - 1);

    stride = std::max(1, stride);
    int yRange = yEnd - yStart + 1;
    int zRange = zEnd - zStart + 1;
    std::vector<uint8_t> Map(yRange * zRange, 0);

    QVector<int> yIndices;
    for (int y = yStart; y <= yEnd; y += stride) {
      yIndices.append(y);
    }
    auto markRow = [&](int y) {
      for (int z = zStart; z <= zEnd; ++z) {
        QVector3D pt = volume.voxelToPatient(sliceIndex, y, z);
        if (isPointInsideROI(pt, static_cast<int>(r))) {
          Map[(y - yStart) * zRange + (z - zStart)] = 1;
        }
      }
    };
    QFutureWatcher<void> watcher;
    watcher.setFuture(QtConcurrent::map(yIndices, markRow));
    watcher.waitForFinished();

    QColor color = QColor::fromHsv((r * 40) % 360, 255, 255, 255);
    QPen pen(color, 1);
    pen.setCosmetic(true);
    p.setPen(pen);
    // 境界抽出：4 近傍の内外判定でエッジを描画
    for (int y = 0; y < yRange; y += stride) {
      for (int z = 0; z < zRange; ++z) {
        if (!Map[(y) * zRange + (z)])
          continue;
        // 上端
        if (z == 0 || !Map[(y) * zRange + (z - 1)]) {
          p.drawLine(
              QPointF(yStart + y, zStart + z),
              QPointF(yStart + std::min(y + stride, yRange), zStart + z));
        }
        // 下端
        if (z >= zRange - 1 || !Map[(y) * zRange + (z + 1)]) {
          p.drawLine(
              QPointF(yStart + y, zStart + z + 1),
              QPointF(yStart + std::min(y + stride, yRange), zStart + z + 1));
        }
        // 左端
        if (y == 0 || !Map[(y - stride) * zRange + (z)]) {
          p.drawLine(QPointF(yStart + y, zStart + z),
                     QPointF(yStart + y, zStart + z + 1));
        }
        // 右端
        if (y >= yRange - stride || !Map[(y + stride) * zRange + (z)]) {
          p.drawLine(
              QPointF(yStart + std::min(y + stride, yRange), zStart + z),
              QPointF(yStart + std::min(y + stride, yRange), zStart + z + 1));
        }
      }
    }
  }

  p.end();
  return img.mirrored(false, true);
}

StructureLineList RTStructureSet::axialContours(const DicomVolume &volume,
                                                int sliceIndex) const {
  StructureLineList lines;
  if (m_rois.empty()) {
    return lines;
  }

  double targetZ = volume.voxelToPatient(volume.width() / 2,
                                         volume.height() / 2,
                                         sliceIndex)
                        .z();

  double w_mm = volume.width() * volume.spacingX();
  double h_mm = volume.height() * volume.spacingY();

  for (size_t r = 0; r < m_rois.size(); ++r) {
    if (!m_rois[r].visible)
      continue;

    QColor color = QColor::fromHsv((r * 40) % 360, 255, 255, 255);
    for (const Contour &c : m_rois[r].contours) {
      if (c.points.empty())
        continue;

      double contourZ = c.points[0].z();
      if (std::abs(contourZ - targetZ) > 1.0)
        continue;

      QVector<QPointF> pts;
      pts.reserve(static_cast<int>(c.points.size()));
      for (const QVector3D &pt : c.points) {
        QVector3D vox = volume.patientToVoxelContinuous(pt);
        float x_mm = vox.x() * volume.spacingX() - w_mm / 2.0f;
        float y_mm = h_mm / 2.0f - vox.y() * volume.spacingY();
        pts.append(QPointF(x_mm, y_mm));
      }
      if (pts.size() >= 2) {
        lines.append({pts, color});
      }
    }
  }

  return lines;
}

StructureLineList RTStructureSet::sagittalContours(const DicomVolume &volume,
                                                   int sliceIndex,
                                                   int stride) const {
  StructureLineList lines;
  if (m_rois.empty()) {
    return lines;
  }

  int cols = volume.height();
  int rows = volume.depth();
  QVector3D slicePos = volume.voxelToPatient(sliceIndex, 0, 0);

  double w_mm = cols * volume.spacingY();
  double h_mm = rows * volume.spacingZ();

  for (size_t r = 0; r < m_rois.size(); ++r) {
    if (!m_rois[r].visible)
      continue;

    QVector3D minPt, maxPt;
    if (!roiBoundingBox(static_cast<int>(r), minPt, maxPt))
      continue;

    if (slicePos.x() < minPt.x() || slicePos.x() > maxPt.x())
      continue;

    QVector3D minVox = volume.patientToVoxelContinuous(
        QVector3D(slicePos.x(), minPt.y(), minPt.z()));
    QVector3D maxVox = volume.patientToVoxelContinuous(
        QVector3D(slicePos.x(), maxPt.y(), maxPt.z()));

    int yStart = std::clamp(
        static_cast<int>(std::floor(std::min(minVox.y(), maxVox.y()))), 0,
        cols - 1);
    int yEnd =
        std::clamp(static_cast<int>(std::ceil(std::max(minVox.y(), maxVox.y()))),
                   0, cols - 1);
    int zStart = std::clamp(
        static_cast<int>(std::floor(std::min(minVox.z(), maxVox.z()))), 0,
        rows - 1);
    int zEnd =
        std::clamp(static_cast<int>(std::ceil(std::max(minVox.z(), maxVox.z()))),
                   0, rows - 1);

    stride = std::max(1, stride);
    int yRange = yEnd - yStart + 1;
    int zRange = zEnd - zStart + 1;
    std::vector<uint8_t> Map(yRange * zRange, 0);

    QVector<int> yIndices;
    for (int y = yStart; y <= yEnd; y += stride) {
      yIndices.append(y);
    }
    auto markRow = [&](int y) {
      for (int z = zStart; z <= zEnd; ++z) {
        QVector3D pt = volume.voxelToPatient(sliceIndex, y, z);
        if (isPointInsideROI(pt, static_cast<int>(r))) {
          Map[(y - yStart) * zRange + (z - zStart)] = 1;
        }
      }
    };
    QFutureWatcher<void> watcher;
    watcher.setFuture(QtConcurrent::map(yIndices, markRow));
    watcher.waitForFinished();

    QColor color = QColor::fromHsv((r * 40) % 360, 255, 255, 255);
    auto toMM = [&](double py, double pz) {
      float x_mm = py * volume.spacingY() - w_mm / 2.0f;
      float y_mm = pz * volume.spacingZ() - h_mm / 2.0f;
      return QPointF(x_mm, y_mm);
    };

    for (int y = 0; y < yRange; y += stride) {
      for (int z = 0; z < zRange; ++z) {
        if (!Map[(y) * zRange + (z)])
          continue;
        if (z == 0 || !Map[(y) * zRange + (z - 1)]) {
          QVector<QPointF> pts{toMM(yStart + y, zStart + z),
                               toMM(yStart + std::min(y + stride, yRange),
                                    zStart + z)};
          lines.append({pts, color});
        }
        if (z >= zRange - 1 || !Map[(y) * zRange + (z + 1)]) {
          QVector<QPointF> pts{toMM(yStart + y, zStart + z + 1),
                               toMM(yStart + std::min(y + stride, yRange),
                                    zStart + z + 1)};
          lines.append({pts, color});
        }
        if (y == 0 || !Map[(y - stride) * zRange + (z)]) {
          QVector<QPointF> pts{toMM(yStart + y, zStart + z),
                               toMM(yStart + y, zStart + z + 1)};
          lines.append({pts, color});
        }
        if (y >= yRange - stride || !Map[(y + stride) * zRange + (z)]) {
          QVector<QPointF> pts{toMM(yStart + std::min(y + stride, yRange),
                                    zStart + z),
                               toMM(yStart + std::min(y + stride, yRange),
                                    zStart + z + 1)};
          lines.append({pts, color});
        }
      }
    }
  }

  return lines;
}

StructureLineList RTStructureSet::coronalContours(const DicomVolume &volume,
                                                  int sliceIndex,
                                                  int stride) const {
  StructureLineList lines;
  if (m_rois.empty()) {
    return lines;
  }

  int cols = volume.width();
  int rows = volume.depth();
  QVector3D slicePos = volume.voxelToPatient(0, sliceIndex, 0);

  double w_mm = cols * volume.spacingX();
  double h_mm = rows * volume.spacingZ();

  for (size_t r = 0; r < m_rois.size(); ++r) {
    if (!m_rois[r].visible)
      continue;

    QVector3D minPt, maxPt;
    if (!roiBoundingBox(static_cast<int>(r), minPt, maxPt))
      continue;

    if (slicePos.y() < minPt.y() || slicePos.y() > maxPt.y())
      continue;

    QVector3D minVox = volume.patientToVoxelContinuous(
        QVector3D(minPt.x(), slicePos.y(), minPt.z()));
    QVector3D maxVox = volume.patientToVoxelContinuous(
        QVector3D(maxPt.x(), slicePos.y(), maxPt.z()));

    int xStart = std::clamp(
        static_cast<int>(std::floor(std::min(minVox.x(), maxVox.x()))), 0,
        cols - 1);
    int xEnd =
        std::clamp(static_cast<int>(std::ceil(std::max(minVox.x(), maxVox.x()))),
                   0, cols - 1);
    int zStart = std::clamp(
        static_cast<int>(std::floor(std::min(minVox.z(), maxVox.z()))), 0,
        rows - 1);
    int zEnd =
        std::clamp(static_cast<int>(std::ceil(std::max(minVox.z(), maxVox.z()))),
                   0, rows - 1);

    stride = std::max(1, stride);
    int xRange = xEnd - xStart + 1;
    int zRange = zEnd - zStart + 1;
    std::vector<uint8_t> Map(xRange * zRange, 0);

    QVector<int> xIndices;
    for (int x = xStart; x <= xEnd; x += stride) {
      xIndices.append(x);
    }
    auto markRow = [&](int x) {
      for (int z = zStart; z <= zEnd; ++z) {
        QVector3D pt = volume.voxelToPatient(x, sliceIndex, z);
        if (isPointInsideROI(pt, static_cast<int>(r))) {
          Map[(x - xStart) * zRange + (z - zStart)] = 1;
        }
      }
    };
    QFutureWatcher<void> watcher;
    watcher.setFuture(QtConcurrent::map(xIndices, markRow));
    watcher.waitForFinished();

    QColor color = QColor::fromHsv((r * 40) % 360, 255, 255, 255);
    auto toMM = [&](double px, double pz) {
      float x_mm = px * volume.spacingX() - w_mm / 2.0f;
      float y_mm = pz * volume.spacingZ() - h_mm / 2.0f;
      return QPointF(x_mm, y_mm);
    };

    for (int x = 0; x < xRange; x += stride) {
      for (int z = 0; z < zRange; ++z) {
        if (!Map[(x) * zRange + (z)])
          continue;
        if (z == 0 || !Map[(x) * zRange + (z - 1)]) {
          QVector<QPointF> pts{toMM(xStart + x, zStart + z),
                               toMM(xStart + std::min(x + stride, xRange),
                                    zStart + z)};
          lines.append({pts, color});
        }
        if (z >= zRange - 1 || !Map[(x) * zRange + (z + 1)]) {
          QVector<QPointF> pts{toMM(xStart + x, zStart + z + 1),
                               toMM(xStart + std::min(x + stride, xRange),
                                    zStart + z + 1)};
          lines.append({pts, color});
        }
        if (x == 0 || !Map[(x - stride) * zRange + (z)]) {
          QVector<QPointF> pts{toMM(xStart + x, zStart + z),
                               toMM(xStart + x, zStart + z + 1)};
          lines.append({pts, color});
        }
        if (x >= xRange - stride || !Map[(x + stride) * zRange + (z)]) {
          QVector<QPointF> pts{toMM(xStart + std::min(x + stride, xRange),
                                    zStart + z),
                               toMM(xStart + std::min(x + stride, xRange),
                                    zStart + z + 1)};
          lines.append({pts, color});
        }
      }
    }
  }

  return lines;
}

StructureLine3DList RTStructureSet::allContours3D(const DicomVolume &volume) const {
  StructureLine3DList lines;
  if (m_rois.empty()) {
    return lines;
  }

  double w_mm = volume.width() * volume.spacingX();
  double h_mm = volume.height() * volume.spacingY();
  double d_mm = volume.depth() * volume.spacingZ();

  for (size_t r = 0; r < m_rois.size(); ++r) {
    if (!m_rois[r].visible)
      continue;

    QColor color = QColor::fromHsv((r * 40) % 360, 255, 255, 255);
    for (const Contour &c : m_rois[r].contours) {
      if (c.points.size() < 2)
        continue;
      QVector<QVector3D> pts;
      pts.reserve(static_cast<int>(c.points.size()));
      for (const QVector3D &pt : c.points) {
        QVector3D vox = volume.patientToVoxelContinuous(pt);
        float x_mm = vox.x() * volume.spacingX() - w_mm / 2.0f;
        float y_mm = h_mm / 2.0f - vox.y() * volume.spacingY();
        float z_mm = vox.z() * volume.spacingZ() - d_mm / 2.0f;
        pts.append(QVector3D(x_mm, y_mm, z_mm));
      }
      if (pts.size() >= 2) {
        lines.append({pts, color});
      }
    }
  }

  return lines;
}

StructureLine3DList RTStructureSet::roiContours3D(int roiIndex, const DicomVolume &volume) const {
  StructureLine3DList lines;

  // Validate ROI index
  if (roiIndex < 0 || roiIndex >= static_cast<int>(m_rois.size())) {
    return lines;
  }

  const ROI& roi = m_rois[roiIndex];
  if (!roi.visible) {
    return lines;
  }

  double w_mm = volume.width() * volume.spacingX();
  double h_mm = volume.height() * volume.spacingY();
  double d_mm = volume.depth() * volume.spacingZ();

  QColor color = QColor::fromHsv((roiIndex * 40) % 360, 255, 255, 255);
  for (const Contour &c : roi.contours) {
    if (c.points.size() < 2)
      continue;
    QVector<QVector3D> pts;
    pts.reserve(static_cast<int>(c.points.size()));
    for (const QVector3D &pt : c.points) {
      QVector3D vox = volume.patientToVoxelContinuous(pt);
      float x_mm = vox.x() * volume.spacingX() - w_mm / 2.0f;
      float y_mm = h_mm / 2.0f - vox.y() * volume.spacingY();
      float z_mm = vox.z() * volume.spacingZ() - d_mm / 2.0f;
      pts.append(QVector3D(x_mm, y_mm, z_mm));
    }
    if (pts.size() >= 2) {
      lines.append({pts, color});
    }
  }

  return lines;
}

StructurePointList RTStructureSet::sagittalVertices(const DicomVolume &volume,
                                                    int sliceIndex,
                                                    int stride) const {
  StructurePointList points;
  if (m_rois.empty()) {
    return points;
  }

  int cols = volume.height();
  int rows = volume.depth();
  QVector3D slicePos = volume.voxelToPatient(sliceIndex, 0, 0);

  double w_mm = cols * volume.spacingY();
  double h_mm = rows * volume.spacingZ();

  for (size_t r = 0; r < m_rois.size(); ++r) {
    if (!m_rois[r].visible)
      continue;

    QVector3D minPt, maxPt;
    if (!roiBoundingBox(static_cast<int>(r), minPt, maxPt))
      continue;

    if (slicePos.x() < minPt.x() || slicePos.x() > maxPt.x())
      continue;

    QVector3D minVox = volume.patientToVoxelContinuous(
        QVector3D(slicePos.x(), minPt.y(), minPt.z()));
    QVector3D maxVox = volume.patientToVoxelContinuous(
        QVector3D(slicePos.x(), maxPt.y(), maxPt.z()));

    int yStart = std::clamp(
        static_cast<int>(std::floor(std::min(minVox.y(), maxVox.y()))), 0,
        cols - 1);
    int yEnd = std::clamp(
        static_cast<int>(std::ceil(std::max(minVox.y(), maxVox.y()))), 0,
        cols - 1);
    int zStart = std::clamp(
        static_cast<int>(std::floor(std::min(minVox.z(), maxVox.z()))), 0,
        rows - 1);
    int zEnd = std::clamp(
        static_cast<int>(std::ceil(std::max(minVox.z(), maxVox.z()))), 0,
        rows - 1);

    stride = std::max(1, stride);
    int yRange = yEnd - yStart + 1;
    int zRange = zEnd - zStart + 1;
    std::vector<uint8_t> Map(yRange * zRange, 0);

    QVector<int> yIndices;
    for (int y = yStart; y <= yEnd; y += stride) {
      yIndices.append(y);
    }
    auto markRow = [&](int y) {
      for (int z = zStart; z <= zEnd; ++z) {
        QVector3D pt = volume.voxelToPatient(sliceIndex, y, z);
        if (isPointInsideROI(pt, static_cast<int>(r))) {
          Map[(y - yStart) * zRange + (z - zStart)] = 1;
        }
      }
    };
    QFutureWatcher<void> watcher;
    watcher.setFuture(QtConcurrent::map(yIndices, markRow));
    watcher.waitForFinished();

    auto toMM = [&](double py, double pz) {
      float x_mm = py * volume.spacingY() - w_mm / 2.0f;
      float y_mm = (pz + 0.5f) * volume.spacingZ() - h_mm / 2.0f;
      return QPointF(x_mm, y_mm);
    };

    QColor color(Qt::red);
    for (int z = 0; z < zRange; ++z) {
      bool prev = false;
      int y;
      for (y = 0; y < yRange; y += stride) {
        bool cur = Map[(y) * zRange + (z)];
        if (cur != prev) {
          points.append({toMM(yStart + y, zStart + z), color});
        }
        prev = cur;
      }
      if (prev) {
        points.append({toMM(yStart + yRange, zStart + z), color});
      }
    }
  }

  return points;
}

StructurePointList RTStructureSet::coronalVertices(const DicomVolume &volume,
                                                   int sliceIndex,
                                                   int stride) const {
  StructurePointList points;
  if (m_rois.empty()) {
    return points;
  }

  int cols = volume.width();
  int rows = volume.depth();
  QVector3D slicePos = volume.voxelToPatient(0, sliceIndex, 0);

  double w_mm = cols * volume.spacingX();
  double h_mm = rows * volume.spacingZ();

  for (size_t r = 0; r < m_rois.size(); ++r) {
    if (!m_rois[r].visible)
      continue;

    QVector3D minPt, maxPt;
    if (!roiBoundingBox(static_cast<int>(r), minPt, maxPt))
      continue;

    if (slicePos.y() < minPt.y() || slicePos.y() > maxPt.y())
      continue;

    QVector3D minVox = volume.patientToVoxelContinuous(
        QVector3D(minPt.x(), slicePos.y(), minPt.z()));
    QVector3D maxVox = volume.patientToVoxelContinuous(
        QVector3D(maxPt.x(), slicePos.y(), maxPt.z()));

    int xStart = std::clamp(
        static_cast<int>(std::floor(std::min(minVox.x(), maxVox.x()))), 0,
        cols - 1);
    int xEnd = std::clamp(
        static_cast<int>(std::ceil(std::max(minVox.x(), maxVox.x()))), 0,
        cols - 1);
    int zStart = std::clamp(
        static_cast<int>(std::floor(std::min(minVox.z(), maxVox.z()))), 0,
        rows - 1);
    int zEnd = std::clamp(
        static_cast<int>(std::ceil(std::max(minVox.z(), maxVox.z()))), 0,
        rows - 1);

    stride = std::max(1, stride);
    int xRange = xEnd - xStart + 1;
    int zRange = zEnd - zStart + 1;
    std::vector<uint8_t> Map(xRange * zRange, 0);

    QVector<int> xIndices;
    for (int x = xStart; x <= xEnd; x += stride) {
      xIndices.append(x);
    }
    auto markRow = [&](int x) {
      for (int z = zStart; z <= zEnd; ++z) {
        QVector3D pt = volume.voxelToPatient(x, sliceIndex, z);
        if (isPointInsideROI(pt, static_cast<int>(r))) {
          Map[(x - xStart) * zRange + (z - zStart)] = 1;
        }
      }
    };
    QFutureWatcher<void> watcher;
    watcher.setFuture(QtConcurrent::map(xIndices, markRow));
    watcher.waitForFinished();

    auto toMM = [&](double px, double pz) {
      float x_mm = px * volume.spacingX() - w_mm / 2.0f;
      float y_mm = (pz + 0.5f) * volume.spacingZ() - h_mm / 2.0f;
      return QPointF(x_mm, y_mm);
    };

    QColor color(Qt::red);
    for (int z = 0; z < zRange; ++z) {
      bool prev = false;
      int x;
      for (x = 0; x < xRange; x += stride) {
        bool cur = Map[(x) * zRange + (z)];
        if (cur != prev) {
          points.append({toMM(xStart + x, zStart + z), color});
        }
        prev = cur;
      }
      if (prev) {
        points.append({toMM(xStart + xRange, zStart + z), color});
      }
    }
  }

  return points;
}
QImage RTStructureSet::coronalOverlay(const DicomVolume &volume, int sliceIndex,
                                      int stride) const {
  QImage img(volume.width(), volume.depth(), QImage::Format_ARGB32);
  img.fill(Qt::transparent);

  if (m_rois.empty()) {
    return img;
  }

  int cols = volume.width();
  int rows = volume.depth();

  QPainter p(&img);
  p.setRenderHint(QPainter::Antialiasing, false);
  p.setBrush(Qt::NoBrush);

  // 現在のスライス位置（患者座標系）
  QVector3D slicePos = volume.voxelToPatient(0, sliceIndex, 0);

  for (size_t r = 0; r < m_rois.size(); ++r) {
    if (!m_rois[r].visible)
      continue;

    QVector3D minPt, maxPt;
    if (!roiBoundingBox(static_cast<int>(r), minPt, maxPt))
      continue;

    // ROI がこのスライスと交差しない場合はスキップ
    if (slicePos.y() < minPt.y() || slicePos.y() > maxPt.y())
      continue;

    // ROI の範囲をボクセル座標系に変換してループを制限
    QVector3D minVox = volume.patientToVoxelContinuous(
        QVector3D(minPt.x(), slicePos.y(), minPt.z()));
    QVector3D maxVox = volume.patientToVoxelContinuous(
        QVector3D(maxPt.x(), slicePos.y(), maxPt.z()));
    int xStart = std::clamp(
        static_cast<int>(std::floor(std::min(minVox.x(), maxVox.x()))), 0,
        cols - 1);
    int xEnd = std::clamp(
        static_cast<int>(std::ceil(std::max(minVox.x(), maxVox.x()))), 0,
        cols - 1);
    int zStart = std::clamp(
        static_cast<int>(std::floor(std::min(minVox.z(), maxVox.z()))), 0,
        rows - 1);
    int zEnd = std::clamp(
        static_cast<int>(std::ceil(std::max(minVox.z(), maxVox.z()))), 0,
        rows - 1);

    stride = std::max(1, stride);
    int xRange = xEnd - xStart + 1;
    int zRange = zEnd - zStart + 1;
    std::vector<uint8_t> Map(xRange * zRange, 0);
    // z方向（Axスライス方向）は1ピクセル刻みで評価し、精度を維持する

    QVector<int> xIndices;
    for (int x = xStart; x <= xEnd; x += stride) {
      xIndices.append(x);
    }
    auto markRow = [&](int x) {
      for (int z = zStart; z <= zEnd; ++z) {
        QVector3D pt = volume.voxelToPatient(x, sliceIndex, z);
        if (isPointInsideROI(pt, static_cast<int>(r))) {
          Map[(x - xStart) * zRange + (z - zStart)] = 1;
        }
      }
    };
    QFutureWatcher<void> watcher;
    watcher.setFuture(QtConcurrent::map(xIndices, markRow));
    watcher.waitForFinished();

    QColor color = QColor::fromHsv((r * 40) % 360, 255, 255, 255);
    QPen pen(color, 1);
    pen.setCosmetic(true);
    p.setPen(pen);
    // 境界抽出：4 近傍の内外判定でエッジを描画
    for (int x = 0; x < xRange; x += stride) {
      for (int z = 0; z < zRange; ++z) {
        if (!Map[(x) * zRange + (z)])
          continue;
        // 上端
        if (z == 0 || !Map[(x) * zRange + (z - 1)]) {
          p.drawLine(
              QPointF(xStart + x, zStart + z),
              QPointF(xStart + std::min(x + stride, xRange), zStart + z));
        }
        // 下端
        if (z >= zRange - 1 || !Map[(x) * zRange + (z + 1)]) {
          p.drawLine(
              QPointF(xStart + x, zStart + z + 1),
              QPointF(xStart + std::min(x + stride, xRange), zStart + z + 1));
        }
        // 左端
        if (x == 0 || !Map[(x - stride) * zRange + (z)]) {
          p.drawLine(QPointF(xStart + x, zStart + z),
                     QPointF(xStart + x, zStart + z + 1));
        }
        // 右端
        if (x >= xRange - stride || !Map[(x + stride) * zRange + (z)]) {
          p.drawLine(
              QPointF(xStart + std::min(x + stride, xRange), zStart + z),
              QPointF(xStart + std::min(x + stride, xRange), zStart + z + 1));
        }
      }
    }
  }

  p.end();
  return img.mirrored(false, true);
}

QString RTStructureSet::roiName(int index) const {
  if (index < 0 || index >= static_cast<int>(m_rois.size()))
    return QString();
  return m_rois[index].name;
}

void RTStructureSet::setROIVisible(int index, bool visible) {
  if (index < 0 || index >= static_cast<int>(m_rois.size()))
    return;
  m_rois[index].visible = visible;
}

bool RTStructureSet::isROIVisible(int index) const {
  if (index < 0 || index >= static_cast<int>(m_rois.size()))
    return false;
  return m_rois[index].visible;
}

bool RTStructureSet::isPointInsideContour(const QVector3D &point,
                                          const Contour &contour) const {
  if (contour.points.size() < 3)
    return false;

  // Z座標が1mm以上離れている場合は判定しない
  double zRef = contour.points.front().z();
  if (std::abs(point.z() - zRef) > 1.0)
    return false;

  bool inside = false;
  const double px = point.x();
  const double py = point.y();

  for (size_t i = 0, j = contour.points.size() - 1; i < contour.points.size();
       j = i++) {
    const QVector3D &pi = contour.points[i];
    const QVector3D &pj = contour.points[j];

    // Ray Casting Algorithm: pointから右方向への水平線との交差数
    bool intersect = ((pi.y() > py) != (pj.y() > py));
    if (intersect) {
      double xint =
          pj.x() + (py - pj.y()) * (pi.x() - pj.x()) / (pi.y() - pj.y());
      if (xint > px)
        inside = !inside;
    }
  }

  return inside;
}

bool RTStructureSet::isPointInsideROI(const QVector3D &point,
                                      int roiIndex) const {
  if (roiIndex < 0 || roiIndex >= static_cast<int>(m_rois.size()))
    return false;

  const ROI &roi = m_rois[roiIndex];
  int zKey = static_cast<int>(std::lround(point.z()));
  // point.z() の前後1スライスのみを走査し、不要な判定を避ける
  for (int dz = -1; dz <= 1; ++dz) {
    auto it = roi.contoursBySlice.find(zKey + dz);
    if (it == roi.contoursBySlice.end())
      continue;

    int insideCount = 0;
    for (size_t idx : it->second) {
      if (isPointInsideContour(point, roi.contours[idx]))
        insideCount++;
    }

    if (insideCount % 2 == 1)
      return true; // 奇数回含まれる場合は内部（中空構造対応）
  }
  return false;
}

bool RTStructureSet::roiBoundingBox(int index, QVector3D &minPoint,
                                    QVector3D &maxPoint) const {
  if (index < 0 || index >= static_cast<int>(m_rois.size()))
    return false;

  const ROI &roi = m_rois[index];
  if (!roi.hasBoundingBox)
    return false;
  minPoint = roi.minPoint;
  maxPoint = roi.maxPoint;
  return true;
}

QVector<QVector<QVector3D>> RTStructureSet::roiContoursPatient(int roiIndex) const {
  QVector<QVector<QVector3D>> contours;

  // Validate ROI index
  if (roiIndex < 0 || roiIndex >= static_cast<int>(m_rois.size())) {
    return contours;
  }

  const ROI& roi = m_rois[roiIndex];

  // Convert std::vector to QVector and return contours in patient coordinate system
  for (const Contour& c : roi.contours) {
    QVector<QVector3D> contour;
    contour.reserve(static_cast<int>(c.points.size()));
    for (const QVector3D& pt : c.points) {
      contour.append(pt);
    }
    if (!contour.isEmpty()) {
      contours.append(contour);
    }
  }

  return contours;
}