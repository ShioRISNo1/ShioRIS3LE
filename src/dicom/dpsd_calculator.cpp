#include "dicom/dpsd_calculator.h"
#include <QDebug>
#include <QList>
#include <QThread>
#include <QtConcurrent>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <functional>

DPSDCalculator::Result DPSDCalculator::calculate(
    const DicomVolume &ctVolume, const DoseResampledVolume &doseVolume,
    const RTStructureSet &structures, int roiIndex, double startDistanceMm,
    double endDistanceMm, double stepMm, Mode mode, int sampleRoiIndex,
    std::atomic_bool *cancel) {
  Result result;
  if (!doseVolume.isResampled()) {
    qWarning() << "DPSD calculation: dose volume is not resampled";
    return result;
  }
  if (roiIndex < 0 || roiIndex >= structures.roiCount()) {
    qWarning() << "DPSD calculation: invalid ROI index" << roiIndex
               << "(ROI count:" << structures.roiCount() << ")";
    return result;
  }
  if (sampleRoiIndex >= structures.roiCount()) {
    qWarning() << "DPSD calculation: invalid sample ROI index" << sampleRoiIndex;
    sampleRoiIndex = -1;
  }
  if (stepMm <= 0.0) {
    stepMm = 1.0;
  }
  if (endDistanceMm <= startDistanceMm) {
    qWarning() << "DPSD calculation: start distance" << startDistanceMm
               << "is not less than end distance" << endDistanceMm;
    return {};
  }
  const int w = ctVolume.width();
  const int h = ctVolume.height();
  const int d = ctVolume.depth();

  // ROIのバウンディングボックスを取得し、計算範囲を縮小
  int x0 = 0, y0 = 0, z0 = 0;
  int x1 = w - 1, y1 = h - 1, z1 = d - 1;
  QVector3D bbMin, bbMax;
  if (structures.roiBoundingBox(roiIndex, bbMin, bbMax)) {
    double margin =
        std::max(std::abs(startDistanceMm), std::abs(endDistanceMm));
    bbMin -= QVector3D(margin, margin, margin);
    bbMax += QVector3D(margin, margin, margin);
    QVector3D voxMin = ctVolume.patientToVoxelContinuous(bbMin);
    QVector3D voxMax = ctVolume.patientToVoxelContinuous(bbMax);
    x0 = std::max(0, static_cast<int>(std::floor(voxMin.x())));
    y0 = std::max(0, static_cast<int>(std::floor(voxMin.y())));
    z0 = std::max(0, static_cast<int>(std::floor(voxMin.z())));
    x1 = std::min(w - 1, static_cast<int>(std::ceil(voxMax.x())));
    y1 = std::min(h - 1, static_cast<int>(std::ceil(voxMax.y())));
    z1 = std::min(d - 1, static_cast<int>(std::ceil(voxMax.z())));
  }
  if (sampleRoiIndex >= 0) {
    QVector3D bbMin2, bbMax2;
    if (structures.roiBoundingBox(sampleRoiIndex, bbMin2, bbMax2)) {
      double margin =
          std::max(std::abs(startDistanceMm), std::abs(endDistanceMm));
      bbMin2 -= QVector3D(margin, margin, margin);
      bbMax2 += QVector3D(margin, margin, margin);
      QVector3D voxMin2 = ctVolume.patientToVoxelContinuous(bbMin2);
      QVector3D voxMax2 = ctVolume.patientToVoxelContinuous(bbMax2);
      x0 = std::min(x0, std::max(0, static_cast<int>(std::floor(voxMin2.x()))));
      y0 = std::min(y0, std::max(0, static_cast<int>(std::floor(voxMin2.y()))));
      z0 = std::min(z0, std::max(0, static_cast<int>(std::floor(voxMin2.z()))));
      x1 = std::max(x1, std::min(w - 1, static_cast<int>(std::ceil(voxMax2.x()))));
      y1 = std::max(y1, std::min(h - 1, static_cast<int>(std::ceil(voxMax2.y()))));
      z1 = std::max(z1, std::min(d - 1, static_cast<int>(std::ceil(voxMax2.z()))));
    }
  }
  const int subW = x1 - x0 + 1;
  const int subH = y1 - y0 + 1;
  const int subD = z1 - z0 + 1;

  // ROIマスク作成
  int sizes[3] = {subD, subH, subW};
  cv::Mat mask(3, sizes, CV_8U, cv::Scalar(0));
  cv::Mat sampleMask;
  if (sampleRoiIndex >= 0)
    sampleMask = cv::Mat(3, sizes, CV_8U, cv::Scalar(0));
  for (int z = 0; z < subD; ++z) {
    int gz = z + z0;
    for (int y = 0; y < subH; ++y) {
      int gy = y + y0;
      for (int x = 0; x < subW; ++x) {
        if (cancel && cancel->load()) {
          qWarning() << "DPSD calculation: cancelled during mask creation";
          return result;
        }
        int gx = x + x0;
        QVector3D p = ctVolume.voxelToPatient(gx, gy, gz);
        if (structures.isPointInsideROI(p, roiIndex)) {
          mask.at<uchar>(z, y, x) = 255;
        }
        if (sampleRoiIndex >= 0 &&
            structures.isPointInsideROI(p, sampleRoiIndex)) {
          sampleMask.at<uchar>(z, y, x) = 255;
        }
      }
    }
  }

  // 距離変換結果を格納する配列（mm単位で保持）
  std::vector<double> distInside(subW * subH * subD,
                                 std::numeric_limits<double>::infinity());
  std::vector<double> distOutside(subW * subH * subD,
                                  std::numeric_limits<double>::infinity());

  auto index = [subW, subH](int x, int y, int z) {
    return z * subW * subH + y * subW + x;
  };

  // 画素間隔（mm）
  const double sx = ctVolume.spacingX();
  const double sy = ctVolume.spacingY();
  const double sz = ctVolume.spacingZ();

  // Dijkstra（境界ボクセルからの最短距離）
  auto run_dijkstra = [&](bool forInside) {
    using Node = std::pair<double, int>; // (dist, idx)
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;

    // 初期境界点の検出
    auto is_boundary = [&](int x, int y, int z) {
      bool inside = mask.at<uchar>(z, y, x) > 0;
      // 2D: 同一zで4近傍のみ境界判定、3D: 6近傍
      static const int nb2D[4][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}};
      static const int nb3D[6][3] = {
          {1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
          {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
      const int (*nb)[3] = (mode == Mode::Mode2D) ? nb2D : nb3D;
      int ncount = (mode == Mode::Mode2D) ? 4 : 6;
      bool boundary = false;
      for (int k = 0; k < ncount; ++k) {
        int nx = x + nb[k][0];
        int ny = y + nb[k][1];
        int nz = z + nb[k][2];
        if (nx < 0 || ny < 0 || nz < 0 || nx >= subW || ny >= subH ||
            nz >= subD) {
          if (inside)
            boundary = true;
          continue;
        }
        bool nInside = mask.at<uchar>(nz, ny, nx) > 0;
        if (inside != nInside)
          boundary = true;
      }
      return boundary;
    };

    auto &dist = forInside ? distInside : distOutside;

    // サンプルROIがある場合、対象ボクセル数を事前カウント
    int targetsRemaining = 0;
    std::vector<unsigned char> targetSeen; // 0/1
    if (sampleRoiIndex >= 0) {
      targetSeen.assign(subW * subH * subD, 0);
      for (int z = 0; z < subD; ++z) {
        for (int y = 0; y < subH; ++y) {
          for (int x = 0; x < subW; ++x) {
            bool inside = mask.at<uchar>(z, y, x) > 0;
            if ((forInside && !inside) || (!forInside && inside))
              continue;
            if (sampleMask.data && sampleMask.at<uchar>(z, y, x) > 0)
              ++targetsRemaining;
          }
        }
      }
    }

    // 探索距離の上限を設定（外側はend、内側は-start）
    const double considerMax = forInside
                                   ? (startDistanceMm < 0
                                          ? (-startDistanceMm + stepMm / 2.0)
                                          : (stepMm / 2.0))
                                   : (endDistanceMm > 0 ? (endDistanceMm + stepMm / 2.0)
                                                        : (stepMm / 2.0));
    for (int z = 0; z < subD; ++z) {
      for (int y = 0; y < subH; ++y) {
        for (int x = 0; x < subW; ++x) {
          bool inside = mask.at<uchar>(z, y, x) > 0;
          if ((forInside && !inside) || (!forInside && inside))
            continue;
          if (!is_boundary(x, y, z))
            continue;
          int idx = index(x, y, z);
          dist[idx] = 0.0;
          pq.emplace(0.0, idx);
        }
      }
    }

    while (!pq.empty()) {
      if (cancel && cancel->load())
        return;
      auto [curDist, uidx] = pq.top();
      pq.pop();
      if (curDist > dist[uidx])
        continue; // stale entry
      int ux = uidx % subW;
      int uy = (uidx / subW) % subH;
      int uz = uidx / (subW * subH);
      // サンプルROIの対象ボクセルに到達したらカウントダウン
      if (sampleRoiIndex >= 0 && sampleMask.data &&
          sampleMask.at<uchar>(uz, uy, ux) > 0 && !targetSeen[uidx]) {
        targetSeen[uidx] = 1;
        if (--targetsRemaining == 0) {
          return; // すべての対象が解決
        }
      }

      // 上限距離を超えたら展開しない
      if (curDist > considerMax)
        continue;
      for (int dz = (mode == Mode::Mode2D ? 0 : -1);
           dz <= (mode == Mode::Mode2D ? 0 : 1); ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0 && dz == 0)
              continue;
            int nx = ux + dx;
            int ny = uy + dy;
            int nz = uz + dz;
            if (nx < 0 || ny < 0 || nz < 0 || nx >= subW || ny >= subH ||
                nz >= subD)
              continue;
            bool nInside = mask.at<uchar>(nz, ny, nx) > 0;
            if ((forInside && !nInside) || (!forInside && nInside))
              continue;
            int nidx = index(nx, ny, nz);
            double step = std::sqrt((dx ? sx : 0.0) * (dx ? sx : 0.0) +
                                    (dy ? sy : 0.0) * (dy ? sy : 0.0) +
                                    (dz ? sz : 0.0) * (dz ? sz : 0.0));
            if (step == 0.0)
              continue;
            double nd = curDist + step;
            if (nd < dist[nidx]) {
              dist[nidx] = nd;
              pq.emplace(nd, nidx);
            }
          }
        }
      }
    }
  };

  // 2D/3Dそれぞれで内・外の距離を計算
  run_dijkstra(true);  // inside
  run_dijkstra(false); // outside

  int binCount =
      static_cast<int>((endDistanceMm - startDistanceMm) / stepMm) + 1;
  result.distancesMm.resize(binCount);
  result.minDoseGy.assign(binCount, std::numeric_limits<double>::max());
  result.maxDoseGy.assign(binCount, std::numeric_limits<double>::lowest());
  result.meanDoseGy.assign(binCount, 0.0);
  std::vector<int> counts(binCount, 0);
  for (int i = 0; i < binCount; ++i) {
    result.distancesMm[i] = startDistanceMm + i * stepMm; // bin center
  }

  const int threadCount = std::max(1, QThread::idealThreadCount());
  std::vector<double> threadMin(threadCount * binCount,
                                std::numeric_limits<double>::max());
  std::vector<double> threadMax(threadCount * binCount,
                                std::numeric_limits<double>::lowest());
  std::vector<double> threadSum(threadCount * binCount, 0.0);
  std::vector<int> threadCounts(threadCount * binCount, 0);

  auto worker = [&](int t) {
    int zStart = t * subD / threadCount;
    int zEnd = (t + 1) * subD / threadCount;
    for (int z = zStart; z < zEnd; ++z) {
      int gz = z + z0;
      for (int y = 0; y < subH; ++y) {
        int gy = y + y0;
        for (int x = 0; x < subW; ++x) {
          int gx = x + x0;
          if (cancel && cancel->load()) {
            return;
          }
          int idx = index(x, y, z);
          bool inside = mask.at<uchar>(z, y, x) > 0;
          if (sampleRoiIndex >= 0 && sampleMask.at<uchar>(z, y, x) == 0)
            continue;
          double dist = inside ? -distInside[idx] : distOutside[idx];
          if (dist < startDistanceMm - stepMm / 2.0 ||
              dist >= endDistanceMm + stepMm / 2.0)
            continue;
          // round to nearest bin so bin 0 covers [-0.5 mm, 0.5 mm)
          int bin = static_cast<int>(
              std::floor((dist - startDistanceMm) / stepMm + 0.5));
          if (bin < 0 || bin >= binCount)
            continue;
          float dose = doseVolume.voxelDose(gx, gy, gz);
          int offset = t * binCount + bin;
          threadCounts[offset]++;
          threadMin[offset] =
              std::min(threadMin[offset], static_cast<double>(dose));
          threadMax[offset] =
              std::max(threadMax[offset], static_cast<double>(dose));
          threadSum[offset] += dose;
        }
      }
    }
  };

  QList<QFuture<void>> futures;
  for (int t = 0; t < threadCount; ++t)
    futures.append(QtConcurrent::run(worker, t));
  for (auto &f : futures)
    f.waitForFinished();

  for (int t = 0; t < threadCount; ++t) {
    for (int b = 0; b < binCount; ++b) {
      int offset = t * binCount + b;
      if (threadCounts[offset] == 0)
        continue;
      counts[b] += threadCounts[offset];
      result.minDoseGy[b] = std::min(result.minDoseGy[b], threadMin[offset]);
      result.maxDoseGy[b] = std::max(result.maxDoseGy[b], threadMax[offset]);
      result.meanDoseGy[b] += threadSum[offset];
    }
  }

  if (std::all_of(counts.begin(), counts.end(), [](int c) { return c == 0; })) {
    qWarning() << "DPSD calculation: no voxels found in specified range";
    return {};
  }

  for (int i = 0; i < binCount; ++i) {
    if (counts[i] > 0)
      result.meanDoseGy[i] /= counts[i];
  }

  std::vector<double> distFiltered;
  std::vector<double> minFiltered;
  std::vector<double> maxFiltered;
  std::vector<double> meanFiltered;
  for (int i = 0; i < binCount; ++i) {
    if (counts[i] == 0)
      continue;
    distFiltered.push_back(result.distancesMm[i]);
    minFiltered.push_back(result.minDoseGy[i]);
    maxFiltered.push_back(result.maxDoseGy[i]);
    meanFiltered.push_back(result.meanDoseGy[i]);
  }
  result.distancesMm = std::move(distFiltered);
  result.minDoseGy = std::move(minFiltered);
  result.maxDoseGy = std::move(maxFiltered);
  result.meanDoseGy = std::move(meanFiltered);
  return result;
}

DPSDCalculator::Result DPSDCalculator::calculateFromRTDose(
    const DicomVolume &ctVolume, const RTDoseVolume &rtDose,
    const RTStructureSet &structures, int roiIndex, double startDistanceMm,
    double endDistanceMm, double stepMm, Mode mode, int sampleRoiIndex,
    std::atomic_bool *cancel) {
  Result result;
  if (rtDose.width() == 0) {
    qWarning() << "DPSD (RTDose): dose not loaded";
    return result;
  }
  if (roiIndex < 0 || roiIndex >= structures.roiCount()) {
    qWarning() << "DPSD (RTDose): invalid ROI index" << roiIndex;
    return result;
  }
  if (sampleRoiIndex >= structures.roiCount()) sampleRoiIndex = -1;
  if (stepMm <= 0.0) stepMm = 1.0;
  if (endDistanceMm <= startDistanceMm) return {};

  const int w = ctVolume.width();
  const int h = ctVolume.height();
  const int d = ctVolume.depth();

  int x0 = 0, y0 = 0, z0 = 0;
  int x1 = w - 1, y1 = h - 1, z1 = d - 1;
  QVector3D bbMin, bbMax;
  if (structures.roiBoundingBox(roiIndex, bbMin, bbMax)) {
    double margin = std::max(std::abs(startDistanceMm), std::abs(endDistanceMm));
    bbMin -= QVector3D(margin, margin, margin);
    bbMax += QVector3D(margin, margin, margin);
    QVector3D voxMin = ctVolume.patientToVoxelContinuous(bbMin);
    QVector3D voxMax = ctVolume.patientToVoxelContinuous(bbMax);
    x0 = std::max(0, static_cast<int>(std::floor(voxMin.x())));
    y0 = std::max(0, static_cast<int>(std::floor(voxMin.y())));
    z0 = std::max(0, static_cast<int>(std::floor(voxMin.z())));
    x1 = std::min(w - 1, static_cast<int>(std::ceil(voxMax.x())));
    y1 = std::min(h - 1, static_cast<int>(std::ceil(voxMax.y())));
    z1 = std::min(d - 1, static_cast<int>(std::ceil(voxMax.z())));
  }
  if (sampleRoiIndex >= 0) {
    QVector3D bbMin2, bbMax2;
    if (structures.roiBoundingBox(sampleRoiIndex, bbMin2, bbMax2)) {
      double margin = std::max(std::abs(startDistanceMm), std::abs(endDistanceMm));
      bbMin2 -= QVector3D(margin, margin, margin);
      bbMax2 += QVector3D(margin, margin, margin);
      QVector3D voxMin2 = ctVolume.patientToVoxelContinuous(bbMin2);
      QVector3D voxMax2 = ctVolume.patientToVoxelContinuous(bbMax2);
      x0 = std::min(x0, std::max(0, static_cast<int>(std::floor(voxMin2.x()))));
      y0 = std::min(y0, std::max(0, static_cast<int>(std::floor(voxMin2.y()))));
      z0 = std::min(z0, std::max(0, static_cast<int>(std::floor(voxMin2.z()))));
      x1 = std::max(x1, std::min(w - 1, static_cast<int>(std::ceil(voxMax2.x()))));
      y1 = std::max(y1, std::min(h - 1, static_cast<int>(std::ceil(voxMax2.y()))));
      z1 = std::max(z1, std::min(d - 1, static_cast<int>(std::ceil(voxMax2.z()))));
    }
  }

  const int subW = x1 - x0 + 1;
  const int subH = y1 - y0 + 1;
  const int subD = z1 - z0 + 1;

  int sizes[3] = {subD, subH, subW};
  cv::Mat mask(3, sizes, CV_8U, cv::Scalar(0));
  cv::Mat sampleMask;
  if (sampleRoiIndex >= 0) sampleMask = cv::Mat(3, sizes, CV_8U, cv::Scalar(0));

  for (int z = 0; z < subD; ++z) {
    int gz = z + z0;
    for (int y = 0; y < subH; ++y) {
      int gy = y + y0;
      for (int x = 0; x < subW; ++x) {
        int gx = x + x0;
        QVector3D p = ctVolume.voxelToPatient(gx, gy, gz);
        if (structures.isPointInsideROI(p, roiIndex)) mask.at<uchar>(z, y, x) = 255;
        if (sampleRoiIndex >= 0 && structures.isPointInsideROI(p, sampleRoiIndex)) sampleMask.at<uchar>(z, y, x) = 255;
      }
    }
  }

  std::vector<double> distInside(subW * subH * subD, std::numeric_limits<double>::infinity());
  std::vector<double> distOutside(subW * subH * subD, std::numeric_limits<double>::infinity());
  auto index = [subW, subH](int x, int y, int z) { return z * subW * subH + y * subW + x; };
  const double sx = ctVolume.spacingX();
  const double sy = ctVolume.spacingY();
  const double sz = ctVolume.spacingZ();

  auto run_dijkstra = [&](bool forInside) {
    auto &dist = forInside ? distInside : distOutside;
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<>> pq;
    auto add_if = [&](int x, int y, int z) {
      if (x < 0 || y < 0 || z < 0 || x >= subW || y >= subH || z >= subD) return;
      bool in = mask.at<uchar>(z, y, x) > 0;
      if ((forInside && !in) || (!forInside && in)) return;
      int idx = index(x, y, z);
      dist[idx] = 0.0;
      pq.emplace(0.0, idx);
    };
    for (int z = 0; z < subD; ++z) {
      for (int y = 0; y < subH; ++y) {
        for (int x = 0; x < subW; ++x) {
          bool in = mask.at<uchar>(z, y, x) > 0;
          if ((forInside && !in) || (!forInside && in)) continue;
          bool boundary = false;
          for (int dz = -1; dz <= 1 && !boundary; ++dz) {
            for (int dy = -1; dy <= 1 && !boundary; ++dy) {
              for (int dx = -1; dx <= 1 && !boundary; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx = x + dx, ny = y + dy, nz = z + dz;
                if (nx < 0 || ny < 0 || nz < 0 || nx >= subW || ny >= subH || nz >= subD) continue;
                bool nin = mask.at<uchar>(nz, ny, nx) > 0;
                if (nin != in) boundary = true;
              }
            }
          }
          if (boundary) add_if(x, y, z);
        }
      }
    }
    while (!pq.empty()) {
      auto [curDist, idx0] = pq.top(); pq.pop();
      if (curDist > dist[idx0]) continue;
      int z = idx0 / (subW * subH);
      int rem = idx0 % (subW * subH);
      int y = rem / subW;
      int x = rem % subW;
      for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0 && dz == 0) continue;
            int nx = x + dx, ny = y + dy, nz = z + dz;
            if (nx < 0 || ny < 0 || nz < 0 || nx >= subW || ny >= subH || nz >= subD) continue;
            bool nin = mask.at<uchar>(nz, ny, nx) > 0;
            if ((forInside && !nin) || (!forInside && nin)) continue;
            int nidx = index(nx, ny, nz);
            double step = std::sqrt((dx ? sx : 0.0) * (dx ? sx : 0.0) + (dy ? sy : 0.0) * (dy ? sy : 0.0) + (dz ? sz : 0.0) * (dz ? sz : 0.0));
            if (step == 0.0) continue;
            double nd = curDist + step;
            if (nd < dist[nidx]) { dist[nidx] = nd; pq.emplace(nd, nidx); }
          }
        }
      }
    }
  };

  run_dijkstra(true);
  run_dijkstra(false);

  int binCount = static_cast<int>((endDistanceMm - startDistanceMm) / stepMm) + 1;
  result.distancesMm.resize(binCount);
  result.minDoseGy.assign(binCount, std::numeric_limits<double>::max());
  result.maxDoseGy.assign(binCount, std::numeric_limits<double>::lowest());
  result.meanDoseGy.assign(binCount, 0.0);
  std::vector<int> counts(binCount, 0);
  for (int i = 0; i < binCount; ++i) result.distancesMm[i] = startDistanceMm + i * stepMm;

  const int threadCount = std::max(1, QThread::idealThreadCount());
  std::vector<double> threadMin(threadCount * binCount, std::numeric_limits<double>::max());
  std::vector<double> threadMax(threadCount * binCount, std::numeric_limits<double>::lowest());
  std::vector<double> threadSum(threadCount * binCount, 0.0);
  std::vector<int> threadCounts(threadCount * binCount, 0);

  auto worker = [&](int t) {
    int zStart = t * subD / threadCount;
    int zEnd = (t + 1) * subD / threadCount;
    for (int z = zStart; z < zEnd; ++z) {
      int gz = z + z0;
      for (int y = 0; y < subH; ++y) {
        int gy = y + y0;
        for (int x = 0; x < subW; ++x) {
          int gx = x + x0;
          if (cancel && cancel->load()) return;
          int idx = index(x, y, z);
          bool inside = mask.at<uchar>(z, y, x) > 0;
          if (sampleRoiIndex >= 0 && sampleMask.at<uchar>(z, y, x) == 0) continue;
          double dist = inside ? -distInside[idx] : distOutside[idx];
          if (dist < startDistanceMm - stepMm / 2.0 || dist >= endDistanceMm + stepMm / 2.0) continue;
          int bin = static_cast<int>(std::floor((dist - startDistanceMm) / stepMm + 0.5));
          if (bin < 0 || bin >= binCount) continue;
          QVector3D patient = ctVolume.voxelToPatient(gx, gy, gz);
          bool inDose = false;
          double dose = rtDose.doseAtPatient(patient, &inDose);
          if (!inDose) continue;
          int offset = t * binCount + bin;
          threadCounts[offset]++;
          threadMin[offset] = std::min(threadMin[offset], dose);
          threadMax[offset] = std::max(threadMax[offset], dose);
          threadSum[offset] += dose;
        }
      }
    }
  };

  QList<QFuture<void>> futures;
  for (int t = 0; t < threadCount; ++t) futures.append(QtConcurrent::run(worker, t));
  for (auto &f : futures) f.waitForFinished();

  for (int t = 0; t < threadCount; ++t) {
    for (int b = 0; b < binCount; ++b) {
      int offset = t * binCount + b;
      if (threadCounts[offset] == 0) continue;
      counts[b] += threadCounts[offset];
      result.minDoseGy[b] = std::min(result.minDoseGy[b], threadMin[offset]);
      result.maxDoseGy[b] = std::max(result.maxDoseGy[b], threadMax[offset]);
      result.meanDoseGy[b] += threadSum[offset];
    }
  }

  if (std::all_of(counts.begin(), counts.end(), [](int c) { return c == 0; })) return {};
  for (int i = 0; i < binCount; ++i) if (counts[i] > 0) result.meanDoseGy[i] /= counts[i];

  std::vector<double> distFiltered;
  std::vector<double> minFiltered;
  std::vector<double> maxFiltered;
  std::vector<double> meanFiltered;
  for (int i = 0; i < binCount; ++i) {
    if (counts[i] == 0) continue;
    distFiltered.push_back(result.distancesMm[i]);
    minFiltered.push_back(result.minDoseGy[i]);
    maxFiltered.push_back(result.maxDoseGy[i]);
    meanFiltered.push_back(result.meanDoseGy[i]);
  }
  result.distancesMm = std::move(distFiltered);
  result.minDoseGy = std::move(minFiltered);
  result.maxDoseGy = std::move(maxFiltered);
  result.meanDoseGy = std::move(meanFiltered);
  return result;
}
