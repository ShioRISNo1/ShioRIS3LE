#include "dicom/dvh_calculator.h"
#include <QMutex>
#include <QMutexLocker>
#include <QApplication>
#include <QDebug>
#include <QVector3D>
#include <QtConcurrent>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <memory>

std::vector<DVHCalculator::DVHData> DVHCalculator::calculateDVH(
    const DicomVolume& ctVolume,
    const DoseResampledVolume& doseVolume,
    const RTStructureSet& structures,
    double binSize,
    std::atomic_bool* cancel,
    std::function<void(int, int)> progressCallback)
{
    std::vector<DVHData> dvhList;
    int roiCount = structures.roiCount();
    
    if (roiCount == 0 || !doseVolume.isResampled()) {
        qWarning() << "DVH calculation: No ROIs or dose volume not resampled";
        return dvhList;
    }

    qDebug() << QString("Starting DVH calculation for %1 ROIs").arg(roiCount);

    // binSizeが0以下なら最大線量の1/200を使用
    if (binSize <= 0.0) {
        binSize = doseVolume.maxDose() / 200.0;
        qDebug() << QString("Auto bin size set to %1 Gy").arg(binSize, 0, 'f', 4);
    }

    // ボリュームフィルタの設定
    const double MAX_VOLUME_CM3 = 2000.0;  // 2000cc以上は大きすぎるとみなす
    const int MAX_VOXEL_COUNT = 10000000;   // 1000万ボクセル以上は処理しない
    
    // スキップするROI名のパターン
    const QStringList SKIP_PATTERNS = {
        "outer", "external", "body", "support", "table", "couch",
        "skin", "surface", "outline", "contour", "air", "background"
    };

    dvhList.resize(roiCount);
    int processedCount = 0;
    int skippedCount = 0;

    // 事前フィルタリング：ROIのサイズを確認
    std::vector<bool> shouldProcess(roiCount, true);
    std::vector<double> estimatedVolumes(roiCount, 0.0);
    
    qDebug() << "=== ROI Pre-filtering ===";
    
    for (int r = 0; r < roiCount; ++r) {
        QString roiName = structures.roiName(r).toLower();
        
        // 名前による除外チェック
        bool skipByName = false;
        for (const QString& pattern : SKIP_PATTERNS) {
            if (roiName.contains(pattern, Qt::CaseInsensitive)) {
                skipByName = true;
                qDebug() << QString("ROI %1 (%2) skipped by name pattern: %3")
                            .arg(r).arg(structures.roiName(r)).arg(pattern);
                break;
            }
        }
        
        if (skipByName) {
            shouldProcess[r] = false;
            skippedCount++;
            continue;
        }
        
        // バウンディングボックスによるサイズ推定
        QVector3D roiMin, roiMax;
        if (structures.roiBoundingBox(r, roiMin, roiMax)) {
            // バウンディングボックスの体積を計算
            QVector3D size = roiMax - roiMin;
            double boundingVolumeCm3 = (size.x() * size.y() * size.z()) / 1000.0;  // mm³ to cm³
            estimatedVolumes[r] = boundingVolumeCm3;
            
            // ボクセル座標での範囲を計算
            QVector3D minVox = ctVolume.patientToVoxelContinuous(roiMin);
            QVector3D maxVox = ctVolume.patientToVoxelContinuous(roiMax);
            
            int x0 = std::clamp(static_cast<int>(std::floor(std::min(minVox.x(), maxVox.x()))), 0, ctVolume.width() - 1);
            int x1 = std::clamp(static_cast<int>(std::ceil(std::max(minVox.x(), maxVox.x()))), 0, ctVolume.width() - 1);
            int y0 = std::clamp(static_cast<int>(std::floor(std::min(minVox.y(), maxVox.y()))), 0, ctVolume.height() - 1);
            int y1 = std::clamp(static_cast<int>(std::ceil(std::max(minVox.y(), maxVox.y()))), 0, ctVolume.height() - 1);
            int z0 = std::clamp(static_cast<int>(std::floor(std::min(minVox.z(), maxVox.z()))), 0, ctVolume.depth() - 1);
            int z1 = std::clamp(static_cast<int>(std::ceil(std::max(minVox.z(), maxVox.z()))), 0, ctVolume.depth() - 1);
            
            int voxelCount = (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1);
            
            qDebug() << QString("ROI %1 (%2): bounding volume = %3 cm³, voxels = %4")
                        .arg(r).arg(structures.roiName(r))
                        .arg(boundingVolumeCm3, 0, 'f', 1)
                        .arg(voxelCount);
            
            // サイズフィルタリング
            if (boundingVolumeCm3 > MAX_VOLUME_CM3) {
                qDebug() << QString("ROI %1 (%2) exceeds volume threshold (%3 cm³ > %4 cm³) but will be processed")
                            .arg(r).arg(structures.roiName(r))
                            .arg(boundingVolumeCm3, 0, 'f', 1).arg(MAX_VOLUME_CM3);
            }

            if (voxelCount > MAX_VOXEL_COUNT) {
                qDebug() << QString("ROI %1 (%2) exceeds voxel threshold (%3 > %4) but will be processed")
                            .arg(r).arg(structures.roiName(r))
                            .arg(voxelCount).arg(MAX_VOXEL_COUNT);
            }
            
        } else {
            qWarning() << QString("ROI %1 (%2): No bounding box available")
                          .arg(r).arg(structures.roiName(r));
            // バウンディングボックスがない場合はスキップ
            shouldProcess[r] = false;
            skippedCount++;
        }
    }
    
    qDebug() << QString("Pre-filtering complete: %1 ROIs will be processed, %2 skipped")
                .arg(roiCount - skippedCount).arg(skippedCount);

    // 実際のDVH計算を並列処理
    std::vector<int> processIndices;
    processIndices.reserve(roiCount - skippedCount);

    for (int r = 0; r < roiCount; ++r) {
        if (!shouldProcess[r]) {
            DVHData skippedResult;
            skippedResult.roiName = structures.roiName(r);
            skippedResult.color = QColor::fromHsv((r * 40) % 360, 255, 200);
            skippedResult.isVisible = false;  // スキップされたROIは非表示
            skippedResult.maxDose = 0.0;
            skippedResult.totalVolume = 0.0;
            dvhList[r] = skippedResult;
        } else {
            processIndices.push_back(r);
        }
    }

    if (progressCallback) {
        progressCallback(0, static_cast<int>(processIndices.size()));
    }

    std::atomic<int> completed{0};
    auto worker = [&](int r) {
        if (cancel && cancel->load()) {
            return;
        }
        dvhList[r] = calculateSingleROI(ctVolume, doseVolume, structures, r, binSize, cancel, {});
        int done = ++completed;
        if (progressCallback) {
            progressCallback(done, static_cast<int>(processIndices.size()));
        }
    };

    QtConcurrent::blockingMap(processIndices, worker);

    if (progressCallback) {
        progressCallback(static_cast<int>(processIndices.size()), static_cast<int>(processIndices.size()));
    }

    qDebug() << QString("DVH calculation completed: %1 ROIs processed, %2 skipped")
                .arg(completed.load()).arg(skippedCount);
    return dvhList;
}

std::vector<DVHCalculator::DVHData> DVHCalculator::calculateDVHFromRTDose(
    const DicomVolume& ctVolume,
    const RTDoseVolume& rtDose,
    const RTStructureSet& structures,
    double binSize,
    std::atomic_bool* cancel,
    std::function<void(int, int)> progressCallback)
{
    std::vector<DVHData> dvhList;
    int roiCount = structures.roiCount();
    if (roiCount == 0 || rtDose.width() == 0) {
        qWarning() << "DVH calculation (RTDose): No ROIs or dose not loaded";
        return dvhList;
    }

    if (binSize <= 0.0) {
        // 粗い推定: RTDOSEの最大線量が分からない場合は1/200 of 100 Gy相当を仮定
        binSize = std::max(0.01, 100.0 / 200.0);
    }

    dvhList.resize(roiCount);
    std::vector<int> processIndices;
    for (int r = 0; r < roiCount; ++r) {
        processIndices.push_back(r);
    }
    if (progressCallback) progressCallback(0, static_cast<int>(processIndices.size()));

    std::atomic<int> completed{0};
    auto worker = [&](int r) {
        if (cancel && cancel->load()) return;
        dvhList[r] = calculateSingleROIFromRTDose(ctVolume, rtDose, structures, r, binSize, cancel, {});
        int done = ++completed;
        if (progressCallback) progressCallback(done, static_cast<int>(processIndices.size()));
    };
    QtConcurrent::blockingMap(processIndices, worker);
    if (progressCallback) progressCallback(static_cast<int>(processIndices.size()), static_cast<int>(processIndices.size()));
    return dvhList;
}

DVHCalculator::DVHData DVHCalculator::calculateSingleROI(
    const DicomVolume& ctVolume,
    const DoseResampledVolume& doseVolume,
    const RTStructureSet& structures,
    int roiIndex,
    double binSize,
    std::atomic_bool* cancel,
    std::function<void(int, int)> progressCallback,
    double maxDoseCap)
{
    DVHData result;
    
    result.roiName = structures.roiName(roiIndex);
    result.color = QColor::fromHsv((roiIndex * 40) % 360, 255, 200);
    result.isVisible = structures.isROIVisible(roiIndex);

    const int w = ctVolume.width();
    const int h = ctVolume.height();
    const int d = ctVolume.depth();
    const double voxelVolume = ctVolume.spacingX() * ctVolume.spacingY() * ctVolume.spacingZ();

    // バウンディングボックスを取得
    QVector3D roiMin, roiMax;
    if (!structures.roiBoundingBox(roiIndex, roiMin, roiMax)) {
        qWarning() << "No bounding box for ROI" << roiIndex;
        return result;
    }

    // ボクセル座標に変換
    QVector3D minVox = ctVolume.patientToVoxelContinuous(roiMin);
    QVector3D maxVox = ctVolume.patientToVoxelContinuous(roiMax);
    
    int x0 = std::clamp(static_cast<int>(std::floor(std::min(minVox.x(), maxVox.x()))), 0, w - 1);
    int x1 = std::clamp(static_cast<int>(std::ceil(std::max(minVox.x(), maxVox.x()))), 0, w - 1);
    int y0 = std::clamp(static_cast<int>(std::floor(std::min(minVox.y(), maxVox.y()))), 0, h - 1);
    int y1 = std::clamp(static_cast<int>(std::ceil(std::max(minVox.y(), maxVox.y()))), 0, h - 1);
    int z0 = std::clamp(static_cast<int>(std::floor(std::min(minVox.z(), maxVox.z()))), 0, d - 1);
    int z1 = std::clamp(static_cast<int>(std::ceil(std::max(minVox.z(), maxVox.z()))), 0, d - 1);

    std::vector<double> doses;
    // メモリ効率的な予約
    int estimatedSize = (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1) / 100;  // 保守的予約
    doses.reserve(std::min(estimatedSize, 2000000));  // 最大2M要素まで

    int totalVoxels = (z1 - z0 + 1) * (y1 - y0 + 1) * (x1 - x0 + 1);
    std::atomic<int> processedVoxels{0};

    if (progressCallback) {
        progressCallback(0, totalVoxels);
    }

    QMutex doseMutex;
    std::vector<int> zIndices(z1 - z0 + 1);
    std::iota(zIndices.begin(), zIndices.end(), z0);

    int stepSize = 1;

    auto sliceFunc = [&](int z) {
        if (cancel && cancel->load()) {
            return;
        }
        std::vector<double> local;
        local.reserve((x1 - x0 + 1) * (y1 - y0 + 1) / 4);
        for (int y = y0; y <= y1 && !(cancel && cancel->load()); ++y) {
            for (int x = x0; x <= x1 && !(cancel && cancel->load()); ++x) {
                QVector3D patient = ctVolume.voxelToPatient(x + 0.5, y + 0.5, z + 0.5);
                if (structures.isPointInsideROI(patient, roiIndex)) {
                    float dose = doseVolume.voxelDose(x, y, z);
                    local.push_back(static_cast<double>(dose));
                }
            }
        }
        {
            QMutexLocker locker(&doseMutex);
            doses.insert(doses.end(), local.begin(), local.end());
        }
        int sliceVoxels = (x1 - x0 + 1) * (y1 - y0 + 1);
        int pv = processedVoxels.fetch_add(sliceVoxels) + sliceVoxels;
        if (progressCallback) {
            progressCallback(pv, totalVoxels);
        }
    };

    QtConcurrent::blockingMap(zIndices, sliceFunc);

    if (progressCallback) {
        progressCallback(totalVoxels, totalVoxels);
    }
    if (cancel && cancel->load()) {
        qDebug() << QString("DVH calculation cancelled for ROI %1").arg(roiIndex);
        return result;
    }

    // サンプリングによる体積補正
    double samplingFactor = stepSize * stepSize * stepSize;
    result.totalVolume = doses.size() * voxelVolume * samplingFactor;
    
    if (doses.empty()) {
        qWarning() << QString("No dose points found for ROI %1").arg(roiIndex);
        return result;
    }

    result.maxDose = *std::max_element(doses.begin(), doses.end());
    result.minDose = *std::min_element(doses.begin(), doses.end());
    result.meanDose = std::accumulate(doses.begin(), doses.end(), 0.0) /
                      static_cast<double>(doses.size());

    // ヒストグラム範囲を設定（CalcMax指定時はそれに合わせる）
    double maxUpper = (maxDoseCap > 0.0) ? maxDoseCap : doseVolume.maxDose();
    int binCount = static_cast<int>(std::round(maxUpper / binSize)) + 1; // 0..N inclusive
    binCount = std::min(binCount, 5000);  // ビン数制限

    std::vector<double> histogram(binCount, 0.0);
    double adjustedVoxelVolume = voxelVolume * samplingFactor;
    
    for (double dose : doses) {
        double capped = (maxDoseCap > 0.0) ? std::min(dose, maxUpper) : dose;
        int bin = static_cast<int>(capped / binSize);
        bin = std::clamp(bin, 0, binCount - 1);
        histogram[bin] += adjustedVoxelVolume;
    }

    // 累積DVHを計算
    double cumulative = 0.0;
    double total = result.totalVolume;
    result.points.reserve(binCount);
    
    for (int i = binCount - 1; i >= 0; --i) {
        cumulative += histogram[i];
        double volPct = (total > 0.0) ? (cumulative / total * 100.0) : 0.0;
        result.points.push_back({ i * binSize, volPct });
    }

    std::reverse(result.points.begin(), result.points.end());

    qDebug() << QString("ROI %1 completed: %2 dose points (step=%3), volume=%4 cm³, %5 DVH points, cap=%6 Gy, binSize=%7")
                .arg(roiIndex).arg(doses.size()).arg(stepSize)
                .arg(result.totalVolume / 1000.0, 0, 'f', 1).arg(result.points.size())
                .arg((maxDoseCap>0.0)?maxUpper:-1.0).arg(binSize);

    return result;
}

DVHCalculator::DVHData DVHCalculator::calculateSingleROIFromRTDose(
    const DicomVolume& ctVolume,
    const RTDoseVolume& rtDose,
    const RTStructureSet& structures,
    int roiIndex,
    double binSize,
    std::atomic_bool* cancel,
    std::function<void(int, int)> progressCallback,
    double maxDoseCap)
{
    DVHData result;
    result.roiName = structures.roiName(roiIndex);
    result.color = QColor::fromHsv((roiIndex * 40) % 360, 255, 200);
    result.isVisible = structures.isROIVisible(roiIndex);

    const int w = ctVolume.width();
    const int h = ctVolume.height();
    const int d = ctVolume.depth();

    QVector3D roiMin, roiMax;
    if (!structures.roiBoundingBox(roiIndex, roiMin, roiMax)) {
        qWarning() << "No bounding box for ROI" << roiIndex;
        return result;
    }

    QVector3D minVox = ctVolume.patientToVoxelContinuous(roiMin);
    QVector3D maxVox = ctVolume.patientToVoxelContinuous(roiMax);
    int x0 = std::clamp(static_cast<int>(std::floor(std::min(minVox.x(), maxVox.x()))), 0, w - 1);
    int x1 = std::clamp(static_cast<int>(std::ceil(std::max(minVox.x(), maxVox.x()))), 0, w - 1);
    int y0 = std::clamp(static_cast<int>(std::floor(std::min(minVox.y(), maxVox.y()))), 0, h - 1);
    int y1 = std::clamp(static_cast<int>(std::ceil(std::max(minVox.y(), maxVox.y()))), 0, h - 1);
    int z0 = std::clamp(static_cast<int>(std::floor(std::min(minVox.z(), maxVox.z()))), 0, d - 1);
    int z1 = std::clamp(static_cast<int>(std::ceil(std::max(minVox.z(), maxVox.z()))), 0, d - 1);

    // バケット準備
    // Use Gy bins up to cap if provided
    double maxUpper = (maxDoseCap > 0.0) ? maxDoseCap : 100.0; // fallback
    const int binCount = std::max(1, static_cast<int>(std::round(maxUpper / binSize)) + 1);
    std::vector<double> bins(binCount, 0.0);
    std::vector<int> counts(binCount, 0);

    int totalVoxels = (z1 - z0 + 1) * (y1 - y0 + 1) * (x1 - x0 + 1);
    if (progressCallback) progressCallback(0, totalVoxels);

    int processed = 0;
    for (int z = z0; z <= z1; ++z) {
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                if (cancel && cancel->load()) return result;
                QVector3D patient = ctVolume.voxelToPatient(x + 0.5, y + 0.5, z + 0.5);
                if (!structures.isPointInsideROI(patient, roiIndex)) continue;
                bool inside = false;
                double dose = static_cast<double>(rtDose.doseAtPatient(patient, &inside));
                if (!inside) continue;
                double capped = (maxDoseCap > 0.0) ? std::min(dose, maxUpper) : dose;
                int bin = static_cast<int>(std::floor(capped / binSize));
                bin = std::clamp(bin, 0, binCount - 1);
                bins[bin] += 1.0;
                counts[bin] += 1;
                if ((++processed % 10000) == 0 && progressCallback) progressCallback(processed, totalVoxels);
            }
        }
    }

    // 累積体積[%]に変換
    int totalCount = std::accumulate(counts.begin(), counts.end(), 0);
    if (totalCount == 0) return result;
    result.points.clear();
    double cumulative = 100.0;
    for (int i = 0; i < binCount; ++i) {
        double doseGy = (i + 1) * binSize;
        int voxelsAtOrAbove = 0;
        for (int j = i; j < binCount; ++j) voxelsAtOrAbove += counts[j];
        double volPercent = 100.0 * voxelsAtOrAbove / totalCount;
        result.points.push_back({doseGy, volPercent});
    }
    result.maxDose = binSize * (binCount - 1);
    result.minDose = 0.0;
    // 平均線量（近似）：ビン中央値 * 出現数で計算
    double sumDose = 0.0;
    for (int i = 0; i < binCount; ++i) sumDose += ((i + 0.5) * binSize) * counts[i];
    result.meanDose = sumDose / totalCount;
    return result;
}

double DVHCalculator::estimateROIVolume(const QVector3D &roiMin,
                                        const QVector3D &roiMax)
{
    QVector3D size = roiMax - roiMin;
    double volumeMm3 = size.x() * size.y() * size.z();
    return volumeMm3 / 1000.0; // mm^3 -> cm^3
}

int DVHCalculator::estimateVoxelCount(const DicomVolume &ctVolume,
                                      const QVector3D &roiMin,
                                      const QVector3D &roiMax)
{
    QVector3D minVox = ctVolume.patientToVoxelContinuous(roiMin);
    QVector3D maxVox = ctVolume.patientToVoxelContinuous(roiMax);

    int x0 = std::clamp(static_cast<int>(std::floor(std::min(minVox.x(), maxVox.x()))),
                        0, ctVolume.width() - 1);
    int x1 = std::clamp(static_cast<int>(std::ceil(std::max(minVox.x(), maxVox.x()))),
                        0, ctVolume.width() - 1);
    int y0 = std::clamp(static_cast<int>(std::floor(std::min(minVox.y(), maxVox.y()))),
                        0, ctVolume.height() - 1);
    int y1 = std::clamp(static_cast<int>(std::ceil(std::max(minVox.y(), maxVox.y()))),
                        0, ctVolume.height() - 1);
    int z0 = std::clamp(static_cast<int>(std::floor(std::min(minVox.z(), maxVox.z()))),
                        0, ctVolume.depth() - 1);
    int z1 = std::clamp(static_cast<int>(std::ceil(std::max(minVox.z(), maxVox.z()))),
                        0, ctVolume.depth() - 1);

    return (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1);
}
