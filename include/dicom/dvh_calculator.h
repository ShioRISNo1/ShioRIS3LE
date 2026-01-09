// include/dicom/dvh_calculator.h の更新版
#ifndef DVH_CALCULATOR_H
#define DVH_CALCULATOR_H

#include <vector>
#include <QString>
#include <QColor>
#include <functional>
#include <atomic>
#include <memory>
#include "dicom/dicom_volume.h"
#include "dicom/dose_resampled_volume.h"
#include "dicom/rtdose_volume.h"
#include "dicom/rtstruct.h"

class DVHCalculator
{
public:
    struct DVHPoint {
        double dose;   // Gy
        double volume; // 累積体積[%]
    };

    struct DVHData {
        QString roiName;
        std::vector<DVHPoint> points;
        QColor color;
        bool isVisible{true};
        double maxDose{0.0};
        double minDose{0.0};
        double meanDose{0.0};
        double totalVolume{0.0};
    };

    // 設定構造体
    struct DVHSettings {
        double maxVolumeCm3{2000.0};        // 最大体積制限 (cm³)
        int maxVoxelCount{10000000};        // 最大ボクセル数制限
        bool skipLargeVolumes{true};        // 大きなボリュームをスキップ
        bool skipSupportStructures{true};  // サポート構造をスキップ
        // 線量ビンサイズ (Gy)。0以下の場合は最大線量の1/200を自動設定
        double binSize{0.0};
        
        // スキップするROI名のパターン
        QStringList skipPatterns{
            "outer", "external", "body", "support", "table", "couch",
            "skin", "surface", "outline", "contour", "air", "background"
        };
    };

    DVHCalculator() = default;
    ~DVHCalculator() = default;

    // コピー・移動の無効化（スレッド安全性のため）
    DVHCalculator(const DVHCalculator&) = delete;
    DVHCalculator& operator=(const DVHCalculator&) = delete;
    DVHCalculator(DVHCalculator&&) = delete;
    DVHCalculator& operator=(DVHCalculator&&) = delete;

    // メインの計算メソッド
    std::vector<DVHData> calculateDVH(const DicomVolume& ctVolume,
                                      const DoseResampledVolume& doseVolume,
                                      const RTStructureSet& structures,
                                      double binSize = 0.0, // 0以下で自動設定
                                      std::atomic_bool* cancel = nullptr,
                                      std::function<void(int, int)> progressCallback = {});

    // 設定付きの計算メソッド
    std::vector<DVHData> calculateDVHWithSettings(const DicomVolume& ctVolume,
                                                  const DoseResampledVolume& doseVolume,
                                                  const RTStructureSet& structures,
                                                  const DVHSettings& settings,
                                                  std::atomic_bool* cancel = nullptr,
                                                  std::function<void(int, int)> progressCallback = {});

    // 直接RTDoseからサンプリングしてDVHを計算（検証・代替経路）
    std::vector<DVHData> calculateDVHFromRTDose(const DicomVolume& ctVolume,
                                                const RTDoseVolume& rtDose,
                                                const RTStructureSet& structures,
                                                double binSize = 0.0,
                                                std::atomic_bool* cancel = nullptr,
                                                std::function<void(int, int)> progressCallback = {});

    // 単一ROI計算用の静的メソッド
    static DVHData calculateSingleROI(const DicomVolume& ctVolume,
                                     const DoseResampledVolume& doseVolume,
                                     const RTStructureSet& structures,
                                     int roiIndex,
                                     double binSize,
                                     std::atomic_bool* cancel,
                                     std::function<void(int, int)> progressCallback = {},
                                     double maxDoseCap = -1.0);

    static DVHData calculateSingleROIFromRTDose(const DicomVolume& ctVolume,
                                               const RTDoseVolume& rtDose,
                                               const RTStructureSet& structures,
                                               int roiIndex,
                                               double binSize,
                                               std::atomic_bool* cancel,
                                               std::function<void(int, int)> progressCallback = {},
                                               double maxDoseCap = -1.0);

    // ROIフィルタリング用のヘルパーメソッド
    static bool shouldSkipROI(const QString& roiName, 
                              const QVector3D& roiMin, 
                              const QVector3D& roiMax,
                              const DVHSettings& settings);

    static double estimateROIVolume(const QVector3D& roiMin, const QVector3D& roiMax);
    static int estimateVoxelCount(const DicomVolume& ctVolume,
                                  const QVector3D& roiMin, 
                                  const QVector3D& roiMax);

private:
    DVHSettings m_defaultSettings;
};

#endif // DVH_CALCULATOR_H
