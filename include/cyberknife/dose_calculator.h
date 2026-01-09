#pragma once

#include "cyberknife/beam_data_manager.h"
#include "cyberknife/geometry_calculator.h"

#ifdef ENABLE_GPU_DOSE_CALCULATION
#include "cyberknife/gpu_dose_backend.h"
#endif

#include <QLoggingCategory>
#include <QPair>
#include <QtGlobal>
#include <QString>
#include <QStringList>
#include <QVector>
#include <atomic>
#include <functional>
#include <memory>
#include <limits>

class DicomVolume;
class RTDoseVolume;

namespace CyberKnife {

class FastDoseLookup;

struct FastDoseLookupDeleter {
    void operator()(FastDoseLookup *ptr) const;
};

using FastDoseLookupPtr = std::unique_ptr<FastDoseLookup, FastDoseLookupDeleter>;

Q_DECLARE_LOGGING_CATEGORY(CyberKnifeDoseLog)

class CyberKnifeDoseCalculator {
public:
    struct ResolutionOptions {
        int stepXY{1};
        int stepZ{1};
        // Dynamic Grid Size Mode: adaptively refines high-dose regions
        bool useDynamicGridMode{true};
        double dynamicThresholdStep2{0.5};  // Refine regions >= 50% of max dose with Step2
        double dynamicThresholdStep1{0.7};  // Refine regions >= 70% of max dose with Step1
        // Recalculation thresholds based on interpolation ratio (should match dose thresholds)
        double dynamicInterpolationThresholdStep2{0.5};  // Recalculate if interpolation ratio > this value
        double dynamicInterpolationThresholdStep1{0.7};  // Warning if interpolation ratio > this value
        // Cancellation flag pointer (set by UI thread, checked by calculation thread)
        std::atomic<bool> *cancelRequested{nullptr};
    };

    struct CalculationPoint {
        QVector3D position;
        double density = 1.0;
    };

    struct RayTracingPointMetrics {
        double ssd = std::numeric_limits<double>::quiet_NaN();
        double depth = std::numeric_limits<double>::quiet_NaN();
        double effectiveDepth = std::numeric_limits<double>::quiet_NaN();
        double offAxis = std::numeric_limits<double>::quiet_NaN();
        double offAxisX = std::numeric_limits<double>::quiet_NaN();
        double offAxisY = std::numeric_limits<double>::quiet_NaN();
        double radiusSad = std::numeric_limits<double>::quiet_NaN();
        double radius800 = std::numeric_limits<double>::quiet_NaN();
        double tmr = std::numeric_limits<double>::quiet_NaN();
        double unitDose = std::numeric_limits<double>::quiet_NaN();
    };

    struct DensityTableInfo {
        QVector<QPair<double, double>> entries;
        double offset = 0.0;
        QString source;

        bool isValid() const { return !entries.isEmpty(); }
    };

    enum class DoseModel {
        PrimaryPlusScatter,
        PrimaryOnly,
        RayTracing,
    };

    bool initialize(const QString &beamDataPath);

    ~CyberKnifeDoseCalculator();

    void setResolutionOptions(const ResolutionOptions &options);
    ResolutionOptions resolutionOptions() const { return m_resolutionOptions; }

    double calculatePointDose(const CalculationPoint &point,
                              const GeometryCalculator::BeamGeometry &beam,
                              double referencedose = 100.0);

    double calculatePointDoseWithGeometry(const CalculationPoint &point,
                                          const GeometryCalculator::BeamGeometry &beam,
                                          double depth,
                                          double offAxis,
                                          double referencedose = 100.0);

    bool computeRayTracingPointMetrics(const CalculationPoint &point,
                                       const GeometryCalculator::BeamGeometry &beam,
                                       const DicomVolume *ctVolume,
                                       RayTracingPointMetrics *outMetrics) const;

    bool calculateVolumeDose(const DicomVolume &ctVolume,
                             const GeometryCalculator::BeamGeometry &beam,
                             RTDoseVolume &resultDose);

    bool calculateVolumeDoseWithProgress(const DicomVolume &ctVolume,
                                         const GeometryCalculator::BeamGeometry &beam,
                                         RTDoseVolume &resultDose,
                                         std::function<void(int)> progressCallback);

    bool calculateMultiBeamVolumeDose(const DicomVolume &ctVolume,
                                      const QVector<GeometryCalculator::BeamGeometry> &beams,
                                      const QVector<double> &beamWeights,
                                      RTDoseVolume &resultDose);

    bool calculateMultiBeamVolumeDoseWithProgress(
        const DicomVolume &ctVolume,
        const QVector<GeometryCalculator::BeamGeometry> &beams,
        const QVector<double> &beamWeights,
        std::function<void(int, int)> progressCallback,
        RTDoseVolume &resultDose);

    bool isReady() const { return m_beamDataManager && m_beamDataManager->isDataLoaded(); }

    const QStringList &lastErrors() const { return m_lastErrors; }

    const BeamDataManager *beamDataManager() const { return m_beamDataManager.get(); }

    bool resolveOcrTableInfo(double collimatorSize,
                             double *matchedCollimator,
                             QString *sourceFile) const;

    struct FastLookupContext {
        const void *coneData = nullptr;
        const BeamDataParser::OCRData *managerOcr = nullptr;
        bool useFastLookup = false;
        bool hasOffAxisLimit = false;
        double offAxisLimit = 0.0;
        double matchedCollimator = 0.0;
        double managerMatchedCollimator = 0.0;
        QString sourceFile;
        QString managerSourceFile;
    };

    FastLookupContext buildFastLookupContext(double collimatorSize) const;
    double calculatePointDoseWithContext(const CalculationPoint &point,
                                         const GeometryCalculator::BeamGeometry &beam,
                                         double depth,
                                         double offAxis,
                                         double referencedose,
                                         const FastLookupContext &context) const;

    DensityTableInfo densityTableInfo() const;

    bool applyDensityTableOverride(const DensityTableInfo &info);
    void clearDensityTableOverride();

    void setDoseModel(DoseModel model);
    DoseModel doseModel() const;

#ifdef ENABLE_GPU_DOSE_CALCULATION
    // GPU acceleration control
    bool initializeGPU(bool useMultiGPU = false, int maxGPUs = 0);
    bool isGPUEnabled() const;
    void setGPUEnabled(bool enabled);
    QString getGPUDeviceInfo() const;
    int getGPUCount() const;
#endif

private:
    std::unique_ptr<BeamDataManager> m_beamDataManager;
    FastDoseLookupPtr m_fastLookup;
    QStringList m_lastErrors;
    ResolutionOptions m_resolutionOptions;
    std::atomic<DoseModel> m_doseModel{DoseModel::RayTracing};  // Default: RayTracing

#ifdef ENABLE_GPU_DOSE_CALCULATION
    std::unique_ptr<IGPUDoseBackend> m_gpuBackend;
    bool m_gpuEnabled{false};

    // GPU helper methods
    GPUBeamData convertBeamDataToGPU(double collimatorSize) const;
    GPUComputeParams createGPUComputeParams(const DicomVolume &volume,
                                            const GeometryCalculator::BeamGeometry &beam,
                                            int stepX, int stepY, int stepZ,
                                            double referenceDose) const;
#endif

    bool resolveDoseFactors(const GeometryCalculator::BeamGeometry &beam,
                            double depth,
                            double radius800,
                            double sad,
                            const FastLookupContext &context,
                            double &outputFactor,
                            double &ocrRatio,
                            double &tmr) const;

    static bool prepareDoseVolumeStorage(const DicomVolume &ctVolume,
                                         RTDoseVolume &doseVolume,
                                         QStringList &errors);
    static bool applyToVolumeParallel(const DicomVolume &volume,
                                      RTDoseVolume &doseVolume,
                                      const std::function<double(int, int, int)> &calculator,
                                      bool accumulate,
                                      const QVector<int> &xIndices,
                                      const QVector<int> &yIndices,
                                      const QVector<int> &zIndices,
                                      QVector<quint8> *computedMask,
                                      std::atomic<bool> *cancelRequested);
};

} // namespace CyberKnife

