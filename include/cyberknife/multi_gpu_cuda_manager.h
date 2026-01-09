#pragma once

#include "cyberknife/gpu_dose_backend.h"

#ifdef USE_CUDA_BACKEND

#include <QVector>
#include <QString>
#include <memory>
#include <vector>
#include <mutex>
#include <atomic>

namespace CyberKnife {

class CUDADoseBackend;

/**
 * @brief Multi-GPU manager for CUDA dose calculation
 *
 * Manages multiple CUDA devices for parallel beam calculations.
 * Distributes beams across available GPUs for maximum throughput.
 */
class MultiGPUCUDAManager : public IGPUDoseBackend {
public:
    /**
     * @brief Create multi-GPU manager
     * @param maxGPUs Maximum number of GPUs to use (0 = use all available)
     */
    explicit MultiGPUCUDAManager(int maxGPUs = 0);
    ~MultiGPUCUDAManager() override;

    // IGPUDoseBackend interface
    bool initialize() override;
    bool isReady() const override;
    GPUBackendType getBackendType() const override;
    QString getBackendName() const override;
    QString getDeviceInfo() const override;

    bool uploadCTVolume(const cv::Mat& volume) override;
    bool uploadBeamData(const GPUBeamData& beamData) override;
    bool calculateDose(const GPUComputeParams& params, cv::Mat& doseVolume) override;
    bool interpolateVolume(cv::Mat& doseVolume,
                          const std::vector<uint8_t>& computedMask,
                          int step) override;
    bool recalculateDoseWithThreshold(const GPUComputeParams& params,
                                      cv::Mat& doseVolume,
                                      float threshold,
                                      int skipStep = 4) override;

    QString getLastError() const override;
    void cleanup() override;

    // Multi-GPU specific methods

    /**
     * @brief Get number of active GPUs
     */
    int getGPUCount() const { return m_backends.size(); }

    /**
     * @brief Calculate dose for multiple beams in parallel
     * @param volume CT volume
     * @param beams List of beam geometries
     * @param beamWeights Weights for each beam
     * @param doseVolume Output dose volume (accumulated)
     * @param progressCallback Progress callback (beamIndex, totalBeams)
     * @return true if successful
     */
    bool calculateMultiBeamDose(const cv::Mat& ctVolume,
                               const QVector<GPUComputeParams>& beamParams,
                               cv::Mat& doseVolume,
                               std::function<void(int, int)> progressCallback = nullptr);

private:
    struct GPUBackendWrapper {
        std::unique_ptr<CUDADoseBackend> backend;
        int deviceId;
        bool busy;
        std::mutex mutex;

        GPUBackendWrapper(int id) : deviceId(id), busy(false) {}
    };

    std::vector<std::unique_ptr<GPUBackendWrapper>> m_backends;
    int m_maxGPUs;
    bool m_initialized;
    mutable std::mutex m_managerMutex;
    QString m_lastError;

    // CT volume cached for all GPUs
    cv::Mat m_cachedCTVolume;
    bool m_ctVolumeUploaded;

    // Beam data cached for all GPUs
    GPUBeamData m_cachedBeamData;
    bool m_beamDataUploaded;

    // Helper methods
    void setError(const QString& error);
    GPUBackendWrapper* acquireGPU();
    void releaseGPU(GPUBackendWrapper* gpu);
    bool initializeGPU(int deviceId);
};

} // namespace CyberKnife

#endif // USE_CUDA_BACKEND
