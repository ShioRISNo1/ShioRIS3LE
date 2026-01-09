#pragma once

#include "cyberknife/gpu_dose_backend.h"

#ifdef USE_CUDA_BACKEND

#include <QString>
#include <vector>
#include <memory>

// Forward declarations for CUDA types (to avoid including cuda_runtime.h in header)
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

namespace CyberKnife {

/**
 * @brief CUDA implementation of GPU dose calculation backend
 *
 * Provides GPU-accelerated dose calculation using NVIDIA CUDA.
 * Optimized for RTX 3090 (Compute Capability 8.6)
 */
class CUDADoseBackend : public IGPUDoseBackend {
public:
    CUDADoseBackend();
    explicit CUDADoseBackend(int deviceId);
    ~CUDADoseBackend() override;

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

private:
    // CUDA device properties
    int m_deviceId;
    cudaStream_t m_stream;

    // Device memory buffers (using void* to avoid CUDA types in header)
    void* m_d_ctVolume;           // short*
    void* m_d_doseVolume;         // float*
    void* m_d_computedMask;       // uint8_t*

    // Beam data buffers on device
    void* m_d_ofTable;            // float*
    void* m_d_ofDepths;           // float*
    void* m_d_ofCollimators;      // float*
    void* m_d_tmrTable;           // float*
    void* m_d_tmrDepths;          // float*
    void* m_d_tmrFieldSizes;      // float*
    void* m_d_ocrTable;           // float*
    void* m_d_ocrDepths;          // float*
    void* m_d_ocrRadii;           // float*
    void* m_d_ocrCollimators;     // float*

    // Cached beam data dimensions
    struct BeamDataDimensions {
        int ofDepthCount = 0;
        int ofCollimatorCount = 0;
        int tmrDepthCount = 0;
        int tmrFieldSizeCount = 0;
        int ocrDepthCount = 0;
        int ocrRadiusCount = 0;
        int ocrCollimatorCount = 0;
    };
    BeamDataDimensions m_beamDataDims;

    // Volume dimensions (cached)
    int m_volumeWidth;
    int m_volumeHeight;
    int m_volumeDepth;

    // State
    bool m_initialized;
    QString m_lastError;

    // Helper methods
    bool selectBestDevice();
    bool createBuffers(int width, int height, int depth);
    void releaseBuffers();
    void setError(const QString& error);
    QString getCudaErrorString(int error) const;

    // Device capability check
    bool checkDeviceCapability();

    // Memory management helpers
    bool allocateDeviceMemory(void** ptr, size_t size, const QString& name);
    bool copyToDevice(void* dst, const void* src, size_t size, const QString& name);
    bool copyFromDevice(void* dst, const void* src, size_t size, const QString& name);
    void freeDeviceMemory(void** ptr);
};

} // namespace CyberKnife

#endif // USE_CUDA_BACKEND
