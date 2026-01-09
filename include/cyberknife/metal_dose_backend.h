#pragma once

#include "cyberknife/gpu_dose_backend.h"

#ifdef USE_METAL_BACKEND

#ifdef __OBJC__
@class MTLDevice;
@class MTLCommandQueue;
@class MTLLibrary;
@class MTLComputePipelineState;
@class MTLBuffer;
#else
typedef void MTLDevice;
typedef void MTLCommandQueue;
typedef void MTLLibrary;
typedef void MTLComputePipelineState;
typedef void MTLBuffer;
#endif

#include <QString>
#include <memory>

namespace CyberKnife {

/**
 * @brief Metal implementation of GPU dose calculation backend (macOS only)
 *
 * Uses Apple's Metal API for GPU compute on macOS.
 * Provides optimal performance on both Intel and Apple Silicon Macs.
 */
class MetalDoseBackend : public IGPUDoseBackend {
public:
    MetalDoseBackend();
    ~MetalDoseBackend() override;

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
    // Metal objects (using void* to avoid Objective-C in header)
    void* m_device;              // id<MTLDevice>
    void* m_commandQueue;        // id<MTLCommandQueue>
    void* m_library;             // id<MTLLibrary>
    void* m_doseKernel;          // id<MTLComputePipelineState>
    void* m_interpolationKernel; // id<MTLComputePipelineState>
    void* m_thresholdRecalcKernel; // id<MTLComputePipelineState>

    // Device buffers
    void* m_ctBuffer;            // id<MTLBuffer>
    void* m_doseBuffer;          // id<MTLBuffer>
    void* m_computedMaskBuffer;  // id<MTLBuffer>

    // Beam data buffers
    void* m_ofTableBuffer;
    void* m_ofDepthsBuffer;
    void* m_ofCollimatorsBuffer;
    void* m_tmrTableBuffer;
    void* m_tmrDepthsBuffer;
    void* m_tmrFieldSizesBuffer;
    void* m_ocrTableBuffer;
    void* m_ocrDepthsBuffer;
    void* m_ocrRadiiBuffer;
    void* m_ocrCollimatorsBuffer;

    // Volume dimensions (cached)
    int m_volumeWidth;
    int m_volumeHeight;
    int m_volumeDepth;

    // Beam data table dimensions (cached from uploadBeamData)
    int m_ofDepthCount;
    int m_ofCollimatorCount;
    int m_tmrDepthCount;
    int m_tmrFieldSizeCount;
    int m_ocrDepthCount;
    int m_ocrRadiusCount;
    int m_ocrCollimatorCount;

    // State
    bool m_initialized;
    QString m_lastError;

    // Helper methods
    bool createMetalDevice();
    bool compileShaders();
    bool createBuffers(int width, int height, int depth);
    void releaseBuffers();
    void releaseMetalResources();
    void setError(const QString& error);
};

} // namespace CyberKnife

#endif // USE_METAL_BACKEND
