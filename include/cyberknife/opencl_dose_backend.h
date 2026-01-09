#pragma once

#include "cyberknife/gpu_dose_backend.h"

#ifdef USE_OPENCL_BACKEND

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <QString>
#include <vector>
#include <memory>

namespace CyberKnife {

/**
 * @brief OpenCL implementation of GPU dose calculation backend
 */
class OpenCLDoseBackend : public IGPUDoseBackend {
public:
    OpenCLDoseBackend();
    ~OpenCLDoseBackend() override;

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
    // OpenCL objects
    cl_platform_id m_platform;
    cl_device_id m_device;
    cl_context m_context;
    cl_command_queue m_commandQueue;
    cl_program m_program;

    // Kernels
    cl_kernel m_doseKernel;
    cl_kernel m_interpolationKernel;
    cl_kernel m_thresholdRecalcKernel;

    // Device buffers
    cl_mem m_ctBuffer;
    cl_mem m_doseBuffer;
    cl_mem m_computedMaskBuffer;

    // Beam data buffers
    cl_mem m_ofTableBuffer;
    cl_mem m_ofDepthsBuffer;
    cl_mem m_ofCollimatorsBuffer;
    cl_mem m_tmrTableBuffer;
    cl_mem m_tmrDepthsBuffer;
    cl_mem m_tmrFieldSizesBuffer;
    cl_mem m_ocrTableBuffer;
    cl_mem m_ocrDepthsBuffer;
    cl_mem m_ocrRadiiBuffer;
    cl_mem m_ocrCollimatorsBuffer;

    // Beam data dimensions (cached from uploadBeamData)
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
    bool compileKernels();
    bool createBuffers(int width, int height, int depth);
    void releaseBuffers();
    void releaseKernels();
    void releaseOpenCLResources();
    void setError(const QString& error);
    QString getOpenCLErrorString(cl_int error) const;

    // Kernel compilation
    QString getKernelSource() const;
};

} // namespace CyberKnife

#endif // USE_OPENCL_BACKEND
