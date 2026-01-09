#pragma once

#include "cyberknife/geometry_calculator.h"
#include <opencv2/core.hpp>
#include <QVector3D>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

namespace CyberKnife {

/**
 * @brief GPU backend types for dose calculation
 */
enum class GPUBackendType {
    None,       // CPU fallback
    OpenCL,     // OpenCL backend (cross-platform)
    Metal,      // Metal backend (macOS, future)
    CUDA        // CUDA backend (NVIDIA, future)
};

/**
 * @brief Parameters for GPU dose calculation
 */
struct GPUComputeParams {
    // Volume dimensions
    int width;
    int height;
    int depth;

    // Voxel spacing in mm
    double spacingX;
    double spacingY;
    double spacingZ;

    // Volume origin in patient coordinates
    double originX;
    double originY;
    double originZ;

    // Image orientation (direction cosines)
    double orientationX[3];  // Row direction
    double orientationY[3];  // Column direction
    double orientationZ[3];  // Slice direction

    // Calculation step size for dynamic grid
    int stepX;
    int stepY;
    int stepZ;

    // Number of coarse grid samples produced for each axis
    int gridCountX = 0;
    int gridCountY = 0;
    int gridCountZ = 0;

    // Reference dose (usually 100.0 cGy)
    double referenceDose;

    // Beam geometry
    GeometryCalculator::BeamGeometry beam;

    // Cancellation support
    bool* cancelRequested = nullptr;

    // Precomputed beam depth profile (optional)
    double depthEntryDistance = std::numeric_limits<double>::quiet_NaN();
    double depthStepSize = 1.0;
    std::vector<float> depthCumulative;
};

/**
 * @brief Beam data tables for GPU upload
 */
struct GPUBeamData {
    // Output Factor table (2D: depth x collimator)
    std::vector<float> ofTable;
    std::vector<float> ofDepths;
    std::vector<float> ofCollimators;
    int ofDepthCount = 0;
    int ofCollimatorCount = 0;

    // TMR table (2D: depth x field size)
    std::vector<float> tmrTable;
    std::vector<float> tmrDepths;
    std::vector<float> tmrFieldSizes;
    int tmrDepthCount = 0;
    int tmrFieldSizeCount = 0;

    // OCR table (3D: depth x radius x collimator)
    std::vector<float> ocrTable;
    std::vector<float> ocrDepths;
    std::vector<float> ocrRadii;
    std::vector<float> ocrCollimators;
    int ocrDepthCount = 0;
    int ocrRadiusCount = 0;
    int ocrCollimatorCount = 0;

    // Matched collimator size for current calculation
    float matchedCollimator = 0.0f;
};

/**
 * @brief Abstract interface for GPU dose calculation backends
 */
class IGPUDoseBackend {
public:
    virtual ~IGPUDoseBackend() = default;

    /**
     * @brief Initialize the GPU backend
     * @return true if initialization successful
     */
    virtual bool initialize() = 0;

    /**
     * @brief Check if backend is ready for computation
     */
    virtual bool isReady() const = 0;

    /**
     * @brief Get backend type
     */
    virtual GPUBackendType getBackendType() const = 0;

    /**
     * @brief Get backend name string
     */
    virtual QString getBackendName() const = 0;

    /**
     * @brief Get GPU device information
     */
    virtual QString getDeviceInfo() const = 0;

    /**
     * @brief Upload CT volume to GPU memory
     * @param volume CT volume in HU values (short/int16)
     * @return true if upload successful
     */
    virtual bool uploadCTVolume(const cv::Mat& volume) = 0;

    /**
     * @brief Upload beam data tables to GPU memory
     * @param beamData Beam data lookup tables
     * @return true if upload successful
     */
    virtual bool uploadBeamData(const GPUBeamData& beamData) = 0;

    /**
     * @brief Calculate dose for entire volume on GPU
     * @param params Computation parameters
     * @param doseVolume Output dose volume (will be accumulated if not empty)
     * @return true if calculation successful
     */
    virtual bool calculateDose(const GPUComputeParams& params,
                               cv::Mat& doseVolume) = 0;

    /**
     * @brief Perform trilinear interpolation on GPU
     * @param doseVolume Input/output dose volume
     * @param computedMask Mask indicating which voxels were computed (vs interpolated)
     * @param step Current step size (4, 2, or 1)
     * @return true if interpolation successful
     */
    virtual bool interpolateVolume(cv::Mat& doseVolume,
                                   const std::vector<uint8_t>& computedMask,
                                   int step) = 0;

    /**
     * @brief Recalculate dose for voxels above threshold
     * @param params Computation parameters
     * @param doseVolume Input/output dose volume
     * @param threshold Only recalculate voxels with dose >= threshold
     * @param skipStep Skip grid points at this step (e.g., 4 for Pass 2, 2 for Pass 3)
     * @return true if recalculation successful
     */
    virtual bool recalculateDoseWithThreshold(const GPUComputeParams& params,
                                              cv::Mat& doseVolume,
                                              float threshold,
                                              int skipStep = 4) = 0;

    /**
     * @brief Get last error message
     */
    virtual QString getLastError() const = 0;

    /**
     * @brief Release GPU resources
     */
    virtual void cleanup() = 0;
};

/**
 * @brief Factory for creating GPU dose backends
 */
class GPUDoseBackendFactory {
public:
    /**
     * @brief Create GPU backend instance
     * @param preferred Preferred backend type (None = auto-detect)
     * @param useMultiGPU Enable multi-GPU mode for CUDA (if available)
     * @param maxGPUs Maximum number of GPUs to use (0 = use all available)
     * @return Unique pointer to backend, or nullptr if unavailable
     */
    static std::unique_ptr<IGPUDoseBackend> createBackend(
        GPUBackendType preferred = GPUBackendType::None,
        bool useMultiGPU = false,
        int maxGPUs = 0);

    /**
     * @brief Detect best available GPU backend
     * @return Best backend type, or None if no GPU available
     */
    static GPUBackendType detectBestBackend();

    /**
     * @brief Check if specific backend is available
     */
    static bool isBackendAvailable(GPUBackendType type);

    /**
     * @brief Get list of available backends
     */
    static QVector<GPUBackendType> getAvailableBackends();

    /**
     * @brief Get number of available CUDA devices
     * @return Number of CUDA GPUs, or 0 if CUDA not available
     */
    static int getCUDADeviceCount();
};

} // namespace CyberKnife
