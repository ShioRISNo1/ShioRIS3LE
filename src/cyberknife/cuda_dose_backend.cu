#include "cyberknife/cuda_dose_backend.h"

#ifdef USE_CUDA_BACKEND

#include <cuda_runtime.h>
#include <QDebug>
#include <algorithm>
#include <cstring>

// Include CUDA kernels from separate file
#include "cuda_kernels.cu"

namespace CyberKnife {

namespace {

int computeCoarseGridCount(int size, int step)
{
    if (size <= 0) {
        return 0;
    }

    step = std::max(1, step);

    int count = 1;
    for (int value = step; value < size - 1; value += step) {
        ++count;
    }

    if (size > 1) {
        ++count;
    }

    return count;
}

} // anonymous namespace

CUDADoseBackend::CUDADoseBackend()
    : m_deviceId(-1)  // -1 means auto-select
    , m_stream(nullptr)
    , m_d_ctVolume(nullptr)
    , m_d_doseVolume(nullptr)
    , m_d_computedMask(nullptr)
    , m_d_ofTable(nullptr)
    , m_d_ofDepths(nullptr)
    , m_d_ofCollimators(nullptr)
    , m_d_tmrTable(nullptr)
    , m_d_tmrDepths(nullptr)
    , m_d_tmrFieldSizes(nullptr)
    , m_d_ocrTable(nullptr)
    , m_d_ocrDepths(nullptr)
    , m_d_ocrRadii(nullptr)
    , m_d_ocrCollimators(nullptr)
    , m_volumeWidth(0)
    , m_volumeHeight(0)
    , m_volumeDepth(0)
    , m_initialized(false)
{
}

CUDADoseBackend::CUDADoseBackend(int deviceId)
    : m_deviceId(deviceId)
    , m_stream(nullptr)
    , m_d_ctVolume(nullptr)
    , m_d_doseVolume(nullptr)
    , m_d_computedMask(nullptr)
    , m_d_ofTable(nullptr)
    , m_d_ofDepths(nullptr)
    , m_d_ofCollimators(nullptr)
    , m_d_tmrTable(nullptr)
    , m_d_tmrDepths(nullptr)
    , m_d_tmrFieldSizes(nullptr)
    , m_d_ocrTable(nullptr)
    , m_d_ocrDepths(nullptr)
    , m_d_ocrRadii(nullptr)
    , m_d_ocrCollimators(nullptr)
    , m_volumeWidth(0)
    , m_volumeHeight(0)
    , m_volumeDepth(0)
    , m_initialized(false)
{
}

CUDADoseBackend::~CUDADoseBackend()
{
    cleanup();
}

bool CUDADoseBackend::initialize()
{
    if (m_initialized) {
        return true;
    }

    qDebug() << "CUDA: Initializing backend...";

    // Select best device
    if (!selectBestDevice()) {
        setError("Failed to select CUDA device");
        return false;
    }

    // Check device capability
    if (!checkDeviceCapability()) {
        return false;
    }

    // Create CUDA stream for async operations
    cudaError_t err = cudaStreamCreate(&m_stream);
    if (err != cudaSuccess) {
        setError(QString("Failed to create CUDA stream: %1").arg(getCudaErrorString(err)));
        return false;
    }

    m_initialized = true;
    qDebug() << "CUDA: Initialization successful";
    qDebug() << "CUDA: Device:" << getDeviceInfo();

    return true;
}

bool CUDADoseBackend::isReady() const
{
    return m_initialized;
}

GPUBackendType CUDADoseBackend::getBackendType() const
{
    return GPUBackendType::CUDA;
}

QString CUDADoseBackend::getBackendName() const
{
    return "CUDA";
}

QString CUDADoseBackend::getDeviceInfo() const
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, m_deviceId);

    if (err != cudaSuccess) {
        return "Unknown CUDA device";
    }

    return QString("%1 (Compute %2.%3, %4 MB)")
        .arg(prop.name)
        .arg(prop.major)
        .arg(prop.minor)
        .arg(prop.totalGlobalMem / (1024 * 1024));
}

bool CUDADoseBackend::selectBestDevice()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    qDebug() << "CUDA: Device detection started";
    qDebug() << "CUDA: Found" << deviceCount << "device(s)";

    if (err != cudaSuccess || deviceCount == 0) {
        setError(QString("No CUDA devices found: %1").arg(getCudaErrorString(err)));
        qWarning() << "CUDA: Device detection failed";
        return false;
    }

    // If device ID was specified in constructor, use it
    if (m_deviceId >= 0) {
        if (m_deviceId >= deviceCount) {
            setError(QString("Requested CUDA device %1 not found (only %2 devices available)")
                         .arg(m_deviceId).arg(deviceCount));
            return false;
        }

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, m_deviceId);
        qDebug() << "CUDA: Using specified device" << m_deviceId << ":" << prop.name
                 << "Compute" << prop.major << "." << prop.minor
                 << "Memory:" << (prop.totalGlobalMem / (1024 * 1024)) << "MB";
    } else {
        // Auto-select: find the device with the highest compute capability
        int bestDevice = 0;
        int maxComputeCapability = 0;

        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);

            int computeCapability = prop.major * 10 + prop.minor;
            qDebug() << "CUDA: Device" << i << ":" << prop.name
                     << "Compute" << prop.major << "." << prop.minor
                     << "Memory:" << (prop.totalGlobalMem / (1024 * 1024)) << "MB";

            if (computeCapability > maxComputeCapability) {
                maxComputeCapability = computeCapability;
                bestDevice = i;
            }
        }

        m_deviceId = bestDevice;
        qDebug() << "CUDA: ✓ Auto-selected device" << m_deviceId;
    }

    err = cudaSetDevice(m_deviceId);
    if (err != cudaSuccess) {
        setError(QString("Failed to set CUDA device %1: %2").arg(m_deviceId).arg(getCudaErrorString(err)));
        return false;
    }

    return true;
}

bool CUDADoseBackend::checkDeviceCapability()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, m_deviceId);

    // Check minimum compute capability (3.0 for atomicAdd on floats)
    if (prop.major < 3) {
        setError(QString("CUDA Compute Capability %1.%2 is too old (minimum 3.0 required)")
                     .arg(prop.major).arg(prop.minor));
        return false;
    }

    qDebug() << "CUDA: Device capability check passed";
    qDebug() << "CUDA: Compute Capability:" << prop.major << "." << prop.minor;
    qDebug() << "CUDA: Total Memory:" << (prop.totalGlobalMem / (1024 * 1024)) << "MB";
    qDebug() << "CUDA: Multiprocessors:" << prop.multiProcessorCount;

    return true;
}

bool CUDADoseBackend::uploadCTVolume(const cv::Mat& volume)
{
    if (!isReady()) {
        setError("Backend not initialized");
        return false;
    }

    // Set the CUDA device for this thread (critical for multi-GPU)
    {
        cudaError_t deviceErr = cudaSetDevice(m_deviceId);
        if (deviceErr != cudaSuccess) {
            setError(QString("Failed to set CUDA device %1: %2").arg(m_deviceId).arg(getCudaErrorString(deviceErr)));
            return false;
        }
    }

    if (volume.empty() || volume.type() != CV_16S) {
        setError("Invalid CT volume format (expected CV_16S)");
        return false;
    }

    int width = volume.size[2];
    int height = volume.size[1];
    int depth = volume.size[0];

    qDebug() << "CUDA: Uploading CT volume on device" << m_deviceId << ":" << width << "x" << height << "x" << depth;

    // Create buffers if needed
    if (width != m_volumeWidth || height != m_volumeHeight || depth != m_volumeDepth) {
        releaseBuffers();
        if (!createBuffers(width, height, depth)) {
            return false;
        }
        m_volumeWidth = width;
        m_volumeHeight = height;
        m_volumeDepth = depth;
    }

    // Upload CT data
    size_t bufferSize = width * height * depth * sizeof(short);
    if (!copyToDevice(m_d_ctVolume, volume.data, bufferSize, "CT volume")) {
        return false;
    }

    qDebug() << "CUDA: CT volume uploaded successfully";
    return true;
}

bool CUDADoseBackend::uploadBeamData(const GPUBeamData& beamData)
{
    if (!isReady()) {
        setError("Backend not initialized");
        return false;
    }

    // Set the CUDA device for this thread (critical for multi-GPU)
    {
        cudaError_t deviceErr = cudaSetDevice(m_deviceId);
        if (deviceErr != cudaSuccess) {
            setError(QString("Failed to set CUDA device %1: %2").arg(m_deviceId).arg(getCudaErrorString(deviceErr)));
            return false;
        }
    }

    qDebug() << "CUDA: Uploading beam data tables on device" << m_deviceId << "...";

    // Store beam data dimensions
    m_beamDataDims.ofDepthCount = beamData.ofDepthCount;
    m_beamDataDims.ofCollimatorCount = beamData.ofCollimatorCount;
    m_beamDataDims.tmrDepthCount = beamData.tmrDepthCount;
    m_beamDataDims.tmrFieldSizeCount = beamData.tmrFieldSizeCount;
    m_beamDataDims.ocrDepthCount = beamData.ocrDepthCount;
    m_beamDataDims.ocrRadiusCount = beamData.ocrRadiusCount;
    m_beamDataDims.ocrCollimatorCount = beamData.ocrCollimatorCount;

    qDebug() << "CUDA: Beam data dimensions:";
    qDebug() << "  OF table:" << beamData.ofDepthCount << "x" << beamData.ofCollimatorCount << "=" << beamData.ofTable.size() << "values";
    qDebug() << "  TMR table:" << beamData.tmrDepthCount << "x" << beamData.tmrFieldSizeCount << "=" << beamData.tmrTable.size() << "values";
    qDebug() << "  OCR table:" << beamData.ocrDepthCount << "x" << beamData.ocrRadiusCount << "x" << beamData.ocrCollimatorCount << "=" << beamData.ocrTable.size() << "values";
    qDebug() << "  Matched collimator:" << beamData.matchedCollimator;

    // Release old buffers
    freeDeviceMemory(&m_d_ofTable);
    freeDeviceMemory(&m_d_ofDepths);
    freeDeviceMemory(&m_d_ofCollimators);
    freeDeviceMemory(&m_d_tmrTable);
    freeDeviceMemory(&m_d_tmrDepths);
    freeDeviceMemory(&m_d_tmrFieldSizes);
    freeDeviceMemory(&m_d_ocrTable);
    freeDeviceMemory(&m_d_ocrDepths);
    freeDeviceMemory(&m_d_ocrRadii);
    freeDeviceMemory(&m_d_ocrCollimators);

    // Upload OF table
    if (!allocateDeviceMemory(&m_d_ofTable, beamData.ofTable.size() * sizeof(float), "OF table")) return false;
    if (!copyToDevice(m_d_ofTable, beamData.ofTable.data(), beamData.ofTable.size() * sizeof(float), "OF table")) return false;

    if (!allocateDeviceMemory(&m_d_ofDepths, beamData.ofDepths.size() * sizeof(float), "OF depths")) return false;
    if (!copyToDevice(m_d_ofDepths, beamData.ofDepths.data(), beamData.ofDepths.size() * sizeof(float), "OF depths")) return false;

    if (!allocateDeviceMemory(&m_d_ofCollimators, beamData.ofCollimators.size() * sizeof(float), "OF collimators")) return false;
    if (!copyToDevice(m_d_ofCollimators, beamData.ofCollimators.data(), beamData.ofCollimators.size() * sizeof(float), "OF collimators")) return false;

    // Upload TMR table
    if (!allocateDeviceMemory(&m_d_tmrTable, beamData.tmrTable.size() * sizeof(float), "TMR table")) return false;
    if (!copyToDevice(m_d_tmrTable, beamData.tmrTable.data(), beamData.tmrTable.size() * sizeof(float), "TMR table")) return false;

    if (!allocateDeviceMemory(&m_d_tmrDepths, beamData.tmrDepths.size() * sizeof(float), "TMR depths")) return false;
    if (!copyToDevice(m_d_tmrDepths, beamData.tmrDepths.data(), beamData.tmrDepths.size() * sizeof(float), "TMR depths")) return false;

    if (!allocateDeviceMemory(&m_d_tmrFieldSizes, beamData.tmrFieldSizes.size() * sizeof(float), "TMR field sizes")) return false;
    if (!copyToDevice(m_d_tmrFieldSizes, beamData.tmrFieldSizes.data(), beamData.tmrFieldSizes.size() * sizeof(float), "TMR field sizes")) return false;

    // Upload OCR table
    if (!allocateDeviceMemory(&m_d_ocrTable, beamData.ocrTable.size() * sizeof(float), "OCR table")) return false;
    if (!copyToDevice(m_d_ocrTable, beamData.ocrTable.data(), beamData.ocrTable.size() * sizeof(float), "OCR table")) return false;

    if (!allocateDeviceMemory(&m_d_ocrDepths, beamData.ocrDepths.size() * sizeof(float), "OCR depths")) return false;
    if (!copyToDevice(m_d_ocrDepths, beamData.ocrDepths.data(), beamData.ocrDepths.size() * sizeof(float), "OCR depths")) return false;

    if (!allocateDeviceMemory(&m_d_ocrRadii, beamData.ocrRadii.size() * sizeof(float), "OCR radii")) return false;
    if (!copyToDevice(m_d_ocrRadii, beamData.ocrRadii.data(), beamData.ocrRadii.size() * sizeof(float), "OCR radii")) return false;

    if (!allocateDeviceMemory(&m_d_ocrCollimators, beamData.ocrCollimators.size() * sizeof(float), "OCR collimators")) return false;
    if (!copyToDevice(m_d_ocrCollimators, beamData.ocrCollimators.data(), beamData.ocrCollimators.size() * sizeof(float), "OCR collimators")) return false;

    qDebug() << "CUDA: Beam data uploaded successfully";
    return true;
}

bool CUDADoseBackend::calculateDose(const GPUComputeParams& params, cv::Mat& doseVolume)
{
    if (!isReady()) {
        setError("Backend not initialized");
        return false;
    }

    // Set the CUDA device for this thread (critical for multi-GPU)
    {
        cudaError_t deviceErr = cudaSetDevice(m_deviceId);
        if (deviceErr != cudaSuccess) {
            setError(QString("Failed to set CUDA device %1: %2").arg(m_deviceId).arg(getCudaErrorString(deviceErr)));
            return false;
        }
    }

    qDebug() << "CUDA: Calculating dose on device" << m_deviceId << "...";

    // Clear computed mask
    size_t maskSize = m_volumeWidth * m_volumeHeight * m_volumeDepth;
    cudaMemsetAsync(m_d_computedMask, 0, maskSize, m_stream);

    // Calculate grid counts
    int gridCountX = params.gridCountX > 0 ? params.gridCountX
                                           : computeCoarseGridCount(params.width, params.stepX);
    int gridCountY = params.gridCountY > 0 ? params.gridCountY
                                           : computeCoarseGridCount(params.height, params.stepY);
    int gridCountZ = params.gridCountZ > 0 ? params.gridCountZ
                                           : computeCoarseGridCount(params.depth, params.stepZ);

    // Configure kernel launch
    dim3 blockSize(8, 8, 8);  // 512 threads per block (optimized for RTX 3090)
    dim3 gridSize(
        (gridCountX + blockSize.x - 1) / blockSize.x,
        (gridCountY + blockSize.y - 1) / blockSize.y,
        (gridCountZ + blockSize.z - 1) / blockSize.z
    );

    qDebug() << "CUDA: Launching kernel with grid" << gridCountX << "x" << gridCountY << "x" << gridCountZ;
    qDebug() << "CUDA: Block size:" << blockSize.x << "x" << blockSize.y << "x" << blockSize.z;
    qDebug() << "CUDA: Grid size:" << gridSize.x << "x" << gridSize.y << "x" << gridSize.z;
    qDebug() << "CUDA: Collimator size:" << params.beam.collimatorSize;
    qDebug() << "CUDA: Step sizes: X=" << params.stepX << "Y=" << params.stepY << "Z=" << params.stepZ;

    // Upload depth profile data (temporary)
    float* d_depthCumulative = nullptr;
    int depthSampleCount = params.depthCumulative.size();
    float depthEntryDistance = static_cast<float>(params.depthEntryDistance);
    float depthStepSize = static_cast<float>(params.depthStepSize);

    if (depthSampleCount > 0) {
        cudaMalloc(&d_depthCumulative, depthSampleCount * sizeof(float));
        cudaMemcpyAsync(d_depthCumulative, params.depthCumulative.data(),
                        depthSampleCount * sizeof(float), cudaMemcpyHostToDevice, m_stream);
        qDebug() << "CUDA: Depth profile uploaded:" << depthSampleCount << "samples, entry ="
                 << depthEntryDistance << "mm, step =" << depthStepSize << "mm";
    }

    // Launch kernel (Note: referenceDose should always be 1.0 to match CPU implementation)
    calculateDoseKernel<<<gridSize, blockSize, 0, m_stream>>>(
        (const short*)m_d_ctVolume,
        (float*)m_d_doseVolume,
        (unsigned char*)m_d_computedMask,
        params.width, params.height, params.depth,
        (float)params.spacingX, (float)params.spacingY, (float)params.spacingZ,
        (float)params.originX, (float)params.originY, (float)params.originZ,
        (float)params.orientationX[0], (float)params.orientationX[1], (float)params.orientationX[2],
        (float)params.orientationY[0], (float)params.orientationY[1], (float)params.orientationY[2],
        (float)params.orientationZ[0], (float)params.orientationZ[1], (float)params.orientationZ[2],
        params.stepX, params.stepY, params.stepZ,
        gridCountX, gridCountY, gridCountZ,
        (float)params.beam.sourcePosition.x(), (float)params.beam.sourcePosition.y(), (float)params.beam.sourcePosition.z(),
        (float)params.beam.targetPosition.x(), (float)params.beam.targetPosition.y(), (float)params.beam.targetPosition.z(),
        (float)params.beam.beamX.x(), (float)params.beam.beamX.y(), (float)params.beam.beamX.z(),
        (float)params.beam.beamY.x(), (float)params.beam.beamY.y(), (float)params.beam.beamY.z(),
        (float)params.beam.beamZ.x(), (float)params.beam.beamZ.y(), (float)params.beam.beamZ.z(),
        (float)params.beam.collimatorSize,
        1.0f,  // referenceDose (scaled to 1.0 for now)
        (const float*)m_d_ofTable, (const float*)m_d_ofDepths, (const float*)m_d_ofCollimators,
        m_beamDataDims.ofDepthCount, m_beamDataDims.ofCollimatorCount,
        (const float*)m_d_tmrTable, (const float*)m_d_tmrDepths, (const float*)m_d_tmrFieldSizes,
        m_beamDataDims.tmrDepthCount, m_beamDataDims.tmrFieldSizeCount,
        (const float*)m_d_ocrTable, (const float*)m_d_ocrDepths, (const float*)m_d_ocrRadii, (const float*)m_d_ocrCollimators,
        m_beamDataDims.ocrDepthCount, m_beamDataDims.ocrRadiusCount, m_beamDataDims.ocrCollimatorCount,
        d_depthCumulative, depthSampleCount, depthEntryDistance, depthStepSize,
        0  // accumulate = false
    );

    // Free temporary depth profile memory
    if (d_depthCumulative) {
        cudaFree(d_depthCumulative);
    }

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        setError(QString("CUDA kernel launch failed: %1").arg(getCudaErrorString(err)));
        return false;
    }

    // Wait for kernel to complete
    err = cudaStreamSynchronize(m_stream);
    if (err != cudaSuccess) {
        setError(QString("CUDA kernel execution failed: %1").arg(getCudaErrorString(err)));
        return false;
    }

    // Download result
    if (doseVolume.empty() || doseVolume.type() != CV_32F ||
        doseVolume.size[0] != m_volumeDepth || doseVolume.size[1] != m_volumeHeight || doseVolume.size[2] != m_volumeWidth) {
        int sizes[] = {m_volumeDepth, m_volumeHeight, m_volumeWidth};
        doseVolume = cv::Mat(3, sizes, CV_32F, cv::Scalar(0));
    }

    size_t doseSize = m_volumeWidth * m_volumeHeight * m_volumeDepth * sizeof(float);
    if (!copyFromDevice(doseVolume.data, m_d_doseVolume, doseSize, "dose volume")) {
        return false;
    }

    // Calculate statistics for debugging
    double minDose, maxDose;
    cv::minMaxIdx(doseVolume, &minDose, &maxDose);
    qDebug() << "CUDA: Dose calculation completed successfully";
    qDebug() << "CUDA: Dose statistics: min =" << minDose << ", max =" << maxDose;

    return true;
}

bool CUDADoseBackend::interpolateVolume(cv::Mat& doseVolume,
                                        const std::vector<uint8_t>& computedMask,
                                        int step)
{
    if (!isReady()) {
        setError("Backend not initialized");
        return false;
    }

    // Set the CUDA device for this thread (critical for multi-GPU)
    {
        cudaError_t deviceErr = cudaSetDevice(m_deviceId);
        if (deviceErr != cudaSuccess) {
            setError(QString("Failed to set CUDA device %1: %2").arg(m_deviceId).arg(getCudaErrorString(deviceErr)));
            return false;
        }
    }

    qDebug() << "CUDA: Interpolating volume on device" << m_deviceId << "with step" << step;

    // Upload mask
    size_t maskSize = m_volumeWidth * m_volumeHeight * m_volumeDepth;
    if (!copyToDevice(m_d_computedMask, computedMask.data(), maskSize, "computed mask")) {
        return false;
    }

    // Upload dose volume
    size_t doseSize = m_volumeWidth * m_volumeHeight * m_volumeDepth * sizeof(float);
    if (!copyToDevice(m_d_doseVolume, doseVolume.data, doseSize, "dose volume")) {
        return false;
    }

    // Configure kernel
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (m_volumeWidth + blockSize.x - 1) / blockSize.x,
        (m_volumeHeight + blockSize.y - 1) / blockSize.y,
        (m_volumeDepth + blockSize.z - 1) / blockSize.z
    );

    // Launch interpolation kernel
    interpolateVolumeKernel<<<gridSize, blockSize, 0, m_stream>>>(
        (float*)m_d_doseVolume,
        (const unsigned char*)m_d_computedMask,
        m_volumeWidth, m_volumeHeight, m_volumeDepth,
        step
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        setError(QString("CUDA interpolation kernel launch failed: %1").arg(getCudaErrorString(err)));
        return false;
    }

    cudaStreamSynchronize(m_stream);

    // Download result
    if (!copyFromDevice(doseVolume.data, m_d_doseVolume, doseSize, "interpolated dose volume")) {
        return false;
    }

    qDebug() << "CUDA: Interpolation completed";
    return true;
}

bool CUDADoseBackend::recalculateDoseWithThreshold(const GPUComputeParams& params,
                                                    cv::Mat& doseVolume,
                                                    float threshold,
                                                    int skipStep)
{
    if (!isReady()) {
        setError("Backend not initialized");
        return false;
    }

    // Set the CUDA device for this thread (critical for multi-GPU)
    {
        cudaError_t deviceErr = cudaSetDevice(m_deviceId);
        if (deviceErr != cudaSuccess) {
            setError(QString("Failed to set CUDA device %1: %2").arg(m_deviceId).arg(getCudaErrorString(deviceErr)));
            return false;
        }
    }

    qDebug() << "CUDA: Recalculating dose on device" << m_deviceId << "with threshold" << threshold << "skipStep" << skipStep;

    // Upload current dose volume
    size_t doseSize = m_volumeWidth * m_volumeHeight * m_volumeDepth * sizeof(float);
    if (!copyToDevice(m_d_doseVolume, doseVolume.data, doseSize, "dose volume")) {
        return false;
    }

    // Calculate grid counts
    int gridCountX = params.gridCountX > 0 ? params.gridCountX
                                           : computeCoarseGridCount(params.width, params.stepX);
    int gridCountY = params.gridCountY > 0 ? params.gridCountY
                                           : computeCoarseGridCount(params.height, params.stepY);
    int gridCountZ = params.gridCountZ > 0 ? params.gridCountZ
                                           : computeCoarseGridCount(params.depth, params.stepZ);

    // Configure kernel
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (gridCountX + blockSize.x - 1) / blockSize.x,
        (gridCountY + blockSize.y - 1) / blockSize.y,
        (gridCountZ + blockSize.z - 1) / blockSize.z
    );

    // Upload depth profile data (temporary)
    float* d_depthCumulative = nullptr;
    int depthSampleCount = params.depthCumulative.size();
    float depthEntryDistance = static_cast<float>(params.depthEntryDistance);
    float depthStepSize = static_cast<float>(params.depthStepSize);

    if (depthSampleCount > 0) {
        cudaMalloc(&d_depthCumulative, depthSampleCount * sizeof(float));
        cudaMemcpyAsync(d_depthCumulative, params.depthCumulative.data(),
                        depthSampleCount * sizeof(float), cudaMemcpyHostToDevice, m_stream);
    }

    // Launch kernel
    recalculateDoseWithThresholdKernel<<<gridSize, blockSize, 0, m_stream>>>(
        (const short*)m_d_ctVolume,
        (float*)m_d_doseVolume,
        (unsigned char*)m_d_computedMask,
        params.width, params.height, params.depth,
        (float)params.spacingX, (float)params.spacingY, (float)params.spacingZ,
        (float)params.originX, (float)params.originY, (float)params.originZ,
        (float)params.orientationX[0], (float)params.orientationX[1], (float)params.orientationX[2],
        (float)params.orientationY[0], (float)params.orientationY[1], (float)params.orientationY[2],
        (float)params.orientationZ[0], (float)params.orientationZ[1], (float)params.orientationZ[2],
        params.stepX, params.stepY, params.stepZ,
        gridCountX, gridCountY, gridCountZ,
        (float)params.beam.sourcePosition.x(), (float)params.beam.sourcePosition.y(), (float)params.beam.sourcePosition.z(),
        (float)params.beam.targetPosition.x(), (float)params.beam.targetPosition.y(), (float)params.beam.targetPosition.z(),
        (float)params.beam.beamX.x(), (float)params.beam.beamX.y(), (float)params.beam.beamX.z(),
        (float)params.beam.beamY.x(), (float)params.beam.beamY.y(), (float)params.beam.beamY.z(),
        (float)params.beam.beamZ.x(), (float)params.beam.beamZ.y(), (float)params.beam.beamZ.z(),
        (float)params.beam.collimatorSize,
        1.0f,
        (const float*)m_d_ofTable, (const float*)m_d_ofDepths, (const float*)m_d_ofCollimators,
        m_beamDataDims.ofDepthCount, m_beamDataDims.ofCollimatorCount,
        (const float*)m_d_tmrTable, (const float*)m_d_tmrDepths, (const float*)m_d_tmrFieldSizes,
        m_beamDataDims.tmrDepthCount, m_beamDataDims.tmrFieldSizeCount,
        (const float*)m_d_ocrTable, (const float*)m_d_ocrDepths, (const float*)m_d_ocrRadii, (const float*)m_d_ocrCollimators,
        m_beamDataDims.ocrDepthCount, m_beamDataDims.ocrRadiusCount, m_beamDataDims.ocrCollimatorCount,
        d_depthCumulative, depthSampleCount, depthEntryDistance, depthStepSize,
        threshold,
        skipStep
    );

    // Free temporary depth profile memory
    if (d_depthCumulative) {
        cudaFree(d_depthCumulative);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        setError(QString("CUDA threshold recalc kernel launch failed: %1").arg(getCudaErrorString(err)));
        return false;
    }

    cudaStreamSynchronize(m_stream);

    // Download result
    if (!copyFromDevice(doseVolume.data, m_d_doseVolume, doseSize, "recalculated dose volume")) {
        return false;
    }

    qDebug() << "CUDA: Threshold recalculation completed";
    return true;
}

QString CUDADoseBackend::getLastError() const
{
    return m_lastError;
}

void CUDADoseBackend::cleanup()
{
    if (!m_initialized) {
        return;
    }

    qDebug() << "CUDA: Cleaning up resources...";

    releaseBuffers();

    if (m_stream) {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }

    cudaDeviceReset();

    m_initialized = false;
    qDebug() << "CUDA: Cleanup complete";
}

bool CUDADoseBackend::createBuffers(int width, int height, int depth)
{
    size_t voxelCount = width * height * depth;
    size_t ctSize = voxelCount * sizeof(short);
    size_t doseSize = voxelCount * sizeof(float);
    size_t maskSize = voxelCount * sizeof(uint8_t);

    qDebug() << "CUDA: Creating buffers for" << width << "x" << height << "x" << depth;
    qDebug() << "CUDA: Total memory required:" << ((ctSize + doseSize + maskSize) / (1024 * 1024)) << "MB";

    if (!allocateDeviceMemory(&m_d_ctVolume, ctSize, "CT volume buffer")) return false;
    if (!allocateDeviceMemory(&m_d_doseVolume, doseSize, "dose volume buffer")) return false;
    if (!allocateDeviceMemory(&m_d_computedMask, maskSize, "computed mask buffer")) return false;

    // Initialize dose volume to zero
    cudaMemsetAsync(m_d_doseVolume, 0, doseSize, m_stream);

    qDebug() << "CUDA: Buffers created successfully";
    return true;
}

void CUDADoseBackend::releaseBuffers()
{
    freeDeviceMemory(&m_d_ctVolume);
    freeDeviceMemory(&m_d_doseVolume);
    freeDeviceMemory(&m_d_computedMask);
}

void CUDADoseBackend::setError(const QString& error)
{
    m_lastError = error;
    qWarning() << "CUDA Error:" << error;
}

QString CUDADoseBackend::getCudaErrorString(int error) const
{
    return QString::fromUtf8(cudaGetErrorString((cudaError_t)error));
}

bool CUDADoseBackend::allocateDeviceMemory(void** ptr, size_t size, const QString& name)
{
    cudaError_t err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
        setError(QString("Failed to allocate %1: %2").arg(name).arg(getCudaErrorString(err)));
        return false;
    }
    return true;
}

bool CUDADoseBackend::copyToDevice(void* dst, const void* src, size_t size, const QString& name)
{
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, m_stream);
    if (err != cudaSuccess) {
        setError(QString("Failed to copy %1 to device: %2").arg(name).arg(getCudaErrorString(err)));
        return false;
    }
    return true;
}

bool CUDADoseBackend::copyFromDevice(void* dst, const void* src, size_t size, const QString& name)
{
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, m_stream);
    if (err != cudaSuccess) {
        setError(QString("Failed to copy %1 from device: %2").arg(name).arg(getCudaErrorString(err)));
        return false;
    }
    // Synchronize to ensure copy is complete before returning
    cudaStreamSynchronize(m_stream);
    return true;
}

void CUDADoseBackend::freeDeviceMemory(void** ptr)
{
    if (ptr && *ptr) {
        cudaFree(*ptr);
        *ptr = nullptr;
    }
}

} // namespace CyberKnife

#endif // USE_CUDA_BACKEND
