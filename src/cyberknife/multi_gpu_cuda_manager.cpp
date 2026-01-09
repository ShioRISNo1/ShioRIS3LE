#include "cyberknife/multi_gpu_cuda_manager.h"

#ifdef USE_CUDA_BACKEND

#include "cyberknife/cuda_dose_backend.h"
#include <cuda_runtime.h>
#include <QDebug>
#include <QtConcurrent/QtConcurrentRun>
#include <QFuture>
#include <QFutureWatcher>
#include <algorithm>

namespace CyberKnife {

MultiGPUCUDAManager::MultiGPUCUDAManager(int maxGPUs)
    : m_maxGPUs(maxGPUs)
    , m_initialized(false)
    , m_ctVolumeUploaded(false)
    , m_beamDataUploaded(false)
{
}

MultiGPUCUDAManager::~MultiGPUCUDAManager()
{
    cleanup();
}

bool MultiGPUCUDAManager::initialize()
{
    if (m_initialized) {
        return true;
    }

    qDebug() << "Multi-GPU CUDA: Initializing manager...";

    // Get device count
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        setError(QString("No CUDA devices found: %1").arg(cudaGetErrorString(err)));
        return false;
    }

    qDebug() << "Multi-GPU CUDA: Found" << deviceCount << "device(s)";

    // Determine how many GPUs to use
    int gpuCount = (m_maxGPUs > 0) ? qMin(m_maxGPUs, deviceCount) : deviceCount;
    qDebug() << "Multi-GPU CUDA: Will use" << gpuCount << "GPU(s)";

    // Initialize each GPU backend
    m_backends.reserve(gpuCount);
    for (int i = 0; i < gpuCount; i++) {
        if (!initializeGPU(i)) {
            qWarning() << "Multi-GPU CUDA: Failed to initialize GPU" << i;
            cleanup();
            return false;
        }
    }

    m_initialized = true;
    qDebug() << "Multi-GPU CUDA: Initialization successful with" << m_backends.size() << "GPU(s)";

    return true;
}

bool MultiGPUCUDAManager::initializeGPU(int deviceId)
{
    auto wrapper = std::make_unique<GPUBackendWrapper>(deviceId);
    wrapper->backend = std::make_unique<CUDADoseBackend>(deviceId);

    if (!wrapper->backend->initialize()) {
        setError(QString("Failed to initialize GPU %1: %2")
                     .arg(deviceId)
                     .arg(wrapper->backend->getLastError()));
        return false;
    }

    qDebug() << "Multi-GPU CUDA: Initialized GPU" << deviceId
             << ":" << wrapper->backend->getDeviceInfo();

    m_backends.push_back(std::move(wrapper));
    return true;
}

bool MultiGPUCUDAManager::isReady() const
{
    return m_initialized && !m_backends.empty();
}

GPUBackendType MultiGPUCUDAManager::getBackendType() const
{
    return GPUBackendType::CUDA;
}

QString MultiGPUCUDAManager::getBackendName() const
{
    return QString("Multi-GPU CUDA (%1 GPUs)").arg(m_backends.size());
}

QString MultiGPUCUDAManager::getDeviceInfo() const
{
    QStringList deviceInfos;
    for (const auto& wrapper : m_backends) {
        deviceInfos.append(QString("GPU %1: %2")
                              .arg(wrapper->deviceId)
                              .arg(wrapper->backend->getDeviceInfo()));
    }
    return deviceInfos.join("; ");
}

bool MultiGPUCUDAManager::uploadCTVolume(const cv::Mat& volume)
{
    if (!isReady()) {
        setError("Manager not initialized");
        return false;
    }

    qDebug() << "Multi-GPU CUDA: Uploading CT volume to all GPUs...";

    // Cache the CT volume
    m_cachedCTVolume = volume.clone();

    // Upload to all GPUs
    for (auto& wrapper : m_backends) {
        std::lock_guard<std::mutex> lock(wrapper->mutex);
        if (!wrapper->backend->uploadCTVolume(volume)) {
            setError(QString("Failed to upload CT volume to GPU %1: %2")
                         .arg(wrapper->deviceId)
                         .arg(wrapper->backend->getLastError()));
            m_ctVolumeUploaded = false;
            return false;
        }
    }

    m_ctVolumeUploaded = true;
    qDebug() << "Multi-GPU CUDA: CT volume uploaded to all GPUs successfully";
    return true;
}

bool MultiGPUCUDAManager::uploadBeamData(const GPUBeamData& beamData)
{
    if (!isReady()) {
        setError("Manager not initialized");
        return false;
    }

    qDebug() << "Multi-GPU CUDA: Uploading beam data to all GPUs...";

    // Cache the beam data
    m_cachedBeamData = beamData;

    // Upload to all GPUs
    for (auto& wrapper : m_backends) {
        std::lock_guard<std::mutex> lock(wrapper->mutex);
        if (!wrapper->backend->uploadBeamData(beamData)) {
            setError(QString("Failed to upload beam data to GPU %1: %2")
                         .arg(wrapper->deviceId)
                         .arg(wrapper->backend->getLastError()));
            m_beamDataUploaded = false;
            return false;
        }
    }

    m_beamDataUploaded = true;
    qDebug() << "Multi-GPU CUDA: Beam data uploaded to all GPUs successfully";
    return true;
}

bool MultiGPUCUDAManager::calculateDose(const GPUComputeParams& params, cv::Mat& doseVolume)
{
    if (!isReady()) {
        setError("Manager not initialized");
        return false;
    }

    // For single beam, use the first available GPU
    GPUBackendWrapper* gpu = acquireGPU();
    if (!gpu) {
        setError("No GPU available");
        return false;
    }

    bool result = gpu->backend->calculateDose(params, doseVolume);
    if (!result) {
        setError(gpu->backend->getLastError());
    }

    releaseGPU(gpu);
    return result;
}

bool MultiGPUCUDAManager::interpolateVolume(cv::Mat& doseVolume,
                                           const std::vector<uint8_t>& computedMask,
                                           int step)
{
    if (!isReady()) {
        setError("Manager not initialized");
        return false;
    }

    // Use the first GPU for interpolation
    GPUBackendWrapper* gpu = acquireGPU();
    if (!gpu) {
        setError("No GPU available");
        return false;
    }

    bool result = gpu->backend->interpolateVolume(doseVolume, computedMask, step);
    if (!result) {
        setError(gpu->backend->getLastError());
    }

    releaseGPU(gpu);
    return result;
}

bool MultiGPUCUDAManager::recalculateDoseWithThreshold(const GPUComputeParams& params,
                                                      cv::Mat& doseVolume,
                                                      float threshold,
                                                      int skipStep)
{
    if (!isReady()) {
        setError("Manager not initialized");
        return false;
    }

    // Use the first GPU for threshold recalculation
    GPUBackendWrapper* gpu = acquireGPU();
    if (!gpu) {
        setError("No GPU available");
        return false;
    }

    bool result = gpu->backend->recalculateDoseWithThreshold(params, doseVolume, threshold, skipStep);
    if (!result) {
        setError(gpu->backend->getLastError());
    }

    releaseGPU(gpu);
    return result;
}

bool MultiGPUCUDAManager::calculateMultiBeamDose(const cv::Mat& ctVolume,
                                                 const QVector<GPUComputeParams>& beamParams,
                                                 cv::Mat& doseVolume,
                                                 std::function<void(int, int)> progressCallback)
{
    if (!isReady()) {
        setError("Manager not initialized");
        return false;
    }

    if (beamParams.isEmpty()) {
        setError("No beams provided");
        return false;
    }

    const int totalBeams = beamParams.size();
    const int numGPUs = m_backends.size();

    qDebug() << "Multi-GPU CUDA: Calculating" << totalBeams << "beams across" << numGPUs << "GPUs";

    // Initialize dose volume if needed
    if (doseVolume.empty()) {
        int sizes[] = {ctVolume.size[0], ctVolume.size[1], ctVolume.size[2]};
        doseVolume = cv::Mat(3, sizes, CV_32F, cv::Scalar(0));
    }

    // Thread-safe progress counter and error handling
    std::atomic<int> completedBeams{0};
    std::atomic<bool> hasError{false};
    std::mutex doseMutex;
    QString errorMsg;

    // Create worker function for each GPU
    auto workerFunc = [this, &beamParams, &doseVolume, &doseMutex, &completedBeams,
                       &hasError, &errorMsg, progressCallback, totalBeams](int gpuIndex) {
        GPUBackendWrapper* gpu = m_backends[gpuIndex].get();

        // Process beams assigned to this GPU (round-robin distribution)
        for (int beamIndex = gpuIndex; beamIndex < totalBeams && !hasError.load(); beamIndex += m_backends.size()) {
            // Calculate dose for this beam
            cv::Mat beamDose;
            bool result = gpu->backend->calculateDose(beamParams[beamIndex], beamDose);

            if (result) {
                // Accumulate dose (thread-safe)
                {
                    std::lock_guard<std::mutex> lock(doseMutex);
                    cv::add(doseVolume, beamDose, doseVolume);
                }

                // Update progress
                int completed = completedBeams.fetch_add(1) + 1;
                if (progressCallback) {
                    progressCallback(completed, totalBeams);
                }

                qDebug() << "Multi-GPU CUDA: Beam" << (beamIndex + 1) << "/" << totalBeams
                         << "completed on GPU" << gpu->deviceId;
            } else {
                hasError.store(true);
                std::lock_guard<std::mutex> lock(doseMutex);
                errorMsg = QString("Beam %1 failed on GPU %2: %3")
                              .arg(beamIndex + 1)
                              .arg(gpu->deviceId)
                              .arg(gpu->backend->getLastError());
                qWarning() << "Multi-GPU CUDA:" << errorMsg;
                break;
            }
        }
    };

    // Launch worker threads for each GPU
    QVector<QFuture<void>> futures;
    for (int i = 0; i < numGPUs; i++) {
        futures.append(QtConcurrent::run(workerFunc, i));
    }

    // Wait for all workers to complete
    for (auto& future : futures) {
        future.waitForFinished();
    }

    if (hasError.load()) {
        setError(errorMsg);
        return false;
    }

    qDebug() << "Multi-GPU CUDA: All beams completed successfully";
    return true;
}

QString MultiGPUCUDAManager::getLastError() const
{
    return m_lastError;
}

void MultiGPUCUDAManager::cleanup()
{
    if (!m_initialized) {
        return;
    }

    qDebug() << "Multi-GPU CUDA: Cleaning up resources...";

    for (auto& wrapper : m_backends) {
        std::lock_guard<std::mutex> lock(wrapper->mutex);
        wrapper->backend->cleanup();
    }

    m_backends.clear();
    m_cachedCTVolume.release();
    m_ctVolumeUploaded = false;
    m_beamDataUploaded = false;
    m_initialized = false;

    qDebug() << "Multi-GPU CUDA: Cleanup complete";
}

void MultiGPUCUDAManager::setError(const QString& error)
{
    m_lastError = error;
    qWarning() << "Multi-GPU CUDA Error:" << error;
}

MultiGPUCUDAManager::GPUBackendWrapper* MultiGPUCUDAManager::acquireGPU()
{
    std::lock_guard<std::mutex> lock(m_managerMutex);

    // Find a free GPU
    for (auto& wrapper : m_backends) {
        if (!wrapper->busy) {
            wrapper->busy = true;
            return wrapper.get();
        }
    }

    return nullptr;  // All GPUs are busy
}

void MultiGPUCUDAManager::releaseGPU(GPUBackendWrapper* gpu)
{
    if (!gpu) return;

    std::lock_guard<std::mutex> lock(m_managerMutex);
    gpu->busy = false;
}

} // namespace CyberKnife

#endif // USE_CUDA_BACKEND
