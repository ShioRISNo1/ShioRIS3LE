#include "cyberknife/gpu_dose_backend.h"

#ifdef USE_OPENCL_BACKEND
#include "cyberknife/opencl_dose_backend.h"
#endif

#ifdef USE_METAL_BACKEND
#include "cyberknife/metal_dose_backend.h"
#endif

#ifdef USE_CUDA_BACKEND
#include "cyberknife/cuda_dose_backend.h"
#include "cyberknife/multi_gpu_cuda_manager.h"
#include <cuda_runtime.h>
#endif

#include <QDebug>

namespace CyberKnife {

std::unique_ptr<IGPUDoseBackend> GPUDoseBackendFactory::createBackend(GPUBackendType preferred,
                                                                      bool useMultiGPU,
                                                                      int maxGPUs)
{
    GPUBackendType targetType = preferred;

    // Auto-detect if None specified
    if (targetType == GPUBackendType::None) {
        targetType = detectBestBackend();
    }

    // Try to create the requested backend
    switch (targetType) {
#ifdef USE_METAL_BACKEND
        case GPUBackendType::Metal: {
            qDebug() << "GPU Backend Factory: Creating Metal backend";
            auto backend = std::make_unique<MetalDoseBackend>();
            if (backend->initialize()) {
                return backend;
            } else {
                qWarning() << "GPU Backend Factory: Metal initialization failed:" << backend->getLastError();
                return nullptr;
            }
        }
#endif

#ifdef USE_OPENCL_BACKEND
        case GPUBackendType::OpenCL: {
            qDebug() << "GPU Backend Factory: Creating OpenCL backend";
            auto backend = std::make_unique<OpenCLDoseBackend>();
            if (backend->initialize()) {
                return backend;
            } else {
                qWarning() << "GPU Backend Factory: OpenCL initialization failed:" << backend->getLastError();
                return nullptr;
            }
        }
#endif

#ifdef USE_CUDA_BACKEND
        case GPUBackendType::CUDA: {
            // Check if multi-GPU mode is requested and multiple GPUs are available
            if (useMultiGPU) {
                int deviceCount = getCUDADeviceCount();
                if (deviceCount > 1) {
                    qDebug() << "GPU Backend Factory: Creating Multi-GPU CUDA backend with"
                             << (maxGPUs > 0 ? maxGPUs : deviceCount) << "GPUs";
                    auto backend = std::make_unique<MultiGPUCUDAManager>(maxGPUs);
                    if (backend->initialize()) {
                        return backend;
                    } else {
                        qWarning() << "GPU Backend Factory: Multi-GPU CUDA initialization failed:" << backend->getLastError();
                        qWarning() << "GPU Backend Factory: Falling back to single-GPU CUDA backend";
                        // Fall through to single-GPU mode
                    }
                } else {
                    qDebug() << "GPU Backend Factory: Multi-GPU requested but only" << deviceCount
                             << "GPU(s) available, using single-GPU mode";
                }
            }

            // Single-GPU CUDA backend
            qDebug() << "GPU Backend Factory: Creating single-GPU CUDA backend";
            auto backend = std::make_unique<CUDADoseBackend>();
            if (backend->initialize()) {
                return backend;
            } else {
                qWarning() << "GPU Backend Factory: CUDA initialization failed:" << backend->getLastError();
                return nullptr;
            }
        }
#endif

        case GPUBackendType::None:
        default:
            qDebug() << "GPU Backend Factory: No GPU backend available, using CPU";
            return nullptr;
    }
}

GPUBackendType GPUDoseBackendFactory::detectBestBackend()
{
    qDebug() << "GPU Backend Factory: Detecting best available backend...";

#ifdef __APPLE__
    qDebug() << "GPU Backend Factory: Platform = macOS";
    // macOS: prefer Metal when implemented
    #ifdef USE_METAL_BACKEND
        qDebug() << "GPU Backend Factory: Checking Metal backend...";
        if (isBackendAvailable(GPUBackendType::Metal)) {
            qDebug() << "GPU Backend Factory: Metal backend available";
            return GPUBackendType::Metal;
        } else {
            qDebug() << "GPU Backend Factory: Metal backend not available";
        }
    #else
        qDebug() << "GPU Backend Factory: Metal backend not compiled";
    #endif

    // Fallback to OpenCL on macOS (though deprecated)
    #ifdef USE_OPENCL_BACKEND
        qDebug() << "GPU Backend Factory: Checking OpenCL backend...";
        if (isBackendAvailable(GPUBackendType::OpenCL)) {
            qWarning() << "GPU Backend: Using deprecated OpenCL on macOS";
            return GPUBackendType::OpenCL;
        } else {
            qDebug() << "GPU Backend Factory: OpenCL backend not available";
        }
    #else
        qDebug() << "GPU Backend Factory: OpenCL backend not compiled";
    #endif
#else
    qDebug() << "GPU Backend Factory: Platform = Windows/Linux";
    // Windows/Linux: Check for CUDA first (NVIDIA GPUs)
    #ifdef USE_CUDA_BACKEND
        qDebug() << "GPU Backend Factory: Checking CUDA backend...";
        if (isBackendAvailable(GPUBackendType::CUDA)) {
            qDebug() << "GPU Backend Factory: CUDA backend available";
            return GPUBackendType::CUDA;
        } else {
            qDebug() << "GPU Backend Factory: CUDA backend not available";
        }
    #else
        qDebug() << "GPU Backend Factory: CUDA backend not compiled";
    #endif

    // Fallback to OpenCL (works with NVIDIA, AMD, Intel)
    #ifdef USE_OPENCL_BACKEND
        qDebug() << "GPU Backend Factory: Checking OpenCL backend...";
        if (isBackendAvailable(GPUBackendType::OpenCL)) {
            qDebug() << "GPU Backend Factory: OpenCL backend available";
            return GPUBackendType::OpenCL;
        } else {
            qDebug() << "GPU Backend Factory: OpenCL backend not available";
        }
    #else
        qDebug() << "GPU Backend Factory: OpenCL backend not compiled";
    #endif
#endif

    // No GPU backend available
    qWarning() << "GPU Backend Factory: No GPU backend available";
    return GPUBackendType::None;
}

bool GPUDoseBackendFactory::isBackendAvailable(GPUBackendType type)
{
    switch (type) {
#ifdef USE_OPENCL_BACKEND
        case GPUBackendType::OpenCL: {
            // Try to create and initialize OpenCL backend
            qDebug() << "GPU Backend Factory: Testing OpenCL availability...";
            OpenCLDoseBackend backend;
            bool success = backend.initialize();
            if (!success) {
                qDebug() << "GPU Backend Factory: OpenCL initialization failed:" << backend.getLastError();
            }
            return success;
        }
#endif

#ifdef USE_METAL_BACKEND
        case GPUBackendType::Metal: {
            // Try to create and initialize Metal backend
            qDebug() << "GPU Backend Factory: Testing Metal availability...";
            MetalDoseBackend backend;
            bool success = backend.initialize();
            if (!success) {
                qDebug() << "GPU Backend Factory: Metal initialization failed:" << backend.getLastError();
            }
            return success;
        }
#endif

#ifdef USE_CUDA_BACKEND
        case GPUBackendType::CUDA: {
            // Try to create and initialize CUDA backend
            qDebug() << "GPU Backend Factory: Testing CUDA availability...";
            CUDADoseBackend backend;
            bool success = backend.initialize();
            if (!success) {
                qDebug() << "GPU Backend Factory: CUDA initialization failed:" << backend.getLastError();
            }
            return success;
        }
#endif

        case GPUBackendType::None:
        default:
            return false;
    }
}

QVector<GPUBackendType> GPUDoseBackendFactory::getAvailableBackends()
{
    QVector<GPUBackendType> backends;

#ifdef USE_OPENCL_BACKEND
    if (isBackendAvailable(GPUBackendType::OpenCL)) {
        backends.append(GPUBackendType::OpenCL);
    }
#endif

#ifdef USE_METAL_BACKEND
    if (isBackendAvailable(GPUBackendType::Metal)) {
        backends.append(GPUBackendType::Metal);
    }
#endif

#ifdef USE_CUDA_BACKEND
    if (isBackendAvailable(GPUBackendType::CUDA)) {
        backends.append(GPUBackendType::CUDA);
    }
#endif

    return backends;
}

int GPUDoseBackendFactory::getCUDADeviceCount()
{
#ifdef USE_CUDA_BACKEND
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        qDebug() << "GPU Backend Factory: cudaGetDeviceCount failed:" << cudaGetErrorString(err);
        return 0;
    }
    return deviceCount;
#else
    return 0;
#endif
}

} // namespace CyberKnife
