#include "cyberknife/opencl_dose_backend.h"

#ifdef USE_OPENCL_BACKEND

#include <QCoreApplication>
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <QDir>
#include <algorithm>
#include <cstring>

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

}

OpenCLDoseBackend::OpenCLDoseBackend()
    : m_platform(nullptr)
    , m_device(nullptr)
    , m_context(nullptr)
    , m_commandQueue(nullptr)
    , m_program(nullptr)
    , m_doseKernel(nullptr)
    , m_interpolationKernel(nullptr)
    , m_thresholdRecalcKernel(nullptr)
    , m_ctBuffer(nullptr)
    , m_doseBuffer(nullptr)
    , m_computedMaskBuffer(nullptr)
    , m_ofTableBuffer(nullptr)
    , m_ofDepthsBuffer(nullptr)
    , m_ofCollimatorsBuffer(nullptr)
    , m_tmrTableBuffer(nullptr)
    , m_tmrDepthsBuffer(nullptr)
    , m_tmrFieldSizesBuffer(nullptr)
    , m_ocrTableBuffer(nullptr)
    , m_ocrDepthsBuffer(nullptr)
    , m_ocrRadiiBuffer(nullptr)
    , m_ocrCollimatorsBuffer(nullptr)
    , m_volumeWidth(0)
    , m_volumeHeight(0)
    , m_volumeDepth(0)
    , m_initialized(false)
{
}

OpenCLDoseBackend::~OpenCLDoseBackend()
{
    cleanup();
}

bool OpenCLDoseBackend::initialize()
{
    if (m_initialized) {
        return true;
    }

    qDebug() << "OpenCL: Initializing backend...";

    // Select best GPU device
    if (!selectBestDevice()) {
        setError("Failed to select OpenCL device");
        return false;
    }

    // Create context
    cl_int err;
    m_context = clCreateContext(nullptr, 1, &m_device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        setError(QString("Failed to create OpenCL context: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    // Create command queue
#ifdef CL_VERSION_2_0
    m_commandQueue = clCreateCommandQueueWithProperties(m_context, m_device, nullptr, &err);
#else
    m_commandQueue = clCreateCommandQueue(m_context, m_device, 0, &err);
#endif
    if (err != CL_SUCCESS) {
        setError(QString("Failed to create command queue: %1").arg(getOpenCLErrorString(err)));
        cleanup();
        return false;
    }

    // Compile kernels
    if (!compileKernels()) {
        cleanup();
        return false;
    }

    m_initialized = true;
    qDebug() << "OpenCL: Initialization successful";
    qDebug() << "OpenCL: Device:" << getDeviceInfo();

    return true;
}

bool OpenCLDoseBackend::isReady() const
{
    return m_initialized && m_context != nullptr && m_commandQueue != nullptr;
}

GPUBackendType OpenCLDoseBackend::getBackendType() const
{
    return GPUBackendType::OpenCL;
}

QString OpenCLDoseBackend::getBackendName() const
{
    return "OpenCL";
}

QString OpenCLDoseBackend::getDeviceInfo() const
{
    if (!m_device) {
        return "No device selected";
    }

    char deviceName[256];
    char deviceVendor[256];
    cl_device_type deviceType;

    clGetDeviceInfo(m_device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    clGetDeviceInfo(m_device, CL_DEVICE_VENDOR, sizeof(deviceVendor), deviceVendor, nullptr);
    clGetDeviceInfo(m_device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr);

    QString typeStr;
    if (deviceType & CL_DEVICE_TYPE_GPU) {
        typeStr = "GPU";
    } else if (deviceType & CL_DEVICE_TYPE_CPU) {
        typeStr = "CPU";
    } else {
        typeStr = "Other";
    }

    return QString("%1 %2 (%3)").arg(deviceVendor).arg(deviceName).arg(typeStr);
}

bool OpenCLDoseBackend::selectBestDevice()
{
    cl_uint platformCount = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platformCount);

    qDebug() << "OpenCL: Platform detection started";
    qDebug() << "OpenCL: Found" << platformCount << "platform(s)";

    if (platformCount == 0 || err != CL_SUCCESS) {
        setError(QString("No OpenCL platforms found (error: %1)").arg(getOpenCLErrorString(err)));
        qWarning() << "OpenCL: Platform detection failed";
        return false;
    }

    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);

    // Debug: List all platforms
    for (cl_uint i = 0; i < platformCount; i++) {
        char platformName[256];
        char platformVendor[256];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(platformVendor), platformVendor, nullptr);
        qDebug() << "OpenCL: Platform" << i << ":" << platformVendor << platformName;
    }

    // Try to find a GPU device first
    for (size_t i = 0; i < platforms.size(); i++) {
        cl_platform_id platform = platforms[i];
        cl_uint deviceCount = 0;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);

        qDebug() << "OpenCL: Platform" << i << "has" << deviceCount << "GPU device(s) (error:" << getOpenCLErrorString(err) << ")";

        if (err == CL_SUCCESS && deviceCount > 0) {
            std::vector<cl_device_id> devices(deviceCount);
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr);

            // Debug: List all GPU devices
            for (cl_uint j = 0; j < deviceCount; j++) {
                char deviceName[256];
                cl_device_type deviceType;
                cl_bool available;
                clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
                clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr);
                clGetDeviceInfo(devices[j], CL_DEVICE_AVAILABLE, sizeof(available), &available, nullptr);
                qDebug() << "  OpenCL: GPU Device" << j << ":" << deviceName
                         << "Available:" << (available ? "Yes" : "No");
            }

            // Select first GPU device
            m_platform = platform;
            m_device = devices[0];

            qDebug() << "OpenCL: ✓ Selected GPU device";
            return true;
        }
    }

    qWarning() << "OpenCL: No GPU devices found, trying CPU fallback";

    // Fallback to CPU if no GPU found
    for (size_t i = 0; i < platforms.size(); i++) {
        cl_platform_id platform = platforms[i];
        cl_uint deviceCount = 0;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &deviceCount);

        qDebug() << "OpenCL: Platform" << i << "has" << deviceCount << "CPU device(s) (error:" << getOpenCLErrorString(err) << ")";

        if (err == CL_SUCCESS && deviceCount > 0) {
            std::vector<cl_device_id> devices(deviceCount);
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, deviceCount, devices.data(), nullptr);

            // Debug: List CPU devices
            for (cl_uint j = 0; j < deviceCount; j++) {
                char deviceName[256];
                clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
                qDebug() << "  OpenCL: CPU Device" << j << ":" << deviceName;
            }

            m_platform = platform;
            m_device = devices[0];

            qWarning() << "OpenCL: ⚠ Selected CPU device (no GPU available)";
            qWarning() << "OpenCL: Note - macOS has deprecated OpenCL for GPU. Consider using Metal backend for GPU acceleration.";
            return true;
        }
    }

    setError("No suitable OpenCL device found (neither GPU nor CPU)");
    qCritical() << "OpenCL: ✗ Device selection failed completely";
    return false;
}

bool OpenCLDoseBackend::compileKernels()
{
    qDebug() << "OpenCL: Compiling kernels...";

    QString kernelSource = getKernelSource();
    if (kernelSource.isEmpty()) {
        setError("Failed to load kernel source");
        return false;
    }

    QByteArray sourceBytes = kernelSource.toUtf8();
    const char* sourcePtr = sourceBytes.constData();
    size_t sourceSize = sourceBytes.size();

    cl_int err;
    m_program = clCreateProgramWithSource(m_context, 1, &sourcePtr, &sourceSize, &err);
    if (err != CL_SUCCESS) {
        setError(QString("Failed to create program: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    err = clBuildProgram(m_program, 1, &m_device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t logSize;
        clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

        std::vector<char> log(logSize);
        clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);

        setError(QString("Failed to build program: %1\nBuild log:\n%2")
                     .arg(getOpenCLErrorString(err))
                     .arg(QString::fromUtf8(log.data())));
        return false;
    }

    // Create kernels
    m_doseKernel = clCreateKernel(m_program, "calculateDoseKernel", &err);
    if (err != CL_SUCCESS) {
        setError(QString("Failed to create dose kernel: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    m_interpolationKernel = clCreateKernel(m_program, "interpolateVolumeKernel", &err);
    if (err != CL_SUCCESS) {
        setError(QString("Failed to create interpolation kernel: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    m_thresholdRecalcKernel = clCreateKernel(m_program, "recalculateDoseWithThresholdKernel", &err);
    if (err != CL_SUCCESS) {
        setError(QString("Failed to create threshold recalculation kernel: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    qDebug() << "OpenCL: Kernels compiled successfully";
    return true;
}

QString OpenCLDoseBackend::getKernelSource() const
{
    // Try multiple possible paths for the kernel file
    QStringList searchPaths = {
        // 1. Same directory as executable (POST_BUILD copy target)
        QCoreApplication::applicationDirPath() + "/opencl_kernels.cl",

        // 2. Current working directory
        QDir::currentPath() + "/opencl_kernels.cl",

        // 3. Source directory (for development)
        QDir::currentPath() + "/src/cyberknife/opencl_kernels.cl",

        // 4. Parent directory (build directory structure)
        "../src/cyberknife/opencl_kernels.cl",

        // 5. Absolute path from source (last resort)
        QString(SHIORIS_SOURCE_DIR) + "/src/cyberknife/opencl_kernels.cl"
    };

    for (const QString& kernelPath : searchPaths) {
        QFile file(kernelPath);
        if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QTextStream in(&file);
            QString source = in.readAll();
            file.close();
            qDebug() << "OpenCL: Loaded kernel from" << kernelPath;
            return source;
        }
    }

    qWarning() << "OpenCL: Could not load kernel source from any search path";
    qWarning() << "OpenCL: Searched paths:" << searchPaths;
    return QString();
}

bool OpenCLDoseBackend::uploadCTVolume(const cv::Mat& volume)
{
    if (!isReady()) {
        setError("Backend not initialized");
        return false;
    }

    if (volume.empty() || volume.type() != CV_16S) {
        setError("Invalid CT volume format (expected CV_16S)");
        return false;
    }

    int width = volume.size[2];
    int height = volume.size[1];
    int depth = volume.size[0];

    qDebug() << "OpenCL: Uploading CT volume:" << width << "x" << height << "x" << depth;

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
    cl_int err = clEnqueueWriteBuffer(m_commandQueue, m_ctBuffer, CL_TRUE,
                                      0, bufferSize, volume.data, 0, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        setError(QString("Failed to upload CT volume: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    qDebug() << "OpenCL: CT volume uploaded successfully";
    return true;
}

bool OpenCLDoseBackend::uploadBeamData(const GPUBeamData& beamData)
{
    if (!isReady()) {
        setError("Backend not initialized");
        return false;
    }

    qDebug() << "OpenCL: Uploading beam data tables...";

    // Store beam data dimensions
    m_beamDataDims.ofDepthCount = beamData.ofDepthCount;
    m_beamDataDims.ofCollimatorCount = beamData.ofCollimatorCount;
    m_beamDataDims.tmrDepthCount = beamData.tmrDepthCount;
    m_beamDataDims.tmrFieldSizeCount = beamData.tmrFieldSizeCount;
    m_beamDataDims.ocrDepthCount = beamData.ocrDepthCount;
    m_beamDataDims.ocrRadiusCount = beamData.ocrRadiusCount;
    m_beamDataDims.ocrCollimatorCount = beamData.ocrCollimatorCount;

    qDebug() << "OpenCL: Beam data dimensions:";
    qDebug() << "  OF table:" << beamData.ofDepthCount << "x" << beamData.ofCollimatorCount;
    qDebug() << "  TMR table:" << beamData.tmrDepthCount << "x" << beamData.tmrFieldSizeCount;
    qDebug() << "  OCR table:" << beamData.ocrDepthCount << "x" << beamData.ocrRadiusCount << "x" << beamData.ocrCollimatorCount;

    cl_int err;

    // Release old buffers
    if (m_ofTableBuffer) clReleaseMemObject(m_ofTableBuffer);
    if (m_ofDepthsBuffer) clReleaseMemObject(m_ofDepthsBuffer);
    if (m_ofCollimatorsBuffer) clReleaseMemObject(m_ofCollimatorsBuffer);
    if (m_tmrTableBuffer) clReleaseMemObject(m_tmrTableBuffer);
    if (m_tmrDepthsBuffer) clReleaseMemObject(m_tmrDepthsBuffer);
    if (m_tmrFieldSizesBuffer) clReleaseMemObject(m_tmrFieldSizesBuffer);
    if (m_ocrTableBuffer) clReleaseMemObject(m_ocrTableBuffer);
    if (m_ocrDepthsBuffer) clReleaseMemObject(m_ocrDepthsBuffer);
    if (m_ocrRadiiBuffer) clReleaseMemObject(m_ocrRadiiBuffer);
    if (m_ocrCollimatorsBuffer) clReleaseMemObject(m_ocrCollimatorsBuffer);

    // Upload OF table
    m_ofTableBuffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     beamData.ofTable.size() * sizeof(float),
                                     (void*)beamData.ofTable.data(), &err);
    if (err != CL_SUCCESS) {
        setError(QString("Failed to create OF table buffer: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    m_ofDepthsBuffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      beamData.ofDepths.size() * sizeof(float),
                                      (void*)beamData.ofDepths.data(), &err);
    m_ofCollimatorsBuffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           beamData.ofCollimators.size() * sizeof(float),
                                           (void*)beamData.ofCollimators.data(), &err);

    // Upload TMR table
    m_tmrTableBuffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      beamData.tmrTable.size() * sizeof(float),
                                      (void*)beamData.tmrTable.data(), &err);
    m_tmrDepthsBuffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       beamData.tmrDepths.size() * sizeof(float),
                                       (void*)beamData.tmrDepths.data(), &err);
    m_tmrFieldSizesBuffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           beamData.tmrFieldSizes.size() * sizeof(float),
                                           (void*)beamData.tmrFieldSizes.data(), &err);

    // Upload OCR table
    m_ocrTableBuffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      beamData.ocrTable.size() * sizeof(float),
                                      (void*)beamData.ocrTable.data(), &err);
    m_ocrDepthsBuffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       beamData.ocrDepths.size() * sizeof(float),
                                       (void*)beamData.ocrDepths.data(), &err);
    m_ocrRadiiBuffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      beamData.ocrRadii.size() * sizeof(float),
                                      (void*)beamData.ocrRadii.data(), &err);
    m_ocrCollimatorsBuffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            beamData.ocrCollimators.size() * sizeof(float),
                                            (void*)beamData.ocrCollimators.data(), &err);

    qDebug() << "OpenCL: Beam data uploaded successfully";
    return true;
}

bool OpenCLDoseBackend::calculateDose(const GPUComputeParams& params, cv::Mat& doseVolume)
{
    if (!isReady()) {
        setError("Backend not initialized");
        return false;
    }

    qDebug() << "OpenCL: Calculating dose...";

    // Clear computed mask
    size_t maskSize = m_volumeWidth * m_volumeHeight * m_volumeDepth;
    std::vector<uint8_t> mask(maskSize, 0);
    clEnqueueWriteBuffer(m_commandQueue, m_computedMaskBuffer, CL_FALSE,
                        0, maskSize, mask.data(), 0, nullptr, nullptr);

    // Set kernel arguments (lots of them!)
    int argIndex = 0;
    cl_int err;

    err = clSetKernelArg(m_doseKernel, argIndex++, sizeof(cl_mem), &m_ctBuffer);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(cl_mem), &m_doseBuffer);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(cl_mem), &m_computedMaskBuffer);

    // Volume dimensions
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &params.width);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &params.height);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &params.depth);

    // Voxel spacing
    float spacingX = static_cast<float>(params.spacingX);
    float spacingY = static_cast<float>(params.spacingY);
    float spacingZ = static_cast<float>(params.spacingZ);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &spacingX);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &spacingY);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &spacingZ);

    // Origin
    float originX = static_cast<float>(params.originX);
    float originY = static_cast<float>(params.originY);
    float originZ = static_cast<float>(params.originZ);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &originX);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &originY);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &originZ);

    // Orientation (9 floats)
    for (int i = 0; i < 3; i++) {
        float val = static_cast<float>(params.orientationX[i]);
        err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &val);
    }
    for (int i = 0; i < 3; i++) {
        float val = static_cast<float>(params.orientationY[i]);
        err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &val);
    }
    for (int i = 0; i < 3; i++) {
        float val = static_cast<float>(params.orientationZ[i]);
        err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &val);
    }

    // Step size
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &params.stepX);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &params.stepY);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &params.stepZ);

    int gridCountX = params.gridCountX > 0 ? params.gridCountX
                                           : computeCoarseGridCount(params.width, params.stepX);
    int gridCountY = params.gridCountY > 0 ? params.gridCountY
                                           : computeCoarseGridCount(params.height, params.stepY);
    int gridCountZ = params.gridCountZ > 0 ? params.gridCountZ
                                           : computeCoarseGridCount(params.depth, params.stepZ);

    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &gridCountX);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &gridCountY);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &gridCountZ);

    // Beam geometry (source, target, basis vectors)
    float beamSourceX = static_cast<float>(params.beam.sourcePosition.x());
    float beamSourceY = static_cast<float>(params.beam.sourcePosition.y());
    float beamSourceZ = static_cast<float>(params.beam.sourcePosition.z());
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &beamSourceX);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &beamSourceY);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &beamSourceZ);

    float beamTargetX = static_cast<float>(params.beam.targetPosition.x());
    float beamTargetY = static_cast<float>(params.beam.targetPosition.y());
    float beamTargetZ = static_cast<float>(params.beam.targetPosition.z());
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &beamTargetX);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &beamTargetY);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &beamTargetZ);

    // Beam basis vectors (9 floats)
    float beamXx = static_cast<float>(params.beam.beamX.x());
    float beamXy = static_cast<float>(params.beam.beamX.y());
    float beamXz = static_cast<float>(params.beam.beamX.z());
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &beamXx);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &beamXy);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &beamXz);

    float beamYx = static_cast<float>(params.beam.beamY.x());
    float beamYy = static_cast<float>(params.beam.beamY.y());
    float beamYz = static_cast<float>(params.beam.beamY.z());
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &beamYx);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &beamYy);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &beamYz);

    float beamZx = static_cast<float>(params.beam.beamZ.x());
    float beamZy = static_cast<float>(params.beam.beamZ.y());
    float beamZz = static_cast<float>(params.beam.beamZ.z());
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &beamZx);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &beamZy);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &beamZz);

    float collimatorSize = static_cast<float>(params.beam.collimatorSize);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &collimatorSize);

    // Reference dose
    // CPU 実装と揃えるため GPU カーネル内では参照線量によるスケーリングを行わない
    float referenceDose = 1.0f;
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &referenceDose);

    // Beam data tables - OF
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(cl_mem), &m_ofTableBuffer);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(cl_mem), &m_ofDepthsBuffer);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(cl_mem), &m_ofCollimatorsBuffer);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &m_beamDataDims.ofDepthCount);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &m_beamDataDims.ofCollimatorCount);

    // Beam data tables - TMR
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(cl_mem), &m_tmrTableBuffer);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(cl_mem), &m_tmrDepthsBuffer);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(cl_mem), &m_tmrFieldSizesBuffer);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &m_beamDataDims.tmrDepthCount);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &m_beamDataDims.tmrFieldSizeCount);

    // Beam data tables - OCR
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(cl_mem), &m_ocrTableBuffer);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(cl_mem), &m_ocrDepthsBuffer);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(cl_mem), &m_ocrRadiiBuffer);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(cl_mem), &m_ocrCollimatorsBuffer);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &m_beamDataDims.ocrDepthCount);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &m_beamDataDims.ocrRadiusCount);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &m_beamDataDims.ocrCollimatorCount);

    // Depth profile for CT-density corrected depth calculation
    cl_mem d_depthCumulative = nullptr;
    int depthSampleCount = params.depthCumulative.size();
    float depthEntryDistance = static_cast<float>(params.depthEntryDistance);
    float depthStepSize = static_cast<float>(params.depthStepSize);

    if (depthSampleCount > 0) {
        d_depthCumulative = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          depthSampleCount * sizeof(float),
                                          (void*)params.depthCumulative.data(), &err);
        if (err != CL_SUCCESS) {
            setError(QString("Failed to create depth profile buffer: %1").arg(getOpenCLErrorString(err)));
            return false;
        }
        qDebug() << "OpenCL: Depth profile uploaded:" << depthSampleCount << "samples, entry ="
                 << depthEntryDistance << "mm, step =" << depthStepSize << "mm";
    } else {
        // Create a dummy buffer (OpenCL requires non-null buffer pointers)
        float dummy = 0.0f;
        d_depthCumulative = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(float), &dummy, &err);
    }

    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(cl_mem), &d_depthCumulative);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &depthSampleCount);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &depthEntryDistance);
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(float), &depthStepSize);

    // Accumulate mode
    int accumulate = 0;
    err |= clSetKernelArg(m_doseKernel, argIndex++, sizeof(int), &accumulate);

    if (err != CL_SUCCESS) {
        setError(QString("Failed to set kernel arguments: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    // Execute kernel
    size_t globalWorkSize[3] = {
        static_cast<size_t>(std::max(1, gridCountX)),
        static_cast<size_t>(std::max(1, gridCountY)),
        static_cast<size_t>(std::max(1, gridCountZ))
    };

    err = clEnqueueNDRangeKernel(m_commandQueue, m_doseKernel, 3, nullptr,
                                globalWorkSize, nullptr, 0, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        setError(QString("Failed to execute dose kernel: %1").arg(getOpenCLErrorString(err)));
        if (d_depthCumulative) clReleaseMemObject(d_depthCumulative);
        return false;
    }

    // Wait for completion
    clFinish(m_commandQueue);

    // Free temporary depth profile buffer
    if (d_depthCumulative) {
        clReleaseMemObject(d_depthCumulative);
    }

    // Download result
    size_t doseSize = m_volumeWidth * m_volumeHeight * m_volumeDepth * sizeof(float);
    err = clEnqueueReadBuffer(m_commandQueue, m_doseBuffer, CL_TRUE,
                             0, doseSize, doseVolume.data, 0, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        setError(QString("Failed to download dose volume: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    qDebug() << "OpenCL: Dose calculation completed";
    return true;
}

bool OpenCLDoseBackend::interpolateVolume(cv::Mat& doseVolume,
                                         const std::vector<uint8_t>& computedMask,
                                         int step)
{
    if (!isReady()) {
        setError("Backend not initialized");
        return false;
    }

    if (doseVolume.empty() || doseVolume.type() != CV_32F) {
        setError("Invalid dose volume format (expected CV_32F)");
        return false;
    }

    qDebug() << "OpenCL: Interpolating volume with step" << step;

    // Upload computed mask to GPU
    size_t maskSize = m_volumeWidth * m_volumeHeight * m_volumeDepth;
    cl_int err = clEnqueueWriteBuffer(m_commandQueue, m_computedMaskBuffer, CL_TRUE,
                                      0, maskSize, computedMask.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        setError(QString("Failed to upload computed mask: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    // Upload current dose volume to GPU
    size_t doseSize = m_volumeWidth * m_volumeHeight * m_volumeDepth * sizeof(float);
    err = clEnqueueWriteBuffer(m_commandQueue, m_doseBuffer, CL_TRUE,
                               0, doseSize, doseVolume.data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        setError(QString("Failed to upload dose volume: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    // Set kernel arguments
    int argIndex = 0;
    err = clSetKernelArg(m_interpolationKernel, argIndex++, sizeof(cl_mem), &m_doseBuffer);
    err |= clSetKernelArg(m_interpolationKernel, argIndex++, sizeof(cl_mem), &m_computedMaskBuffer);
    err |= clSetKernelArg(m_interpolationKernel, argIndex++, sizeof(int), &m_volumeWidth);
    err |= clSetKernelArg(m_interpolationKernel, argIndex++, sizeof(int), &m_volumeHeight);
    err |= clSetKernelArg(m_interpolationKernel, argIndex++, sizeof(int), &m_volumeDepth);
    err |= clSetKernelArg(m_interpolationKernel, argIndex++, sizeof(int), &step);

    if (err != CL_SUCCESS) {
        setError(QString("Failed to set interpolation kernel arguments: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    // Execute kernel - one thread per voxel
    size_t globalWorkSize[3] = {
        static_cast<size_t>(m_volumeWidth),
        static_cast<size_t>(m_volumeHeight),
        static_cast<size_t>(m_volumeDepth)
    };

    err = clEnqueueNDRangeKernel(m_commandQueue, m_interpolationKernel, 3, nullptr,
                                 globalWorkSize, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        setError(QString("Failed to execute interpolation kernel: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    // Wait for completion
    clFinish(m_commandQueue);

    // Download result
    err = clEnqueueReadBuffer(m_commandQueue, m_doseBuffer, CL_TRUE,
                             0, doseSize, doseVolume.data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        setError(QString("Failed to download interpolated volume: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    qDebug() << "OpenCL: Interpolation completed";
    return true;
}

bool OpenCLDoseBackend::recalculateDoseWithThreshold(const GPUComputeParams& params,
                                                     cv::Mat& doseVolume,
                                                     float threshold,
                                                     int skipStep)
{
    if (!isReady()) {
        setError("Backend not initialized");
        return false;
    }

    qDebug() << "OpenCL: Recalculating dose with threshold" << threshold << "skipStep" << skipStep;

    // Upload current dose volume to GPU
    size_t doseSize = m_volumeWidth * m_volumeHeight * m_volumeDepth * sizeof(float);
    cl_int err = clEnqueueWriteBuffer(m_commandQueue, m_doseBuffer, CL_TRUE,
                                      0, doseSize, doseVolume.data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        setError(QString("Failed to upload dose volume: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    // Set kernel arguments (similar to calculateDose, but with threshold)
    int argIndex = 0;
    err = clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(cl_mem), &m_ctBuffer);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(cl_mem), &m_doseBuffer);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(cl_mem), &m_computedMaskBuffer);

    // Volume dimensions
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &params.width);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &params.height);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &params.depth);

    // Voxel spacing
    float spacingX = static_cast<float>(params.spacingX);
    float spacingY = static_cast<float>(params.spacingY);
    float spacingZ = static_cast<float>(params.spacingZ);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &spacingX);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &spacingY);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &spacingZ);

    // Origin
    float originX = static_cast<float>(params.originX);
    float originY = static_cast<float>(params.originY);
    float originZ = static_cast<float>(params.originZ);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &originX);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &originY);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &originZ);

    // Orientation (9 floats)
    for (int i = 0; i < 3; i++) {
        float val = static_cast<float>(params.orientationX[i]);
        err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &val);
    }
    for (int i = 0; i < 3; i++) {
        float val = static_cast<float>(params.orientationY[i]);
        err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &val);
    }
    for (int i = 0; i < 3; i++) {
        float val = static_cast<float>(params.orientationZ[i]);
        err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &val);
    }

    // Step size and grid count
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &params.stepX);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &params.stepY);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &params.stepZ);

    int gridCountX = params.gridCountX;
    int gridCountY = params.gridCountY;
    int gridCountZ = params.gridCountZ;
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &gridCountX);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &gridCountY);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &gridCountZ);

    // Beam geometry
    float beamSourceX = static_cast<float>(params.beam.sourcePosition.x());
    float beamSourceY = static_cast<float>(params.beam.sourcePosition.y());
    float beamSourceZ = static_cast<float>(params.beam.sourcePosition.z());
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &beamSourceX);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &beamSourceY);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &beamSourceZ);

    float beamTargetX = static_cast<float>(params.beam.targetPosition.x());
    float beamTargetY = static_cast<float>(params.beam.targetPosition.y());
    float beamTargetZ = static_cast<float>(params.beam.targetPosition.z());
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &beamTargetX);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &beamTargetY);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &beamTargetZ);

    // Beam basis vectors (9 floats)
    float beamXx = static_cast<float>(params.beam.beamX.x());
    float beamXy = static_cast<float>(params.beam.beamX.y());
    float beamXz = static_cast<float>(params.beam.beamX.z());
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &beamXx);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &beamXy);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &beamXz);

    float beamYx = static_cast<float>(params.beam.beamY.x());
    float beamYy = static_cast<float>(params.beam.beamY.y());
    float beamYz = static_cast<float>(params.beam.beamY.z());
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &beamYx);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &beamYy);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &beamYz);

    float beamZx = static_cast<float>(params.beam.beamZ.x());
    float beamZy = static_cast<float>(params.beam.beamZ.y());
    float beamZz = static_cast<float>(params.beam.beamZ.z());
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &beamZx);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &beamZy);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &beamZz);

    float collimatorSize = static_cast<float>(params.beam.collimatorSize);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &collimatorSize);

    // Reference dose
    float referenceDose = 1.0f;
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &referenceDose);

    // Beam data tables - OF
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(cl_mem), &m_ofTableBuffer);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(cl_mem), &m_ofDepthsBuffer);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(cl_mem), &m_ofCollimatorsBuffer);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &m_beamDataDims.ofDepthCount);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &m_beamDataDims.ofCollimatorCount);

    // Beam data tables - TMR
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(cl_mem), &m_tmrTableBuffer);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(cl_mem), &m_tmrDepthsBuffer);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(cl_mem), &m_tmrFieldSizesBuffer);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &m_beamDataDims.tmrDepthCount);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &m_beamDataDims.tmrFieldSizeCount);

    // Beam data tables - OCR
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(cl_mem), &m_ocrTableBuffer);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(cl_mem), &m_ocrDepthsBuffer);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(cl_mem), &m_ocrRadiiBuffer);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(cl_mem), &m_ocrCollimatorsBuffer);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &m_beamDataDims.ocrDepthCount);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &m_beamDataDims.ocrRadiusCount);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &m_beamDataDims.ocrCollimatorCount);

    // Depth profile for CT-density corrected depth calculation
    cl_mem d_depthCumulative = nullptr;
    int depthSampleCount = params.depthCumulative.size();
    float depthEntryDistance = static_cast<float>(params.depthEntryDistance);
    float depthStepSize = static_cast<float>(params.depthStepSize);

    if (depthSampleCount > 0) {
        d_depthCumulative = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          depthSampleCount * sizeof(float),
                                          (void*)params.depthCumulative.data(), &err);
        if (err != CL_SUCCESS) {
            setError(QString("Failed to create depth profile buffer: %1").arg(getOpenCLErrorString(err)));
            return false;
        }
    } else {
        // Create a dummy buffer
        float dummy = 0.0f;
        d_depthCumulative = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(float), &dummy, &err);
    }

    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(cl_mem), &d_depthCumulative);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &depthSampleCount);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &depthEntryDistance);
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &depthStepSize);

    // Threshold parameter
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(float), &threshold);

    // Skip step parameter
    err |= clSetKernelArg(m_thresholdRecalcKernel, argIndex++, sizeof(int), &skipStep);

    if (err != CL_SUCCESS) {
        setError(QString("Failed to set kernel arguments: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    // Execute kernel
    size_t globalWorkSize[3] = {
        static_cast<size_t>(std::max(1, gridCountX)),
        static_cast<size_t>(std::max(1, gridCountY)),
        static_cast<size_t>(std::max(1, gridCountZ))
    };

    err = clEnqueueNDRangeKernel(m_commandQueue, m_thresholdRecalcKernel, 3, nullptr,
                                globalWorkSize, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        setError(QString("Failed to execute threshold recalc kernel: %1").arg(getOpenCLErrorString(err)));
        if (d_depthCumulative) clReleaseMemObject(d_depthCumulative);
        return false;
    }

    // Wait for completion
    clFinish(m_commandQueue);

    // Free temporary depth profile buffer
    if (d_depthCumulative) {
        clReleaseMemObject(d_depthCumulative);
    }

    // Download result
    err = clEnqueueReadBuffer(m_commandQueue, m_doseBuffer, CL_TRUE,
                             0, doseSize, doseVolume.data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        setError(QString("Failed to download dose volume: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    qDebug() << "OpenCL: Threshold recalculation completed";
    return true;
}

bool OpenCLDoseBackend::createBuffers(int width, int height, int depth)
{
    qDebug() << "OpenCL: Creating buffers for" << width << "x" << height << "x" << depth;

    cl_int err;
    size_t ctSize = width * height * depth * sizeof(short);
    size_t doseSize = width * height * depth * sizeof(float);
    size_t maskSize = width * height * depth * sizeof(uint8_t);

    m_ctBuffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY, ctSize, nullptr, &err);
    if (err != CL_SUCCESS) {
        setError(QString("Failed to create CT buffer: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    m_doseBuffer = clCreateBuffer(m_context, CL_MEM_READ_WRITE, doseSize, nullptr, &err);
    if (err != CL_SUCCESS) {
        setError(QString("Failed to create dose buffer: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    m_computedMaskBuffer = clCreateBuffer(m_context, CL_MEM_READ_WRITE, maskSize, nullptr, &err);
    if (err != CL_SUCCESS) {
        setError(QString("Failed to create mask buffer: %1").arg(getOpenCLErrorString(err)));
        return false;
    }

    return true;
}

void OpenCLDoseBackend::releaseBuffers()
{
    if (m_ctBuffer) {
        clReleaseMemObject(m_ctBuffer);
        m_ctBuffer = nullptr;
    }
    if (m_doseBuffer) {
        clReleaseMemObject(m_doseBuffer);
        m_doseBuffer = nullptr;
    }
    if (m_computedMaskBuffer) {
        clReleaseMemObject(m_computedMaskBuffer);
        m_computedMaskBuffer = nullptr;
    }

    // Release beam data buffers
    if (m_ofTableBuffer) clReleaseMemObject(m_ofTableBuffer);
    if (m_ofDepthsBuffer) clReleaseMemObject(m_ofDepthsBuffer);
    if (m_ofCollimatorsBuffer) clReleaseMemObject(m_ofCollimatorsBuffer);
    if (m_tmrTableBuffer) clReleaseMemObject(m_tmrTableBuffer);
    if (m_tmrDepthsBuffer) clReleaseMemObject(m_tmrDepthsBuffer);
    if (m_tmrFieldSizesBuffer) clReleaseMemObject(m_tmrFieldSizesBuffer);
    if (m_ocrTableBuffer) clReleaseMemObject(m_ocrTableBuffer);
    if (m_ocrDepthsBuffer) clReleaseMemObject(m_ocrDepthsBuffer);
    if (m_ocrRadiiBuffer) clReleaseMemObject(m_ocrRadiiBuffer);
    if (m_ocrCollimatorsBuffer) clReleaseMemObject(m_ocrCollimatorsBuffer);

    m_ofTableBuffer = nullptr;
    m_ofDepthsBuffer = nullptr;
    m_ofCollimatorsBuffer = nullptr;
    m_tmrTableBuffer = nullptr;
    m_tmrDepthsBuffer = nullptr;
    m_tmrFieldSizesBuffer = nullptr;
    m_ocrTableBuffer = nullptr;
    m_ocrDepthsBuffer = nullptr;
    m_ocrRadiiBuffer = nullptr;
    m_ocrCollimatorsBuffer = nullptr;
}

void OpenCLDoseBackend::releaseKernels()
{
    if (m_doseKernel) {
        clReleaseKernel(m_doseKernel);
        m_doseKernel = nullptr;
    }
    if (m_interpolationKernel) {
        clReleaseKernel(m_interpolationKernel);
        m_interpolationKernel = nullptr;
    }
    if (m_thresholdRecalcKernel) {
        clReleaseKernel(m_thresholdRecalcKernel);
        m_thresholdRecalcKernel = nullptr;
    }
    if (m_program) {
        clReleaseProgram(m_program);
        m_program = nullptr;
    }
}

void OpenCLDoseBackend::releaseOpenCLResources()
{
    releaseKernels();
    releaseBuffers();

    if (m_commandQueue) {
        clReleaseCommandQueue(m_commandQueue);
        m_commandQueue = nullptr;
    }
    if (m_context) {
        clReleaseContext(m_context);
        m_context = nullptr;
    }
}

void OpenCLDoseBackend::cleanup()
{
    qDebug() << "OpenCL: Cleaning up resources";
    releaseOpenCLResources();
    m_initialized = false;
}

QString OpenCLDoseBackend::getLastError() const
{
    return m_lastError;
}

void OpenCLDoseBackend::setError(const QString& error)
{
    m_lastError = error;
    qWarning() << "OpenCL Error:" << error;
}

QString OpenCLDoseBackend::getOpenCLErrorString(cl_int error) const
{
    switch (error) {
        case CL_SUCCESS: return "Success";
        case CL_DEVICE_NOT_FOUND: return "Device not found";
        case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE: return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES: return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
        case CL_BUILD_PROGRAM_FAILURE: return "Build program failure";
        case CL_INVALID_VALUE: return "Invalid value";
        case CL_INVALID_DEVICE: return "Invalid device";
        case CL_INVALID_CONTEXT: return "Invalid context";
        case CL_INVALID_KERNEL: return "Invalid kernel";
        case CL_INVALID_WORK_GROUP_SIZE: return "Invalid work group size";
        default: return QString("Unknown error (%1)").arg(error);
    }
}

} // namespace CyberKnife

#endif // USE_OPENCL_BACKEND
