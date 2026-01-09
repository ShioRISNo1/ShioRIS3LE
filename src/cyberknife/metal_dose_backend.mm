#include "cyberknife/metal_dose_backend.h"

#ifdef USE_METAL_BACKEND

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <simd/simd.h>

#include <QDebug>
#include <QFile>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace CyberKnife {

static int computeCoarseGridCount(int size, int step)
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

// Helper to convert void* to id<MTLDevice>
static inline id<MTLDevice> toDevice(void* ptr) {
    return (__bridge id<MTLDevice>)ptr;
}

static inline id<MTLCommandQueue> toCommandQueue(void* ptr) {
    return (__bridge id<MTLCommandQueue>)ptr;
}

static inline id<MTLLibrary> toLibrary(void* ptr) {
    return (__bridge id<MTLLibrary>)ptr;
}

static inline id<MTLComputePipelineState> toPipeline(void* ptr) {
    return (__bridge id<MTLComputePipelineState>)ptr;
}

static inline id<MTLBuffer> toBuffer(void* ptr) {
    return (__bridge id<MTLBuffer>)ptr;
}

MetalDoseBackend::MetalDoseBackend()
    : m_device(nullptr)
    , m_commandQueue(nullptr)
    , m_library(nullptr)
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
    , m_ofDepthCount(0)
    , m_ofCollimatorCount(0)
    , m_tmrDepthCount(0)
    , m_tmrFieldSizeCount(0)
    , m_ocrDepthCount(0)
    , m_ocrRadiusCount(0)
    , m_ocrCollimatorCount(0)
    , m_initialized(false)
{
}

MetalDoseBackend::~MetalDoseBackend()
{
    cleanup();
}

bool MetalDoseBackend::initialize()
{
    if (m_initialized) {
        return true;
    }

    qDebug() << "Metal: Initializing backend...";

    // Create Metal device
    if (!createMetalDevice()) {
        return false;
    }

    // Compile shaders
    if (!compileShaders()) {
        cleanup();
        return false;
    }

    m_initialized = true;
    qDebug() << "Metal: Initialization successful";
    qDebug() << "Metal: Device:" << getDeviceInfo();

    return true;
}

bool MetalDoseBackend::createMetalDevice()
{
    @autoreleasepool {
        // Get the default Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        if (!device) {
            setError("No Metal device found. Metal requires macOS 10.11 or later.");
            qCritical() << "Metal: ✗ No device available";
            return false;
        }

        m_device = (__bridge_retained void*)device;

        // Create command queue
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            setError("Failed to create Metal command queue");
            qCritical() << "Metal: ✗ Failed to create command queue";
            return false;
        }

        m_commandQueue = (__bridge_retained void*)queue;

        qDebug() << "Metal: ✓ Device created:" << QString::fromNSString([device name]);
        qDebug() << "Metal: Supports GPU Family:" << [device supportsFamily:MTLGPUFamilyApple1];

        return true;
    }
}

bool MetalDoseBackend::compileShaders()
{
    @autoreleasepool {
        qDebug() << "Metal: Compiling shaders...";

        id<MTLDevice> device = toDevice(m_device);

        // Try to load from file first
        QString metalPath = QStringLiteral(SHIORIS_SOURCE_DIR) + "/src/cyberknife/metal_kernels.metal";
        QFile file(metalPath);

        NSString* source = nil;

        if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QString sourceCode = QString::fromUtf8(file.readAll());
            file.close();
            source = sourceCode.toNSString();
            qDebug() << "Metal: Loaded shader from" << metalPath;
        } else {
            setError(QString("Failed to load Metal shader from %1").arg(metalPath));
            return false;
        }

        // Compile the shader source
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                       options:nil
                                                         error:&error];

        if (!library || error) {
            QString errorMsg = QString::fromNSString([error localizedDescription]);
            setError(QString("Failed to compile Metal shaders: %1").arg(errorMsg));
            qCritical() << "Metal: ✗ Shader compilation failed:" << errorMsg;
            return false;
        }

        m_library = (__bridge_retained void*)library;

        // Create dose calculation pipeline
        id<MTLFunction> doseFunc = [library newFunctionWithName:@"calculateDoseKernel"];
        if (!doseFunc) {
            setError("Failed to find calculateDoseKernel function in Metal library");
            return false;
        }

        id<MTLComputePipelineState> dosePipeline = [device newComputePipelineStateWithFunction:doseFunc
                                                                                          error:&error];
        if (!dosePipeline || error) {
            QString errorMsg = QString::fromNSString([error localizedDescription]);
            setError(QString("Failed to create dose pipeline: %1").arg(errorMsg));
            return false;
        }

        m_doseKernel = (__bridge_retained void*)dosePipeline;

        // Create interpolation pipeline
        id<MTLFunction> interpFunc = [library newFunctionWithName:@"interpolateVolumeKernel"];
        if (!interpFunc) {
            setError("Failed to find interpolateVolumeKernel function in Metal library");
            return false;
        }

        id<MTLComputePipelineState> interpPipeline = [device newComputePipelineStateWithFunction:interpFunc
                                                                                            error:&error];
        if (!interpPipeline || error) {
            QString errorMsg = QString::fromNSString([error localizedDescription]);
            setError(QString("Failed to create interpolation pipeline: %1").arg(errorMsg));
            return false;
        }

        m_interpolationKernel = (__bridge_retained void*)interpPipeline;

        // Create threshold recalculation pipeline
        id<MTLFunction> thresholdFunc = [library newFunctionWithName:@"recalculateDoseWithThresholdKernel"];
        if (!thresholdFunc) {
            setError("Failed to find recalculateDoseWithThresholdKernel function in Metal library");
            return false;
        }

        id<MTLComputePipelineState> thresholdPipeline = [device newComputePipelineStateWithFunction:thresholdFunc
                                                                                               error:&error];
        if (!thresholdPipeline || error) {
            QString errorMsg = QString::fromNSString([error localizedDescription]);
            setError(QString("Failed to create threshold recalculation pipeline: %1").arg(errorMsg));
            return false;
        }

        m_thresholdRecalcKernel = (__bridge_retained void*)thresholdPipeline;

        qDebug() << "Metal: ✓ Shaders compiled successfully (dose, interpolation, threshold recalc)";
        return true;
    }
}

bool MetalDoseBackend::isReady() const
{
    return m_initialized && m_device != nullptr && m_commandQueue != nullptr;
}

GPUBackendType MetalDoseBackend::getBackendType() const
{
    return GPUBackendType::Metal;
}

QString MetalDoseBackend::getBackendName() const
{
    return "Metal";
}

QString MetalDoseBackend::getDeviceInfo() const
{
    @autoreleasepool {
        if (!m_device) {
            return "No device";
        }

        id<MTLDevice> device = toDevice(m_device);
        return QString::fromNSString([device name]);
    }
}

bool MetalDoseBackend::createBuffers(int width, int height, int depth)
{
    @autoreleasepool {
        qDebug() << "Metal: Creating buffers for" << width << "x" << height << "x" << depth;

        id<MTLDevice> device = toDevice(m_device);

        size_t ctSize = width * height * depth * sizeof(short);
        size_t doseSize = width * height * depth * sizeof(float);
        size_t maskSize = width * height * depth * sizeof(uint8_t);

        id<MTLBuffer> ctBuf = [device newBufferWithLength:ctSize
                                                  options:MTLResourceStorageModeShared];
        if (!ctBuf) {
            setError("Failed to create CT buffer");
            return false;
        }
        m_ctBuffer = (__bridge_retained void*)ctBuf;

        id<MTLBuffer> doseBuf = [device newBufferWithLength:doseSize
                                                     options:MTLResourceStorageModeShared];
        if (!doseBuf) {
            setError("Failed to create dose buffer");
            return false;
        }
        m_doseBuffer = (__bridge_retained void*)doseBuf;

        id<MTLBuffer> maskBuf = [device newBufferWithLength:maskSize
                                                     options:MTLResourceStorageModeShared];
        if (!maskBuf) {
            setError("Failed to create mask buffer");
            return false;
        }
        m_computedMaskBuffer = (__bridge_retained void*)maskBuf;

        qDebug() << "Metal: ✓ Buffers created successfully";
        return true;
    }
}

bool MetalDoseBackend::uploadCTVolume(const cv::Mat& volume)
{
    @autoreleasepool {
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

        qDebug() << "Metal: Uploading CT volume:" << width << "x" << height << "x" << depth;

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

        // Copy data to buffer
        id<MTLBuffer> ctBuf = toBuffer(m_ctBuffer);
        size_t bufferSize = width * height * depth * sizeof(short);
        memcpy([ctBuf contents], volume.data, bufferSize);

        qDebug() << "Metal: ✓ CT volume uploaded successfully";
        return true;
    }
}

bool MetalDoseBackend::uploadBeamData(const GPUBeamData& beamData)
{
    @autoreleasepool {
        if (!isReady()) {
            setError("Backend not initialized");
            return false;
        }

        qDebug() << "Metal: Uploading beam data tables...";

        id<MTLDevice> device = toDevice(m_device);

        // Release old buffers
        if (m_ofTableBuffer) CFRelease(m_ofTableBuffer);
        if (m_ofDepthsBuffer) CFRelease(m_ofDepthsBuffer);
        if (m_ofCollimatorsBuffer) CFRelease(m_ofCollimatorsBuffer);
        if (m_tmrTableBuffer) CFRelease(m_tmrTableBuffer);
        if (m_tmrDepthsBuffer) CFRelease(m_tmrDepthsBuffer);
        if (m_tmrFieldSizesBuffer) CFRelease(m_tmrFieldSizesBuffer);
        if (m_ocrTableBuffer) CFRelease(m_ocrTableBuffer);
        if (m_ocrDepthsBuffer) CFRelease(m_ocrDepthsBuffer);
        if (m_ocrRadiiBuffer) CFRelease(m_ocrRadiiBuffer);
        if (m_ocrCollimatorsBuffer) CFRelease(m_ocrCollimatorsBuffer);

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

        // Store table dimensions
        m_ofDepthCount = beamData.ofDepthCount;
        m_ofCollimatorCount = beamData.ofCollimatorCount;
        m_tmrDepthCount = beamData.tmrDepthCount;
        m_tmrFieldSizeCount = beamData.tmrFieldSizeCount;
        m_ocrDepthCount = beamData.ocrDepthCount;
        m_ocrRadiusCount = beamData.ocrRadiusCount;
        m_ocrCollimatorCount = beamData.ocrCollimatorCount;

        // Create and upload OF table
        size_t ofTableSize = beamData.ofTable.size() * sizeof(float);
        if (ofTableSize > 0) {
            id<MTLBuffer> buf = [device newBufferWithBytes:beamData.ofTable.data()
                                                    length:ofTableSize
                                                   options:MTLResourceStorageModeShared];
            m_ofTableBuffer = (__bridge_retained void*)buf;
        }

        // Create and upload OF depths
        size_t ofDepthsSize = beamData.ofDepths.size() * sizeof(float);
        if (ofDepthsSize > 0) {
            id<MTLBuffer> buf = [device newBufferWithBytes:beamData.ofDepths.data()
                                                    length:ofDepthsSize
                                                   options:MTLResourceStorageModeShared];
            m_ofDepthsBuffer = (__bridge_retained void*)buf;
        }

        // Create and upload OF collimators
        size_t ofCollimatorsSize = beamData.ofCollimators.size() * sizeof(float);
        if (ofCollimatorsSize > 0) {
            id<MTLBuffer> buf = [device newBufferWithBytes:beamData.ofCollimators.data()
                                                    length:ofCollimatorsSize
                                                   options:MTLResourceStorageModeShared];
            m_ofCollimatorsBuffer = (__bridge_retained void*)buf;
        }

        // Create and upload TMR table
        size_t tmrTableSize = beamData.tmrTable.size() * sizeof(float);
        if (tmrTableSize > 0) {
            id<MTLBuffer> buf = [device newBufferWithBytes:beamData.tmrTable.data()
                                                    length:tmrTableSize
                                                   options:MTLResourceStorageModeShared];
            m_tmrTableBuffer = (__bridge_retained void*)buf;
        }

        // Create and upload TMR depths
        size_t tmrDepthsSize = beamData.tmrDepths.size() * sizeof(float);
        if (tmrDepthsSize > 0) {
            id<MTLBuffer> buf = [device newBufferWithBytes:beamData.tmrDepths.data()
                                                    length:tmrDepthsSize
                                                   options:MTLResourceStorageModeShared];
            m_tmrDepthsBuffer = (__bridge_retained void*)buf;
        }

        // Create and upload TMR field sizes
        size_t tmrFieldSizesSize = beamData.tmrFieldSizes.size() * sizeof(float);
        if (tmrFieldSizesSize > 0) {
            id<MTLBuffer> buf = [device newBufferWithBytes:beamData.tmrFieldSizes.data()
                                                    length:tmrFieldSizesSize
                                                   options:MTLResourceStorageModeShared];
            m_tmrFieldSizesBuffer = (__bridge_retained void*)buf;
        }

        // Create and upload OCR table
        size_t ocrTableSize = beamData.ocrTable.size() * sizeof(float);
        if (ocrTableSize > 0) {
            id<MTLBuffer> buf = [device newBufferWithBytes:beamData.ocrTable.data()
                                                    length:ocrTableSize
                                                   options:MTLResourceStorageModeShared];
            m_ocrTableBuffer = (__bridge_retained void*)buf;
        }

        // Create and upload OCR depths
        size_t ocrDepthsSize = beamData.ocrDepths.size() * sizeof(float);
        if (ocrDepthsSize > 0) {
            id<MTLBuffer> buf = [device newBufferWithBytes:beamData.ocrDepths.data()
                                                    length:ocrDepthsSize
                                                   options:MTLResourceStorageModeShared];
            m_ocrDepthsBuffer = (__bridge_retained void*)buf;
        }

        // Create and upload OCR radii
        size_t ocrRadiiSize = beamData.ocrRadii.size() * sizeof(float);
        if (ocrRadiiSize > 0) {
            id<MTLBuffer> buf = [device newBufferWithBytes:beamData.ocrRadii.data()
                                                    length:ocrRadiiSize
                                                   options:MTLResourceStorageModeShared];
            m_ocrRadiiBuffer = (__bridge_retained void*)buf;
        }

        // Create and upload OCR collimators
        size_t ocrCollimatorsSize = beamData.ocrCollimators.size() * sizeof(float);
        if (ocrCollimatorsSize > 0) {
            id<MTLBuffer> buf = [device newBufferWithBytes:beamData.ocrCollimators.data()
                                                    length:ocrCollimatorsSize
                                                   options:MTLResourceStorageModeShared];
            m_ocrCollimatorsBuffer = (__bridge_retained void*)buf;
        }

        qDebug() << "Metal: ✓ Beam data uploaded successfully";
        qDebug() << "Metal: OF:" << m_ofDepthCount << "x" << m_ofCollimatorCount;
        qDebug() << "Metal: TMR:" << m_tmrDepthCount << "x" << m_tmrFieldSizeCount;
        qDebug() << "Metal: OCR:" << m_ocrDepthCount << "x" << m_ocrRadiusCount << "x" << m_ocrCollimatorCount;
        return true;
    }
}

bool MetalDoseBackend::calculateDose(const GPUComputeParams& params, cv::Mat& doseVolume)
{
    @autoreleasepool {
        if (!isReady()) {
            setError("Backend not initialized");
            return false;
        }

        qDebug() << "Metal: Calculating dose...";

        id<MTLDevice> device = toDevice(m_device);
        id<MTLCommandQueue> queue = toCommandQueue(m_commandQueue);
        id<MTLComputePipelineState> pipeline = toPipeline(m_doseKernel);

        // Create DoseParams structure matching Metal kernel
        struct DoseParams {
            int width, height, depth;
            float spacingX, spacingY, spacingZ;
            float originX, originY, originZ;
            float orientationX[3], orientationY[3], orientationZ[3];
            int stepX, stepY, stepZ;
            int gridCountX, gridCountY, gridCountZ;
            float beamSourceX, beamSourceY, beamSourceZ;
            float beamTargetX, beamTargetY, beamTargetZ;
            float beamBasisX[3], beamBasisY[3], beamBasisZ[3];
            float collimatorSize;
            float referenceDose;
            float depthEntryDistance;
            float depthStepSize;
            int depthSampleCount;
            int depthValid;
            int ofDepthCount, ofCollimatorCount;
            int tmrDepthCount, tmrFieldSizeCount;
            int ocrDepthCount, ocrRadiusCount, ocrCollimatorCount;
            int accumulate;  // Whether to accumulate dose (1) or replace (0)
        };

        DoseParams doseParams;
        doseParams.width = params.width;
        doseParams.height = params.height;
        doseParams.depth = params.depth;
        doseParams.spacingX = static_cast<float>(params.spacingX);
        doseParams.spacingY = static_cast<float>(params.spacingY);
        doseParams.spacingZ = static_cast<float>(params.spacingZ);
        doseParams.originX = static_cast<float>(params.originX);
        doseParams.originY = static_cast<float>(params.originY);
        doseParams.originZ = static_cast<float>(params.originZ);

        for (int i = 0; i < 3; i++) {
            doseParams.orientationX[i] = static_cast<float>(params.orientationX[i]);
            doseParams.orientationY[i] = static_cast<float>(params.orientationY[i]);
            doseParams.orientationZ[i] = static_cast<float>(params.orientationZ[i]);
        }

        doseParams.stepX = params.stepX;
        doseParams.stepY = params.stepY;
        doseParams.stepZ = params.stepZ;

        doseParams.gridCountX = params.gridCountX > 0
                                     ? params.gridCountX
                                     : computeCoarseGridCount(m_volumeWidth, params.stepX);
        doseParams.gridCountY = params.gridCountY > 0
                                     ? params.gridCountY
                                     : computeCoarseGridCount(m_volumeHeight, params.stepY);
        doseParams.gridCountZ = params.gridCountZ > 0
                                     ? params.gridCountZ
                                     : computeCoarseGridCount(m_volumeDepth, params.stepZ);

        doseParams.beamSourceX = static_cast<float>(params.beam.sourcePosition.x());
        doseParams.beamSourceY = static_cast<float>(params.beam.sourcePosition.y());
        doseParams.beamSourceZ = static_cast<float>(params.beam.sourcePosition.z());
        doseParams.beamTargetX = static_cast<float>(params.beam.targetPosition.x());
        doseParams.beamTargetY = static_cast<float>(params.beam.targetPosition.y());
        doseParams.beamTargetZ = static_cast<float>(params.beam.targetPosition.z());

        doseParams.beamBasisX[0] = static_cast<float>(params.beam.beamX.x());
        doseParams.beamBasisX[1] = static_cast<float>(params.beam.beamX.y());
        doseParams.beamBasisX[2] = static_cast<float>(params.beam.beamX.z());
        doseParams.beamBasisY[0] = static_cast<float>(params.beam.beamY.x());
        doseParams.beamBasisY[1] = static_cast<float>(params.beam.beamY.y());
        doseParams.beamBasisY[2] = static_cast<float>(params.beam.beamY.z());
        doseParams.beamBasisZ[0] = static_cast<float>(params.beam.beamZ.x());
        doseParams.beamBasisZ[1] = static_cast<float>(params.beam.beamZ.y());
        doseParams.beamBasisZ[2] = static_cast<float>(params.beam.beamZ.z());

        doseParams.collimatorSize = static_cast<float>(params.beam.collimatorSize);
        // CPU 実装と一致させるため、参照線量による追加スケーリングは適用しない
        doseParams.referenceDose = 1.0f;
        const bool depthValid = !params.depthCumulative.empty() && std::isfinite(params.depthEntryDistance)
                                && params.depthStepSize > 0.0;
        doseParams.depthEntryDistance = depthValid ? static_cast<float>(params.depthEntryDistance) : 0.0f;
        doseParams.depthStepSize = depthValid ? static_cast<float>(params.depthStepSize) : 1.0f;
        doseParams.depthSampleCount = depthValid ? static_cast<int>(params.depthCumulative.size()) : 0;
        doseParams.depthValid = depthValid ? 1 : 0;

        doseParams.ofDepthCount = m_ofDepthCount;
        doseParams.ofCollimatorCount = m_ofCollimatorCount;
        doseParams.tmrDepthCount = m_tmrDepthCount;
        doseParams.tmrFieldSizeCount = m_tmrFieldSizeCount;
        doseParams.ocrDepthCount = m_ocrDepthCount;
        doseParams.ocrRadiusCount = m_ocrRadiusCount;
        doseParams.ocrCollimatorCount = m_ocrCollimatorCount;
        doseParams.accumulate = 0;  // Replace mode - accumulation handled by CPU

        // Create parameters buffer
        id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&doseParams
                                                         length:sizeof(DoseParams)
                                                        options:MTLResourceStorageModeShared];
        if (!paramsBuffer) {
            setError("Failed to create parameters buffer");
            return false;
        }

        // Clear dose/mask buffers to remove stale data from previous runs
        if (m_doseBuffer) {
            id<MTLBuffer> doseBuf = toBuffer(m_doseBuffer);
            if (doseBuf) {
                size_t doseSize = static_cast<size_t>(m_volumeWidth)
                                   * static_cast<size_t>(m_volumeHeight)
                                   * static_cast<size_t>(m_volumeDepth)
                                   * sizeof(float);
                memset([doseBuf contents], 0, doseSize);
            }
        }
        if (m_computedMaskBuffer) {
            id<MTLBuffer> maskBuf = toBuffer(m_computedMaskBuffer);
            if (maskBuf) {
                size_t maskSize = static_cast<size_t>(m_volumeWidth)
                                   * static_cast<size_t>(m_volumeHeight)
                                   * static_cast<size_t>(m_volumeDepth)
                                   * sizeof(uint8_t);
                memset([maskBuf contents], 0, maskSize);
            }
        }

        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        if (!commandBuffer) {
            setError("Failed to create command buffer");
            return false;
        }

        // Create compute encoder
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            setError("Failed to create compute encoder");
            return false;
        }

        [encoder setComputePipelineState:pipeline];

        // Validate that all required buffers are available
        if (!m_ofTableBuffer || !m_ofDepthsBuffer || !m_ofCollimatorsBuffer ||
            !m_tmrTableBuffer || !m_tmrDepthsBuffer || !m_tmrFieldSizesBuffer ||
            !m_ocrTableBuffer || !m_ocrDepthsBuffer || !m_ocrRadiiBuffer || !m_ocrCollimatorsBuffer) {
            setError("Beam data not uploaded. Call uploadBeamData() first.");
            qCritical() << "Metal: ✗ Beam data buffers not initialized";
            return false;
        }

        // Set buffers according to Metal kernel signature
        [encoder setBuffer:toBuffer(m_ctBuffer) offset:0 atIndex:0];
        [encoder setBuffer:toBuffer(m_doseBuffer) offset:0 atIndex:1];
        [encoder setBuffer:toBuffer(m_computedMaskBuffer) offset:0 atIndex:2];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:3];

        // Set beam data buffers
        id<MTLBuffer> depthBuffer = nil;
        if (depthValid) {
            depthBuffer = [device newBufferWithBytes:params.depthCumulative.data()
                                               length:params.depthCumulative.size() * sizeof(float)
                                              options:MTLResourceStorageModeShared];
            if (!depthBuffer) {
                setError("Failed to create depth profile buffer");
                return false;
            }
        }

        [encoder setBuffer:toBuffer(m_ofTableBuffer) offset:0 atIndex:4];
        [encoder setBuffer:toBuffer(m_ofDepthsBuffer) offset:0 atIndex:5];
        [encoder setBuffer:toBuffer(m_ofCollimatorsBuffer) offset:0 atIndex:6];
        [encoder setBuffer:toBuffer(m_tmrTableBuffer) offset:0 atIndex:7];
        [encoder setBuffer:toBuffer(m_tmrDepthsBuffer) offset:0 atIndex:8];
        [encoder setBuffer:toBuffer(m_tmrFieldSizesBuffer) offset:0 atIndex:9];
        [encoder setBuffer:toBuffer(m_ocrTableBuffer) offset:0 atIndex:10];
        [encoder setBuffer:toBuffer(m_ocrDepthsBuffer) offset:0 atIndex:11];
        [encoder setBuffer:toBuffer(m_ocrRadiiBuffer) offset:0 atIndex:12];
        [encoder setBuffer:toBuffer(m_ocrCollimatorsBuffer) offset:0 atIndex:13];
        [encoder setBuffer:depthBuffer offset:0 atIndex:14];

        // Calculate thread groups
        NSUInteger gridCountX = static_cast<NSUInteger>(qMax(1, doseParams.gridCountX));
        NSUInteger gridCountY = static_cast<NSUInteger>(qMax(1, doseParams.gridCountY));
        NSUInteger gridCountZ = static_cast<NSUInteger>(qMax(1, doseParams.gridCountZ));

        MTLSize threadsPerGrid = MTLSizeMake(gridCountX, gridCountY, gridCountZ);

        NSUInteger threadExecutionWidth = pipeline.threadExecutionWidth;
        MTLSize threadsPerThreadgroup = MTLSizeMake(
            std::min<NSUInteger>(threadExecutionWidth, threadsPerGrid.width),
            1,
            1
        );

        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];

        // Commit and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Check for errors
        if (commandBuffer.status == MTLCommandBufferStatusError) {
            NSError* error = commandBuffer.error;
            QString errorMsg = QString::fromNSString([error localizedDescription]);
            setError(QString("GPU computation failed: %1").arg(errorMsg));
            return false;
        }

        // Copy result back
        id<MTLBuffer> doseBuf = toBuffer(m_doseBuffer);
        size_t doseSize = m_volumeWidth * m_volumeHeight * m_volumeDepth * sizeof(float);
        memcpy(doseVolume.data, [doseBuf contents], doseSize);

        qDebug() << "Metal: ✓ Dose calculation completed";
        return true;
    }
}

bool MetalDoseBackend::interpolateVolume(cv::Mat& doseVolume,
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

    qDebug() << "Metal: Interpolating volume with step" << step;

    @autoreleasepool {
        id<MTLDevice> device = toDevice(m_device);
        id<MTLCommandQueue> queue = toCommandQueue(m_commandQueue);
        id<MTLComputePipelineState> pipeline = toPipeline(m_interpolationKernel);

        // Upload current dose volume to GPU
        size_t doseSize = m_volumeWidth * m_volumeHeight * m_volumeDepth * sizeof(float);
        id<MTLBuffer> doseBuffer = toBuffer(m_doseBuffer);

        memcpy([doseBuffer contents], doseVolume.data, doseSize);

        // Upload computed mask
        id<MTLBuffer> maskBuffer = toBuffer(m_computedMaskBuffer);
        size_t maskSize = m_volumeWidth * m_volumeHeight * m_volumeDepth;
        memcpy([maskBuffer contents], computedMask.data(), maskSize);

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:doseBuffer offset:0 atIndex:0];
        [encoder setBuffer:maskBuffer offset:0 atIndex:1];
        [encoder setBytes:&m_volumeWidth length:sizeof(int) atIndex:2];
        [encoder setBytes:&m_volumeHeight length:sizeof(int) atIndex:3];
        [encoder setBytes:&m_volumeDepth length:sizeof(int) atIndex:4];
        [encoder setBytes:&step length:sizeof(int) atIndex:5];

        // Calculate thread groups
        MTLSize threadsPerGrid = MTLSizeMake(m_volumeWidth, m_volumeHeight, m_volumeDepth);
        NSUInteger w = pipeline.threadExecutionWidth;
        NSUInteger h = pipeline.maxTotalThreadsPerThreadgroup / w;
        MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);

        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Check for errors
        if (commandBuffer.status == MTLCommandBufferStatusError) {
            NSError* error = commandBuffer.error;
            QString errorMsg = QString::fromNSString([error localizedDescription]);
            setError(QString("Metal interpolation failed: %1").arg(errorMsg));
            return false;
        }

        // Download result
        memcpy(doseVolume.data, [doseBuffer contents], doseSize);

        qDebug() << "Metal: Interpolation completed";
        return true;
    }
}

bool MetalDoseBackend::recalculateDoseWithThreshold(const GPUComputeParams& params,
                                                    cv::Mat& doseVolume,
                                                    float threshold,
                                                    int skipStep)
{
    if (!isReady()) {
        setError("Backend not initialized");
        return false;
    }

    qDebug() << "Metal: Recalculating dose with threshold" << threshold << "skipStep" << skipStep;

    @autoreleasepool {
        id<MTLDevice> device = toDevice(m_device);
        id<MTLCommandQueue> queue = toCommandQueue(m_commandQueue);
        id<MTLComputePipelineState> pipeline = toPipeline(m_thresholdRecalcKernel);

        // Upload current dose volume to GPU
        size_t doseSize = m_volumeWidth * m_volumeHeight * m_volumeDepth * sizeof(float);
        id<MTLBuffer> doseBuffer = toBuffer(m_doseBuffer);
        memcpy([doseBuffer contents], doseVolume.data, doseSize);

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        // Set buffers
        [encoder setBuffer:toBuffer(m_ctBuffer) offset:0 atIndex:0];
        [encoder setBuffer:doseBuffer offset:0 atIndex:1];
        [encoder setBuffer:toBuffer(m_computedMaskBuffer) offset:0 atIndex:2];

        // Prepare VolumeParams struct
        struct {
            int width, height, depth;
            float spacingX, spacingY, spacingZ;
            float originX, originY, originZ;
            simd_float3 orientationX, orientationY, orientationZ;
            int stepX, stepY, stepZ;
            int gridCountX, gridCountY, gridCountZ;
            // Depth profile parameters
            float depthEntryDistance;
            float depthStepSize;
            int depthSampleCount;
            int depthValid;
        } volumeParams;

        volumeParams.width = params.width;
        volumeParams.height = params.height;
        volumeParams.depth = params.depth;
        volumeParams.spacingX = static_cast<float>(params.spacingX);
        volumeParams.spacingY = static_cast<float>(params.spacingY);
        volumeParams.spacingZ = static_cast<float>(params.spacingZ);
        volumeParams.originX = static_cast<float>(params.originX);
        volumeParams.originY = static_cast<float>(params.originY);
        volumeParams.originZ = static_cast<float>(params.originZ);
        volumeParams.orientationX = simd_make_float3(params.orientationX[0], params.orientationX[1], params.orientationX[2]);
        volumeParams.orientationY = simd_make_float3(params.orientationY[0], params.orientationY[1], params.orientationY[2]);
        volumeParams.orientationZ = simd_make_float3(params.orientationZ[0], params.orientationZ[1], params.orientationZ[2]);
        volumeParams.stepX = params.stepX;
        volumeParams.stepY = params.stepY;
        volumeParams.stepZ = params.stepZ;
        volumeParams.gridCountX = params.gridCountX;
        volumeParams.gridCountY = params.gridCountY;
        volumeParams.gridCountZ = params.gridCountZ;
        // Set depth profile parameters
        volumeParams.depthEntryDistance = static_cast<float>(params.depthEntryDistance);
        volumeParams.depthStepSize = static_cast<float>(params.depthStepSize);
        volumeParams.depthSampleCount = static_cast<int>(params.depthCumulative.size());
        volumeParams.depthValid = (!params.depthCumulative.empty() && params.depthStepSize > 0.0) ? 1 : 0;

        [encoder setBytes:&volumeParams length:sizeof(volumeParams) atIndex:3];

        // Prepare BeamParams struct
        struct {
            simd_float3 source, target;
            simd_float3 basisX, basisY, basisZ;
            float collimatorSize;
        } beamParams;

        beamParams.source = simd_make_float3(params.beam.sourcePosition.x(), params.beam.sourcePosition.y(), params.beam.sourcePosition.z());
        beamParams.target = simd_make_float3(params.beam.targetPosition.x(), params.beam.targetPosition.y(), params.beam.targetPosition.z());
        beamParams.basisX = simd_make_float3(params.beam.beamX.x(), params.beam.beamX.y(), params.beam.beamX.z());
        beamParams.basisY = simd_make_float3(params.beam.beamY.x(), params.beam.beamY.y(), params.beam.beamY.z());
        beamParams.basisZ = simd_make_float3(params.beam.beamZ.x(), params.beam.beamZ.y(), params.beam.beamZ.z());
        beamParams.collimatorSize = static_cast<float>(params.beam.collimatorSize);

        [encoder setBytes:&beamParams length:sizeof(beamParams) atIndex:4];

        // Reference dose
        float referenceDose = 1.0f;
        [encoder setBytes:&referenceDose length:sizeof(float) atIndex:5];

        // Set lookup tables
        [encoder setBuffer:toBuffer(m_ofTableBuffer) offset:0 atIndex:6];
        [encoder setBuffer:toBuffer(m_ofDepthsBuffer) offset:0 atIndex:7];
        [encoder setBuffer:toBuffer(m_ofCollimatorsBuffer) offset:0 atIndex:8];

        [encoder setBuffer:toBuffer(m_tmrTableBuffer) offset:0 atIndex:9];
        [encoder setBuffer:toBuffer(m_tmrDepthsBuffer) offset:0 atIndex:10];
        [encoder setBuffer:toBuffer(m_tmrFieldSizesBuffer) offset:0 atIndex:11];

        [encoder setBuffer:toBuffer(m_ocrTableBuffer) offset:0 atIndex:12];
        [encoder setBuffer:toBuffer(m_ocrDepthsBuffer) offset:0 atIndex:13];
        [encoder setBuffer:toBuffer(m_ocrRadiiBuffer) offset:0 atIndex:14];
        [encoder setBuffer:toBuffer(m_ocrCollimatorsBuffer) offset:0 atIndex:15];

        // Prepare LookupTableCounts struct
        struct {
            int ofDepthCount, ofCollimatorCount;
            int tmrDepthCount, tmrFieldSizeCount;
            int ocrDepthCount, ocrRadiusCount, ocrCollimatorCount;
        } tableCounts;

        tableCounts.ofDepthCount = m_ofDepthCount;
        tableCounts.ofCollimatorCount = m_ofCollimatorCount;
        tableCounts.tmrDepthCount = m_tmrDepthCount;
        tableCounts.tmrFieldSizeCount = m_tmrFieldSizeCount;
        tableCounts.ocrDepthCount = m_ocrDepthCount;
        tableCounts.ocrRadiusCount = m_ocrRadiusCount;
        tableCounts.ocrCollimatorCount = m_ocrCollimatorCount;

        [encoder setBytes:&tableCounts length:sizeof(tableCounts) atIndex:16];

        // Set threshold
        [encoder setBytes:&threshold length:sizeof(float) atIndex:17];

        // Set depth profile buffer
        id<MTLBuffer> depthProfileBuffer = nil;
        if (!params.depthCumulative.empty()) {
            size_t depthProfileSize = params.depthCumulative.size() * sizeof(float);
            depthProfileBuffer = [device newBufferWithBytes:params.depthCumulative.data()
                                                     length:depthProfileSize
                                                    options:MTLResourceStorageModeShared];
            [encoder setBuffer:depthProfileBuffer offset:0 atIndex:18];
        } else {
            // Create a dummy buffer with a single zero value
            float dummy = 0.0f;
            depthProfileBuffer = [device newBufferWithBytes:&dummy
                                                     length:sizeof(float)
                                                    options:MTLResourceStorageModeShared];
            [encoder setBuffer:depthProfileBuffer offset:0 atIndex:18];
        }

        // Set skipStep parameter
        [encoder setBytes:&skipStep length:sizeof(int) atIndex:19];

        // Calculate thread groups
        MTLSize threadsPerGrid = MTLSizeMake(volumeParams.gridCountX, volumeParams.gridCountY, volumeParams.gridCountZ);
        NSUInteger w = pipeline.threadExecutionWidth;
        NSUInteger h = pipeline.maxTotalThreadsPerThreadgroup / w;
        MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);

        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Check for errors
        if (commandBuffer.status == MTLCommandBufferStatusError) {
            NSError* error = commandBuffer.error;
            QString errorMsg = QString::fromNSString([error localizedDescription]);
            setError(QString("Metal threshold recalculation failed: %1").arg(errorMsg));
            return false;
        }

        // Download result
        memcpy(doseVolume.data, [doseBuffer contents], doseSize);

        qDebug() << "Metal: Threshold recalculation completed";
        return true;
    }
}

void MetalDoseBackend::releaseBuffers()
{
    if (m_ctBuffer) {
        CFRelease(m_ctBuffer);
        m_ctBuffer = nullptr;
    }
    if (m_doseBuffer) {
        CFRelease(m_doseBuffer);
        m_doseBuffer = nullptr;
    }
    if (m_computedMaskBuffer) {
        CFRelease(m_computedMaskBuffer);
        m_computedMaskBuffer = nullptr;
    }

    // Release beam data buffers
    if (m_ofTableBuffer) CFRelease(m_ofTableBuffer);
    if (m_ofDepthsBuffer) CFRelease(m_ofDepthsBuffer);
    if (m_ofCollimatorsBuffer) CFRelease(m_ofCollimatorsBuffer);
    if (m_tmrTableBuffer) CFRelease(m_tmrTableBuffer);
    if (m_tmrDepthsBuffer) CFRelease(m_tmrDepthsBuffer);
    if (m_tmrFieldSizesBuffer) CFRelease(m_tmrFieldSizesBuffer);
    if (m_ocrTableBuffer) CFRelease(m_ocrTableBuffer);
    if (m_ocrDepthsBuffer) CFRelease(m_ocrDepthsBuffer);
    if (m_ocrRadiiBuffer) CFRelease(m_ocrRadiiBuffer);
    if (m_ocrCollimatorsBuffer) CFRelease(m_ocrCollimatorsBuffer);

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

void MetalDoseBackend::releaseMetalResources()
{
    releaseBuffers();

    if (m_doseKernel) {
        CFRelease(m_doseKernel);
        m_doseKernel = nullptr;
    }
    if (m_interpolationKernel) {
        CFRelease(m_interpolationKernel);
        m_interpolationKernel = nullptr;
    }
    if (m_thresholdRecalcKernel) {
        CFRelease(m_thresholdRecalcKernel);
        m_thresholdRecalcKernel = nullptr;
    }
    if (m_library) {
        CFRelease(m_library);
        m_library = nullptr;
    }
    if (m_commandQueue) {
        CFRelease(m_commandQueue);
        m_commandQueue = nullptr;
    }
    if (m_device) {
        CFRelease(m_device);
        m_device = nullptr;
    }
}

void MetalDoseBackend::cleanup()
{
    qDebug() << "Metal: Cleaning up resources";
    releaseMetalResources();
    m_initialized = false;
}

QString MetalDoseBackend::getLastError() const
{
    return m_lastError;
}

void MetalDoseBackend::setError(const QString& error)
{
    m_lastError = error;
    qWarning() << "Metal Error:" << error;
}

} // namespace CyberKnife

#endif // USE_METAL_BACKEND
