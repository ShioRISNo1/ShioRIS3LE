#include "ai/onnx_segmenter.h"
#include <QDebug>
#include <QStringList>
#include <cstring>
#include <string>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <functional>
#include <thread>
#include <chrono>
#include <limits>
#include <numeric>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace {
#ifndef NDEBUG
// Diagnostic function for debugging volume depth statistics
void logVolumeDepthStats(const cv::Mat &volume, const char *label) {
    if (volume.dims != 3) {
        qWarning() << label << "volume is not 3D";
        return;
    }
    int depth = volume.size[0];
    int height = volume.size[1];
    int width = volume.size[2];
    std::vector<double> sliceMeans(depth), sliceMins(depth), sliceMaxs(depth);
    for (int z = 0; z < depth; ++z) {
        cv::Mat slice(height, width, volume.type(), const_cast<uchar *>(volume.ptr(z)));
        cv::Mat sliceF;
        slice.convertTo(sliceF, CV_32F);
        double mn, mx;
        cv::Scalar mean = cv::mean(sliceF);
        cv::minMaxLoc(sliceF, &mn, &mx);
        sliceMeans[z] = mean[0];
        sliceMins[z] = mn;
        sliceMaxs[z] = mx;
    }
    double meanRange = *std::max_element(sliceMeans.begin(), sliceMeans.end()) -
                       *std::min_element(sliceMeans.begin(), sliceMeans.end());
    double minRange = *std::max_element(sliceMins.begin(), sliceMins.end()) -
                      *std::min_element(sliceMins.begin(), sliceMins.end());
    double maxRange = *std::max_element(sliceMaxs.begin(), sliceMaxs.end()) -
                      *std::min_element(sliceMaxs.begin(), sliceMaxs.end());
    double maxMeanDiff = 0.0, maxMinDiff = 0.0, maxMaxDiff = 0.0;
    for (int z = 1; z < depth; ++z) {
        maxMeanDiff = std::max(maxMeanDiff, std::abs(sliceMeans[z] - sliceMeans[z - 1]));
        maxMinDiff = std::max(maxMinDiff, std::abs(sliceMins[z] - sliceMins[z - 1]));
        maxMaxDiff = std::max(maxMaxDiff, std::abs(sliceMaxs[z] - sliceMaxs[z - 1]));
    }
    qDebug() << label << "depth stats - mean range:" << meanRange
             << "min range:" << minRange
             << "max range:" << maxRange;
    qDebug() << label << "max consecutive diffs - mean:" << maxMeanDiff
             << "min:" << maxMinDiff
             << "max:" << maxMaxDiff;
    if (maxMeanDiff < 1e-5 && maxMinDiff < 1e-5 && maxMaxDiff < 1e-5) {
        qWarning() << label << "volume slices show almost no variation";
    }
}
#endif
}

#ifdef USE_ONNXRUNTIME
OnnxSegmenter::OnnxSegmenter()
#ifndef NDEBUG
    : m_env(ORT_LOGGING_LEVEL_VERBOSE, "shioris3")  // Verbose logging in debug builds
#else
    : m_env(ORT_LOGGING_LEVEL_WARNING, "shioris3")  // Warning level in release builds
#endif
{
    // セッションオプションの基本設定
    m_sessionOptions.SetIntraOpNumThreads(4);
    m_sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // プラットフォーム別の実行プロバイダー設定
    #ifdef __APPLE__
    try {
        // CoreML EPを試行（macOS用）
        std::unordered_map<std::string, std::string> coreml_options;
        coreml_options["ComputeUnits"] = "0"; // All compute units
        m_sessionOptions.AppendExecutionProvider("CoreMLExecutionProvider", coreml_options);
        qDebug() << "CoreML Execution Provider enabled";
    } catch (const std::exception& e) {
        qDebug() << "CoreML EP not available, using optimized CPU:" << e.what();
        // CPU専用最適化
        m_sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        m_sessionOptions.EnableCpuMemArena();
        m_sessionOptions.EnableMemPattern();
    }
    
    #elif defined(ONNXRUNTIME_USE_CUDA)
    // Linux CUDA設定
    qDebug() << "=== CUDA EXECUTION PROVIDER INITIALIZATION ===";
    qDebug() << "Build configuration: ONNXRUNTIME_USE_CUDA is defined";

    try {
        // 利用可能なプロバイダーを確認
        auto available_providers = Ort::GetAvailableProviders();
        qDebug() << "Available execution providers:";
        bool cuda_available = false;
        bool tensorrt_available = false;

        for (const auto& provider : available_providers) {
            qDebug() << "  -" << QString::fromStdString(provider);
            if (provider == "CUDAExecutionProvider") {
                cuda_available = true;
            }
            if (provider == "TensorrtExecutionProvider") {
                tensorrt_available = true;
            }
        }

        if (cuda_available) {
            qDebug() << "✓ CUDAExecutionProvider is available";

            // CUDA Execution Providerの設定
            try {
                // 環境変数でGPUデバイスを選択可能に (デフォルト: 0)
                int gpu_device = 0;
                const char* device_env = std::getenv("SHIORIS_GPU_DEVICE");
                if (device_env) {
                    gpu_device = std::atoi(device_env);
                    qDebug() << "GPU device from environment: SHIORIS_GPU_DEVICE=" << gpu_device;
                } else {
                    qDebug() << "Using default GPU device: 0 (set SHIORIS_GPU_DEVICE to change)";
                }

                // デバイスが有効かチェック（簡易版）
                if (gpu_device < 0) {
                    qWarning() << "Invalid GPU device" << gpu_device << ", using 0";
                    gpu_device = 0;
                }

                // CUDA EPを有効化
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(m_sessionOptions, gpu_device));
                m_cudaEnabled = true;
                qDebug() << "✓ CUDA Execution Provider successfully enabled (GPU device" << gpu_device << ")";

                // GPU最適化のためのセッション設定
                m_sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

                // VRAM最適化: メモリアリーナの設定を改善
                // CPUメモリアリーナは有効化してCPU-GPU転送を効率化
                m_sessionOptions.EnableCpuMemArena();
                m_sessionOptions.EnableMemPattern();

                // GPU メモリ制限を環境変数で設定可能に（デフォルト: 8GB）
                size_t gpu_mem_limit_bytes = 8ULL * 1024 * 1024 * 1024; // 8GB
                const char* mem_limit_env = std::getenv("SHIORIS_GPU_MEM_LIMIT_GB");
                if (mem_limit_env) {
                    int limit_gb = std::atoi(mem_limit_env);
                    if (limit_gb > 0 && limit_gb <= 24) {
                        gpu_mem_limit_bytes = static_cast<size_t>(limit_gb) * 1024 * 1024 * 1024;
                        qDebug() << "GPU memory limit from environment:" << limit_gb << "GB";
                    }
                } else {
                    qDebug() << "Using default GPU memory limit: 8GB (set SHIORIS_GPU_MEM_LIMIT_GB to change)";
                }

                std::string mem_limit_str = std::to_string(gpu_mem_limit_bytes);
                m_sessionOptions.AddConfigEntry("gpu_mem_limit", mem_limit_str.c_str());
                m_sessionOptions.AddConfigEntry("arena_extend_strategy", "kSameAsRequested");

                // 追加のメモリ最適化オプション
                m_sessionOptions.AddConfigEntry("gpu_external_alloc", "0");
                m_sessionOptions.AddConfigEntry("gpu_external_free", "0");

                qDebug() << "✓ GPU-optimized session options configured with VRAM management";
                qDebug() << "  GPU memory limit:" << (gpu_mem_limit_bytes / (1024*1024*1024)) << "GB";

            } catch (const Ort::Exception& ort_ex) {
                qCritical() << "❌ Failed to append CUDA EP:" << ort_ex.what();
                throw;
            }

        } else {
            qWarning() << "❌ CUDAExecutionProvider not in available providers list";
            qWarning() << "This may indicate:";
            qWarning() << "  1. CUDA provider library (libonnxruntime_providers_cuda.so) not found";
            qWarning() << "  2. CUDA runtime libraries not properly installed";
            qWarning() << "  3. GPU drivers not properly configured";
            m_cudaEnabled = false;
            throw std::runtime_error("CUDAExecutionProvider not available in runtime");
        }

    } catch (const std::exception& e) {
        qCritical() << "❌ CUDA EP initialization failed:" << e.what();
        qWarning() << "Falling back to CPU execution provider";
        m_cudaEnabled = false;

        // CPUにフォールバック
        m_sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        m_sessionOptions.EnableCpuMemArena();
        m_sessionOptions.EnableMemPattern();
        qDebug() << "✓ CPU execution provider configured";
    }

    qDebug() << "=== CUDA EP INITIALIZATION COMPLETE ===";
    qDebug() << "CUDA Enabled:" << (m_cudaEnabled ? "YES" : "NO");

    #else
    // その他のプラットフォームはCPU最適化
    qDebug() << "Using optimized CPU execution";
    m_cudaEnabled = false;
    m_sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    m_sessionOptions.EnableCpuMemArena();
    m_sessionOptions.EnableMemPattern();
    #endif

    // 利用可能なプロバイダーをログ出力
    qDebug() << "=== EXECUTION PROVIDERS CONFIGURED ===";
}

bool OnnxSegmenter::loadModel(const std::string &modelPath) {
#ifdef USE_ONNXRUNTIME
    try {
        qDebug() << "Loading ONNX model:" << QString::fromStdString(modelPath);

        if (!m_cudaEnabled) {
            unsigned int hwThreads = std::max(1u, std::thread::hardware_concurrency());
            m_sessionOptions.SetIntraOpNumThreads(static_cast<int>(hwThreads));
            m_sessionOptions.SetInterOpNumThreads(1);
            m_sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
            m_sessionOptions.EnableCpuMemArena();
            m_sessionOptions.EnableMemPattern();
            m_sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            qDebug() << "ONNX Runtime configured for CPU execution";
        } else {
            qDebug() << "ONNX Runtime will use CUDA execution provider";
        }

        // セッション作成
#ifdef _WIN32
        std::wstring wModelPath(modelPath.begin(), modelPath.end());
        m_session = std::make_unique<Ort::Session>(m_env, wModelPath.c_str(), m_sessionOptions);
#else
        m_session = std::make_unique<Ort::Session>(m_env, modelPath.c_str(), m_sessionOptions);
#endif
        
        qDebug() << "ONNX session created successfully";
        
        // 入力・出力情報の取得（既存コードをそのまま使用）
        Ort::AllocatorWithDefaultOptions allocator;
        
        // 入力情報を取得
        size_t numInputNodes = m_session->GetInputCount();
        m_inputNames.resize(numInputNodes);
        
        for (size_t i = 0; i < numInputNodes; ++i) {
            Ort::AllocatedStringPtr name = m_session->GetInputNameAllocated(i, allocator);
            m_inputNames[i] = name.get();
            qDebug() << "Input name:" << QString::fromStdString(m_inputNames[i]);
            
            Ort::TypeInfo typeInfo = m_session->GetInputTypeInfo(i);
            Ort::ConstTensorTypeAndShapeInfo tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            auto shape = tensorInfo.GetShape();
            m_inputDims = shape;
            
            std::ostringstream oss;
            for (size_t j = 0; j < shape.size(); ++j) {
                if (j) oss << "x";
                oss << (shape[j] == -1 ? "dynamic" : std::to_string(shape[j]));
            }
            qDebug() << "Input dims:" << QString::fromStdString(oss.str());
        }
        
        // 出力情報を取得
        size_t numOutputNodes = m_session->GetOutputCount();
        m_outputNames.resize(numOutputNodes);
        
        for (size_t i = 0; i < numOutputNodes; ++i) {
            Ort::AllocatedStringPtr name = m_session->GetOutputNameAllocated(i, allocator);
            m_outputNames[i] = name.get();
            qDebug() << "Output name:" << QString::fromStdString(m_outputNames[i]);
        }
        
        qDebug() << "ONNX model loaded successfully";
        return true;

    } catch (const Ort::Exception &e) {
        qCritical() << "ONNX model load failed:" << e.what();
        m_session.reset();
        return false;
    } catch (const std::exception &e) {
        qCritical() << "Model load failed:" << e.what();
        m_session.reset();
        return false;
    }
#else
    return false;
#endif
}

bool OnnxSegmenter::isLoaded() const {
    return m_session != nullptr;
}


cv::Mat OnnxSegmenter::predict(const cv::Mat &slice) {
    if (!m_session) {
        qWarning() << "No model loaded";
        return cv::Mat();
    }

    try {
        qDebug() << "=== 2D SLICE SEGMENTATION ===";
        qDebug() << "=== ENHANCED DEBUG SEGMENTATION ===";
        qDebug() << "Input slice:" << slice.cols << "x" << slice.rows << "type:" << slice.type();
        
        // Step 1: 入力データの詳細診断
        cv::Scalar mean, stddev;
        cv::meanStdDev(slice, mean, stddev);
        double minVal, maxVal;
        cv::minMaxLoc(slice, &minVal, &maxVal);
        qDebug() << "Input stats - Min:" << minVal << "Max:" << maxVal << "Mean:" << mean[0] << "StdDev:" << stddev[0];
        
        // ★★★ 重要：モデルの入力次元情報を詳細に確認 ★★★
        qDebug() << "=== MODEL INPUT SPECIFICATIONS ===";
        qDebug() << "Model expects" << m_inputDims.size() << "D input:";
        for (size_t i = 0; i < m_inputDims.size(); ++i) {
            qDebug() << "  dim[" << i << "]:" << m_inputDims[i];
        }
        
        // ★★★ 前処理の実行と確認 ★★★
        cv::Mat preprocessed = preprocessCTSliceForModel(slice);
        if (preprocessed.empty()) {
            qWarning() << "CT preprocessing failed";
            return cv::Mat();
        }
        
        cv::meanStdDev(preprocessed, mean, stddev);
        cv::minMaxLoc(preprocessed, &minVal, &maxVal);
        qDebug() << "Preprocessed stats - Min:" << minVal << "Max:" << maxVal << "Mean:" << mean[0];
        
        int nonZeroCount = cv::countNonZero(preprocessed > 0.01f);
        qDebug() << "Non-zero pixels:" << nonZeroCount << "out of" << preprocessed.total();
        
        if (nonZeroCount == 0) {
            qWarning() << "No significant pixel values found after preprocessing";
            return cv::Mat();
        }
        
        // ★★★ 重要：実際のモデル入力次元に基づいて動的に設定 ★★★
        int FIXED_HEIGHT, FIXED_WIDTH, FIXED_DEPTH;
        
        if (m_inputDims.size() == 5) {
            // 5D入力: [batch, channel, depth, height, width]
            FIXED_DEPTH = static_cast<int>(m_inputDims[2]);
            FIXED_HEIGHT = static_cast<int>(m_inputDims[3]);
            FIXED_WIDTH = static_cast<int>(m_inputDims[4]);
        } else if (m_inputDims.size() == 4) {
            // 4D入力: [batch, channel, height, width]  
            FIXED_DEPTH = 1;
            FIXED_HEIGHT = static_cast<int>(m_inputDims[2]);
            FIXED_WIDTH = static_cast<int>(m_inputDims[3]);
        } else {
            qWarning() << "Unsupported input dimension count:" << m_inputDims.size();
            return cv::Mat();
        }
        
        qDebug() << "Target dimensions - D:" << FIXED_DEPTH << "H:" << FIXED_HEIGHT << "W:" << FIXED_WIDTH;
        
        cv::Mat resizedForModel;
        if (preprocessed.rows != FIXED_HEIGHT || preprocessed.cols != FIXED_WIDTH) {
            cv::resize(preprocessed, resizedForModel, cv::Size(FIXED_WIDTH, FIXED_HEIGHT), 0, 0, cv::INTER_LINEAR);
            qDebug() << "Resized to model expected size:" << FIXED_WIDTH << "x" << FIXED_HEIGHT;
        } else {
            resizedForModel = preprocessed;
        }
        
        // ★★★ テンソル形状を動的に設定 ★★★
        std::vector<int64_t> inputShape;
        if (m_inputDims.size() == 5) {
            inputShape = {1, 1, FIXED_DEPTH, FIXED_HEIGHT, FIXED_WIDTH};
        } else {
            inputShape = {1, 1, FIXED_HEIGHT, FIXED_WIDTH};
        }
        
        qDebug() << "Actual input shape to be used:";
        for (size_t i = 0; i < inputShape.size(); ++i) {
            qDebug() << "  [" << i << "]:" << inputShape[i];
        }
        
        // ★★★ 入力データの準備 ★★★
        size_t planeElements = FIXED_HEIGHT * FIXED_WIDTH;
        size_t totalElements = (m_inputDims.size() == 5) ? (planeElements * FIXED_DEPTH) : planeElements;
        std::vector<float> inputData(totalElements);

        if (resizedForModel.type() != CV_32FC1) {
            qWarning() << "Unexpected preprocessed data type:" << resizedForModel.type();
            return cv::Mat();
        }
        
        // データのコピー
        cv::Mat continuousData = resizedForModel.isContinuous() ? resizedForModel : resizedForModel.clone();
        const float* srcData = continuousData.ptr<float>(0);
        
        if (m_inputDims.size() == 5) {
            // 5Dの場合：同じスライスを深度方向に複製
            for (int d = 0; d < FIXED_DEPTH; ++d) {
                std::memcpy(inputData.data() + d * planeElements,
                           srcData, planeElements * sizeof(float));
            }
            qDebug() << "Created 5D tensor with replicated slices";
        } else {
            // 4Dの場合：そのままコピー
            std::memcpy(inputData.data(), srcData, planeElements * sizeof(float));
            qDebug() << "Created 4D tensor";
        }
        
        // ★★★ 入力データの統計確認 ★★★
        auto minMaxIter = std::minmax_element(inputData.begin(), inputData.end());
        double sum = std::accumulate(inputData.begin(), inputData.end(), 0.0);
        double tensorMean = sum / inputData.size();
        qDebug() << "Tensor stats - Min:" << *minMaxIter.first << "Max:" << *minMaxIter.second << "Mean:" << tensorMean;
        
        // ★★★ ONNX推論の実行 ★★★
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputData.data(), totalElements,
            inputShape.data(), inputShape.size());

        std::vector<const char *> inputNames;
        std::vector<const char *> outputNames;
        for (const auto &s : m_inputNames) inputNames.push_back(s.c_str());
        for (const auto &s : m_outputNames) outputNames.push_back(s.c_str());

        qDebug() << "Starting inference...";
        qDebug() << "Input names count:" << inputNames.size() << "Output names count:" << outputNames.size();
        
        auto start = std::chrono::high_resolution_clock::now();
        
        auto outputTensors = m_session->Run(Ort::RunOptions{nullptr}, 
                                           inputNames.data(), &inputTensor, 1, 
                                           outputNames.data(), m_outputNames.size());
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        qDebug() << "Inference completed in" << duration.count() << "ms";
        
        if (outputTensors.empty()) {
            qWarning() << "No output tensors received";
            return cv::Mat();
        }
        
        // ★★★ 出力テンソルの詳細分析 ★★★
        qDebug() << "=== OUTPUT TENSOR ANALYSIS ===";
        auto &outputTensor = outputTensors.front();
        auto outputShape = outputTensor.GetTensorTypeAndShapeInfo().GetShape();
        const float* outputData = outputTensor.GetTensorData<float>();
        
        qDebug() << "Output shape:";
        for (size_t i = 0; i < outputShape.size(); ++i) {
            qDebug() << "  [" << i << "]:" << outputShape[i];
        }
        
        // 出力データの統計
        size_t outputTotalElements = 1;
        for (auto dim : outputShape) {
            outputTotalElements *= static_cast<size_t>(dim);
        }
        
        qDebug() << "Total output elements:" << outputTotalElements;
        
        if (outputTotalElements > 0) {
            // 出力データの統計計算
            float outputMin = outputData[0], outputMax = outputData[0];
            double outputSum = 0.0;
            int significantValues = 0;
            
            for (size_t i = 0; i < std::min(outputTotalElements, size_t(10000)); ++i) {
                float val = outputData[i];
                outputMin = std::min(outputMin, val);
                outputMax = std::max(outputMax, val);
                outputSum += val;
                if (std::abs(val) > 0.01f) significantValues++;
            }
            
            double outputMean = outputSum / std::min(outputTotalElements, size_t(10000));
            qDebug() << "Output stats - Min:" << outputMin << "Max:" << outputMax << "Mean:" << outputMean;
            qDebug() << "Significant values count:" << significantValues;
            
            // 先頭の数値を表示
            qDebug() << "First 10 output values:";
            for (int i = 0; i < std::min(10, static_cast<int>(outputTotalElements)); ++i) {
                qDebug() << "  [" << i << "]:" << outputData[i];
            }
        }
        
        // ★★★ 出力の後処理 ★★★
        cv::Size originalImageSize(slice.cols, slice.rows);
        cv::Mat result;

        if (outputShape.size() == 5) {
            int originalSize[3] = {1, originalImageSize.height, originalImageSize.width};
            result = processAbdomenSegmentationOutput(outputTensor, originalSize);
        } else if (outputShape.size() == 4) {
            result = processOutput4D(outputTensor, outputShape, originalImageSize);
        } else {
            qWarning() << "Unsupported output dimension count:" << outputShape.size();
            return cv::Mat();
        }

        if (!result.empty()) {
            // 結果の統計
            double resultMin = 0, resultMax = 0;
            if (result.dims == 2) {
                cv::minMaxLoc(result, &resultMin, &resultMax);
            } else {
                cv::minMaxIdx(result, &resultMin, &resultMax);
            }
            qDebug() << "Final result - Min label:" << resultMin << "Max label:" << resultMax;

            // ラベル分布
            std::vector<int> labelCounts(4, 0);
            if (result.dims == 2) {
                for (int y = 0; y < result.rows; ++y) {
                    for (int x = 0; x < result.cols; ++x) {
                        int label = result.at<uchar>(y, x);
                        if (label >= 0 && label < 4) {
                            labelCounts[label]++;
                        }
                    }
                }
            } else if (result.dims == 3) {
                int depth = result.size[0];
                int height = result.size[1];
                int width = result.size[2];
                for (int z = 0; z < depth; ++z) {
                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            int label = result.at<uchar>(z, y, x);
                            if (label >= 0 && label < 4) {
                                labelCounts[label]++;
                            }
                        }
                    }
                }
            }

            qDebug() << "FINAL RESULT LABEL DISTRIBUTION:";
            auto organLabels = OnnxSegmenter::getOrganLabels();
            int totalUnits;
            if (result.dims == 2) {
                totalUnits = result.rows * result.cols;
            } else {
                totalUnits = result.size[0] * result.size[1] * result.size[2];
            }
            int nonBackgroundUnits = totalUnits - labelCounts[0];

            for (int i = 0; i < 4; ++i) {
                double percentage = (labelCounts[i] * 100.0) / totalUnits;
                qDebug() << QString("  %1: %2 %3 (%4%)")
                            .arg(QString::fromStdString(organLabels[i]))
                            .arg(labelCounts[i])
                            .arg(result.dims == 2 ? "pixels" : "voxels")
                            .arg(percentage, 0, 'f', 2);
            }

            qDebug() << QString("Non-background %1: %2 out of %3 (%4%)")
                        .arg(result.dims == 2 ? "pixels" : "voxels")
                        .arg(nonBackgroundUnits).arg(totalUnits)
                        .arg((nonBackgroundUnits * 100.0) / totalUnits, 0, 'f', 2);

            if (nonBackgroundUnits == 0) {
                qWarning() << "*** NO STRUCTURES DETECTED - ALL BACKGROUND ***";
            } else {
                qDebug() << "*** STRUCTURES DETECTED SUCCESSFULLY ***";
            }
        } else {
            qWarning() << "Result matrix is empty";
        }

        qDebug() << "=== ENHANCED DEBUG COMPLETED ===";
        return result;

    } catch (const Ort::Exception &e) {
        qWarning() << "ONNX inference failed:" << e.what();
        return cv::Mat();
    } catch (const std::exception &e) {
        qWarning() << "Inference failed:" << e.what();
        return cv::Mat();
    }
}

cv::Mat OnnxSegmenter::buildInputVolume(const cv::Mat &volume) {
    qDebug() << "==================================================";
    qDebug() << "=== BUILD INPUT VOLUME - START ===";
    qDebug() << "==================================================";
    
    if (volume.dims != 3) {
        qWarning() << "[buildInputVolume] ERROR: expects 3D volume, got" << volume.dims << "dims";
        return cv::Mat();
    }
    
    qDebug() << "[buildInputVolume] Input volume dimensions:" << volume.size[0] << "x" << volume.size[1] << "x" << volume.size[2];
    qDebug() << "[buildInputVolume] Input volume type:" << volume.type();
    qDebug() << "[buildInputVolume] Input volume total elements:" << volume.total();

#ifndef NDEBUG
    logVolumeDepthStats(volume, "[buildInputVolume] original");
#endif

    qDebug() << "[buildInputVolume] Calling resampleVolumeFor3D...";
    cv::Mat res = resampleVolumeFor3D(volume);

    if (!res.empty()) {
        qDebug() << "[buildInputVolume] Resampling SUCCESS";
        qDebug() << "[buildInputVolume] Result dimensions:" << res.size[0] << "x" << res.size[1] << "x" << res.size[2];
        qDebug() << "[buildInputVolume] Result type:" << res.type();
        qDebug() << "[buildInputVolume] Result total elements:" << res.total();
#ifndef NDEBUG
        logVolumeDepthStats(res, "[buildInputVolume] resampled");
#endif
    } else {
        qCritical() << "[buildInputVolume] FAILED: resampleVolumeFor3D returned empty matrix!";
    }
    
    qDebug() << "==================================================";
    qDebug() << "=== BUILD INPUT VOLUME - END ===";
    qDebug() << "==================================================";
    
    return res;
}


//=============================================================================
// ファイル: src/ai/onnx_segmenter.cpp
// 安全版: preprocessCTSliceForModel 関数（クラッシュ回避）
//=============================================================================




#ifndef NDEBUG
bool OnnxSegmenter::diagnoseInputData(const cv::Mat &slice) {
    qDebug() << "=== INPUT DATA DIAGNOSIS ===";
    
    if (slice.empty()) {
        qWarning() << "Input slice is empty";
        return false;
    }
    
    qDebug() << "Slice dimensions:" << slice.cols << "x" << slice.rows;
    qDebug() << "Slice type:" << slice.type();
    qDebug() << "Expected types: CV_16SC1=" << CV_16SC1 << ", CV_8UC1=" << CV_8UC1;
    
    double minVal, maxVal;
    cv::Scalar mean, stddev;
    cv::minMaxLoc(slice, &minVal, &maxVal);
    cv::meanStdDev(slice, mean, stddev);
    
    qDebug() << "Value statistics:";
    qDebug() << "  Min:" << minVal;
    qDebug() << "  Max:" << maxVal;
    qDebug() << "  Mean:" << mean[0];
    qDebug() << "  StdDev:" << stddev[0];
    
    // CT値の妥当性チェック
    if (slice.type() == CV_16SC1) {
        if (minVal < -2000 || maxVal > 4000) {
            qWarning() << "Unusual CT values detected - possible data corruption";
            return false;
        }
        
        // 典型的な人体CT値の範囲チェック
        bool hasAir = minVal < -800;      // 空気
        bool hasSoftTissue = mean[0] > -200 && mean[0] < 200;  // 軟部組織
        bool hasBone = maxVal > 200;      // 骨
        
        qDebug() << "CT value analysis:";
        qDebug() << "  Contains air:" << hasAir;
        qDebug() << "  Contains soft tissue:" << hasSoftTissue;
        qDebug() << "  Contains bone:" << hasBone;
        
        if (!hasSoftTissue) {
            qWarning() << "No soft tissue detected - may not be suitable for abdominal segmentation";
            return false;
        }
    }
    
    // 画像の情報量チェック
    cv::Mat hist;
    int histSize = 256;
    float range[] = {static_cast<float>(minVal), static_cast<float>(maxVal)};
    const float* histRange = {range};
    cv::calcHist(&slice, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    
    // エントロピー計算（簡易版）
    double entropy = 0.0;
    int totalPixels = slice.rows * slice.cols;
    for (int i = 0; i < histSize; ++i) {
        float count = hist.at<float>(i);
        if (count > 0) {
            double prob = count / totalPixels;
            entropy -= prob * log2(prob);
        }
    }
    
    qDebug() << "Image entropy:" << entropy;
    if (entropy < 3.0) {
        qWarning() << "Low entropy detected - image may lack sufficient detail";
        return false;
    }
    
    qDebug() << "Input data diagnosis: PASSED";
    return true;
}

bool OnnxSegmenter::diagnoseModelOutput(const cv::Mat &result) {
    qDebug() << "=== OUTPUT DATA DIAGNOSIS ===";
    
    if (result.empty()) {
        qWarning() << "Result is empty";
        return false;
    }
    
    qDebug() << "Result dimensions:" << result.cols << "x" << result.rows;
    qDebug() << "Result type:" << result.type();
    
    double minLabel, maxLabel;
    cv::minMaxLoc(result, &minLabel, &maxLabel);
    qDebug() << "Label range:" << minLabel << "to" << maxLabel;
    
    // ラベル分布の詳細分析
    auto organLabels = OnnxSegmenter::getOrganLabels();  // staticメソッドとして呼び出し
    std::vector<int> labelCounts(organLabels.size(), 0);
    
    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {
            int label = result.at<uchar>(y, x);
            if (label >= 0 && label < static_cast<int>(organLabels.size())) {
                labelCounts[label]++;
            }
        }
    }
    
    int totalPixels = result.rows * result.cols;
    int nonBackgroundPixels = totalPixels - labelCounts[0];
    
    qDebug() << "Label distribution:";
    for (size_t i = 0; i < organLabels.size(); ++i) {
        double percentage = (labelCounts[i] * 100.0) / totalPixels;
        qDebug() << QString("  %1: %2 pixels (%3%)")
                    .arg(QString::fromStdString(organLabels[i]))
                    .arg(labelCounts[i])
                    .arg(percentage, 0, 'f', 2);
    }
    
    qDebug() << QString("Non-background ratio: %1%")
                .arg((nonBackgroundPixels * 100.0) / totalPixels, 0, 'f', 2);
    
    if (nonBackgroundPixels == 0) {
        qWarning() << "No structures detected in segmentation result";
        return false;
    }
    
    if (nonBackgroundPixels < totalPixels * 0.01) {
        qWarning() << "Very few structures detected - possible segmentation failure";
        return false;
    }
    
    qDebug() << "Output diagnosis: PASSED";
    return true;
}
#endif  // NDEBUG

cv::Mat OnnxSegmenter::preprocessCTSliceAlternative(const cv::Mat &slice) {
    cv::Mat result;
    
    try {
        qDebug() << "=== ALTERNATIVE CT PREPROCESSING (Pretrained Model) ===";
        
        cv::Mat normalized;
        
        if (slice.type() == CV_16SC1 || slice.type() == CV_16UC1) {
            cv::Mat windowed;
            slice.convertTo(windowed, CV_32FC1);
            
            // **代替手法1**: ImageNetスタイルの正規化
            // 多くの事前訓練済みモデルが期待する正規化
            double MEAN_HU = 0.0;     // CT画像の平均HU値
            double STD_HU = 1000.0;   // CT画像の標準偏差
            
            qDebug() << "Applying ImageNet-style normalization";
            
            windowed = (windowed - MEAN_HU) / STD_HU;
            
            // [-1, 1]に正規化
            double minVal, maxVal;
            cv::minMaxLoc(windowed, &minVal, &maxVal);
            if (maxVal > minVal) {
                windowed = 2.0 * (windowed - minVal) / (maxVal - minVal) - 1.0;
            }
            
            // [0, 1]に変換
            normalized = (windowed + 1.0) / 2.0;
            
        } else if (slice.type() == CV_8UC1) {
            slice.convertTo(normalized, CV_32FC1, 1.0/255.0);
        } else {
            slice.convertTo(normalized, CV_32FC1);
        }
        
        // 256x256に固定リサイズ
        cv::resize(normalized, result, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);
        
        // 統計確認
        double minVal, maxVal;
        cv::Scalar mean;
        cv::minMaxLoc(result, &minVal, &maxVal);
        cv::meanStdDev(result, mean, cv::Scalar());
        qDebug() << "Alternative result - Range:[" << minVal << "," << maxVal << "] Mean:" << mean[0];
        qDebug() << "=== END ALTERNATIVE PREPROCESSING ===";
        
        return result;
        
    } catch (const std::exception &e) {
        qWarning() << "Alternative preprocessing error:" << e.what();
        return cv::Mat();
    }
}

// 元の前処理関数（互換性のため保持）
cv::Mat OnnxSegmenter::preprocessCTSlice(const cv::Mat &slice) {
    cv::Mat result;
    
    try {
        qDebug() << "=== CT PREPROCESSING ===";
        qDebug() << "Input slice type:" << slice.type() << "CV_16SC1=" << CV_16SC1 << "CV_8UC1=" << CV_8UC1;
        qDebug() << "Original size:" << slice.cols << "x" << slice.rows;
        
        // 入力データの初期統計
        double minVal, maxVal;
        cv::Scalar mean, stddev;
        cv::minMaxLoc(slice, &minVal, &maxVal);
        cv::meanStdDev(slice, mean, stddev);
        qDebug() << "Input stats - Min:" << minVal << "Max:" << maxVal << "Mean:" << mean[0];
        
        // CT画像の前処理パラメータ（より適切な範囲に調整）
        cv::Mat normalized;
        
        if (slice.type() == CV_16SC1 || slice.type() == CV_16UC1) {
            qDebug() << "Processing 16-bit CT data";
            
            cv::Mat windowed;
            slice.convertTo(windowed, CV_32FC1);
            
            // 腹部臓器用のより適切なウィンドウ設定
            double WINDOW_WIDTH = 150.0;   
            double WINDOW_LEVEL = 30.0;    
            
            double minWindow = WINDOW_LEVEL - WINDOW_WIDTH / 2.0;  // -45
            double maxWindow = WINDOW_LEVEL + WINDOW_WIDTH / 2.0;  // 105
            
            qDebug() << "Applying liver window: [" << minWindow << "," << maxWindow << "]";
            
            // ウィンドウ適用
            windowed = (windowed - minWindow) / (maxWindow - minWindow);
            
            // 0-1にクランプ
            cv::threshold(windowed, windowed, 0.0, 0.0, cv::THRESH_TOZERO);
            cv::threshold(windowed, windowed, 1.0, 1.0, cv::THRESH_TRUNC);
            
            normalized = windowed;
            
        } else if (slice.type() == CV_8UC1) {
            qDebug() << "Processing 8-bit image data";
            slice.convertTo(normalized, CV_32FC1, 1.0/255.0);
            
        } else if (slice.type() == CV_32FC1) {
            qDebug() << "Processing 32-bit float data";
            normalized = slice.clone();
            
        } else {
            qWarning() << "Unsupported image type:" << slice.type();
            return cv::Mat();
        }
        
        // 正規化後の統計確認
        cv::minMaxLoc(normalized, &minVal, &maxVal);
        cv::meanStdDev(normalized, mean, stddev);
        qDebug() << "After normalization - Range:[" << minVal << "," << maxVal << "] Mean:" << mean[0] << "StdDev:" << stddev[0];
        
        // 値がほとんど0の場合の対処
        if (mean[0] < 0.01) {
            qWarning() << "Very low mean value detected, adjusting window parameters";
            // より広いウィンドウを試行
            if (slice.type() == CV_16SC1 || slice.type() == CV_16UC1) {
                cv::Mat windowed;
                slice.convertTo(windowed, CV_32FC1);
                
                // より広いウィンドウ（全腹部用）
                double WIDE_WINDOW_WIDTH = 400.0;   
                double WIDE_WINDOW_LEVEL = 40.0;    
                
                double minWindow = WIDE_WINDOW_LEVEL - WIDE_WINDOW_WIDTH / 2.0;  // -160
                double maxWindow = WIDE_WINDOW_LEVEL + WIDE_WINDOW_WIDTH / 2.0;  // 240
                
                qDebug() << "Applying wide abdomen window: [" << minWindow << "," << maxWindow << "]";
                
                windowed = (windowed - minWindow) / (maxWindow - minWindow);
                cv::threshold(windowed, windowed, 0.0, 0.0, cv::THRESH_TOZERO);
                cv::threshold(windowed, windowed, 1.0, 1.0, cv::THRESH_TRUNC);
                
                normalized = windowed;
                
                // 再度統計確認
                cv::minMaxLoc(normalized, &minVal, &maxVal);
                cv::meanStdDev(normalized, mean, stddev);
                qDebug() << "After wide window - Range:[" << minVal << "," << maxVal << "] Mean:" << mean[0];
            }
        }
        
        // 適応的サイズ調整（元のサイズを維持または32の倍数に調整）
        cv::Size targetSize;
        
        // 32の倍数に調整（多くのCNNモデルの要件）
        int targetW = ((slice.cols + 31) / 32) * 32;
        int targetH = ((slice.rows + 31) / 32) * 32;
        
        // しかし、極端に大きくならないよう制限
        targetW = std::min(targetW, 512);
        targetH = std::min(targetH, 512);
        targetSize = cv::Size(targetW, targetH);
        
        qDebug() << "Target size:" << targetW << "x" << targetH;
        
        if (normalized.size() != targetSize) {
            qDebug() << "Resizing from" << normalized.cols << "x" << normalized.rows << "to" << targetSize.width << "x" << targetSize.height;
            cv::resize(normalized, result, targetSize, 0, 0, cv::INTER_LINEAR);
        } else {
            result = normalized;
        }
        
        // 最終確認
        cv::minMaxLoc(result, &minVal, &maxVal);
        cv::meanStdDev(result, mean, stddev);
        qDebug() << "Final preprocessed - Range: [" << minVal << "," << maxVal << "] Mean:" << mean[0] << "StdDev:" << stddev[0];
        qDebug() << "Final size:" << result.cols << "x" << result.rows;
        qDebug() << "Final type:" << result.type() << "(CV_32FC1=" << CV_32FC1 << ")";
        qDebug() << "Continuous:" << result.isContinuous();
        qDebug() << "=== END PREPROCESSING ===";
        
        return result;
        
    } catch (const std::exception &e) {
        qWarning() << "CT preprocessing error:" << e.what();
        return cv::Mat();
    }
}

cv::Mat OnnxSegmenter::processAbdomenSegmentationOutput(Ort::Value &outputTensor, const int *originalSize) {
    try {
        auto outputShape = outputTensor.GetTensorTypeAndShapeInfo().GetShape();
        if (outputShape.size() != 5) {
            qWarning() << "Expected 5D output tensor";
            return cv::Mat();
        }

        int channels = static_cast<int>(outputShape[1]);
        int depth    = static_cast<int>(outputShape[2]);
        int height   = static_cast<int>(outputShape[3]);
        int width    = static_cast<int>(outputShape[4]);
        const float* data = outputTensor.GetTensorData<float>();

        int sizes[3] = {depth, height, width};
        cv::Mat mask(3, sizes, CV_8UC1);
        int outClasses = (channels == 105) ? 4 : channels;
        std::vector<int> labelCounts(outClasses, 0);

        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int bestLabel = 0;
                    float bestScore = data[y * width + x + z * height * width];
                    for (int c = 1; c < channels; ++c) {
                        int idx = c * depth * height * width + z * height * width + y * width + x;
                        float val = data[idx];
                        if (val > bestScore) {
                            bestScore = val;
                            bestLabel = c;
                        }
                    }

                    uchar finalLabel;
                    if (channels == 105) {
                        if (bestLabel == 0) {
                            finalLabel = 0; // Background
                        } else if (bestLabel <= 10) {
                            finalLabel = 1; // Abdominal Organs
                        } else if (bestLabel <= 30) {
                            finalLabel = 2; // Other Soft Tissue
                        } else {
                            finalLabel = 3; // Other Structures
                        }
                    } else {
                        finalLabel = static_cast<uchar>(bestLabel);
                    }

                    mask.at<uchar>(z, y, x) = finalLabel;
                    labelCounts[finalLabel]++;
                }
            }
        }

        qDebug() << "Label distribution:";
        auto organLabels = OnnxSegmenter::getOrganLabels();  // staticメソッドとして呼び出し
        int totalVoxels = depth * height * width;
        for (int i = 0; i < outClasses; ++i) {
            double percentage = (labelCounts[i] * 100.0) / totalVoxels;
            QString labelName = (i < static_cast<int>(organLabels.size()))
                                ? QString::fromStdString(organLabels[i])
                                : QString("Label%1").arg(i);
            qDebug() << QString("  %1: %2 voxels (%3%)")
                        .arg(labelName)
                        .arg(labelCounts[i])
                        .arg(percentage, 0, 'f', 2);
        }

        cv::Mat result = resample3DToOriginalSize(mask, originalSize);
        return result;

    } catch (const std::exception &e) {
        qWarning() << "Abdomen output processing error:" << e.what();
        return cv::Mat();
    }
}


// 4D出力処理のヘルパー関数
cv::Mat OnnxSegmenter::processOutput4D(Ort::Value &outputTensor, const std::vector<int64_t> &outShape, const cv::Size &originalSize) {
    try {
        qDebug() << "=== 4D OUTPUT PROCESSING DIAGNOSIS ===";
        
        int batch = static_cast<int>(outShape[0]);
        int channels = static_cast<int>(outShape[1]);
        int height = static_cast<int>(outShape[2]);
        int width = static_cast<int>(outShape[3]);
        
        qDebug() << "Processing 4D output format";
        qDebug() << "4D Output dimensions:" << batch << "x" << channels << "x" << height << "x" << width;
        
        float *outputData = outputTensor.GetTensorMutableData<float>();
        
        // 各チャンネルの詳細統計
        qDebug() << "DETAILED CHANNEL ANALYSIS:";
        for (int c = 0; c < channels; ++c) {
            float minVal = outputData[c * height * width];
            float maxVal = minVal;
            float sum = 0.0f;
            int nonZeroCount = 0;
            
            for (int i = 0; i < height * width; ++i) {
                float val = outputData[c * height * width + i];
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
                sum += val;
                if (val > 0.001f) nonZeroCount++;
            }
            float mean = sum / (height * width);
            
            QString channelName;
            switch(c) {
                case 0: channelName = "Background"; break;
                case 1: channelName = "Liver"; break;
                case 2: channelName = "Right Kidney"; break;
                case 3: channelName = "Left Kidney/Spleen"; break;
                default: channelName = QString("Channel%1").arg(c); break;
            }
            
            qDebug() << QString("  %1 (ch%2): min=%3 max=%4 mean=%5 nonZero=%6")
                        .arg(channelName).arg(c).arg(minVal, 0, 'f', 6).arg(maxVal, 0, 'f', 6)
                        .arg(mean, 0, 'f', 6).arg(nonZeroCount);
        }
        
        // 中央ピクセルの各チャンネル値の詳細
        int centerY = height / 2;
        int centerX = width / 2;
        qDebug() << QString("Center pixel (%1,%2) detailed values:").arg(centerX).arg(centerY);
        for (int c = 0; c < channels; ++c) {
            int idx = c * height * width + centerY * width + centerX;
            float val = outputData[idx];
            qDebug() << QString("  channel%1: %2").arg(c).arg(val, 0, 'f', 6);
        }
        
        // 複数の位置での確認
        qDebug() << "Multi-position analysis:";
        std::vector<std::pair<int, int>> testPositions = {
            {width/4, height/4},     // 左上領域
            {3*width/4, height/4},   // 右上領域
            {width/4, 3*height/4},   // 左下領域
            {3*width/4, 3*height/4}, // 右下領域
            {width/2, height/2}      // 中央
        };
        
        for (const auto &pos : testPositions) {
            int x = pos.first, y = pos.second;
            qDebug() << QString("Position (%1,%2):").arg(x).arg(y);
            
            float maxProb = 0.0f;
            int maxChannel = 0;
            
            for (int c = 0; c < channels; ++c) {
                int idx = c * height * width + y * width + x;
                float val = outputData[idx];
                if (val > maxProb) {
                    maxProb = val;
                    maxChannel = c;
                }
                qDebug() << QString("    ch%1: %2").arg(c).arg(val, 0, 'f', 4);
            }
            qDebug() << QString("    -> Best: ch%1 with prob %2").arg(maxChannel).arg(maxProb, 0, 'f', 4);
        }
        
        cv::Mat segmentationMask(height, width, CV_8UC1, cv::Scalar(0));
        
        // Simple argmax across channels per pixel
        qDebug() << "Performing per-pixel argmax classification...";
        std::vector<int> labelCounts(channels, 0);
        int totalPixels = height * width;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int bestLabel = 0;
                float bestVal = outputData[y * width + x]; // channel 0
                for (int c = 1; c < channels; ++c) {
                    int idx = c * height * width + y * width + x;
                    float val = outputData[idx];
                    if (val > bestVal) {
                        bestVal = val;
                        bestLabel = c;
                    }
                }
                segmentationMask.at<uchar>(y, x) = static_cast<uchar>(bestLabel);
                labelCounts[bestLabel]++;
            }
        }
        
        // 調整結果の確認
        qDebug() << "ADJUSTED LABEL DISTRIBUTION:";
        for (int i = 0; i < channels; ++i) {
            double percentage = (labelCounts[i] * 100.0) / totalPixels;
            QString labelName;
            switch(i) {
                case 0: labelName = "Background"; break;
                case 1: labelName = "Liver"; break;
                case 2: labelName = "Right Kidney"; break;
                case 3: labelName = "Left Kidney/Spleen"; break;
                default: labelName = QString("Label%1").arg(i); break;
            }
            qDebug() << QString("  %1 (label%2): %3 pixels (%4%)")
                        .arg(labelName).arg(i).arg(labelCounts[i]).arg(percentage, 0, 'f', 2);
        }
        
        // Report non-background ratio (for debugging)
        int adjustedNonBackgroundPixels = totalPixels - labelCounts[0];
        qDebug() << QString("Non-background pixels: %1 out of %2 (%3%)")
                    .arg(adjustedNonBackgroundPixels).arg(totalPixels)
                    .arg((adjustedNonBackgroundPixels * 100.0) / totalPixels, 0, 'f', 2);
        
        // 元のサイズにリサイズ
        cv::Mat result;
        cv::Size originalCvSize(originalSize.width, originalSize.height);
        if (segmentationMask.size() != originalCvSize) {
            cv::resize(segmentationMask, result, originalCvSize, 0, 0, cv::INTER_NEAREST);
        } else {
            result = segmentationMask;
        }
        
        qDebug() << QString("Final result size: %1x%2").arg(result.cols).arg(result.rows);
        
        // 最終確認: リサイズ後のラベル分布
        std::vector<int> finalLabelCounts(channels, 0);
        for (int y = 0; y < result.rows; ++y) {
            for (int x = 0; x < result.cols; ++x) {
                int label = result.at<uchar>(y, x);
                if (label >= 0 && label < channels) {
                    finalLabelCounts[label]++;
                }
            }
        }
        
        qDebug() << "Final result label distribution:";
        for (int i = 0; i < channels; ++i) {
            double percentage = (finalLabelCounts[i] * 100.0) / (result.rows * result.cols);
            qDebug() << QString("  Label%1: %2 pixels (%3%)")
                        .arg(i).arg(finalLabelCounts[i]).arg(percentage, 0, 'f', 2);
        }
        
        qDebug() << "=== END 4D OUTPUT PROCESSING DIAGNOSIS ===";
        
        return result;
        
    } catch (const std::exception &e) {
        qWarning() << "4D output processing error:" << e.what();
        return cv::Mat();
    }
}

cv::Mat OnnxSegmenter::predictVolume(const cv::Mat &volume96, const int *originalSize) {
#ifdef USE_ONNXRUNTIME
    qDebug() << "=== PREDICT VOLUME START ===";

    // VRAM使用量の初期ログ
    #ifdef __linux__
    if (m_cudaEnabled) {
        qDebug() << "VRAM usage before predictVolume:";
        int ret = system("nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | head -2");
        if (ret != 0) {
            qDebug() << "nvidia-smi command failed with code:" << ret;
        }
    }
    #endif

    if (!m_session) {
        qCritical() << "[predictVolume] CRITICAL: No ONNX session loaded";
        return cv::Mat();
    }

    if (volume96.dims != 3) {
        qCritical() << "[predictVolume] CRITICAL: expects 3D volume, got" << volume96.dims << "dims";
        return cv::Mat();
    }

    try {
        int inputDepth = volume96.size[0];
        int inputHeight = volume96.size[1];
        int inputWidth = volume96.size[2];

        qDebug() << "[predictVolume] Input volume:" << inputDepth << "x" << inputHeight << "x" << inputWidth;
        qDebug() << "[predictVolume] Target original:" << originalSize[0] << "x" << originalSize[1] << "x" << originalSize[2];

        // メモリ使用量の推定
        size_t estimated_memory_mb = (inputDepth * inputHeight * inputWidth * sizeof(float)) / (1024 * 1024);
        qDebug() << "[predictVolume] Estimated input memory:" << estimated_memory_mb << "MB";
        
        // CT値の基本チェック（型に応じた正しいアクセス）
        double minVal = std::numeric_limits<double>::max();
        double maxVal = std::numeric_limits<double>::lowest();

        // 簡単な統計計算
        for (int z = 0; z < inputDepth && z < 10; ++z) {  // 最初の10スライスのみチェック
            for (int y = 0; y < inputHeight && y < 100; y += 10) {
                for (int x = 0; x < inputWidth && x < 100; x += 10) {
                    double val = 0.0;
                    if (volume96.type() == CV_16SC1) {
                        val = static_cast<double>(volume96.at<int16_t>(z, y, x));
                    } else if (volume96.type() == CV_32FC1) {
                        val = static_cast<double>(volume96.at<float>(z, y, x));
                    } else if (volume96.type() == CV_8UC1) {
                        val = static_cast<double>(volume96.at<uint8_t>(z, y, x));
                    }
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                }
            }
        }

        qDebug() << "[predictVolume] Input volume type:" << volume96.type()
                 << "(CV_16SC1=" << CV_16SC1 << ", CV_32FC1=" << CV_32FC1 << ")";
        qDebug() << "[predictVolume] Sample HU range:" << minVal << "to" << maxVal;

        if (maxVal == minVal) {
            qCritical() << "[predictVolume] CRITICAL: All values are the same";
            return cv::Mat();
        }
        
        // ONNX推論の準備
        cv::Mat inputVolume;
        if (volume96.type() != CV_32FC1) {
            volume96.convertTo(inputVolume, CV_32FC1);
        } else {
            inputVolume = volume96;
        }
        
        if (!inputVolume.isContinuous()) {
            inputVolume = inputVolume.clone();
        }
        
        // 入力テンソルの作成
        size_t totalElements = 1 * 1 * inputDepth * inputHeight * inputWidth;
        std::vector<int64_t> inputShape = {1, 1, inputDepth, inputHeight, inputWidth};
        
        qDebug() << "[predictVolume] Creating tensor with" << totalElements << "elements";
        
        std::vector<float> inputData;
        inputData.reserve(totalElements);

        // Check if data is already normalized (0-1 range) or raw HU values
        // 既に正規化済みか（0-1範囲）、未正規化のHU値かをチェック
        bool alreadyNormalized = (minVal >= -1.0 && maxVal <= 1.5);

        if (alreadyNormalized) {
            qDebug() << "[predictVolume] Data already normalized (range [" << minVal << "," << maxVal << "])";
            qDebug() << "  Skipping normalization, using data as-is";

            // データをそのままコピー
            for (int z = 0; z < inputDepth; ++z) {
                for (int y = 0; y < inputHeight; ++y) {
                    for (int x = 0; x < inputWidth; ++x) {
                        inputData.push_back(inputVolume.at<float>(z, y, x));
                    }
                }
            }
        } else {
            // MONAI標準の腹部CT前処理を適用
            // BTCV (Beyond the Cranial Vault) データセットの標準前処理に準拠
            // 参考: MONAI公式実装
            // https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb
            //
            // 公式BTCV前処理パラメータ:
            // ScaleIntensityRanged(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True)
            //
            // IMPORTANT: Use BTCV standard range (-175 to 250) to match model training data
            // モデルの訓練データと一致させるため、BTCV標準範囲（-175～250）を使用

            float CLIP_MIN = -175.0f;  // BTCV標準：腹部軟部組織の下限
            float CLIP_MAX = 250.0f;   // BTCV標準：軟部組織の上限

            // 環境変数で上書き可能（腹部のみの場合は -175 を指定）
            const char* clip_min_env = std::getenv("SHIORIS_HU_CLIP_MIN");
            const char* clip_max_env = std::getenv("SHIORIS_HU_CLIP_MAX");
            if (clip_min_env) {
                CLIP_MIN = std::atof(clip_min_env);
            }
            if (clip_max_env) {
                CLIP_MAX = std::atof(clip_max_env);
            }

            const float CLIP_RANGE = CLIP_MAX - CLIP_MIN;

            qDebug() << "[predictVolume] Applying MONAI BTCV standard preprocessing:";
            qDebug() << "  Clipping HU values to [" << CLIP_MIN << "," << CLIP_MAX << "]";
            qDebug() << "  Normalizing to [0, 1] range";

            for (int z = 0; z < inputDepth; ++z) {
                for (int y = 0; y < inputHeight; ++y) {
                    for (int x = 0; x < inputWidth; ++x) {
                        float rawValue = inputVolume.at<float>(z, y, x);

                        // ステップ1: HU値をクリッピング
                        float clippedValue = std::max(CLIP_MIN, std::min(CLIP_MAX, rawValue));

                        // ステップ2: [0, 1] に正規化
                        float normalizedValue = (clippedValue - CLIP_MIN) / CLIP_RANGE;

                        inputData.push_back(normalizedValue);
                    }
                }
            }
        }

        // 正規化後のデータ統計を確認（デバッグ用）
        float minNorm = *std::min_element(inputData.begin(), inputData.end());
        float maxNorm = *std::max_element(inputData.begin(), inputData.end());
        double sumNorm = std::accumulate(inputData.begin(), inputData.end(), 0.0);
        float meanNorm = sumNorm / inputData.size();

        qDebug() << "[predictVolume] Normalized data statistics:";
        qDebug() << "  Min:" << minNorm << "Max:" << maxNorm << "Mean:" << meanNorm;
        qDebug() << "  Total elements:" << inputData.size();

        qDebug() << "[predictVolume] Input data prepared, running inference...";

        // ✓ 入力サイズの検証（セグメンテーションフォルト防止）
        if (inputShape.size() != m_inputDims.size()) {
            qCritical() << "[predictVolume] ❌ ERROR: Input dimension count mismatch!";
            qCritical() << "  Expected:" << m_inputDims.size() << "dimensions";
            qCritical() << "  Got:" << inputShape.size() << "dimensions";
            qCritical() << "\n⚠ Please regenerate the model with matching input size or set environment variables.";
            return cv::Mat();
        }

        for (size_t i = 0; i < inputShape.size(); ++i) {
            if (inputShape[i] != m_inputDims[i]) {
                qCritical() << "[predictVolume] ❌ ERROR: Input size mismatch at dimension" << i << "!";
                qCritical() << "  Model expects: [" << m_inputDims[0] << "," << m_inputDims[1] << ","
                           << m_inputDims[2] << "," << m_inputDims[3] << "," << m_inputDims[4] << "]";
                qCritical() << "  Got:           [" << inputShape[0] << "," << inputShape[1] << ","
                           << inputShape[2] << "," << inputShape[3] << "," << inputShape[4] << "]";
                qCritical() << "\n⚠ FIX OPTIONS:";
                qCritical() << "  1. Regenerate model with: --input-size" << inputShape[2] << inputShape[3] << inputShape[4];
                qCritical() << "     python3 export_monai_to_onnx.py --input-size" << inputShape[2] << inputShape[3] << inputShape[4];
                qCritical() << "\n  2. Or set environment variables (current default:" << inputShape[2] << "x" << inputShape[3] << "x" << inputShape[4] << "):";
                qCritical() << "     export SHIORIS_MODEL_INPUT_DEPTH=" << m_inputDims[2];
                qCritical() << "     export SHIORIS_MODEL_INPUT_HEIGHT=" << m_inputDims[3];
                qCritical() << "     export SHIORIS_MODEL_INPUT_WIDTH=" << m_inputDims[4];
                qCritical() << "     ./ShioRIS3";
                return cv::Mat();
            }
        }

        // ONNX推論の実行
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputData.data(), totalElements,
            inputShape.data(), inputShape.size());

        std::vector<const char *> inputNames;
        std::vector<const char *> outputNames;
        for (const auto &s : m_inputNames) inputNames.push_back(s.c_str());
        for (const auto &s : m_outputNames) outputNames.push_back(s.c_str());

        auto outputTensors = m_session->Run(Ort::RunOptions{nullptr},
                                           inputNames.data(), &inputTensor, 1,
                                           outputNames.data(), m_outputNames.size());

        qDebug() << "[predictVolume] Inference completed";

        if (outputTensors.empty()) {
            qWarning() << "[predictVolume] No output tensors received";
            return cv::Mat();
        }

        // 出力の処理
        auto &outTensor = outputTensors.front();
        auto outShape = outTensor.GetTensorTypeAndShapeInfo().GetShape();
        
        qDebug() << "[predictVolume] Output shape size:" << outShape.size();
        
        int channels = static_cast<int>(outShape[1]);
        cv::Mat result;

        if (channels == 4) {
            result = process3DSegmentationOutput(outTensor, originalSize);
        } else {
            result = processAbdomenSegmentationOutput(outTensor, originalSize);
        }

        // VRAM使用量の最終ログ
        #ifdef __linux__
        if (m_cudaEnabled) {
            qDebug() << "VRAM usage after predictVolume:";
            int ret = system("nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | head -2");
            if (ret != 0) {
                qDebug() << "nvidia-smi command failed with code:" << ret;
            }
        }
        #endif

        qDebug() << "=== PREDICT VOLUME END ===";
        return result;

    } catch (const Ort::Exception &e) {
        qCritical() << "[predictVolume] ONNX error:" << e.what();
        return cv::Mat();
    } catch (const std::exception &e) {
        qCritical() << "[predictVolume] Standard error:" << e.what();
        return cv::Mat();
    }
#else
    return cv::Mat();
#endif
}
//=============================================================================
// ファイル: src/ai/onnx_segmenter.cpp
// 緊急修正: minMaxLoc クラッシュの解決
// 対象関数: resampleVolumeFor3D
//=============================================================================

cv::Mat OnnxSegmenter::predict3D(const cv::Mat &volume) {
#ifdef USE_ONNXRUNTIME
    if (!m_session) {
        qWarning() << "No model loaded for 3D prediction";
        return cv::Mat();
    }

    if (volume.dims != 3) {
        qWarning() << "Input must be 3D volume for predict3D";
        return cv::Mat();
    }

    try {
        qDebug() << "=== 3D VOLUME SEGMENTATION ===";
        updateProgress(0.0f, "Starting 3D segmentation");

        // VRAM使用量の初期ログ（CUDA有効時）
        #ifdef __linux__
        if (m_cudaEnabled) {
            qDebug() << "VRAM usage before inference (check with nvidia-smi)";
            int ret = system("nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | head -2");
            if (ret != 0) {
                qDebug() << "nvidia-smi command failed with code:" << ret;
            }
        }
        #endif

        int originalDepth = volume.size[0];
        int originalHeight = volume.size[1];
        int originalWidth = volume.size[2];

        qDebug() << "Original volume dimensions:" << originalDepth << "x" << originalHeight << "x" << originalWidth;
        updateProgress(0.05f, "Volume loaded");

        // Early quality mode check
        int originalSize[3] = {originalDepth, originalHeight, originalWidth};
        std::string qualityMode = getPredictionQualityMode();
        qDebug() << "Quality mode:" << QString::fromStdString(qualityMode);

        // For high and ultra modes, use specialized inference paths
        if (qualityMode == "high" || qualityMode == "ultra") {
            qDebug() << "Using high-quality inference path - bypassing standard flow";
            updateProgress(0.1f, "Preprocessing volume");

            // Step 1: ボリューム前処理とリサンプリング
            cv::Mat processedVolume = resampleVolumeFor3D(volume);

            if (processedVolume.empty()) {
                qWarning() << "Failed to preprocess volume";
                return cv::Mat();
            }

            updateProgress(0.2f, "Volume preprocessed");

            // リサンプリング後のサイズを使用（Ultra/Highモード用）
            int processedSize[3] = {processedVolume.size[0], processedVolume.size[1], processedVolume.size[2]};
            qDebug() << "Processed volume size for inference:" << processedSize[0] << "x" << processedSize[1] << "x" << processedSize[2];

            cv::Mat segmentationResult;

            if (qualityMode == "ultra") {
                // Ultra: TTA + スライディングウィンドウ
                const char* overlap_env = std::getenv("SHIORIS_SLIDING_WINDOW_OVERLAP");
                float overlap = overlap_env ? std::atof(overlap_env) : 0.5f;
                qDebug() << "Using ULTRA quality mode with overlap:" << overlap;
                updateProgress(0.25f, "Starting Ultra inference (TTA + Sliding Window)");
                // ✓ processedSizeを渡してリサンプリングをスキップ
                segmentationResult = predictVolumeUltra(processedVolume, processedSize, overlap);
            } else {
                // High: TTA のみ
                qDebug() << "Using HIGH quality mode (TTA)";
                updateProgress(0.25f, "Starting High inference (TTA)");
                // ✓ processedSizeを渡してリサンプリングをスキップ
                segmentationResult = predictVolumeWithTTA(processedVolume, processedSize);
            }

            if (segmentationResult.empty()) {
                qWarning() << "High-quality segmentation failed";
                return cv::Mat();
            }

            updateProgress(0.9f, "Postprocessing results");

            // ✓ TTA/Ultra ensemble結果（192x192x192）を元のサイズ（178x512x512）にリサンプリング
            qDebug() << "Resampling TTA/Ultra result back to original size:" << originalSize[0] << "x" << originalSize[1] << "x" << originalSize[2];
            cv::Mat finalResult = resample3DToOriginalSize(segmentationResult, originalSize);

            updateProgress(1.0f, "Segmentation completed");
            qDebug() << "High-quality 3D segmentation completed successfully";
            return finalResult;
        }

        // Standard quality mode - continue with normal flow
        qDebug() << "Using STANDARD quality mode";
        updateProgress(0.1f, "Preprocessing volume");

        // Step 1: ボリューム前処理とリサンプリング
        cv::Mat processedVolume = resampleVolumeFor3D(volume);

        if (processedVolume.empty()) {
            qWarning() << "Failed to preprocess volume";
            return cv::Mat();
        }

        int processedDepth = processedVolume.size[0];
        int processedHeight = processedVolume.size[1];
        int processedWidth = processedVolume.size[2];

        qDebug() << "Processed volume dimensions:" << processedDepth << "x" << processedHeight << "x" << processedWidth;
        updateProgress(0.2f, "Volume preprocessed");

        // Step 2: ONNX入力次元更新
        if (!updateInputDimensions(processedVolume)) {
            qWarning() << "Failed to update input dimensions";
            return cv::Mat();
        }

        // Step 3: 入力テンソル作成
        updateProgress(0.3f, "Preparing input tensor");
        std::vector<int64_t> inputShape = {1, 1, processedDepth, processedHeight, processedWidth};
        size_t inputTensorSize = processedDepth * processedHeight * processedWidth;

        std::vector<float> inputTensorValues(inputTensorSize);
        
        // ボリュームデータを正規化してテンソルにコピー
        for (int z = 0; z < processedDepth; ++z) {
            for (int y = 0; y < processedHeight; ++y) {
                for (int x = 0; x < processedWidth; ++x) {
                    float value = 0.0f;
                    
                    if (processedVolume.type() == CV_32FC1) {
                        value = processedVolume.at<float>(z, y, x);
                    } else if (processedVolume.type() == CV_16UC1) {
                        value = processedVolume.at<uint16_t>(z, y, x) / 65535.0f;
                    } else if (processedVolume.type() == CV_16SC1) {
                        // CTの一般的な範囲 [-1024, 3071] を [0, 1] に正規化
                        float rawValue = processedVolume.at<int16_t>(z, y, x);
                        value = std::max(0.0f, std::min(1.0f, (rawValue + 1024.0f) / 4095.0f));
                    } else if (processedVolume.type() == CV_8UC1) {
                        value = processedVolume.at<uint8_t>(z, y, x) / 255.0f;
                    }
                    
                    size_t idx = z * processedHeight * processedWidth + y * processedWidth + x;
                    inputTensorValues[idx] = value;
                }
            }
        }
        
        // 入力データの統計確認
        auto minMax = std::minmax_element(inputTensorValues.begin(), inputTensorValues.end());
        float mean = std::accumulate(inputTensorValues.begin(), inputTensorValues.end(), 0.0f) / inputTensorValues.size();
        qDebug() << "Input tensor stats - Min:" << *minMax.first << "Max:" << *minMax.second << "Mean:" << mean;
        
        // Step 4: ONNX推論実行
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        
        std::vector<Ort::Value> inputTensors;
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize, inputShape.data(), inputShape.size()));
        
        std::vector<const char*> inputNames{m_inputNames[0].c_str()};
        std::vector<const char*> outputNames{m_outputNames[0].c_str()};

        qDebug() << "Running ONNX inference...";
        updateProgress(0.5f, "Running AI inference");
        auto outputTensors = m_session->Run(Ort::RunOptions{nullptr},
                                          inputNames.data(), inputTensors.data(), 1,
                                          outputNames.data(), 1);

        if (outputTensors.empty()) {
            qWarning() << "ONNX inference returned no output";
            return cv::Mat();
        }

        qDebug() << "ONNX inference completed successfully";
        updateProgress(0.8f, "Processing results");

        // Step 5: 出力処理
        cv::Mat segmentationResult = process3DSegmentationOutput(outputTensors[0], originalSize);

        if (segmentationResult.empty()) {
            qWarning() << "Failed to process segmentation output";
            return cv::Mat();
        }

        updateProgress(1.0f, "Segmentation completed");
        qDebug() << "3D segmentation completed successfully";

        // VRAM使用量の最終ログとクリーンアップ
        #ifdef __linux__
        if (m_cudaEnabled) {
            qDebug() << "VRAM usage after inference:";
            int ret = system("nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | head -2");
            if (ret != 0) {
                qDebug() << "nvidia-smi command failed with code:" << ret;
            }

            // CUDAメモリキャッシュをクリーンアップ（環境変数で有効化）
            const char* clear_cache_env = std::getenv("SHIORIS_CLEAR_CUDA_CACHE");
            if (clear_cache_env && std::atoi(clear_cache_env) == 1) {
                qDebug() << "Clearing CUDA cache (SHIORIS_CLEAR_CUDA_CACHE=1)...";
                // Note: ONNX Runtimeは独自のメモリ管理をしているため、
                // ここではログのみ。実際のクリーンアップはORT側で行われる
                qDebug() << "CUDA cache cleanup requested";
            }
        }
        #endif

#ifndef NDEBUG
        // Detailed diagnostic for debugging
        diagnoseProcessingPipeline(volume, processedVolume, segmentationResult);
#endif

        return segmentationResult;
        
    } catch (const Ort::Exception &e) {
        qCritical() << "ONNX runtime error in predict3D:" << e.what();
        return cv::Mat();
    } catch (const std::exception &e) {
        qCritical() << "Standard exception in predict3D:" << e.what();
        return cv::Mat();
    }
#else
    Q_UNUSED(volume);
    qWarning() << "ONNX Runtime not available";
    return cv::Mat();
#endif
}

cv::Mat OnnxSegmenter::resampleVolumeFor3D(const cv::Mat &volume) {
    if (volume.dims != 3) {
        qWarning() << "Volume must be 3D for resampling";
        return cv::Mat();
    }

    int originalDepth = volume.size[0];
    int originalHeight = volume.size[1];
    int originalWidth = volume.size[2];

    qDebug() << "Input volume for resampling:" << originalDepth << "x" << originalHeight << "x" << originalWidth;

    int minX, maxX, minY, maxY, minZ, maxZ;
    int bodyVoxelCount = 0;
    int totalVoxels = originalDepth * originalHeight * originalWidth;
    bool manualOverride = false;

    // ✓ RTSS Structure-based cropping (最優先)
    if (m_useExternalBoundingBox) {
        qDebug() << "=== Using RTSS Structure-based bounding box ===";
        minX = std::max(0, m_externalBBoxMinX);
        maxX = std::min(originalWidth - 1, m_externalBBoxMaxX);
        minY = std::max(0, m_externalBBoxMinY);
        maxY = std::min(originalHeight - 1, m_externalBBoxMaxY);
        minZ = std::max(0, m_externalBBoxMinZ);
        maxZ = std::min(originalDepth - 1, m_externalBBoxMaxZ);

        qDebug() << "  RTSS bounding box (voxel): X[" << minX << "-" << maxX << "] Y[" << minY << "-" << maxY << "] Z[" << minZ << "-" << maxZ << "]";

        // 体積を計算
        bodyVoxelCount = (maxX - minX + 1) * (maxY - minY + 1) * (maxZ - minZ + 1);
        manualOverride = true; // Skip HU-based detection and margin addition
    } else {
        // ✓ HUしきい値による自動検出（フォールバック）
        // 放射線治療用CTは診断用CTよりFOVが広く、空気や治療台が多く含まれる
        // そのため、体の部分だけを抽出してからリサンプリングする
        // HU値参考: 空気=-1000, 肺=-900~-500, 脂肪=-100~-50, 水=0, 軟部組織=20~60, 治療台=-100~0

        // CRITICAL FIX for abdominal segmentation: Exclude lungs
        // 腹部セグメンテーションでは肺を除外する必要がある
        // -900だと肺が含まれてしまうため、-300に変更（軟部組織と脂肪のみ）
        const int16_t BODY_THRESHOLD = -300;  // 腹部軟部組織のみ（肺を除外）
        minX = originalWidth;
        maxX = 0;
        minY = originalHeight;
        maxY = 0;
        minZ = originalDepth;
        maxZ = 0;

        qDebug() << "=== Detecting body boundaries using HU threshold ===";

        for (int z = 0; z < originalDepth; ++z) {
            for (int y = 0; y < originalHeight; ++y) {
                for (int x = 0; x < originalWidth; ++x) {
                    int16_t value = volume.at<int16_t>(z, y, x);
                    if (value > BODY_THRESHOLD) {
                        bodyVoxelCount++;
                        minX = std::min(minX, x);
                        maxX = std::max(maxX, x);
                        minY = std::min(minY, y);
                        maxY = std::max(maxY, y);
                        minZ = std::min(minZ, z);
                        maxZ = std::max(maxZ, z);
                    }
                }
            }
        }

        double bodyPercentage = (bodyVoxelCount * 100.0) / totalVoxels;
        qDebug() << "Body detection results:";
        qDebug() << "  Body voxels:" << bodyVoxelCount << "(" << bodyPercentage << "% of total volume)";
        qDebug() << "  Body bounds: X[" << minX << "-" << maxX << "] Y[" << minY << "-" << maxY << "] Z[" << minZ << "-" << maxZ << "]";

        // CRITICAL FIX: For abdominal segmentation, limit Z to middle portion
        // 腹部セグメンテーションでは、Z方向を中央部分に制限（胸部/肺と骨盤部を除外）
        //
        // Typical abdominal organs (liver, kidneys, spleen) are in the middle 40-60% of CT volume
        // Z coordinate: 0=feet/pelvis, max=head/chest
        // 腹部臓器（肝臓、腎臓、脾臓）は通常、CTボリュームの中央40-60%に位置
        // Z座標: 0=足側/骨盤部、max=頭側/胸部

        int abdominalStartZ = originalDepth / 3;      // 上1/3をスキップ（胸部/肺）
        int abdominalEndZ = originalDepth * 5 / 6;    // 下1/6をスキップ（骨盤部）

        bool adjusted = false;
        if (minZ < abdominalStartZ || maxZ > abdominalEndZ) {
            qDebug() << "  ⚠ Adjusting Z range to focus on abdominal region:";
            qDebug() << "    Original Z: [" << minZ << "-" << maxZ << "]";

            if (minZ < abdominalStartZ) {
                minZ = abdominalStartZ;
                adjusted = true;
            }
            if (maxZ > abdominalEndZ) {
                maxZ = abdominalEndZ;
                adjusted = true;
            }

            qDebug() << "    Adjusted Z: [" << minZ << "-" << maxZ << "] (middle 50% of volume)";
            qDebug() << "    Excluded: upper 1/3 (chest/lungs) + lower 1/6 (pelvis)";
        }
    }

    // ✓ 環境変数による手動オーバーライド（RT-CT対応）
    // RTSSのbounding boxが設定されていない場合のみ有効
    // 自動検出がうまくいかない場合、手動で境界を指定可能
    if (!m_useExternalBoundingBox) {
        const char* manual_minX = std::getenv("SHIORIS_CROP_MIN_X");
        const char* manual_maxX = std::getenv("SHIORIS_CROP_MAX_X");
        const char* manual_minY = std::getenv("SHIORIS_CROP_MIN_Y");
        const char* manual_maxY = std::getenv("SHIORIS_CROP_MAX_Y");
        const char* manual_minZ = std::getenv("SHIORIS_CROP_MIN_Z");
        const char* manual_maxZ = std::getenv("SHIORIS_CROP_MAX_Z");

        if (manual_minX || manual_maxX || manual_minY || manual_maxY || manual_minZ || manual_maxZ) {
            qDebug() << "\n⚠ Manual crop override detected!";
            manualOverride = true;

            if (manual_minX) { minX = std::atoi(manual_minX); qDebug() << "  minX overridden to:" << minX; }
            if (manual_maxX) { maxX = std::atoi(manual_maxX); qDebug() << "  maxX overridden to:" << maxX; }
            if (manual_minY) { minY = std::atoi(manual_minY); qDebug() << "  minY overridden to:" << minY; }
            if (manual_maxY) { maxY = std::atoi(manual_maxY); qDebug() << "  maxY overridden to:" << maxY; }
            if (manual_minZ) { minZ = std::atoi(manual_minZ); qDebug() << "  minZ overridden to:" << minZ; }
            if (manual_maxZ) { maxZ = std::atoi(manual_maxZ); qDebug() << "  maxZ overridden to:" << maxZ; }

            qDebug() << "  Final bounds: X[" << minX << "-" << maxX << "] Y[" << minY << "-" << maxY << "] Z[" << minZ << "-" << maxZ << "]";
        }
    }

    // ボディ領域のサイズ
    int bodyWidth = maxX - minX + 1;
    int bodyHeight = maxY - minY + 1;
    int bodyDepth = maxZ - minZ + 1;
    qDebug() << "  Body size:" << bodyDepth << "x" << bodyHeight << "x" << bodyWidth;

    // マージンを追加（体の境界を少し広げる）
    // 手動オーバーライド時はマージンをスキップ
    if (!manualOverride) {
        const int MARGIN_XY = 20;  // 横方向マージン（ピクセル）- 増やして臓器の取りこぼしを防ぐ
        const int MARGIN_Z = 5;    // スライス方向マージン - 増やして臓器の取りこぼしを防ぐ

        minX = std::max(0, minX - MARGIN_XY);
        maxX = std::min(originalWidth - 1, maxX + MARGIN_XY);
        minY = std::max(0, minY - MARGIN_XY);
        maxY = std::min(originalHeight - 1, maxY + MARGIN_XY);
        minZ = std::max(0, minZ - MARGIN_Z);
        maxZ = std::min(originalDepth - 1, maxZ + MARGIN_Z);

        qDebug() << "After adding margins:";
    } else {
        qDebug() << "Margins skipped (manual override):";
    }

    int croppedWidth = maxX - minX + 1;
    int croppedHeight = maxY - minY + 1;
    int croppedDepth = maxZ - minZ + 1;

    qDebug() << "  Cropped region:" << croppedDepth << "x" << croppedHeight << "x" << croppedWidth;
    qDebug() << "  Data retention:" << ((croppedDepth * croppedHeight * croppedWidth * 100.0) / totalVoxels) << "%";

    // クロップされた領域を抽出
    int croppedSizes[3] = {croppedDepth, croppedHeight, croppedWidth};
    cv::Mat croppedVolume(3, croppedSizes, volume.type());

    for (int z = 0; z < croppedDepth; ++z) {
        for (int y = 0; y < croppedHeight; ++y) {
            for (int x = 0; x < croppedWidth; ++x) {
                int16_t value = volume.at<int16_t>(minZ + z, minY + y, minX + x);
                croppedVolume.at<int16_t>(z, y, x) = value;
            }
        }
    }

    qDebug() << "✓ Body region extracted, proceeding with resampling...";

    // クロップオフセットを保存（後でセグメンテーション結果を元の座標系に戻すため）
    m_cropOffsetX = minX;
    m_cropOffsetY = minY;
    m_cropOffsetZ = minZ;
    m_croppedWidth = croppedWidth;
    m_croppedHeight = croppedHeight;
    m_croppedDepth = croppedDepth;

    qDebug() << "Saved crop offsets for coordinate restoration:";
    qDebug() << "  Offset: (" << m_cropOffsetX << "," << m_cropOffsetY << "," << m_cropOffsetZ << ")";

    // クロップされたボリュームを使用（元のボリューム変数を上書き）
    cv::Mat volumeToResample = croppedVolume;
    int resampleDepth = croppedDepth;
    int resampleHeight = croppedHeight;
    int resampleWidth = croppedWidth;

    // ✓ モデルから期待サイズを自動検出（環境変数不要）
    // モデルロード時に m_inputDims に保存されている: [batch, channel, depth, height, width]
    int targetDepth = 96;   // フォールバック値
    int targetHeight = 96;
    int targetWidth = 96;

    if (m_inputDims.size() >= 5) {
        // m_inputDims = [batch=1, channel=1, depth, height, width]
        targetDepth = static_cast<int>(m_inputDims[2]);
        targetHeight = static_cast<int>(m_inputDims[3]);
        targetWidth = static_cast<int>(m_inputDims[4]);
        qDebug() << "✓ Auto-detected model input size from loaded model:"
                 << targetDepth << "x" << targetHeight << "x" << targetWidth;
    } else if (m_inputDims.size() == 4) {
        // 2Dモデルの場合: [batch=1, channel=1, height, width]
        targetHeight = static_cast<int>(m_inputDims[2]);
        targetWidth = static_cast<int>(m_inputDims[3]);
        qDebug() << "✓ Auto-detected 2D model input size:" << targetHeight << "x" << targetWidth;
    } else {
        qWarning() << "⚠ Could not auto-detect model size (m_inputDims.size=" << m_inputDims.size() << ")";
        qWarning() << "  Using fallback: 96x96x96";
    }

    // 環境変数で上書き可能（通常は不要だが、デバッグ用に残す）
    const char* model_depth_env = std::getenv("SHIORIS_MODEL_INPUT_DEPTH");
    const char* model_height_env = std::getenv("SHIORIS_MODEL_INPUT_HEIGHT");
    const char* model_width_env = std::getenv("SHIORIS_MODEL_INPUT_WIDTH");

    if (model_depth_env) {
        targetDepth = std::atoi(model_depth_env);
        qDebug() << "  Overridden by env var: DEPTH=" << targetDepth;
    }
    if (model_height_env) {
        targetHeight = std::atoi(model_height_env);
        qDebug() << "  Overridden by env var: HEIGHT=" << targetHeight;
    }
    if (model_width_env) {
        targetWidth = std::atoi(model_width_env);
        qDebug() << "  Overridden by env var: WIDTH=" << targetWidth;
    }

    qDebug() << "Target resampling size: D=" << targetDepth << " H=" << targetHeight << " W=" << targetWidth;

    // メモリ使用量の推定と警告
    size_t estimated_input_mb = (static_cast<size_t>(targetDepth) * targetHeight * targetWidth * sizeof(float)) / (1024 * 1024);
    // Transformer モデルは中間層で入力の10-20倍のメモリを使用する可能性がある
    size_t estimated_peak_mb = estimated_input_mb * 15; // 保守的な推定

    qDebug() << "Estimated memory usage:";
    qDebug() << "  Input tensor:" << estimated_input_mb << "MB";
    qDebug() << "  Peak (estimated):" << estimated_peak_mb << "MB";

    if (estimated_peak_mb > 8000) {
        qWarning() << "⚠ WARNING: Estimated peak memory usage (" << estimated_peak_mb << "MB) may exceed 8GB!";
        qWarning() << "  This model size requires a lot of VRAM.";
        qWarning() << "  Consider re-exporting the model with smaller input size:";
        qWarning() << "    python scripts/model_tools/export_monai_to_onnx.py --input-size 64 64 64";
        qWarning() << "  Or use CPU inference (slower but no VRAM limit).";
    }

    // IMPORTANT: Swin UNETRは固定サイズを期待するため、
    // アスペクト比を無視して強制的にtargetサイズにリサンプリングする
    // （元のアスペクト比を保持する処理は削除）

    qDebug() << "Resampling to fixed model size (ignoring aspect ratio):";
    qDebug() << "  From (cropped):" << resampleDepth << "x" << resampleHeight << "x" << resampleWidth;
    qDebug() << "  To (model):" << targetDepth << "x" << targetHeight << "x" << targetWidth;
    qDebug() << "Scaling factors - Z:" << (float)targetDepth/resampleDepth
             << "Y:" << (float)targetHeight/resampleHeight
             << "X:" << (float)targetWidth/resampleWidth;

    // ✓ CRITICAL FIX: Normalize HU values BEFORE resampling
    // リサンプリング前に正規化を行う（トリリニア補間で極端な値が混ざるのを防ぐ）
    // MONAI BTCV標準前処理: ScaleIntensityRanged(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True)
    //
    // IMPORTANT: Use BTCV standard range (-175 to 250) to match model training data
    // モデルの訓練データと一致させるため、BTCV標準範囲（-175～250）を使用
    // Lungs/chest are already excluded by Z-axis cropping, so we don't need extended range
    // 肺/胸部はZ軸クロッピングで既に除外されているため、拡張範囲は不要

    float CLIP_MIN = -175.0f;  // BTCV標準：腹部軟部組織の下限
    float CLIP_MAX = 250.0f;   // BTCV標準：軟部組織の上限

    // 環境変数で上書き可能（腹部のみの場合は -175 を指定）
    const char* clip_min_env = std::getenv("SHIORIS_HU_CLIP_MIN");
    const char* clip_max_env = std::getenv("SHIORIS_HU_CLIP_MAX");
    if (clip_min_env) {
        CLIP_MIN = std::atof(clip_min_env);
        qDebug() << "  ⚙ HU_CLIP_MIN overridden to:" << CLIP_MIN;
    }
    if (clip_max_env) {
        CLIP_MAX = std::atof(clip_max_env);
        qDebug() << "  ⚙ HU_CLIP_MAX overridden to:" << CLIP_MAX;
    }

    const float CLIP_RANGE = CLIP_MAX - CLIP_MIN;

    qDebug() << "=== NORMALIZING HU VALUES BEFORE RESAMPLING ===";
    qDebug() << "  Clipping HU to [" << CLIP_MIN << "," << CLIP_MAX << "]";
    qDebug() << "  Normalizing to [0, 1] range";

    // 正規化前のHU値統計を確認
    int16_t minHU = 32767, maxHU = -32768;
    double sumHU = 0.0;
    int sampleCount = 0;
    for (int z = 0; z < croppedDepth; ++z) {
        for (int y = 0; y < croppedHeight; ++y) {
            for (int x = 0; x < croppedWidth; ++x) {
                int16_t val = volumeToResample.at<int16_t>(z, y, x);
                minHU = std::min(minHU, val);
                maxHU = std::max(maxHU, val);
                sumHU += val;
                sampleCount++;
            }
        }
    }
    double meanHU = sumHU / sampleCount;
    qDebug() << "  BEFORE normalization - HU range: [" << minHU << "," << maxHU << "] mean:" << meanHU;

    // int16_t → float32に変換して正規化
    cv::Mat normalizedVolume(3, croppedSizes, CV_32FC1);
    for (int z = 0; z < croppedDepth; ++z) {
        for (int y = 0; y < croppedHeight; ++y) {
            for (int x = 0; x < croppedWidth; ++x) {
                int16_t rawValue = volumeToResample.at<int16_t>(z, y, x);

                // ステップ1: HU値をクリッピング
                float clippedValue = std::max(CLIP_MIN, std::min(CLIP_MAX, static_cast<float>(rawValue)));

                // ステップ2: [0, 1] に正規化
                float normalizedValue = (clippedValue - CLIP_MIN) / CLIP_RANGE;

                normalizedVolume.at<float>(z, y, x) = normalizedValue;
            }
        }
    }

    // 正規化後のボリュームでリサンプリングを行う
    volumeToResample = normalizedVolume;
    qDebug() << "✓ HU normalization completed, proceeding with resampling";

    // リサンプリング実行
    int sizes[] = {targetDepth, targetHeight, targetWidth};
    cv::Mat resampledVolume(3, sizes, CV_32FC1);  // CV_32FC1に変更

    for (int z = 0; z < targetDepth; ++z) {
        for (int y = 0; y < targetHeight; ++y) {
            for (int x = 0; x < targetWidth; ++x) {
                // 対応する元の座標を計算（バイリニア補間）
                float srcZ = (z + 0.5f) * resampleDepth / targetDepth - 0.5f;
                float srcY = (y + 0.5f) * resampleHeight / targetHeight - 0.5f;
                float srcX = (x + 0.5f) * resampleWidth / targetWidth - 0.5f;

                // 境界チェック
                srcZ = std::max(0.0f, std::min(static_cast<float>(resampleDepth - 1), srcZ));
                srcY = std::max(0.0f, std::min(static_cast<float>(resampleHeight - 1), srcY));
                srcX = std::max(0.0f, std::min(static_cast<float>(resampleWidth - 1), srcX));

                // トリリニア補間（3D線形補間）で精度向上
                int z0 = static_cast<int>(std::floor(srcZ));
                int y0 = static_cast<int>(std::floor(srcY));
                int x0 = static_cast<int>(std::floor(srcX));
                int z1 = std::min(z0 + 1, resampleDepth - 1);
                int y1 = std::min(y0 + 1, resampleHeight - 1);
                int x1 = std::min(x0 + 1, resampleWidth - 1);

                float dz = srcZ - z0;
                float dy = srcY - y0;
                float dx = srcX - x0;

                // 値を補間
                if (volumeToResample.type() == CV_16SC1) {
                    float c000 = volumeToResample.at<int16_t>(z0, y0, x0);
                    float c001 = volumeToResample.at<int16_t>(z0, y0, x1);
                    float c010 = volumeToResample.at<int16_t>(z0, y1, x0);
                    float c011 = volumeToResample.at<int16_t>(z0, y1, x1);
                    float c100 = volumeToResample.at<int16_t>(z1, y0, x0);
                    float c101 = volumeToResample.at<int16_t>(z1, y0, x1);
                    float c110 = volumeToResample.at<int16_t>(z1, y1, x0);
                    float c111 = volumeToResample.at<int16_t>(z1, y1, x1);

                    float c00 = c000 * (1 - dx) + c001 * dx;
                    float c01 = c010 * (1 - dx) + c011 * dx;
                    float c10 = c100 * (1 - dx) + c101 * dx;
                    float c11 = c110 * (1 - dx) + c111 * dx;

                    float c0 = c00 * (1 - dy) + c01 * dy;
                    float c1 = c10 * (1 - dy) + c11 * dy;

                    float value = c0 * (1 - dz) + c1 * dz;
                    resampledVolume.at<int16_t>(z, y, x) = static_cast<int16_t>(std::round(value));
                } else if (volumeToResample.type() == CV_16UC1) {
                    float c000 = volumeToResample.at<uint16_t>(z0, y0, x0);
                    float c001 = volumeToResample.at<uint16_t>(z0, y0, x1);
                    float c010 = volumeToResample.at<uint16_t>(z0, y1, x0);
                    float c011 = volumeToResample.at<uint16_t>(z0, y1, x1);
                    float c100 = volumeToResample.at<uint16_t>(z1, y0, x0);
                    float c101 = volumeToResample.at<uint16_t>(z1, y0, x1);
                    float c110 = volumeToResample.at<uint16_t>(z1, y1, x0);
                    float c111 = volumeToResample.at<uint16_t>(z1, y1, x1);

                    float c00 = c000 * (1 - dx) + c001 * dx;
                    float c01 = c010 * (1 - dx) + c011 * dx;
                    float c10 = c100 * (1 - dx) + c101 * dx;
                    float c11 = c110 * (1 - dx) + c111 * dx;

                    float c0 = c00 * (1 - dy) + c01 * dy;
                    float c1 = c10 * (1 - dy) + c11 * dy;

                    float value = c0 * (1 - dz) + c1 * dz;
                    resampledVolume.at<uint16_t>(z, y, x) = static_cast<uint16_t>(std::round(value));
                } else if (volumeToResample.type() == CV_8UC1) {
                    float c000 = volumeToResample.at<uint8_t>(z0, y0, x0);
                    float c001 = volumeToResample.at<uint8_t>(z0, y0, x1);
                    float c010 = volumeToResample.at<uint8_t>(z0, y1, x0);
                    float c011 = volumeToResample.at<uint8_t>(z0, y1, x1);
                    float c100 = volumeToResample.at<uint8_t>(z1, y0, x0);
                    float c101 = volumeToResample.at<uint8_t>(z1, y0, x1);
                    float c110 = volumeToResample.at<uint8_t>(z1, y1, x0);
                    float c111 = volumeToResample.at<uint8_t>(z1, y1, x1);

                    float c00 = c000 * (1 - dx) + c001 * dx;
                    float c01 = c010 * (1 - dx) + c011 * dx;
                    float c10 = c100 * (1 - dx) + c101 * dx;
                    float c11 = c110 * (1 - dx) + c111 * dx;

                    float c0 = c00 * (1 - dy) + c01 * dy;
                    float c1 = c10 * (1 - dy) + c11 * dy;

                    float value = c0 * (1 - dz) + c1 * dz;
                    resampledVolume.at<uint8_t>(z, y, x) = static_cast<uint8_t>(std::round(value));
                } else if (volumeToResample.type() == CV_32FC1) {
                    float c000 = volumeToResample.at<float>(z0, y0, x0);
                    float c001 = volumeToResample.at<float>(z0, y0, x1);
                    float c010 = volumeToResample.at<float>(z0, y1, x0);
                    float c011 = volumeToResample.at<float>(z0, y1, x1);
                    float c100 = volumeToResample.at<float>(z1, y0, x0);
                    float c101 = volumeToResample.at<float>(z1, y0, x1);
                    float c110 = volumeToResample.at<float>(z1, y1, x0);
                    float c111 = volumeToResample.at<float>(z1, y1, x1);

                    float c00 = c000 * (1 - dx) + c001 * dx;
                    float c01 = c010 * (1 - dx) + c011 * dx;
                    float c10 = c100 * (1 - dx) + c101 * dx;
                    float c11 = c110 * (1 - dx) + c111 * dx;

                    float c0 = c00 * (1 - dy) + c01 * dy;
                    float c1 = c10 * (1 - dy) + c11 * dy;

                    float value = c0 * (1 - dz) + c1 * dz;
                    resampledVolume.at<float>(z, y, x) = value;
                }
            }
        }
    }
    
    // 統計確認
    cv::Scalar mean, stddev;
    cv::meanStdDev(resampledVolume, mean, stddev);
    qDebug() << "Resampled volume stats - Mean:" << mean[0] << "StdDev:" << stddev[0];
    
    return resampledVolume;
}

cv::Mat OnnxSegmenter::process3DSegmentationOutput(Ort::Value &outputTensor, const int* originalSize) {
#ifdef USE_ONNXRUNTIME
    try {
        // 出力テンソル情報取得
        Ort::TensorTypeAndShapeInfo outputInfo = outputTensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> outputShape = outputInfo.GetShape();
        
        qDebug() << "Output tensor shape:";
        for (size_t i = 0; i < outputShape.size(); ++i) {
            qDebug() << "  dim" << i << ":" << outputShape[i];
        }
        
        if (outputShape.size() != 5) {
            qWarning() << "Expected 5D output tensor [batch, channels, depth, height, width], got" << outputShape.size() << "D";
            return cv::Mat();
        }
        
        int batch = static_cast<int>(outputShape[0]);
        int channels = static_cast<int>(outputShape[1]);
        int depth = static_cast<int>(outputShape[2]);
        int height = static_cast<int>(outputShape[3]);
        int width = static_cast<int>(outputShape[4]);
        
        qDebug() << "Output dimensions:" << batch << "x" << channels << "x" << depth << "x" << height << "x" << width;
        
        if (batch != 1) {
            qWarning() << "Expected batch size 1, got" << batch;
            return cv::Mat();
        }

        // チャンネル数の検証
        // - 4チャンネル: 簡易モデル（Background, Liver, Right Kidney, Left Kidney/Spleen）
        // - 14チャンネル: BTCV標準（Background + 13臓器）
        // - その他: カスタムモデル
        if (channels != 4 && channels != 14) {
            qWarning() << "Unexpected number of channels:" << channels;
            qWarning() << "Expected 4 (simple model) or 14 (BTCV dataset)";
            qWarning() << "Will process as" << channels << "channel model";
        }

        qDebug() << "Processing model with" << channels << "output channels";
        
        // 出力データ取得
        float* outputData = outputTensor.GetTensorMutableData<float>();
        if (!outputData) {
            qWarning() << "Failed to get output tensor data";
            return cv::Mat();
        }
        
        // 各チャンネルの統計確認
        qDebug() << "Output channels analysis:";
        for (int c = 0; c < channels; ++c) {
            float minVal = std::numeric_limits<float>::max();
            float maxVal = std::numeric_limits<float>::lowest();
            float sum = 0.0f;
            int nonZeroCount = 0;
            
            for (int z = 0; z < depth; ++z) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        size_t idx = c * depth * height * width + z * height * width + y * width + x;
                        float val = outputData[idx];
                        
                        minVal = std::min(minVal, val);
                        maxVal = std::max(maxVal, val);
                        sum += val;
                        if (val > 0.001f) nonZeroCount++;
                    }
                }
            }
            
            float mean = sum / (depth * height * width);
            qDebug() << QString("  Channel %1: range=[%2, %3], mean=%4, nonzero=%5")
                        .arg(c).arg(minVal, 0, 'f', 6).arg(maxVal, 0, 'f', 6)
                        .arg(mean, 0, 'f', 6).arg(nonZeroCount);
        }
        
        // セグメンテーションマスク生成（argmax）
        int segSizes[] = {depth, height, width};
        cv::Mat segmentationMask(3, segSizes, CV_8UC1, cv::Scalar(0));
        
        qDebug() << "Performing per-voxel argmax classification...";
        std::vector<int> labelCounts(channels, 0);
        
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int bestLabel = 0;
                    float bestProb = outputData[z * height * width + y * width + x]; // channel 0
                    
                    for (int c = 1; c < channels; ++c) {
                        size_t idx = c * depth * height * width + z * height * width + y * width + x;
                        float prob = outputData[idx];
                        if (prob > bestProb) {
                            bestProb = prob;
                            bestLabel = c;
                        }
                    }
                    
                    segmentationMask.at<uchar>(z, y, x) = static_cast<uchar>(bestLabel);
                    labelCounts[bestLabel]++;
                }
            }
        }
        
        // 予測結果の統計
        qDebug() << "Predicted label distribution:";
        auto organLabels = OnnxSegmenter::getOrganLabels();  // staticメソッドとして呼び出し
        int totalVoxels = depth * height * width;
        for (int i = 0; i < channels; ++i) {
            double percentage = (labelCounts[i] * 100.0) / totalVoxels;
            qDebug() << QString("  %1: %2 voxels (%3%)")
                        .arg(QString::fromStdString(organLabels[i]))
                        .arg(labelCounts[i])
                        .arg(percentage, 0, 'f', 2);
        }

        // 元のサイズにリサンプリング（サイズが異なる場合のみ）
        // High/Ultraモードでは、processedSizeが渡されるため、リサンプリングをスキップ
        if (originalSize[0] == depth && originalSize[1] == height && originalSize[2] == width) {
            qDebug() << "Segmentation mask already at target size, skipping resample";
            qDebug() << "3D segmentation output processing completed";
            return segmentationMask;
        }

        cv::Mat finalResult = resample3DToOriginalSize(segmentationMask, originalSize);

        qDebug() << "3D segmentation output processing completed";
        return finalResult;
        
    } catch (const std::exception &e) {
        qCritical() << "Error processing 3D segmentation output:" << e.what();
        return cv::Mat();
    }
#else
    Q_UNUSED(outputTensor);
    Q_UNUSED(originalSize);
    return cv::Mat();
#endif
}

cv::Mat OnnxSegmenter::resample3DToOriginalSize(const cv::Mat &segmentation3D, const int* originalSize) {
    if (segmentation3D.dims != 3) {
        qWarning() << "Input must be 3D segmentation volume";
        return cv::Mat();
    }

    int currentDepth = segmentation3D.size[0];
    int currentHeight = segmentation3D.size[1];
    int currentWidth = segmentation3D.size[2];

    int targetDepth = originalSize[0];
    int targetHeight = originalSize[1];
    int targetWidth = originalSize[2];

    qDebug() << "Resampling 3D segmentation with coordinate restoration:";
    qDebug() << "  From (model output):" << currentDepth << "x" << currentHeight << "x" << currentWidth;
    qDebug() << "  To (original volume):" << targetDepth << "x" << targetHeight << "x" << targetWidth;

    // ✓ TTA/Ultraモード対応: サイズが同じ場合はリサンプリングをスキップ
    // （各augmentationの結果を192x192x192のまま返すため）
    if (currentDepth == targetDepth && currentHeight == targetHeight && currentWidth == targetWidth) {
        qDebug() << "  Sizes match, skipping resampling (returning as-is for TTA ensemble)";
        return segmentation3D.clone();
    }

    qDebug() << "  Crop offset: (" << m_cropOffsetX << "," << m_cropOffsetY << "," << m_cropOffsetZ << ")";
    qDebug() << "  Cropped region: " << m_croppedDepth << "x" << m_croppedHeight << "x" << m_croppedWidth;

    // ✓ Step 1: モデル出力をクロップ領域のサイズにリサンプリング
    int croppedSizes[3] = {m_croppedDepth, m_croppedHeight, m_croppedWidth};
    cv::Mat croppedResult(3, croppedSizes, CV_8UC1, cv::Scalar(0));

    float scaleZ = static_cast<float>(currentDepth) / m_croppedDepth;
    float scaleY = static_cast<float>(currentHeight) / m_croppedHeight;
    float scaleX = static_cast<float>(currentWidth) / m_croppedWidth;

    qDebug() << "  Scale factors (model→cropped): Z=" << scaleZ << " Y=" << scaleY << " X=" << scaleX;

    for (int z = 0; z < m_croppedDepth; ++z) {
        for (int y = 0; y < m_croppedHeight; ++y) {
            for (int x = 0; x < m_croppedWidth; ++x) {
                // モデル出力座標を計算
                int srcZ = static_cast<int>(std::round(z * scaleZ));
                int srcY = static_cast<int>(std::round(y * scaleY));
                int srcX = static_cast<int>(std::round(x * scaleX));

                // 境界チェック
                srcZ = std::min(srcZ, currentDepth - 1);
                srcY = std::min(srcY, currentHeight - 1);
                srcX = std::min(srcX, currentWidth - 1);

                // ラベル値をコピー
                uchar label = segmentation3D.at<uchar>(srcZ, srcY, srcX);
                croppedResult.at<uchar>(z, y, x) = label;
            }
        }
    }

    qDebug() << "  ✓ Resampled to cropped region size";

    // ✓ Step 2: 元のボリュームサイズの空白マット（全て背景=0）を作成
    int resultSizes[3] = {targetDepth, targetHeight, targetWidth};
    cv::Mat result(3, resultSizes, CV_8UC1, cv::Scalar(0));  // 背景で初期化

    // ✓ Step 3: クロップ領域を元の位置（オフセット適用）に配置
    qDebug() << "  Placing cropped result at correct coordinates...";

    for (int z = 0; z < m_croppedDepth; ++z) {
        int targetZ = z + m_cropOffsetZ;
        if (targetZ < 0 || targetZ >= targetDepth) continue;

        for (int y = 0; y < m_croppedHeight; ++y) {
            int targetY = y + m_cropOffsetY;
            if (targetY < 0 || targetY >= targetHeight) continue;

            for (int x = 0; x < m_croppedWidth; ++x) {
                int targetX = x + m_cropOffsetX;
                if (targetX < 0 || targetX >= targetWidth) continue;

                uchar label = croppedResult.at<uchar>(z, y, x);
                result.at<uchar>(targetZ, targetY, targetX) = label;
            }
        }
    }

    qDebug() << "  ✓ Segmentation placed at correct coordinates";
    
    // 最終結果の統計
    auto organLabels = OnnxSegmenter::getOrganLabels();
    int maxLabels = static_cast<int>(organLabels.size());
    std::vector<int> finalLabelCounts(maxLabels, 0);
    int totalVoxels = targetDepth * targetHeight * targetWidth;

    for (int z = 0; z < targetDepth; ++z) {
        for (int y = 0; y < targetHeight; ++y) {
            for (int x = 0; x < targetWidth; ++x) {
                int label = result.at<uchar>(z, y, x);
                if (label >= 0 && label < maxLabels) {
                    finalLabelCounts[label]++;
                }
            }
        }
    }

    // ✓ 左右反転チェック: 中央スライスの左右端を比較
    if (targetDepth > 0 && targetHeight > 0 && targetWidth > 10) {
        int midZ = targetDepth / 2;
        int midY = targetHeight / 2;
        int leftX = targetWidth / 4;      // 左側（X軸の小さい方）
        int rightX = 3 * targetWidth / 4;  // 右側（X軸の大きい方）

        uchar leftLabel = result.at<uchar>(midZ, midY, leftX);
        uchar rightLabel = result.at<uchar>(midZ, midY, rightX);

        qDebug() << "=== LEFT-RIGHT FLIP CHECK ===";
        qDebug() << QString("  Left side (x=%1): %2").arg(leftX)
                    .arg(leftLabel < organLabels.size() ? QString::fromStdString(organLabels[leftLabel]) : QString::number(leftLabel));
        qDebug() << QString("  Right side (x=%1): %2").arg(rightX)
                    .arg(rightLabel < organLabels.size() ? QString::fromStdString(organLabels[rightLabel]) : QString::number(rightLabel));
        qDebug() << "  ⚠ NOTE: In DICOM, X increases from patient's RIGHT to LEFT";
        qDebug() << "  Small X = Patient RIGHT, Large X = Patient LEFT";
        qDebug() << "  Expected: Liver on RIGHT (small X), Spleen on LEFT (large X)";
    }

    qDebug() << "Final resampled label distribution:";
    for (int i = 0; i < maxLabels; ++i) {
        if (finalLabelCounts[i] > 0) {  // 存在するラベルのみ表示
            double percentage = (finalLabelCounts[i] * 100.0) / totalVoxels;
            qDebug() << QString("  %1: %2 voxels (%3%)")
                        .arg(QString::fromStdString(organLabels[i]))
                        .arg(finalLabelCounts[i])
                        .arg(percentage, 0, 'f', 2);
        }
    }
    
    qDebug() << "3D resampling completed";
    return result;
}

bool OnnxSegmenter::updateInputDimensions(const cv::Mat &volume) {
    if (volume.dims != 3) {
        qWarning() << "Volume must be 3D for dimension update";
        return false;
    }
    
    int depth = volume.size[0];
    int height = volume.size[1];
    int width = volume.size[2];
    
    qDebug() << "Updating ONNX input dimensions to:" << depth << "x" << height << "x" << width;
    
    // 新しい入力次元を設定
    m_inputDims = {1, 1, depth, height, width};  // [batch, channel, depth, height, width]
    
    qDebug() << "Input dimensions updated successfully";
    return true;
}

#ifndef NDEBUG
void OnnxSegmenter::diagnoseProcessingPipeline(const cv::Mat &originalVolume, const cv::Mat &processedVolume, const cv::Mat &result) {
    qDebug() << "=== PROCESSING PIPELINE DIAGNOSIS ===";
    
    if (!originalVolume.empty()) {
        double origMin, origMax;
        cv::Scalar origMean;
        cv::Mat flatOrig = originalVolume.reshape(1, originalVolume.total());
        cv::minMaxLoc(flatOrig, &origMin, &origMax);
        cv::meanStdDev(flatOrig, origMean, cv::noArray());
        
        qDebug() << "Original volume:";
        qDebug() << "  Dimensions:" << originalVolume.size[0] << "x" << originalVolume.size[1] << "x" << originalVolume.size[2];
        qDebug() << "  Range:" << origMin << "to" << origMax;
        qDebug() << "  Mean:" << origMean[0];
    }
    
    if (!processedVolume.empty()) {
        double procMin, procMax;
        cv::Scalar procMean;
        cv::Mat flatProc = processedVolume.reshape(1, processedVolume.total());
        cv::minMaxLoc(flatProc, &procMin, &procMax);
        cv::meanStdDev(flatProc, procMean, cv::noArray());
        
        qDebug() << "Processed volume:";
        qDebug() << "  Dimensions:" << processedVolume.size[0] << "x" << processedVolume.size[1] << "x" << processedVolume.size[2];
        qDebug() << "  Range:" << procMin << "to" << procMax;
        qDebug() << "  Mean:" << procMean[0];
        
        // クリティカルチェック
        if (procMax == procMin) {
            qCritical() << "*** PROCESSING PIPELINE ERROR: Constant values after preprocessing ***";
        }
    }
    
    if (!result.empty()) {
        std::vector<int> labelCounts(4, 0);
        int totalVoxels = result.total();
        
        if (result.dims == 3) {
            for (int z = 0; z < result.size[0]; ++z) {
                for (int y = 0; y < result.size[1]; ++y) {
                    for (int x = 0; x < result.size[2]; ++x) {
                        int label = result.at<uchar>(z, y, x);
                        if (label >= 0 && label < 4) {
                            labelCounts[label]++;
                        }
                    }
                }
            }
        }
        
        qDebug() << "Segmentation result:";
        qDebug() << "  Dimensions:" << result.size[0] << "x" << result.size[1] << "x" << result.size[2];
        qDebug() << "  Background ratio:" << (labelCounts[0] * 100.0) / totalVoxels << "%";
        qDebug() << "  Structure ratio:" << ((totalVoxels - labelCounts[0]) * 100.0) / totalVoxels << "%";
    }
    
    qDebug() << "=== DIAGNOSIS COMPLETED ===";
}
#endif  // NDEBUG

// ▼ 新しい関数: 処理パイプライン全体の診断
cv::Mat OnnxSegmenter::preprocessCTSliceForModel(const cv::Mat &slice) {
    qDebug() << "=== EMERGENCY BYPASS PREPROCESSING ===";
    
    try {
        if (slice.empty()) {
            qWarning() << "Empty input slice";
            return cv::Mat();
        }
        
        qDebug() << "Input type:" << slice.type() << "Size:" << slice.cols << "x" << slice.rows;
        
        cv::Mat result;
        
        if (slice.type() == CV_32FC1) {
            qDebug() << "Input already CV_32FC1, cloning directly";
            result = slice.clone();
        } else {
            qDebug() << "Converting to CV_32FC1 using basic convertTo";
            slice.convertTo(result, CV_32FC1);
        }
        
        qDebug() << "Basic type conversion completed";
        qDebug() << "=== EMERGENCY BYPASS COMPLETED ===";
        return result;
        
    } catch (...) {
        qCritical() << "Emergency bypass failed, returning zero matrix";
        return cv::Mat::zeros(slice.size(), CV_32FC1);
    }
}

//=============================================================================
// Test Time Augmentation (TTA) Implementation
//=============================================================================

cv::Mat OnnxSegmenter::flipVolumeX(const cv::Mat &volume) {
    if (volume.dims != 3) {
        qWarning() << "[flipVolumeX] Input must be 3D volume";
        return cv::Mat();
    }

    int depth = volume.size[0];
    int height = volume.size[1];
    int width = volume.size[2];

    int sizes[3] = {depth, height, width};
    cv::Mat flipped(3, sizes, volume.type());

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int flippedX = width - 1 - x;
                if (volume.type() == CV_16SC1) {
                    flipped.at<int16_t>(z, y, flippedX) = volume.at<int16_t>(z, y, x);
                } else if (volume.type() == CV_32FC1) {
                    flipped.at<float>(z, y, flippedX) = volume.at<float>(z, y, x);
                } else if (volume.type() == CV_8UC1) {
                    flipped.at<uint8_t>(z, y, flippedX) = volume.at<uint8_t>(z, y, x);
                }
            }
        }
    }

    return flipped;
}

cv::Mat OnnxSegmenter::flipVolumeY(const cv::Mat &volume) {
    if (volume.dims != 3) {
        qWarning() << "[flipVolumeY] Input must be 3D volume";
        return cv::Mat();
    }

    int depth = volume.size[0];
    int height = volume.size[1];
    int width = volume.size[2];

    int sizes[3] = {depth, height, width};
    cv::Mat flipped(3, sizes, volume.type());

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int flippedY = height - 1 - y;
                if (volume.type() == CV_16SC1) {
                    flipped.at<int16_t>(z, flippedY, x) = volume.at<int16_t>(z, y, x);
                } else if (volume.type() == CV_32FC1) {
                    flipped.at<float>(z, flippedY, x) = volume.at<float>(z, y, x);
                } else if (volume.type() == CV_8UC1) {
                    flipped.at<uint8_t>(z, flippedY, x) = volume.at<uint8_t>(z, y, x);
                }
            }
        }
    }

    return flipped;
}

cv::Mat OnnxSegmenter::flipVolumeZ(const cv::Mat &volume) {
    if (volume.dims != 3) {
        qWarning() << "[flipVolumeZ] Input must be 3D volume";
        return cv::Mat();
    }

    int depth = volume.size[0];
    int height = volume.size[1];
    int width = volume.size[2];

    int sizes[3] = {depth, height, width};
    cv::Mat flipped(3, sizes, volume.type());

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int flippedZ = depth - 1 - z;
                if (volume.type() == CV_16SC1) {
                    flipped.at<int16_t>(flippedZ, y, x) = volume.at<int16_t>(z, y, x);
                } else if (volume.type() == CV_32FC1) {
                    flipped.at<float>(flippedZ, y, x) = volume.at<float>(z, y, x);
                } else if (volume.type() == CV_8UC1) {
                    flipped.at<uint8_t>(flippedZ, y, x) = volume.at<uint8_t>(z, y, x);
                }
            }
        }
    }

    return flipped;
}

std::string OnnxSegmenter::getPredictionQualityMode() const {
    const char* mode_env = std::getenv("SHIORIS_AI_QUALITY_MODE");
    if (mode_env) {
        std::string mode(mode_env);
        if (mode == "high" || mode == "ultra" || mode == "standard") {
            return mode;
        }
    }
    // デフォルトは高精度（TTA）モード。計算時間は増えるが精度向上を優先する。
    return "high";
}

cv::Mat OnnxSegmenter::predictVolumeWithTTA(const cv::Mat &volumeInput, const int *originalSize) {
    qDebug() << "=== TTA (Test Time Augmentation) INFERENCE ===";

    auto start = std::chrono::high_resolution_clock::now();

    // 1. 元画像で推論
    qDebug() << "[TTA] Inference 1/4: Original orientation";
    updateProgress(0.3f, "TTA: Inference 1/4 (Original)");
    cv::Mat result1 = predictVolume(volumeInput, originalSize);
    if (result1.empty()) {
        qWarning() << "[TTA] Original inference failed";
        return cv::Mat();
    }

    // 2. X軸反転（左右反転）
    qDebug() << "[TTA] Inference 2/4: Flip X (left-right)";
    updateProgress(0.45f, "TTA: Inference 2/4 (Flip X)");
    cv::Mat flippedX = flipVolumeX(volumeInput);
    cv::Mat result2 = predictVolume(flippedX, originalSize);
    if (!result2.empty()) {
        result2 = flipVolumeX(result2);  // 結果も反転して戻す
    }

    // 3. Y軸反転（上下反転）
    qDebug() << "[TTA] Inference 3/4: Flip Y (up-down)";
    updateProgress(0.6f, "TTA: Inference 3/4 (Flip Y)");
    cv::Mat flippedY = flipVolumeY(volumeInput);
    cv::Mat result3 = predictVolume(flippedY, originalSize);
    if (!result3.empty()) {
        result3 = flipVolumeY(result3);  // 結果も反転して戻す
    }

    // 4. Z軸反転（前後反転）
    qDebug() << "[TTA] Inference 4/4: Flip Z (front-back)";
    updateProgress(0.75f, "TTA: Inference 4/4 (Flip Z)");
    cv::Mat flippedZ = flipVolumeZ(volumeInput);
    cv::Mat result4 = predictVolume(flippedZ, originalSize);
    if (!result4.empty()) {
        result4 = flipVolumeZ(result4);  // 結果も反転して戻す
    }

    // アンサンブル統合（投票方式）
    qDebug() << "[TTA] Ensemble integration (voting)";
    updateProgress(0.85f, "TTA: Ensemble integration");

    int depth = result1.size[0];
    int height = result1.size[1];
    int width = result1.size[2];

    int sizes[3] = {depth, height, width};
    cv::Mat ensemble(3, sizes, CV_8UC1);

    // 環境変数でアンサンブル方式を選択
    const char* ensemble_method_env = std::getenv("SHIORIS_TTA_ENSEMBLE_METHOD");
    std::string ensemble_method = ensemble_method_env ? std::string(ensemble_method_env) : "vote";

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                uint8_t val1 = result1.at<uint8_t>(z, y, x);
                uint8_t val2 = result2.empty() ? val1 : result2.at<uint8_t>(z, y, x);
                uint8_t val3 = result3.empty() ? val1 : result3.at<uint8_t>(z, y, x);
                uint8_t val4 = result4.empty() ? val1 : result4.at<uint8_t>(z, y, x);

                uint8_t finalValue;
                if (ensemble_method == "average") {
                    // 平均値
                    finalValue = static_cast<uint8_t>((static_cast<int>(val1) + val2 + val3 + val4) / 4);
                } else {
                    // 投票（最頻値）
                    std::unordered_map<uint8_t, int> votes;
                    votes[val1]++;
                    votes[val2]++;
                    votes[val3]++;
                    votes[val4]++;

                    uint8_t maxVote = val1;
                    int maxCount = votes[val1];
                    for (const auto &pair : votes) {
                        if (pair.second > maxCount) {
                            maxCount = pair.second;
                            maxVote = pair.first;
                        }
                    }
                    finalValue = maxVote;
                }

                ensemble.at<uint8_t>(z, y, x) = finalValue;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    qDebug() << "[TTA] Total time:" << duration.count() << "ms";
    qDebug() << "=== TTA INFERENCE COMPLETED ===";

    return ensemble;
}

cv::Mat OnnxSegmenter::predictVolumeWithSlidingWindow(const cv::Mat &volumeInput, const int *originalSize, float overlap) {
    qDebug() << "=== SLIDING WINDOW INFERENCE ===";
    qDebug() << "[SlidingWindow] Overlap:" << overlap;

    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat workingVolume = volumeInput;

    int inputDepth = workingVolume.size[0];
    int inputHeight = workingVolume.size[1];
    int inputWidth = workingVolume.size[2];

    // モデルの期待サイズ取得
    int windowDepth = static_cast<int>(m_inputDims[2]);
    int windowHeight = static_cast<int>(m_inputDims[3]);
    int windowWidth = static_cast<int>(m_inputDims[4]);

    qDebug() << "[SlidingWindow] Input size:" << inputDepth << "x" << inputHeight << "x" << inputWidth;
    qDebug() << "[SlidingWindow] Window size:" << windowDepth << "x" << windowHeight << "x" << windowWidth;

    // 入力がモデルウィンドウと同じサイズの場合、スライディングが1タイルで終わってしまう。
    // ULTRAモードの長考を強制するため、周囲にリフレクションパディングを施して
    // 複数タイルのアンサンブルが必ず走るようにする。
    int padZ = 0, padY = 0, padX = 0;  // パディングオフセットを保存
    if (inputDepth <= windowDepth && inputHeight <= windowHeight && inputWidth <= windowWidth) {
        padZ = std::max(1, windowDepth / 4);
        padY = std::max(1, windowHeight / 4);
        padX = std::max(1, windowWidth / 4);

        int paddedSizes[3] = {inputDepth + 2 * padZ, inputHeight + 2 * padY, inputWidth + 2 * padX};
        cv::Mat paddedVolume(3, paddedSizes, workingVolume.type());

        auto reflectIndex = [](int idx, int maxIdx) {
            if (idx < 0) return -idx - 1;
            if (idx >= maxIdx) return 2 * maxIdx - idx - 1;
            return idx;
        };

        for (int z = 0; z < paddedSizes[0]; ++z) {
            int srcZ = reflectIndex(z - padZ, inputDepth);
            for (int y = 0; y < paddedSizes[1]; ++y) {
                int srcY = reflectIndex(y - padY, inputHeight);
                for (int x = 0; x < paddedSizes[2]; ++x) {
                    int srcX = reflectIndex(x - padX, inputWidth);

                    if (workingVolume.type() == CV_16SC1) {
                        paddedVolume.at<int16_t>(z, y, x) = workingVolume.at<int16_t>(srcZ, srcY, srcX);
                    } else if (workingVolume.type() == CV_32FC1) {
                        paddedVolume.at<float>(z, y, x) = workingVolume.at<float>(srcZ, srcY, srcX);
                    } else if (workingVolume.type() == CV_8UC1) {
                        paddedVolume.at<uint8_t>(z, y, x) = workingVolume.at<uint8_t>(srcZ, srcY, srcX);
                    }
                }
            }
        }

        workingVolume = paddedVolume;
        inputDepth = workingVolume.size[0];
        inputHeight = workingVolume.size[1];
        inputWidth = workingVolume.size[2];

        qDebug() << "[SlidingWindow] Applied reflection padding to enforce multi-tile inference:";
        qDebug() << "  Original:" << volumeInput.size[0] << "x" << volumeInput.size[1] << "x" << volumeInput.size[2];
        qDebug() << "  Padded:  " << inputDepth << "x" << inputHeight << "x" << inputWidth;
        qDebug() << "  Padding offset: (" << padZ << "," << padY << "," << padX << ")";
    }

    // ストライド計算（オーバーラップを考慮）
    int strideDepth = static_cast<int>(windowDepth * (1.0f - overlap));
    int strideHeight = static_cast<int>(windowHeight * (1.0f - overlap));
    int strideWidth = static_cast<int>(windowWidth * (1.0f - overlap));

    strideDepth = std::max(1, strideDepth);
    strideHeight = std::max(1, strideHeight);
    strideWidth = std::max(1, strideWidth);

    qDebug() << "[SlidingWindow] Stride: D=" << strideDepth << " H=" << strideHeight << " W=" << strideWidth;

    // 累積用のマトリクス（確率の合計と重み）
    int origSizes[3] = {originalSize[0], originalSize[1], originalSize[2]};
    cv::Mat sumProb(3, origSizes, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat weightSum(3, origSizes, CV_32FC1, cv::Scalar(0.0f));

    // ウィンドウ数をカウント
    int windowCount = 0;
    for (int z = 0; z + windowDepth <= inputDepth; z += strideDepth) {
        for (int y = 0; y + windowHeight <= inputHeight; y += strideHeight) {
            for (int x = 0; x + windowWidth <= inputWidth; x += strideWidth) {
                windowCount++;
            }
        }
    }

    qDebug() << "[SlidingWindow] Total windows:" << windowCount;

    int currentWindow = 0;

    // スライディングウィンドウで推論
    for (int z = 0; z + windowDepth <= inputDepth; z += strideDepth) {
        for (int y = 0; y + windowHeight <= inputHeight; y += strideHeight) {
            for (int x = 0; x + windowWidth <= inputWidth; x += strideWidth) {
                currentWindow++;
                qDebug() << "[SlidingWindow] Processing window" << currentWindow << "/" << windowCount
                         << "at position (" << z << "," << y << "," << x << ")";

                // ウィンドウ領域を抽出
                int windowSizes[3] = {windowDepth, windowHeight, windowWidth};
                cv::Mat windowVolume(3, windowSizes, workingVolume.type());

                for (int wz = 0; wz < windowDepth; ++wz) {
                    for (int wy = 0; wy < windowHeight; ++wy) {
                        for (int wx = 0; wx < windowWidth; ++wx) {
                            if (workingVolume.type() == CV_16SC1) {
                                windowVolume.at<int16_t>(wz, wy, wx) = workingVolume.at<int16_t>(z + wz, y + wy, x + wx);
                            } else if (workingVolume.type() == CV_32FC1) {
                                windowVolume.at<float>(wz, wy, wx) = workingVolume.at<float>(z + wz, y + wy, x + wx);
                            } else if (workingVolume.type() == CV_8UC1) {
                                windowVolume.at<uint8_t>(wz, wy, wx) = workingVolume.at<uint8_t>(z + wz, y + wy, x + wx);
                            }
                        }
                    }
                }

                // このウィンドウで推論
                cv::Mat windowResult = predictVolume(windowVolume, originalSize);

                if (windowResult.empty()) {
                    qWarning() << "[SlidingWindow] Window inference failed, skipping";
                    continue;
                }

                // ガウシアン重み（中心が高く、端が低い）
                // ウィンドウ結果を正しい位置に配置（ウィンドウ座標とパディングオフセットを考慮）
                for (int wz = 0; wz < windowResult.size[0]; ++wz) {
                    for (int wy = 0; wy < windowResult.size[1]; ++wy) {
                        for (int wx = 0; wx < windowResult.size[2]; ++wx) {
                            // パディング後の座標系での位置を計算
                            int paddedZ = z + wz;
                            int paddedY = y + wy;
                            int paddedX = x + wx;

                            // パディングオフセットを引いて元のボリュームの座標系に変換
                            int globalZ = paddedZ - padZ;
                            int globalY = paddedY - padY;
                            int globalX = paddedX - padX;

                            // 境界チェック：累積バッファの範囲内かどうか
                            if (globalZ < 0 || globalZ >= origSizes[0] ||
                                globalY < 0 || globalY >= origSizes[1] ||
                                globalX < 0 || globalX >= origSizes[2]) {
                                continue;  // パディング領域は無視
                            }

                            // ウィンドウ内での中心からの距離に基づく重み
                            float distZ = std::abs(wz - windowResult.size[0] / 2.0f) / (windowResult.size[0] / 2.0f);
                            float distY = std::abs(wy - windowResult.size[1] / 2.0f) / (windowResult.size[1] / 2.0f);
                            float distX = std::abs(wx - windowResult.size[2] / 2.0f) / (windowResult.size[2] / 2.0f);
                            float dist = std::sqrt(distZ * distZ + distY * distY + distX * distX);
                            float weight = std::exp(-dist * dist / 2.0f);  // ガウシアン重み

                            uint8_t label = windowResult.at<uint8_t>(wz, wy, wx);
                            sumProb.at<float>(globalZ, globalY, globalX) += label * weight;
                            weightSum.at<float>(globalZ, globalY, globalX) += weight;
                        }
                    }
                }
            }
        }
    }

    // 加重平均を計算
    qDebug() << "[SlidingWindow] Computing weighted average";
    cv::Mat result(3, origSizes, CV_8UC1);

    for (int z = 0; z < originalSize[0]; ++z) {
        for (int y = 0; y < originalSize[1]; ++y) {
            for (int x = 0; x < originalSize[2]; ++x) {
                float weight = weightSum.at<float>(z, y, x);
                if (weight > 0) {
                    float avgProb = sumProb.at<float>(z, y, x) / weight;
                    result.at<uint8_t>(z, y, x) = static_cast<uint8_t>(std::round(avgProb));
                } else {
                    result.at<uint8_t>(z, y, x) = 0;
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    qDebug() << "[SlidingWindow] Total time:" << duration.count() << "ms";
    qDebug() << "=== SLIDING WINDOW INFERENCE COMPLETED ===";

    return result;
}

cv::Mat OnnxSegmenter::predictVolumeUltra(const cv::Mat &volumeInput, const int *originalSize, float overlap) {
    qDebug() << "=== ULTRA QUALITY INFERENCE (TTA + Sliding Window) ===";

    auto start = std::chrono::high_resolution_clock::now();

    struct FlipTransform {
        QString name;
        std::function<cv::Mat(const cv::Mat &)> forward;
        std::function<cv::Mat(const cv::Mat &)> inverse;
    };

    // 8方向のTTA（オリジナル + 全軸の組み合わせフリップ）
    std::vector<FlipTransform> transforms = {
        {"Original", [](const cv::Mat &v) { return v; }, [](const cv::Mat &v) { return v; }},
        {"Flip X", [this](const cv::Mat &v) { return flipVolumeX(v); }, [this](const cv::Mat &v) { return flipVolumeX(v); }},
        {"Flip Y", [this](const cv::Mat &v) { return flipVolumeY(v); }, [this](const cv::Mat &v) { return flipVolumeY(v); }},
        {"Flip Z", [this](const cv::Mat &v) { return flipVolumeZ(v); }, [this](const cv::Mat &v) { return flipVolumeZ(v); }},
        {"Flip XY", [this](const cv::Mat &v) { return flipVolumeY(flipVolumeX(v)); }, [this](const cv::Mat &v) { return flipVolumeX(flipVolumeY(v)); }},
        {"Flip XZ", [this](const cv::Mat &v) { return flipVolumeZ(flipVolumeX(v)); }, [this](const cv::Mat &v) { return flipVolumeX(flipVolumeZ(v)); }},
        {"Flip YZ", [this](const cv::Mat &v) { return flipVolumeZ(flipVolumeY(v)); }, [this](const cv::Mat &v) { return flipVolumeY(flipVolumeZ(v)); }},
        {"Flip XYZ", [this](const cv::Mat &v) { return flipVolumeZ(flipVolumeY(flipVolumeX(v))); }, [this](const cv::Mat &v) { return flipVolumeX(flipVolumeY(flipVolumeZ(v))); }}
    };

    std::vector<cv::Mat> results;
    results.reserve(transforms.size());

    int depth = originalSize[0];
    int height = originalSize[1];
    int width = originalSize[2];

    int sizes[3] = {depth, height, width};
    cv::Mat ensemble(3, sizes, CV_8UC1, cv::Scalar(0));

    for (size_t idx = 0; idx < transforms.size(); ++idx) {
        const auto &transform = transforms[idx];
        float progress = 0.3f + (idx * 0.5f / transforms.size());
        qDebug() << "[Ultra] Sliding Window" << (idx + 1) << "/" << transforms.size() << ":" << transform.name;
        updateProgress(progress, QString("Ultra: SW %1/%2 (%3)").arg(idx + 1).arg(transforms.size()).arg(transform.name).toStdString());

        cv::Mat augmentedVolume = transform.forward(volumeInput);
        cv::Mat augmentedResult = predictVolumeWithSlidingWindow(augmentedVolume, originalSize, overlap);

        if (augmentedResult.empty()) {
            qWarning() << "[Ultra]" << transform.name << "inference returned empty result, skipping";
            results.emplace_back();
            continue;
        }

        cv::Mat restoredResult = transform.inverse(augmentedResult);
        results.push_back(restoredResult);
    }

    // 最終アンサンブル
    qDebug() << "[Ultra] Final ensemble integration";
    updateProgress(0.85f, "Ultra: Ensemble integration");

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                std::unordered_map<uint8_t, int> votes;

                for (const auto &res : results) {
                    if (!res.empty()) {
                        uint8_t val = res.at<uint8_t>(z, y, x);
                        votes[val]++;
                    }
                }

                uint8_t finalLabel = 0;
                int maxCount = -1;
                for (const auto &pair : votes) {
                    if (pair.second > maxCount) {
                        maxCount = pair.second;
                        finalLabel = pair.first;
                    }
                }

                ensemble.at<uint8_t>(z, y, x) = finalLabel;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    qDebug() << "[Ultra] Total time:" << duration.count() << "ms (" << (duration.count() / 1000.0) << " seconds)";
    qDebug() << "=== ULTRA QUALITY INFERENCE COMPLETED ===";

    return ensemble;
}

#else
// ONNX Runtime が無効な場合のスタブ実装
OnnxSegmenter::OnnxSegmenter() {}
bool OnnxSegmenter::loadModel(const std::string &) { return false; }
bool OnnxSegmenter::isLoaded() const { return false; }
cv::Mat OnnxSegmenter::predict(const cv::Mat &) { return cv::Mat(); }
cv::Mat OnnxSegmenter::buildInputVolume(const cv::Mat &) { return cv::Mat(); }
cv::Mat OnnxSegmenter::predictVolume(const cv::Mat &, const int *) { return cv::Mat(); }
cv::Mat OnnxSegmenter::preprocessCTSlice(const cv::Mat &) { return cv::Mat(); }
cv::Mat OnnxSegmenter::preprocessCTSliceForModel(const cv::Mat &) { return cv::Mat(); }
cv::Mat OnnxSegmenter::preprocessCTSliceAlternative(const cv::Mat &) { return cv::Mat(); }
cv::Mat OnnxSegmenter::processAbdomenSegmentationOutput(Ort::Value &, const int *) { return cv::Mat(); }
cv::Mat OnnxSegmenter::processOutput4D(Ort::Value &, const std::vector<int64_t> &, const cv::Size &) { return cv::Mat(); }
cv::Mat OnnxSegmenter::resampleVolumeFor3D(const cv::Mat &) { return cv::Mat(); }
cv::Mat OnnxSegmenter::process3DSegmentationOutput(Ort::Value &, const int *) { return cv::Mat(); }
cv::Mat OnnxSegmenter::resample3DToOriginalSize(const cv::Mat &, const int *) { return cv::Mat(); }
bool OnnxSegmenter::updateInputDimensions(const cv::Mat &) { return false; }
void OnnxSegmenter::diagnoseProcessingPipeline(const cv::Mat &, const cv::Mat &, const cv::Mat &) {}
cv::Mat OnnxSegmenter::flipVolumeX(const cv::Mat &) { return cv::Mat(); }
cv::Mat OnnxSegmenter::flipVolumeY(const cv::Mat &) { return cv::Mat(); }
cv::Mat OnnxSegmenter::flipVolumeZ(const cv::Mat &) { return cv::Mat(); }
cv::Mat OnnxSegmenter::predictVolumeWithTTA(const cv::Mat &, const int *) { return cv::Mat(); }
cv::Mat OnnxSegmenter::predictVolumeWithSlidingWindow(const cv::Mat &, const int *, float) { return cv::Mat(); }
cv::Mat OnnxSegmenter::predictVolumeUltra(const cv::Mat &, const int *, float) { return cv::Mat(); }
std::string OnnxSegmenter::getPredictionQualityMode() const { return "high"; }

#endif
