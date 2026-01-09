#include "ai/linux_auto_segmenter.h"
#include "ai/onnx_segmenter.h"
#include <QProcess>
#include <QDebug>
#include <QFile>
#include <QFileInfo>

#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

LinuxAutoSegmenter::LinuxAutoSegmenter()
    : m_segmenter(std::make_unique<OnnxSegmenter>())
    , m_modelLoaded(false)
{
}

LinuxAutoSegmenter::~LinuxAutoSegmenter()
{
}

bool LinuxAutoSegmenter::loadModel(const QString &modelPath)
{
    m_lastError.clear();

    // モデルファイルの存在確認
    QFileInfo fileInfo(modelPath);
    if (!fileInfo.exists()) {
        m_lastError = QString("Model file not found: %1").arg(modelPath);
        qWarning() << m_lastError;
        m_modelLoaded = false;
        return false;
    }

    if (!fileInfo.isReadable()) {
        m_lastError = QString("Model file not readable: %1").arg(modelPath);
        qWarning() << m_lastError;
        m_modelLoaded = false;
        return false;
    }

    // OnnxSegmenterでモデルをロード
    try {
        bool success = m_segmenter->loadModel(modelPath.toStdString());
        if (!success) {
            m_lastError = "Failed to load ONNX model";
            qWarning() << m_lastError;
            m_modelLoaded = false;
            return false;
        }

        m_modelLoaded = true;
        qInfo() << "Successfully loaded ONNX model:" << modelPath;
        return true;

    } catch (const std::exception &e) {
        m_lastError = QString("Exception while loading model: %1").arg(e.what());
        qWarning() << m_lastError;
        m_modelLoaded = false;
        return false;
    }
}

bool LinuxAutoSegmenter::isModelLoaded() const
{
    return m_modelLoaded && m_segmenter && m_segmenter->isLoaded();
}

QFuture<SegmentationResult> LinuxAutoSegmenter::segmentVolumeAsync(
    const cv::Mat &volume,
    ProgressCallback progressCallback)
{
    return QtConcurrent::run([this, volume, progressCallback]() -> SegmentationResult {
        SegmentationResult result;

        // モデルがロードされているかチェック
        if (!isModelLoaded()) {
            result.success = false;
            result.errorMessage = "Model not loaded";
            qWarning() << result.errorMessage;
            return result;
        }

        // 入力ボリュームの検証
        if (volume.empty()) {
            result.success = false;
            result.errorMessage = "Input volume is empty";
            qWarning() << result.errorMessage;
            return result;
        }

        if (volume.dims < 3) {
            result.success = false;
            result.errorMessage = QString("Invalid volume dimensions: %1 (expected 3D)").arg(volume.dims);
            qWarning() << result.errorMessage;
            return result;
        }

        try {
            // 進捗: 開始
            if (progressCallback) {
                progressCallback(10);
            }

            qInfo() << "Starting segmentation for volume:"
                    << "dims=" << volume.dims
                    << "size=" << volume.size[0] << "x" << volume.size[1] << "x" << volume.size[2]
                    << "type=" << volume.type();

            // 進捗: 前処理完了
            if (progressCallback) {
                progressCallback(30);
            }

            // OnnxSegmenterで3Dセグメンテーションを実行
            cv::Mat segmentationMask = m_segmenter->predict3D(volume);

            // 進捗: 推論完了
            if (progressCallback) {
                progressCallback(80);
            }

            // 結果の検証
            if (segmentationMask.empty()) {
                result.success = false;
                result.errorMessage = "Segmentation returned empty result";
                qWarning() << result.errorMessage;
                return result;
            }

            qInfo() << "Segmentation completed successfully:"
                    << "result dims=" << segmentationMask.dims
                    << "size=" << segmentationMask.size[0] << "x"
                    << segmentationMask.size[1] << "x"
                    << segmentationMask.size[2];

            // 結果を設定
            result.mask = segmentationMask;
            result.success = true;

            // 進捗: 完了
            if (progressCallback) {
                progressCallback(100);
            }

            return result;

        } catch (const std::exception &e) {
            result.success = false;
            result.errorMessage = QString("Exception during segmentation: %1").arg(e.what());
            qWarning() << result.errorMessage;
            return result;
        } catch (...) {
            result.success = false;
            result.errorMessage = "Unknown exception during segmentation";
            qWarning() << result.errorMessage;
            return result;
        }
    });
}

bool LinuxAutoSegmenter::validateEnvironment()
{
    m_lastError.clear();

#ifndef USE_ONNXRUNTIME
    m_lastError = "ONNX Runtime support not compiled in";
    qWarning() << m_lastError;
    return false;
#else
    try {
        // ONNX Runtimeのバージョン情報を取得
        std::string version = Ort::GetVersionString();
        qInfo() << "ONNX Runtime version:" << version.c_str();

        // 利用可能なプロバイダーを取得
        Ort::SessionOptions options;
        auto available_providers = Ort::GetAvailableProviders();

        qInfo() << "Available execution providers:";
        for (const auto& provider : available_providers) {
            qInfo() << "  -" << provider.c_str();
        }

        // 基本的な環境チェック成功
        return true;

    } catch (const Ort::Exception &e) {
        m_lastError = QString("ONNX Runtime error: %1").arg(e.what());
        qWarning() << m_lastError;
        return false;
    } catch (const std::exception &e) {
        m_lastError = QString("Exception during environment validation: %1").arg(e.what());
        qWarning() << m_lastError;
        return false;
    }
#endif
}

QString LinuxAutoSegmenter::getGPUInfo()
{
    qDebug() << "=== GPU Information Detection ===";

    // nvidia-smiの複数のパスを試行
    QStringList nvidiaSmiPaths = {
        "/usr/bin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
        "/bin/nvidia-smi",
        "nvidia-smi"  // PATH内を検索
    };

    QString nvidiaSmiPath;
    for (const QString& path : nvidiaSmiPaths) {
        QFileInfo fileInfo(path);
        if (fileInfo.exists() && fileInfo.isExecutable()) {
            nvidiaSmiPath = path;
            qDebug() << "Found nvidia-smi at:" << nvidiaSmiPath;
            break;
        }
    }

    if (nvidiaSmiPath.isEmpty()) {
        // PATHから検索
        QProcess which;
        which.start("which", QStringList() << "nvidia-smi");
        if (which.waitForFinished(1000)) {
            nvidiaSmiPath = QString::fromLocal8Bit(which.readAllStandardOutput()).trimmed();
            if (!nvidiaSmiPath.isEmpty()) {
                qDebug() << "Found nvidia-smi via which:" << nvidiaSmiPath;
            }
        }
    }

    if (nvidiaSmiPath.isEmpty()) {
        qWarning() << "nvidia-smi not found in common paths";
        qDebug() << "GPU detection: FAILED (nvidia-smi not available)";
        return QStringLiteral("CPU");
    }

    // nvidia-smiコマンドを実行してGPU情報を取得
    QProcess process;
    process.start(nvidiaSmiPath, QStringList()
        << "--query-gpu=name,driver_version,memory.total,compute_cap"
        << "--format=csv,noheader");

    if (!process.waitForStarted(2000)) {
        qWarning() << "Failed to start nvidia-smi";
        qDebug() << "GPU detection: FAILED (cannot start nvidia-smi)";
        return QStringLiteral("CPU");
    }

    if (!process.waitForFinished(3000)) {
        qWarning() << "nvidia-smi timeout";
        process.kill();
        qDebug() << "GPU detection: FAILED (timeout)";
        return QStringLiteral("CPU");
    }

    if (process.exitCode() != 0) {
        QString errorOutput = QString::fromLocal8Bit(process.readAllStandardError());
        qWarning() << "nvidia-smi failed with exit code:" << process.exitCode();
        qWarning() << "Error output:" << errorOutput;
        qDebug() << "GPU detection: FAILED (exit code" << process.exitCode() << ")";
        return QStringLiteral("CPU");
    }

    QString output = QString::fromLocal8Bit(process.readAllStandardOutput()).trimmed();

    if (output.isEmpty()) {
        qWarning() << "nvidia-smi returned empty output";
        qDebug() << "GPU detection: FAILED (empty output)";
        return QStringLiteral("CPU");
    }

    qDebug() << "nvidia-smi output:" << output;

    // GPU情報をパース
    QStringList lines = output.split('\n');
    if (lines.isEmpty()) {
        qDebug() << "GPU detection: FAILED (no lines in output)";
        return QStringLiteral("CPU");
    }

    // 最初のGPUの情報を取得
    QString firstGPU = lines.first().trimmed();
    QStringList parts = firstGPU.split(',');

    QString result;
    if (parts.size() >= 4) {
        QString name = parts[0].trimmed();
        QString driver = parts[1].trimmed();
        QString memory = parts[2].trimmed();
        QString computeCap = parts[3].trimmed();

        result = QString("GPU: %1 (Driver: %2, Memory: %3, Compute: %4)")
            .arg(name)
            .arg(driver)
            .arg(memory)
            .arg(computeCap);
    } else if (parts.size() >= 3) {
        QString name = parts[0].trimmed();
        QString driver = parts[1].trimmed();
        QString memory = parts[2].trimmed();

        result = QString("GPU: %1 (Driver: %2, Memory: %3)")
            .arg(name)
            .arg(driver)
            .arg(memory);
    } else if (parts.size() >= 1) {
        result = QString("GPU: %1").arg(parts[0].trimmed());
    } else {
        qDebug() << "GPU detection: FAILED (invalid format)";
        return QStringLiteral("CPU");
    }

    qDebug() << "GPU detection: SUCCESS";
    qDebug() << "GPU Info:" << result;
    qDebug() << "=== GPU Information Detection Complete ===";

    return result;
}

QString LinuxAutoSegmenter::getLastError() const
{
    return m_lastError;
}
