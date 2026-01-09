// パス: src/ai/whisper_client.cpp
#include "ai/whisper_client.h"
#include "whisper.h"
#include <QFile>
#include <QDir>
#include <QStandardPaths>
#include <QDebug>
#include <cmath>
#include <algorithm>

WhisperClient::WhisperClient(QObject* parent)
    : QObject(parent)
{
}

WhisperClient::~WhisperClient() {
    if (m_context) {
        whisper_free(m_context);
        m_context = nullptr;
    }
}

bool WhisperClient::loadModel(const QString& modelPath) {
    // 既存のモデルを解放
    if (m_context) {
        whisper_free(m_context);
        m_context = nullptr;
    }

    // モデルファイルの存在確認
    if (!QFile::exists(modelPath)) {
        QString errorMsg = QString("Whisper model file not found: %1").arg(modelPath);
        qWarning() << errorMsg;
        emit error(errorMsg);
        emit modelLoaded(false);
        return false;
    }

    // モデルをロード
    qDebug() << "Loading Whisper model:" << modelPath;
    auto cparams = whisper_context_default_params();
    m_context = whisper_init_from_file_with_params(modelPath.toUtf8().constData(), cparams);

    if (!m_context) {
        QString errorMsg = QString("Failed to load Whisper model: %1").arg(modelPath);
        qWarning() << errorMsg;
        emit error(errorMsg);
        emit modelLoaded(false);
        return false;
    }

    m_modelPath = modelPath;
    qDebug() << "Whisper model loaded successfully";
    emit modelLoaded(true);
    return true;
}

bool WhisperClient::loadDefaultModel(ModelSize size) {
    QString modelPath = getDefaultModelPath(size);
    return loadModel(modelPath);
}

bool WhisperClient::loadAnyAvailableModel() {
    // 試すモデルサイズの優先順位（精度が高いものから、ただしLargeは遅すぎるので後回し）
    // Base/Smallは精度とスピードのバランスが良い
    QVector<ModelSize> modelSizes = {
        ModelSize::Base,   // 精度とスピードのバランスが良い（推奨）
        ModelSize::Small,  // より高精度だが遅い
        ModelSize::Tiny,   // 最速だが精度は低い（フォールバック用）
        ModelSize::Medium, // 高精度だがかなり遅い
        ModelSize::Large   // 最高精度だが非常に遅い
    };

    QStringList attemptedPaths;

    // 各モデルサイズについて、複数のパスを試す
    for (ModelSize size : modelSizes) {
        QVector<QString> pathsToTry;

        // 標準パス
        QString defaultPath = getDefaultModelPath(size);
        pathsToTry.append(defaultPath);

        // 代替パス（組織名なし）
        QString modelFileName;
        switch (size) {
            case ModelSize::Tiny:
                modelFileName = "ggml-tiny.bin";
                break;
            case ModelSize::Base:
                modelFileName = "ggml-base.bin";
                break;
            case ModelSize::Small:
                modelFileName = "ggml-small.bin";
                break;
            case ModelSize::Medium:
                modelFileName = "ggml-medium.bin";
                break;
            case ModelSize::Large:
                modelFileName = "ggml-large-v3.bin";
                break;
        }

#ifdef Q_OS_MAC
        // macOSの場合、組織名なしのパスも試す
        QString homeDir = QDir::homePath();
        QString altPath = homeDir + "/Library/Application Support/ShioRIS3/whisper/models/" + modelFileName;
        pathsToTry.append(altPath);
#endif

        // 各パスを試す（ファイルが存在する場合のみ）
        for (const QString& path : pathsToTry) {
            attemptedPaths.append(path);
            if (QFile::exists(path)) {
                qDebug() << "Found Whisper model file:" << path;
                if (loadModel(path)) {
                    qDebug() << "Successfully loaded Whisper model:" << modelSizeToString(size);
                    return true;
                }
            }
        }
    }

    // すべて失敗 - 試したパスを含む詳細なエラーメッセージ
    QString errorMsg = "No Whisper model found. Attempted paths:\n";
    for (const QString& path : attemptedPaths) {
        errorMsg += "  - " + path + "\n";
    }
    qWarning() << errorMsg;
    emit error(errorMsg);
    emit modelLoaded(false);
    return false;
}

bool WhisperClient::isModelLoaded() const {
    return m_context != nullptr;
}

QString WhisperClient::transcribeFromFile(const QString& audioFilePath) {
    if (!isModelLoaded()) {
        QString errorMsg = "Whisper model not loaded";
        qWarning() << errorMsg;
        emit error(errorMsg);
        return QString();
    }

    if (!QFile::exists(audioFilePath)) {
        QString errorMsg = QString("Audio file not found: %1").arg(audioFilePath);
        qWarning() << errorMsg;
        emit error(errorMsg);
        return QString();
    }

    // WAVファイルを読み込み
    QFile file(audioFilePath);
    if (!file.open(QIODevice::ReadOnly)) {
        QString errorMsg = QString("Failed to open audio file: %1").arg(audioFilePath);
        qWarning() << errorMsg;
        emit error(errorMsg);
        return QString();
    }

    QByteArray wavData = file.readAll();
    file.close();

    return transcribeFromWav(wavData);
}

QString WhisperClient::transcribeFromPCM(const QVector<float>& pcmData) {
    if (!isModelLoaded()) {
        QString errorMsg = "Whisper model not loaded";
        qWarning() << errorMsg;
        emit error(errorMsg);
        return QString();
    }

    if (pcmData.isEmpty()) {
        QString errorMsg = "Empty PCM data";
        qWarning() << errorMsg;
        emit error(errorMsg);
        return QString();
    }

    // Whisperパラメータを設定（BEAM_SEARCHで高精度化）
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
    params.beam_search.beam_size = 5;  // ビーム幅（精度とスピードのトレードオフ）

    // 言語設定
    if (m_language == Language::Japanese) {
        params.language = "ja";
    } else if (m_language == Language::English) {
        params.language = "en";
    } else {
        params.language = "auto";
    }

    params.print_progress = false;
    params.print_timestamps = m_timestampsEnabled;
    params.print_special = false;
    params.translate = false;
    params.single_segment = false;
    params.max_len = 0;  // セグメント長の制限なし

    // Hallucination（幻聴）対策パラメータ
    params.no_speech_thold = 0.6f;              // 無音検出の閾値（デフォルト: 0.6）
    params.entropy_thold = 2.4f;                // エントロピー閾値（高いと不確実な出力を抑制）
    params.logprob_thold = -1.0f;               // 確率閾値
    params.suppress_blank = true;               // 空白発話を抑制

    // Initial prompt（前回の文字起こし結果）を設定して文脈を維持
    QByteArray promptBytes = m_initialPrompt.toUtf8();
    if (!m_initialPrompt.isEmpty()) {
        params.initial_prompt = promptBytes.constData();
    }

    // 文字起こし実行
    qDebug() << "Starting Whisper transcription, samples:" << pcmData.size();
    int result = whisper_full(m_context, params, pcmData.constData(), pcmData.size());

    if (result != 0) {
        QString errorMsg = QString("Whisper transcription failed with code: %1").arg(result);
        qWarning() << errorMsg;
        emit error(errorMsg);
        return QString();
    }

    // 結果を取得
    QString transcription;
    const int n_segments = whisper_full_n_segments(m_context);

    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(m_context, i);
        if (text) {
            transcription += QString::fromUtf8(text);
        }
    }

    qDebug() << "Transcription completed:" << transcription;
    emit transcriptionReady(transcription);
    return transcription.trimmed();
}

QString WhisperClient::transcribeFromWav(const QByteArray& wavData) {
    if (wavData.isEmpty()) {
        QString errorMsg = "Empty WAV data";
        qWarning() << errorMsg;
        emit error(errorMsg);
        return QString();
    }

    // WAVからPCMデータを抽出
    QVector<float> pcmData = extractPCMFromWav(wavData);

    if (pcmData.isEmpty()) {
        QString errorMsg = "Failed to extract PCM data from WAV";
        qWarning() << errorMsg;
        emit error(errorMsg);
        return QString();
    }

    return transcribeFromPCM(pcmData);
}

void WhisperClient::setLanguage(Language lang) {
    m_language = lang;
}

WhisperClient::Language WhisperClient::getLanguage() const {
    return m_language;
}

void WhisperClient::setTimestampsEnabled(bool enable) {
    m_timestampsEnabled = enable;
}

void WhisperClient::setInitialPrompt(const QString& prompt) {
    m_initialPrompt = prompt;
}

QString WhisperClient::getInitialPrompt() const {
    return m_initialPrompt;
}

QString WhisperClient::getModelPath() const {
    return m_modelPath;
}

QString WhisperClient::getDefaultModelPath(ModelSize size) {
    // モデルの保存場所を決定（プラットフォーム依存）
    QDir modelsDir;

#ifdef Q_OS_MAC
    modelsDir = QDir(QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/whisper/models");
#elif defined(Q_OS_LINUX)
    modelsDir = QDir(QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/whisper/models");
#elif defined(Q_OS_WIN)
    modelsDir = QDir(QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/whisper/models");
#endif

    // モデルファイル名を決定
    QString modelFileName;
    switch (size) {
        case ModelSize::Tiny:
            modelFileName = "ggml-tiny.bin";
            break;
        case ModelSize::Base:
            modelFileName = "ggml-base.bin";
            break;
        case ModelSize::Small:
            modelFileName = "ggml-small.bin";
            break;
        case ModelSize::Medium:
            modelFileName = "ggml-medium.bin";
            break;
        case ModelSize::Large:
            modelFileName = "ggml-large-v3.bin";
            break;
    }

    return modelsDir.absoluteFilePath(modelFileName);
}

QString WhisperClient::modelSizeToString(ModelSize size) {
    switch (size) {
        case ModelSize::Tiny: return "tiny";
        case ModelSize::Base: return "base";
        case ModelSize::Small: return "small";
        case ModelSize::Medium: return "medium";
        case ModelSize::Large: return "large";
        default: return "base";
    }
}

WhisperClient::ModelSize WhisperClient::stringToModelSize(const QString& sizeStr) {
    QString lower = sizeStr.toLower();
    if (lower == "tiny") return ModelSize::Tiny;
    if (lower == "base") return ModelSize::Base;
    if (lower == "small") return ModelSize::Small;
    if (lower == "medium") return ModelSize::Medium;
    if (lower == "large") return ModelSize::Large;
    return ModelSize::Base;
}

QVector<float> WhisperClient::extractPCMFromWav(const QByteArray& wavData) {
    QVector<float> pcmData;

    // WAVヘッダーの最小サイズチェック（44バイト）
    if (wavData.size() < 44) {
        qWarning() << "WAV data too small";
        return pcmData;
    }

    // WAVヘッダー解析
    const char* data = wavData.constData();

    // RIFF header
    if (strncmp(data, "RIFF", 4) != 0) {
        qWarning() << "Invalid WAV file: missing RIFF header";
        return pcmData;
    }

    // WAVE header
    if (strncmp(data + 8, "WAVE", 4) != 0) {
        qWarning() << "Invalid WAV file: missing WAVE header";
        return pcmData;
    }

    // fmt chunk
    if (strncmp(data + 12, "fmt ", 4) != 0) {
        qWarning() << "Invalid WAV file: missing fmt chunk";
        return pcmData;
    }

    // オーディオフォーマット情報を取得
    int channels = *reinterpret_cast<const uint16_t*>(data + 22);
    int sampleRate = *reinterpret_cast<const uint32_t*>(data + 24);
    int bitsPerSample = *reinterpret_cast<const uint16_t*>(data + 34);

    qDebug() << "WAV format - Channels:" << channels
             << "Sample rate:" << sampleRate
             << "Bits per sample:" << bitsPerSample;

    // データチャンクを探す
    int dataOffset = 36;
    qDebug() << "Searching for data chunk, total WAV size:" << wavData.size() << "bytes";

    while (dataOffset < wavData.size() - 8) {
        QString chunkId = QString::fromLatin1(data + dataOffset, 4);
        uint32_t chunkSize = *reinterpret_cast<const uint32_t*>(data + dataOffset + 4);
        qDebug() << "Offset" << dataOffset << ": chunk '" << chunkId << "' size:" << chunkSize;

        if (strncmp(data + dataOffset, "data", 4) == 0) {
            qDebug() << "Found data chunk at offset:" << dataOffset;
            break;
        }
        // 現在のチャンクをスキップ: チャンクID(4) + チャンクサイズ(4) + チャンクデータ
        dataOffset += 8 + chunkSize;
    }

    if (dataOffset >= wavData.size() - 8) {
        qWarning() << "Invalid WAV file: data chunk not found. Final offset:" << dataOffset
                   << "WAV size:" << wavData.size();
        // ヘッダーの詳細をダンプ
        qWarning() << "WAV header dump (first 44 bytes):";
        for (int i = 0; i < qMin(44, wavData.size()); i += 4) {
            qWarning() << QString("  [%1]: %2 %3 %4 %5")
                .arg(i, 2)
                .arg((unsigned char)data[i], 2, 16, QChar('0'))
                .arg((unsigned char)data[i+1], 2, 16, QChar('0'))
                .arg((unsigned char)data[i+2], 2, 16, QChar('0'))
                .arg((unsigned char)data[i+3], 2, 16, QChar('0'));
        }
        return pcmData;
    }

    int dataSize = *reinterpret_cast<const uint32_t*>(data + dataOffset + 4);
    const char* audioData = data + dataOffset + 8;

    // PCMデータを16kHz monoのfloat配列に変換
    int sampleCount = dataSize / (bitsPerSample / 8) / channels;
    QVector<float> rawPcm(sampleCount);

    for (int i = 0; i < sampleCount; ++i) {
        float sample = 0.0f;

        if (bitsPerSample == 16) {
            // 16-bit PCM
            int16_t s = *reinterpret_cast<const int16_t*>(audioData + i * channels * 2);
            sample = static_cast<float>(s) / 32768.0f;
        } else if (bitsPerSample == 32) {
            // 32-bit PCM
            int32_t s = *reinterpret_cast<const int32_t*>(audioData + i * channels * 4);
            sample = static_cast<float>(s) / 2147483648.0f;
        }

        // ステレオの場合は平均を取る
        if (channels == 2) {
            float sample2 = 0.0f;
            if (bitsPerSample == 16) {
                int16_t s2 = *reinterpret_cast<const int16_t*>(audioData + i * channels * 2 + 2);
                sample2 = static_cast<float>(s2) / 32768.0f;
            } else if (bitsPerSample == 32) {
                int32_t s2 = *reinterpret_cast<const int32_t*>(audioData + i * channels * 4 + 4);
                sample2 = static_cast<float>(s2) / 2147483648.0f;
            }
            sample = (sample + sample2) / 2.0f;
        }

        rawPcm[i] = sample;
    }

    // 16kHzにリサンプル
    if (sampleRate != 16000) {
        pcmData = resampleTo16kHz(rawPcm, sampleRate);
    } else {
        pcmData = rawPcm;
    }

    qDebug() << "Extracted PCM samples:" << pcmData.size();
    return pcmData;
}

QVector<float> WhisperClient::resampleTo16kHz(const QVector<float>& pcmData, int originalSampleRate) {
    if (originalSampleRate == 16000) {
        return pcmData;
    }

    double ratio = 16000.0 / originalSampleRate;
    int newSize = static_cast<int>(pcmData.size() * ratio);
    QVector<float> resampled(newSize);

    // シンプルな線形補間によるリサンプリング
    for (int i = 0; i < newSize; ++i) {
        double srcIndex = i / ratio;
        int idx = static_cast<int>(srcIndex);
        double frac = srcIndex - idx;

        if (idx + 1 < pcmData.size()) {
            resampled[i] = pcmData[idx] * (1.0 - frac) + pcmData[idx + 1] * frac;
        } else {
            resampled[i] = pcmData[idx];
        }
    }

    qDebug() << "Resampled from" << originalSampleRate << "Hz to 16000Hz:"
             << pcmData.size() << "->" << resampled.size() << "samples";
    return resampled;
}
