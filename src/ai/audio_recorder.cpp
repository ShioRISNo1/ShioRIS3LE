// パス: src/ai/audio_recorder.cpp
#include "ai/audio_recorder.h"
#include <QAudioDevice>
#include <QMediaDevices>
#include <QDebug>
#include <QDateTime>
#include <QElapsedTimer>
#include <cmath>

AudioRecorder::AudioRecorder(QObject* parent)
    : QObject(parent)
{
    // すべての利用可能な入力デバイスを列挙
    qDebug() << "=== Available Audio Input Devices ===";
    const auto audioDevices = QMediaDevices::audioInputs();
    for (int i = 0; i < audioDevices.size(); ++i) {
        const QAudioDevice& device = audioDevices[i];
        qDebug() << "Device" << i << ":";
        qDebug() << "  Description:" << device.description();
        qDebug() << "  ID:" << device.id();
        qDebug() << "  isDefault:" << (device == QMediaDevices::defaultAudioInput());
        qDebug() << "  isNull:" << device.isNull();

        // サポートされているフォーマットの確認
        QAudioFormat testFormat;
        testFormat.setSampleRate(16000);
        testFormat.setChannelCount(1);
        testFormat.setSampleFormat(QAudioFormat::Int16);
        qDebug() << "  Supports 16kHz mono Int16:" << device.isFormatSupported(testFormat);

        // Preferred format の表示
        QAudioFormat preferred = device.preferredFormat();
        qDebug() << "  Preferred format:" << preferred.sampleRate() << "Hz,"
                 << preferred.channelCount() << "channels,"
                 << preferred.sampleFormat();
    }
    qDebug() << "=====================================";

    // デフォルトの入力デバイスを取得
    m_currentDevice = QMediaDevices::defaultAudioInput();
    if (m_currentDevice.isNull()) {
        qWarning() << "No audio input device available";
    } else {
        qDebug() << "Default audio input:" << m_currentDevice.description();
    }

    // チャンクタイマーの初期化
    m_chunkTimer = new QTimer(this);
    connect(m_chunkTimer, &QTimer::timeout, this, &AudioRecorder::handleChunkTimer);
}

AudioRecorder::~AudioRecorder() {
    if (m_isRecording) {
        stopRecording();
    }
}

bool AudioRecorder::startRecording() {
    if (m_isRecording) {
        qWarning() << "Already recording";
        return false;
    }

    if (m_currentDevice.isNull()) {
        QString errorMsg = "No audio input device available";
        qWarning() << errorMsg;
        emit error(errorMsg);
        return false;
    }

    // オーディオバッファをクリア
    m_audioBuffer.clear();
    m_fullAudioBuffer.clear();  // 全体バッファもクリア

    // デバイス情報をログ
    qDebug() << "Audio input device:" << m_currentDevice.description();

    // macOS対策: Preferred formatを使用する
    // isFormatSupported()がtrueでも実際には動作しないケースがあるため
    QAudioFormat format = m_currentDevice.preferredFormat();
    qDebug() << "Using device preferred format:" << format.sampleRate() << "Hz,"
             << format.channelCount() << "channels,"
             << format.sampleFormat();

    // モノラルに設定（Whisperはモノラルが必要）
    if (format.channelCount() != 1) {
        qDebug() << "Converting to mono (from" << format.channelCount() << "channels)";
        format.setChannelCount(1);
    }

    // フォーマット確認
    if (!m_currentDevice.isFormatSupported(format)) {
        qWarning() << "Preferred mono format not supported, trying alternatives...";

        // 元のPreferred formatに戻す
        format = m_currentDevice.preferredFormat();
        qDebug() << "Using original preferred format with all channels:"
                 << format.channelCount() << "channels";
    }

    // QAudioSourceを作成
    m_audioSource = std::make_unique<QAudioSource>(m_currentDevice, format, this);

    // 実際に使用されているフォーマットを保存
    m_recordingFormat = m_audioSource->format();
    qDebug() << "QAudioSource created with buffer size:" << m_audioSource->bufferSize();
    qDebug() << "Actual recording format:" << m_recordingFormat.sampleRate() << "Hz,"
             << m_recordingFormat.channelCount() << "channels,"
             << m_recordingFormat.sampleFormat();

    // バッファサイズを大きくする（macOS での問題回避）
    int desiredBufferSize = 32000; // 1秒分のバッファ（16kHz * 2 bytes）
    m_audioSource->setBufferSize(desiredBufferSize);
    qDebug() << "Buffer size set to:" << m_audioSource->bufferSize();

    // ボリュームを最大に設定（macOS での問題回避）
    m_audioSource->setVolume(1.0);
    qDebug() << "Volume set to:" << m_audioSource->volume();

    // 状態変化シグナルを接続
    connect(m_audioSource.get(), &QAudioSource::stateChanged,
            this, &AudioRecorder::handleStateChanged);

    // 録音開始
    m_audioInput = m_audioSource->start();

    if (!m_audioInput) {
        QString errorMsg = QString("Failed to start audio recording. Error: %1")
                              .arg(m_audioSource->error());
        qWarning() << errorMsg;
        emit error(errorMsg);
        m_audioSource.reset();
        return false;
    }

    // オーディオソースの状態確認
    qDebug() << "Audio source state:" << m_audioSource->state();
    qDebug() << "Audio source error:" << m_audioSource->error();
    qDebug() << "Audio input device open mode:" << m_audioInput->openMode();

    // データ受信シグナルを接続
    connect(m_audioInput, &QIODevice::readyRead,
            this, &AudioRecorder::handleAudioData, Qt::UniqueConnection);

    m_isRecording = true;
    m_recordingStartTime = QDateTime::currentMSecsSinceEpoch();
    m_lastChunkTime = m_recordingStartTime;

    // リアルタイム処理が有効な場合、チャンクタイマーを開始
    if (m_realtimeProcessing) {
        m_overlapBuffer.clear();
        m_chunkTimer->start(m_chunkIntervalMs);
        qDebug() << "Realtime processing enabled - chunk interval:" << m_chunkIntervalMs << "ms";
    }

    qDebug() << "Recording started successfully";
    qDebug() << "Initial check - bytes available:" << m_audioInput->bytesAvailable();
    qDebug() << "Audio source state after start:" << m_audioSource->state();
    qDebug() << "Audio source error after start:" << m_audioSource->error();

    // macOS対策: IdleStateの場合、resume()を試す
    if (m_audioSource->state() == QAudio::IdleState) {
        qDebug() << "Audio source is in IdleState, trying resume()...";
        m_audioSource->resume();
        qDebug() << "After resume() - state:" << m_audioSource->state();
    }

    // デバッグ: 500ms後に状態を再確認
    QTimer::singleShot(500, this, [this]() {
        if (m_isRecording && m_audioSource) {
            qDebug() << "=== 500ms after recording start ===";
            qDebug() << "  Audio source state:" << m_audioSource->state();
            qDebug() << "  Audio source error:" << m_audioSource->error();
            qDebug() << "  Bytes available:" << (m_audioInput ? m_audioInput->bytesAvailable() : -1);
            qDebug() << "  Buffer size so far:" << m_audioBuffer.size();
            qDebug() << "  handleAudioData called:" << (m_audioBuffer.size() > 0 ? "YES" : "NO");

            // IdleStateのままの場合は、もう一度resume()を試してから警告
            if (m_audioSource->state() == QAudio::IdleState) {
                qDebug() << "Still in IdleState after 500ms, trying resume() again...";
                m_audioSource->resume();

                // 少し待ってから再度チェック
                QTimer::singleShot(200, this, [this]() {
                    if (m_isRecording && m_audioSource) {
                        qDebug() << "After second resume() - state:" << m_audioSource->state();

                        if (m_audioSource->state() == QAudio::IdleState) {
                            QString warningMsg = "WARNING: Audio source is still in IDLE state after multiple attempts. "
                                                "This usually indicates:\n"
                                                "  1. Microphone is muted in System Settings\n"
                                                "  2. Another application is using the microphone\n"
                                                "  3. Qt/macOS compatibility issue\n"
                                                "\n"
                                                "Please check:\n"
                                                "  - System Settings > Sound > Input - ensure microphone is not muted\n"
                                                "  - Close other applications that might use the microphone\n"
                                                "  - Speak into the microphone and check if the input level bar moves";
                            qWarning() << warningMsg;
                            emit error(tr("Microphone not responding - please check system settings"));
                        }
                    }
                });
            }
        }
    });

    emit recordingStarted();
    return true;
}

void AudioRecorder::stopRecording() {
    if (!m_isRecording) {
        return;
    }

    // リアルタイム処理のタイマーを停止
    if (m_chunkTimer->isActive()) {
        m_chunkTimer->stop();
    }

    if (m_audioSource) {
        m_audioSource->stop();
    }

    m_audioInput = nullptr;
    m_isRecording = false;

    qDebug() << "Recording stopped, full buffer size:" << m_fullAudioBuffer.size() << "bytes";

    // WAVファイルを生成（全体バッファを使用）
    QByteArray wavData = createWavHeader(m_fullAudioBuffer.size()) + m_fullAudioBuffer;

    emit recordingStopped(wavData);
}

bool AudioRecorder::isRecording() const {
    return m_isRecording;
}

QByteArray AudioRecorder::getAudioData() const {
    if (m_audioBuffer.isEmpty()) {
        return QByteArray();
    }

    // WAVヘッダー + PCMデータ
    return createWavHeader(m_audioBuffer.size()) + m_audioBuffer;
}

void AudioRecorder::clearAudioData() {
    m_audioBuffer.clear();
}

QStringList AudioRecorder::getAvailableDevices() {
    QStringList devices;
    const auto audioDevices = QMediaDevices::audioInputs();

    for (const auto& device : audioDevices) {
        devices.append(device.description());
    }

    return devices;
}

bool AudioRecorder::setInputDevice(const QString& deviceName) {
    const auto audioDevices = QMediaDevices::audioInputs();

    for (const auto& device : audioDevices) {
        if (device.description() == deviceName) {
            m_currentDevice = device;
            qDebug() << "Audio input device set to:" << deviceName;
            return true;
        }
    }

    qWarning() << "Audio device not found:" << deviceName;
    return false;
}

QString AudioRecorder::getCurrentDeviceName() const {
    if (m_currentDevice.isNull()) {
        return QString();
    }
    return m_currentDevice.description();
}

qint64 AudioRecorder::getRecordingDuration() const {
    if (!m_isRecording) {
        return 0;
    }
    return QDateTime::currentMSecsSinceEpoch() - m_recordingStartTime;
}

void AudioRecorder::handleAudioData() {
    if (!m_audioInput || !m_isRecording) {
        qWarning() << "handleAudioData called but not recording:"
                   << "m_audioInput=" << (m_audioInput != nullptr)
                   << "m_isRecording=" << m_isRecording;
        return;
    }

    // 利用可能なデータを全て読み込む
    qint64 bytesAvailable = m_audioInput->bytesAvailable();
    qDebug() << "Audio data available:" << bytesAvailable << "bytes";

    QByteArray data = m_audioInput->readAll();

    if (data.isEmpty()) {
        qWarning() << "Read data is empty despite" << bytesAvailable << "bytes available";
        return;
    }

    qDebug() << "Read" << data.size() << "bytes of audio data";

    // バッファに追加
    m_audioBuffer.append(data);
    m_fullAudioBuffer.append(data);  // 全体バッファにも追加
    qDebug() << "Chunk buffer size:" << m_audioBuffer.size() << "bytes";
    qDebug() << "Full buffer size:" << m_fullAudioBuffer.size() << "bytes";

    // 音声レベルを計算してシグナルを発行
    float level = calculateAudioLevel(data);
    emit audioLevelChanged(level);
}

void AudioRecorder::handleStateChanged(QAudio::State state) {
    qDebug() << "AudioRecorder::handleStateChanged() - New state:" << state;

    switch (state) {
        case QAudio::StoppedState:
            qDebug() << "  State: STOPPED";
            if (m_audioSource->error() != QAudio::NoError) {
                QString errorMsg = QString("Audio error occurred: %1").arg(m_audioSource->error());
                qWarning() << errorMsg;
                emit error(errorMsg);
            } else {
                qDebug() << "  Stopped normally (no error)";
            }
            break;

        case QAudio::ActiveState:
            qDebug() << "  State: ACTIVE - Recording should be working";
            qDebug() << "  Buffer size:" << m_audioSource->bufferSize();
            qDebug() << "  Bytes available:" << (m_audioInput ? m_audioInput->bytesAvailable() : -1);
            break;

        case QAudio::IdleState:
            qDebug() << "  State: IDLE - No data available";
            qDebug() << "  This might indicate no audio input from microphone";
            break;

        case QAudio::SuspendedState:
            qDebug() << "  State: SUSPENDED";
            break;

        default:
            qDebug() << "  State: UNKNOWN";
            break;
    }
}

QAudioFormat AudioRecorder::getAudioFormat() const {
    QAudioFormat format;
    format.setSampleRate(16000);        // Whisperは16kHzを推奨
    format.setChannelCount(1);          // モノラル
    format.setSampleFormat(QAudioFormat::Int16);  // 16-bit PCM
    return format;
}

float AudioRecorder::calculateAudioLevel(const QByteArray& data) const {
    if (data.isEmpty()) {
        return 0.0f;
    }

    // 16-bit PCMとして解析
    const int16_t* samples = reinterpret_cast<const int16_t*>(data.constData());
    int sampleCount = data.size() / sizeof(int16_t);

    // RMS（二乗平均平方根）を計算
    double sum = 0.0;
    for (int i = 0; i < sampleCount; ++i) {
        double sample = static_cast<double>(samples[i]) / 32768.0;
        sum += sample * sample;
    }

    double rms = std::sqrt(sum / sampleCount);

    // 0.0 - 1.0 の範囲に正規化
    return static_cast<float>(std::min(rms * 5.0, 1.0));
}

QByteArray AudioRecorder::createWavHeader(int dataSize) const {
    QByteArray header;
    QDataStream stream(&header, QIODevice::WriteOnly);
    stream.setByteOrder(QDataStream::LittleEndian);

    // 実際の録音フォーマットから情報を取得
    int sampleRate = m_recordingFormat.sampleRate();
    int numChannels = m_recordingFormat.channelCount();
    int bitsPerSample = m_recordingFormat.bytesPerSample() * 8;

    // WAVフォーマットコードを決定 (1=PCM integer, 3=IEEE float)
    quint16 audioFormat = 1;  // デフォルトはPCM
    if (m_recordingFormat.sampleFormat() == QAudioFormat::Float) {
        audioFormat = 3;  // IEEE Float
    }

    int byteRate = sampleRate * numChannels * (bitsPerSample / 8);
    int blockAlign = numChannels * (bitsPerSample / 8);

    qDebug() << "Creating WAV header:"
             << "format=" << audioFormat
             << "channels=" << numChannels
             << "rate=" << sampleRate
             << "bits=" << bitsPerSample;

    // RIFF header
    stream.writeRawData("RIFF", 4);
    stream << static_cast<quint32>(dataSize + 36);  // ChunkSize
    stream.writeRawData("WAVE", 4);

    // fmt chunk
    stream.writeRawData("fmt ", 4);
    stream << static_cast<quint32>(16);                      // Subchunk1Size (16 for PCM/Float)
    stream << audioFormat;                                   // AudioFormat (1=PCM, 3=Float)
    stream << static_cast<quint16>(numChannels);             // NumChannels
    stream << static_cast<quint32>(sampleRate);              // SampleRate
    stream << static_cast<quint32>(byteRate);                // ByteRate
    stream << static_cast<quint16>(blockAlign);              // BlockAlign
    stream << static_cast<quint16>(bitsPerSample);           // BitsPerSample

    // data chunk
    stream.writeRawData("data", 4);
    stream << static_cast<quint32>(dataSize);                // Subchunk2Size

    return header;
}

void AudioRecorder::setRealtimeProcessing(bool enable) {
    m_realtimeProcessing = enable;
    qDebug() << "Realtime processing" << (enable ? "enabled" : "disabled");
}

bool AudioRecorder::isRealtimeProcessingEnabled() const {
    return m_realtimeProcessing;
}

void AudioRecorder::setChunkInterval(int intervalMs) {
    m_chunkIntervalMs = intervalMs;
    qDebug() << "Chunk interval set to" << intervalMs << "ms";
}

void AudioRecorder::setOverlapDuration(int overlapMs) {
    m_overlapDurationMs = overlapMs;
    qDebug() << "Overlap duration set to" << overlapMs << "ms";
}

void AudioRecorder::handleChunkTimer() {
    if (!m_isRecording || m_audioBuffer.isEmpty()) {
        return;
    }

    qDebug() << "Chunk timer triggered - buffer size:" << m_audioBuffer.size();

    // オーバーラップサイズを計算（バイト単位）
    // 実際の録音フォーマットを使用
    int sampleRate = m_recordingFormat.sampleRate();
    int channels = m_recordingFormat.channelCount();
    int bytesPerSample = m_recordingFormat.bytesPerSample();
    int bytesPerSecond = sampleRate * channels * bytesPerSample;
    int overlapBytes = (m_overlapDurationMs * bytesPerSecond) / 1000;

    qDebug() << "Format:" << sampleRate << "Hz," << channels << "ch," << bytesPerSample << "bytes/sample"
             << "-> bytesPerSecond:" << bytesPerSecond;

    // 現在のバッファから送信するチャンクを作成
    // オーバーラップバッファ + 新しいデータ
    QByteArray chunkData = m_overlapBuffer + m_audioBuffer;

    // WAVヘッダーを追加
    QByteArray wavChunk = createWavHeader(chunkData.size()) + chunkData;

    qDebug() << "Sending audio chunk - overlap:" << m_overlapBuffer.size()
             << "bytes, new data:" << m_audioBuffer.size()
             << "bytes, total:" << chunkData.size() << "bytes";

    // チャンクを送信
    emit audioChunkReady(wavChunk);

    // 次のオーバーラップのために、現在のバッファの最後の部分を保存
    if (m_audioBuffer.size() >= overlapBytes) {
        m_overlapBuffer = m_audioBuffer.right(overlapBytes);
    } else {
        m_overlapBuffer = m_audioBuffer;
    }

    // バッファをクリア（オーバーラップ部分は保持済み）
    m_audioBuffer.clear();

    m_lastChunkTime = QDateTime::currentMSecsSinceEpoch();
}
