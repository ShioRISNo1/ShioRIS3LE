// パス: include/ai/audio_recorder.h
#ifndef AUDIO_RECORDER_H
#define AUDIO_RECORDER_H

#include <QObject>
#include <QAudioSource>
#include <QAudioFormat>
#include <QIODevice>
#include <QByteArray>
#include <QMediaDevices>
#include <QTimer>
#include <memory>

/**
 * @brief オーディオ録音クラス
 *
 * Qt Multimediaを使用してマイクから音声を録音します。
 * Whisper向けに16kHz mono PCM形式で録音します。
 */
class AudioRecorder : public QObject {
    Q_OBJECT

public:
    explicit AudioRecorder(QObject* parent = nullptr);
    ~AudioRecorder();

    /**
     * @brief 録音を開始
     * @return 成功した場合true
     */
    bool startRecording();

    /**
     * @brief 録音を停止
     */
    void stopRecording();

    /**
     * @brief 録音中かどうか
     */
    bool isRecording() const;

    /**
     * @brief 録音したオーディオデータを取得（WAV形式）
     * @return WAV形式のオーディオデータ
     */
    QByteArray getAudioData() const;

    /**
     * @brief 録音したオーディオデータをクリア
     */
    void clearAudioData();

    /**
     * @brief 利用可能な入力デバイスのリストを取得
     */
    static QStringList getAvailableDevices();

    /**
     * @brief 入力デバイスを設定
     * @param deviceName デバイス名
     * @return 成功した場合true
     */
    bool setInputDevice(const QString& deviceName);

    /**
     * @brief 現在の入力デバイス名を取得
     */
    QString getCurrentDeviceName() const;

    /**
     * @brief 録音時間を取得（ミリ秒）
     */
    qint64 getRecordingDuration() const;

    /**
     * @brief リアルタイム処理を有効化/無効化
     * @param enable true: チャンク単位でリアルタイム送信, false: 録音停止時に一括送信
     */
    void setRealtimeProcessing(bool enable);

    /**
     * @brief リアルタイム処理が有効かどうか
     */
    bool isRealtimeProcessingEnabled() const;

    /**
     * @brief チャンク送信間隔を設定（ミリ秒）
     * @param intervalMs チャンク間隔（デフォルト: 3000ms = 3秒）
     */
    void setChunkInterval(int intervalMs);

    /**
     * @brief オーバーラップ時間を設定（ミリ秒）
     * @param overlapMs オーバーラップ時間（デフォルト: 1000ms = 1秒）
     */
    void setOverlapDuration(int overlapMs);

signals:
    /**
     * @brief 録音開始シグナル
     */
    void recordingStarted();

    /**
     * @brief 録音停止シグナル
     * @param audioData 録音されたWAVデータ
     */
    void recordingStopped(const QByteArray& audioData);

    /**
     * @brief リアルタイムチャンクシグナル（リアルタイム処理用）
     * @param audioChunk WAV形式の音声チャンク（オーバーラップ含む）
     */
    void audioChunkReady(const QByteArray& audioChunk);

    /**
     * @brief 音声レベル変化シグナル（音量メーター用）
     * @param level 音声レベル（0.0 - 1.0）
     */
    void audioLevelChanged(float level);

    /**
     * @brief エラーシグナル
     * @param errorMsg エラーメッセージ
     */
    void error(const QString& errorMsg);

private slots:
    /**
     * @brief オーディオデータ受信スロット
     */
    void handleAudioData();

    /**
     * @brief オーディオ状態変化スロット
     * @param state 新しい状態
     */
    void handleStateChanged(QAudio::State state);

    /**
     * @brief チャンク送信タイマーのスロット
     */
    void handleChunkTimer();

private:
    /**
     * @brief オーディオフォーマットを初期化（16kHz mono PCM）
     */
    QAudioFormat getAudioFormat() const;

    /**
     * @brief 音声レベルを計算
     * @param data PCMデータ
     * @return 音声レベル（0.0 - 1.0）
     */
    float calculateAudioLevel(const QByteArray& data) const;

    /**
     * @brief WAVヘッダーを生成
     * @param dataSize PCMデータサイズ
     * @return WAVヘッダー
     */
    QByteArray createWavHeader(int dataSize) const;

    std::unique_ptr<QAudioSource> m_audioSource;
    QIODevice* m_audioInput = nullptr;
    QByteArray m_audioBuffer;      // リアルタイムチャンク用の一時バッファ
    QByteArray m_fullAudioBuffer;  // 録音全体を保持するバッファ
    QAudioDevice m_currentDevice;
    QAudioFormat m_recordingFormat;    // 実際に使用している録音フォーマット
    bool m_isRecording = false;
    qint64 m_recordingStartTime = 0;

    // リアルタイム処理用
    bool m_realtimeProcessing = false;
    int m_chunkIntervalMs = 3000;      // チャンク間隔（デフォルト3秒）
    int m_overlapDurationMs = 1000;    // オーバーラップ時間（デフォルト1秒）
    QTimer* m_chunkTimer = nullptr;
    QByteArray m_overlapBuffer;        // 前のチャンクとのオーバーラップ用バッファ
    qint64 m_lastChunkTime = 0;        // 最後にチャンクを送信した時刻
};

#endif // AUDIO_RECORDER_H
