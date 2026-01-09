// パス: include/ai/whisper_client.h
#ifndef WHISPER_CLIENT_H
#define WHISPER_CLIENT_H

#include <QObject>
#include <QString>
#include <QByteArray>
#include <memory>

// Forward declaration for whisper.cpp structures
struct whisper_context;
struct whisper_full_params;

/**
 * @brief Whisper音声認識クライアント
 *
 * whisper.cppを使用した音声からテキストへの変換を提供します。
 * ローカルで動作し、プライバシーを保護します。
 */
class WhisperClient : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Whisperモデルサイズ
     */
    enum class ModelSize {
        Tiny,      // 39M params, ~75MB
        Base,      // 74M params, ~142MB
        Small,     // 244M params, ~466MB
        Medium,    // 769M params, ~1.5GB
        Large      // 1550M params, ~2.9GB
    };

    /**
     * @brief 文字起こし言語
     */
    enum class Language {
        Auto,      // 自動検出
        Japanese,  // 日本語
        English,   // 英語
    };

    explicit WhisperClient(QObject* parent = nullptr);
    ~WhisperClient();

    /**
     * @brief Whisperモデルをロード
     * @param modelPath モデルファイルのパス（.binファイル）
     * @return 成功した場合true
     */
    bool loadModel(const QString& modelPath);

    /**
     * @brief デフォルトモデルをロード
     * @param size モデルサイズ
     * @return 成功した場合true
     */
    bool loadDefaultModel(ModelSize size = ModelSize::Base);

    /**
     * @brief 利用可能なモデルを自動検出してロード
     * 複数のモデルサイズと複数のパスを試す
     * @return 成功した場合true
     */
    bool loadAnyAvailableModel();

    /**
     * @brief モデルがロード済みかチェック
     */
    bool isModelLoaded() const;

    /**
     * @brief 音声ファイルから文字起こし
     * @param audioFilePath 音声ファイルパス（WAV推奨）
     * @return 文字起こし結果
     */
    QString transcribeFromFile(const QString& audioFilePath);

    /**
     * @brief PCMオーディオデータから文字起こし
     * @param pcmData 16-bit PCM audio data (16kHz mono)
     * @return 文字起こし結果
     */
    QString transcribeFromPCM(const QVector<float>& pcmData);

    /**
     * @brief WAVオーディオバッファから文字起こし
     * @param wavData WAV形式のオーディオデータ
     * @return 文字起こし結果
     */
    QString transcribeFromWav(const QByteArray& wavData);

    /**
     * @brief 言語設定
     * @param lang 認識言語
     */
    void setLanguage(Language lang);

    /**
     * @brief 現在の言語設定を取得
     */
    Language getLanguage() const;

    /**
     * @brief タイムスタンプを有効化
     * @param enable タイムスタンプ付き文字起こしを有効にする
     */
    void setTimestampsEnabled(bool enable);

    /**
     * @brief Initial promptを設定（文脈継続用）
     * @param prompt 前回の文字起こし結果などを設定すると精度が向上
     */
    void setInitialPrompt(const QString& prompt);

    /**
     * @brief Initial promptを取得
     */
    QString getInitialPrompt() const;

    /**
     * @brief モデルパスを取得
     */
    QString getModelPath() const;

    /**
     * @brief モデルサイズから推奨パスを取得
     * @param size モデルサイズ
     * @return モデルファイルパス
     */
    static QString getDefaultModelPath(ModelSize size);

    /**
     * @brief モデルサイズを文字列に変換
     */
    static QString modelSizeToString(ModelSize size);

    /**
     * @brief 文字列からモデルサイズに変換
     */
    static ModelSize stringToModelSize(const QString& sizeStr);

signals:
    /**
     * @brief 文字起こし完了シグナル
     * @param text 文字起こし結果
     */
    void transcriptionReady(const QString& text);

    /**
     * @brief 文字起こし進捗シグナル（部分結果）
     * @param partialText 部分的な文字起こし結果
     */
    void transcriptionProgress(const QString& partialText);

    /**
     * @brief エラーシグナル
     * @param errorMsg エラーメッセージ
     */
    void error(const QString& errorMsg);

    /**
     * @brief モデルロード完了シグナル
     * @param success 成功した場合true
     */
    void modelLoaded(bool success);

private:
    /**
     * @brief WAVヘッダーを解析してPCMデータを抽出
     * @param wavData WAVデータ
     * @return PCMデータ（16kHz mono, float形式）
     */
    QVector<float> extractPCMFromWav(const QByteArray& wavData);

    /**
     * @brief PCMデータを16kHzにリサンプル
     * @param pcmData 元のPCMデータ
     * @param originalSampleRate 元のサンプリングレート
     * @return リサンプル後のPCMデータ
     */
    QVector<float> resampleTo16kHz(const QVector<float>& pcmData, int originalSampleRate);

    whisper_context* m_context = nullptr;
    QString m_modelPath;
    Language m_language = Language::Auto;
    bool m_timestampsEnabled = false;
    QString m_initialPrompt;  // 文脈継続用のprompt
};


#endif // WHISPER_CLIENT_H
