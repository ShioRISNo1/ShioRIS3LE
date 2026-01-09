#ifndef LINUX_AUTO_SEGMENTER_H
#define LINUX_AUTO_SEGMENTER_H

#include <QtConcurrent>
#include <QFuture>
#include <QString>
#include <QObject>
#include <opencv2/core.hpp>
#include <memory>
#include <functional>

// Forward declaration
class OnnxSegmenter;

struct SegmentationResult {
    cv::Mat mask;
    bool success = false;
    QString errorMessage;
};

/**
 * @brief Linux環境でのAIセグメンテーション処理クラス
 *
 * OnnxSegmenterを使用して非同期でセグメンテーションを実行し、
 * GPU情報の取得や環境検証機能を提供します。
 */
class LinuxAutoSegmenter {
public:
    using ProgressCallback = std::function<void(int)>;

    LinuxAutoSegmenter();
    ~LinuxAutoSegmenter();

    /**
     * @brief ONNXモデルをロード
     * @param modelPath モデルファイルのパス
     * @return 成功時true
     */
    bool loadModel(const QString &modelPath);

    /**
     * @brief モデルがロード済みかチェック
     * @return ロード済みの場合true
     */
    bool isModelLoaded() const;

    /**
     * @brief 3Dボリュームに対して非同期でセグメンテーションを実行
     * @param volume 入力ボリューム（CV_16SC1またはCV_8UC1）
     * @param progressCallback 進捗コールバック（オプション）
     * @return セグメンテーション結果のFuture
     */
    QFuture<SegmentationResult> segmentVolumeAsync(
        const cv::Mat &volume,
        ProgressCallback progressCallback = nullptr
    );

    /**
     * @brief ONNX Runtime環境を検証
     * @return 環境が正しく設定されている場合true
     */
    bool validateEnvironment();

    /**
     * @brief GPU情報を取得
     * @return GPU情報文字列（利用可能な場合GPUモデル名、そうでない場合"CPU"）
     */
    QString getGPUInfo();

    /**
     * @brief 最後のエラーメッセージを取得
     * @return エラーメッセージ
     */
    QString getLastError() const;

private:
    std::unique_ptr<OnnxSegmenter> m_segmenter;
    QString m_lastError;
    bool m_modelLoaded;
};

#endif // LINUX_AUTO_SEGMENTER_H
