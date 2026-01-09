//=============================================================================
// ファイル: include/ai/ai_segmentation_helper.h
// 修正内容: AI機能のヘルパークラス（テスト・デバッグ用）
//=============================================================================

#ifndef AI_SEGMENTATION_HELPER_H
#define AI_SEGMENTATION_HELPER_H

#include <QString>
#include <QDir>
#include <QDebug>
#include <QStandardPaths>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QProgressDialog>
#include <QApplication>
#include <opencv2/opencv.hpp>

/**
 * @brief AI Segmentation機能のヘルパークラス
 * 
 * 機能:
 * - ONNXモデルファイルの管理
 * - サンプルデータの生成
 * - テスト用ダミーセグメンテーション
 * - モデルファイルのダウンロード支援
 */
class AISegmentationHelper {
public:
    /**
     * @brief AIモデル用ディレクトリの初期化
     */
    static bool initializeAIDirectories();
    
    /**
     * @brief デフォルトのONNXモデルパスを取得
     */
    static QString getDefaultModelPath();
    
    /**
     * @brief サンプル3Dボリュームの生成（テスト用）
     */
    static cv::Mat generateSampleVolume(int depth = 64, int height = 256, int width = 256);
    
    /**
     * @brief ダミーセグメンテーション結果の生成
     * @param volume 入力ボリューム
     * @return ダミーのセグメンテーション結果
     */
    static cv::Mat generateDummySegmentation(const cv::Mat &volume);
    
    /**
     * @brief ボリュームの前処理（正規化・リサイズ）
     */
    static cv::Mat preprocessVolume(const cv::Mat &volume);
    
    /**
     * @brief セグメンテーション後処理（スムージング・穴埋め）
     */
    static cv::Mat postprocessSegmentation(const cv::Mat &segmentation);
    
    /**
     * @brief ONNXモデルファイルの検証
     */
    static bool validateONNXModel(const QString &modelPath);
    
    /**
     * @brief AI機能のシステム要件チェック
     */
    static QString checkSystemRequirements();
    
    /**
     * @brief おすすめのONNXモデル情報を取得
     */
    static QStringList getRecommendedModels();
    
    /**
     * @brief モデルファイルのメタデータ取得
     */
    static QString getModelMetadata(const QString &modelPath);

private:
    static constexpr const char* AI_MODELS_DIR = "AIModels";
    static constexpr const char* ONNX_DIR = "onnx";
    static constexpr const char* TEMP_DIR = "temp";
    static constexpr const char* SAMPLES_DIR = "samples";
    cv::Mat extractSliceFrom3D(const cv::Mat &volume3D, int sliceIndex);
};

/**
 * @brief AI機能のテスト・デモ用クラス
 */
class AISegmentationDemo {
public:
    /**
     * @brief デモ用セグメンテーション実行
     */
    static void runDemoSegmentation();
    
    /**
     * @brief パフォーマンステスト実行
     */
    static void runPerformanceTest();
    
    /**
     * @brief 様々なボリュームサイズでのテスト
     */
    static void runVolumeVariationTest();
    
    /**
     * @brief メモリ使用量テスト
     */
    static void runMemoryUsageTest();

private:
    static void logTestResult(const QString &testName, bool success, const QString &details = "");
};

// インライン実装（簡単な関数）

inline QString AISegmentationHelper::getDefaultModelPath() {
    QString documentsPath = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    QDir aiDir(QDir(documentsPath).filePath("ShioRIS3/" + QString(AI_MODELS_DIR) + "/" + QString(ONNX_DIR)));
    return aiDir.filePath("abdominal_segmentation.onnx");
}

inline QStringList AISegmentationHelper::getRecommendedModels() {
    return {
        "Abdominal Multi-Organ Segmentation (CT)",
        "TotalSegmentator (Multi-organ)",
        "nnU-Net Abdominal Organs",
        "MONAI Abdominal Segmentation"
    };
}

#endif // AI_SEGMENTATION_HELPER_H