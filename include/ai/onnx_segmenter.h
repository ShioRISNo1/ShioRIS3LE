//=============================================================================
// ファイル: include/ai/onnx_segmenter.h
// 修正内容: 新しい関数宣言の追加
//=============================================================================

#ifndef ONNX_SEGMENTER_H
#define ONNX_SEGMENTER_H

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <opencv2/core.hpp>
#include <QImage>

#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

// プログレスコールバック型定義
// パラメータ: float progress (0.0-1.0), std::string message
using ProgressCallback = std::function<void(float, const std::string&)>;

/**
 * @brief 腹部多臓器セグメンテーション用ONNXモデルの推論クラス
 *
 * Version 3.0 - BTCV 14臓器対応・固定サイズモデル対応
 *
 * モデル仕様（MONAI Swin UNETR on BTCV dataset）:
 * - 入力: [batch, 1, depth, height, width] - CT画像ボリューム（固定サイズ: 96x96x96）
 * - 出力: [batch, 14, depth, height, width] - 14クラスセグメンテーション（BTCV標準）
 *   0: Background, 1: Spleen, 2: Right Kidney, 3: Left Kidney,
 *   4: Gallbladder, 5: Esophagus, 6: Liver, 7: Stomach,
 *   8: Aorta, 9: Inferior Vena Cava, 10: Portal/Splenic Veins,
 *   11: Pancreas, 12: Right Adrenal Gland, 13: Left Adrenal Gland
 *
 * 注意: Swin UNETRモデルは固定サイズ入力を期待します。
 *       エクスポート時に指定したサイズ（--input-size）と推論時の入力サイズを一致させる必要があります。
 */
class OnnxSegmenter {
public:
    OnnxSegmenter();
    ~OnnxSegmenter() = default;
    cv::Mat predict3D(const cv::Mat &volume);
    /**
     * @brief 腹部セグメンテーションモデルをロード
     * @param modelPath ONNXモデルファイルのパス
     * @return 成功時true
     */
    bool loadModel(const std::string &modelPath);
    
    /**
     * @brief モデルがロード済みかチェック
     * @return ロード済みの場合true
     */
    bool isLoaded() const;
    
    /**
     * @brief CTスライスに対して腹部臓器セグメンテーションを実行
     * @param slice 入力CTスライス（CV_16SC1またはCV_8UC1）
     * @return セグメンテーション結果（CV_8UC1, 0-3のラベル値）
     */
    cv::Mat predict(const cv::Mat &slice);

    /**
     * @brief DICOM 連番や 3D 画像から高解像度ボリュームを構築
     * @param volume 元のCTボリューム
     * @return 高品質リサンプリング済みボリューム（動的サイズ）
     */
    cv::Mat buildInputVolume(const cv::Mat &volume);

    /**
     * @brief 高解像度ボリュームでのセグメンテーション推論
     * @param volumeInput 高解像度リサンプリング済みボリューム
     * @param originalSize 元のボリュームサイズ [depth, height, width]
     * @return 元サイズにリサンプリングしたセグメンテーション結果
     */
    cv::Mat predictVolume(const cv::Mat &volumeInput, const int *originalSize);
    
    /**
     * @brief サポートする臓器ラベルを取得
     * @return 臓器名のベクター（BTCVデータセット 14クラス）
     */
    static std::vector<std::string> getOrganLabels() {
        return {
            "0: Background",
            "1: Spleen",
            "2: Right Kidney",
            "3: Left Kidney",
            "4: Gallbladder",
            "5: Esophagus",
            "6: Liver",
            "7: Stomach",
            "8: Aorta",
            "9: Inferior Vena Cava",
            "10: Portal/Splenic Veins",
            "11: Pancreas",
            "12: Right Adrenal Gland",
            "13: Left Adrenal Gland"
        };
    }

    /**
     * @brief CUDA execution providerが有効か確認
     * @return CUDA有効時true、無効時false
     */
    bool isCudaEnabled() const {
#ifdef USE_ONNXRUNTIME
        return m_cudaEnabled;
#else
        return false;
#endif
    }

    /**
     * @brief 実行プロバイダー情報を取得
     * @return 実行プロバイダーの説明文字列
     */
    std::string getExecutionProviderInfo() const {
#ifdef USE_ONNXRUNTIME
        if (m_cudaEnabled) {
            return "CUDA Execution Provider (GPU)";
        } else {
            return "CPU Execution Provider";
        }
#else
        return "ONNX Runtime not available";
#endif
    }

    /**
     * @brief 環境変数から品質モードを取得
     * @return 品質モード文字列（"standard", "high", "ultra"）
     */
    std::string getPredictionQualityMode() const;

    /**
     * @brief プログレスコールバックを設定
     * @param callback プログレス更新時に呼ばれるコールバック関数
     */
    void setProgressCallback(ProgressCallback callback) {
        m_progressCallback = callback;
    }

    /**
     * @brief プログレスコールバックをクリア
     */
    void clearProgressCallback() {
        m_progressCallback = nullptr;
    }

    /**
     * @brief TTAを使用した高精度推論
     * @param volumeInput 前処理済みボリューム
     * @param originalSize 元のボリュームサイズ [depth, height, width]
     * @return セグメンテーション結果
     */
    cv::Mat predictVolumeWithTTA(const cv::Mat &volumeInput, const int *originalSize);

    /**
     * @brief TTA + スライディングウィンドウを使用した最高精度推論
     * @param volumeInput 前処理済みボリューム
     * @param originalSize 元のボリュームサイズ [depth, height, width]
     * @param overlap オーバーラップ率（0.0-1.0）
     * @return セグメンテーション結果
     */
    cv::Mat predictVolumeUltra(const cv::Mat &volumeInput, const int *originalSize, float overlap);

    /**
     * @brief 3Dセグメンテーション結果を元サイズにリサンプリング
     * @param segmentation3D セグメンテーション結果
     * @param originalSize 元のボリュームサイズ [depth, height, width]
     * @return 元サイズのセグメンテーション結果
     */
    cv::Mat resample3DToOriginalSize(const cv::Mat &segmentation3D, const int* originalSize);

    /**
     * @brief RTSSからのbounding box情報を設定（ボクセル座標系）
     * @param minX, minY, minZ 最小座標（ボクセル）
     * @param maxX, maxY, maxZ 最大座標（ボクセル）
     */
    void setBoundingBoxVoxel(int minX, int minY, int minZ, int maxX, int maxY, int maxZ) {
        m_useExternalBoundingBox = true;
        m_externalBBoxMinX = minX;
        m_externalBBoxMinY = minY;
        m_externalBBoxMinZ = minZ;
        m_externalBBoxMaxX = maxX;
        m_externalBBoxMaxY = maxY;
        m_externalBBoxMaxZ = maxZ;
    }

    /**
     * @brief External bounding boxをクリア（HUしきい値方式に戻す）
     */
    void clearBoundingBox() {
        m_useExternalBoundingBox = false;
    }

private:
    /**
     * @brief CTスライス専用の前処理（従来版）
     * @param slice 生CTスライス
     * @return 正規化済み浮動小数点画像
     */
    cv::Mat preprocessCTSlice(const cv::Mat &slice);
    
    /**
     * @brief 高品質CT前処理（改良版）- 腹部多臓器セグメンテーション最適化
     * @param slice 生CTスライス
     * @return 最適化された正規化済み浮動小数点画像
     */
    cv::Mat preprocessCTSliceForModel(const cv::Mat &slice);
    
    /**
     * @brief 代替CT前処理（フォールバック用）
     * @param slice 生CTスライス
     * @return 正規化済み浮動小数点画像
     */
    cv::Mat preprocessCTSliceAlternative(const cv::Mat &slice);

    /**
     * @brief 腹部セグメンテーション出力の後処理（3D対応）
     * @param outputTensor ONNX出力テンソル
     * @param originalSize 元のボリュームサイズ [depth, height, width]
     * @return セグメンテーションボリューム
     */
    cv::Mat processAbdomenSegmentationOutput(Ort::Value &outputTensor, const int *originalSize);

    /**
     * @brief 4D出力フォーマット用のヘルパー関数
     */
    cv::Mat processOutput4D(Ort::Value &outputTensor, const std::vector<int64_t> &outShape, const cv::Size &originalSize);
    
#ifndef NDEBUG
    /**
     * @brief Diagnostic functions for debugging (debug builds only)
     */
    bool diagnoseInputData(const cv::Mat &slice);
    bool diagnoseModelOutput(const cv::Mat &result);
#endif
    
    // ▼ 新しい関数宣言（動的サイズ対応）
    
    /**
     * @brief 動的サイズ対応の高品質3Dボリュームリサンプリング
     * @param volume 元の3Dボリューム
     * @return 高解像度リサンプリング済みボリューム（元解像度に近い動的サイズ）
     */
    cv::Mat resampleVolumeFor3D(const cv::Mat &volume);
    
    /**
     * @brief 3Dセグメンテーション出力の後処理
     * @param outputTensor ONNX出力テンソル
     * @param originalSize 元のボリュームサイズ [depth, height, width]
     * @return セグメンテーションボリューム
     */
    cv::Mat process3DSegmentationOutput(Ort::Value &outputTensor, const int* originalSize);

    /**
     * @brief ONNX入力次元を動的に更新
     * @param volume 入力ボリューム
     * @return 成功時true
     */
    bool updateInputDimensions(const cv::Mat &volume);

    // ========== Test Time Augmentation (TTA) Functions ==========

    /**
     * @brief 3Dボリュームを左右反転（X軸）
     * @param volume 入力ボリューム
     * @return 反転されたボリューム
     */
    cv::Mat flipVolumeX(const cv::Mat &volume);

    /**
     * @brief 3Dボリュームを上下反転（Y軸）
     * @param volume 入力ボリューム
     * @return 反転されたボリューム
     */
    cv::Mat flipVolumeY(const cv::Mat &volume);

    /**
     * @brief 3Dボリュームを前後反転（Z軸）
     * @param volume 入力ボリューム
     * @return 反転されたボリューム
     */
    cv::Mat flipVolumeZ(const cv::Mat &volume);

    /**
     * @brief スライディングウィンドウを使用した推論
     * @param volumeInput 前処理済みボリューム
     * @param originalSize 元のボリュームサイズ [depth, height, width]
     * @param overlap オーバーラップ率（0.0-1.0）
     * @return セグメンテーション結果
     */
    cv::Mat predictVolumeWithSlidingWindow(const cv::Mat &volumeInput, const int *originalSize, float overlap);

#ifndef NDEBUG
    /**
     * @brief Diagnostic output for entire processing pipeline (debug builds only)
     * @param originalVolume Original volume
     * @param processedVolume Preprocessed volume
     * @param result Segmentation result
     */
    void diagnoseProcessingPipeline(const cv::Mat &originalVolume, const cv::Mat &processedVolume, const cv::Mat &result);
#endif

    /**
     * @brief プログレス更新ヘルパー関数
     * @param progress 進捗率（0.0-1.0）
     * @param message 進捗メッセージ
     */
    void updateProgress(float progress, const std::string& message = "") {
        if (m_progressCallback) {
            m_progressCallback(progress, message);
        }
    }

#ifdef USE_ONNXRUNTIME
    Ort::Env m_env;
    Ort::SessionOptions m_sessionOptions;
    std::unique_ptr<Ort::Session> m_session;
    std::vector<std::string> m_inputNames;
    std::vector<std::string> m_outputNames;
    std::vector<int64_t> m_inputDims;
    bool m_cudaEnabled{false};

    // RT-CT auto-cropping: store crop offsets for coordinate restoration
    int m_cropOffsetX{0};
    int m_cropOffsetY{0};
    int m_cropOffsetZ{0};
    int m_croppedWidth{0};
    int m_croppedHeight{0};
    int m_croppedDepth{0};

    // External bounding box (RTSS Structure-based cropping)
    bool m_useExternalBoundingBox{false};
    int m_externalBBoxMinX{0};
    int m_externalBBoxMinY{0};
    int m_externalBBoxMinZ{0};
    int m_externalBBoxMaxX{0};
    int m_externalBBoxMaxY{0};
    int m_externalBBoxMaxZ{0};
#endif

    // Progress callback
    ProgressCallback m_progressCallback;
};

#endif // ONNX_SEGMENTER_H