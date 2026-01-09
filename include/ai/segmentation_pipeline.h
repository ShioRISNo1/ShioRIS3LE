#ifndef SEGMENTATION_PIPELINE_H
#define SEGMENTATION_PIPELINE_H

#include <QString>
#include <QVector3D>
#include <opencv2/core.hpp>
#include <vector>
#include <memory>

// Forward declarations
class DicomVolume;
struct SegmentationResult;

/**
 * @brief セグメンテーション結果を含む構造体
 */
struct SegmentationVolumeData {
    cv::Mat mask;                          // セグメンテーションマスク（3D）
    std::vector<QString> organLabels;      // 臓器ラベル名
    QString frameOfReferenceUID;           // フレーム参照UID
    double spacingX = 1.0;                 // X方向のボクセル間隔
    double spacingY = 1.0;                 // Y方向のボクセル間隔
    double spacingZ = 1.0;                 // Z方向のボクセル間隔
    double originX = 0.0;                  // 原点X座標
    double originY = 0.0;                  // 原点Y座標
    double originZ = 0.0;                  // 原点Z座標
    double rowDir[3] = {1.0, 0.0, 0.0};   // 行方向ベクトル
    double colDir[3] = {0.0, 1.0, 0.0};   // 列方向ベクトル
    double sliceDir[3] = {0.0, 0.0, 1.0}; // スライス方向ベクトル
};

/**
 * @brief DICOMセグメンテーションパイプライン
 *
 * DICOM画像とセグメンテーション結果の相互変換を行います。
 * - ボリュームデータの保存・読み込み
 * - セグメンテーション結果からRT Structure Setへの変換
 */
class SegmentationPipeline {
public:
    SegmentationPipeline();
    ~SegmentationPipeline();

    /**
     * @brief DicomVolumeからボリュームデータを保存
     * @param volume DICOMボリューム
     * @param path 保存先パス（バイナリ形式）
     * @return 成功時true
     */
    bool saveVolumeData(const DicomVolume& volume, const QString& path);

    /**
     * @brief 保存したボリュームデータを読み込み
     * @param path 保存されたデータのパス
     * @param volume 読み込み先ボリューム
     * @return 成功時true
     */
    bool loadVolumeData(const QString& path, DicomVolume& volume);

    /**
     * @brief セグメンテーション結果をファイルに保存
     * @param result セグメンテーション結果
     * @param resultPath 保存先パス
     * @return 成功時true
     */
    bool saveSegmentationResult(const SegmentationResult& result,
                                const QString& resultPath);

    /**
     * @brief セグメンテーション結果をファイルから読み込み
     * @param resultPath 結果ファイルのパス
     * @return セグメンテーション結果
     */
    SegmentationResult loadSegmentationResult(const QString& resultPath);

    /**
     * @brief セグメンテーション結果からRT Structure Setを作成してエクスポート
     * @param result セグメンテーション結果
     * @param referenceVolume 参照DICOMボリューム
     * @param outputPath 出力先パス
     * @param organLabels 臓器ラベル名のリスト
     * @return 成功時true
     */
    bool exportToRTStructureSet(const SegmentationResult& result,
                               const DicomVolume& referenceVolume,
                               const QString& outputPath,
                               const std::vector<QString>& organLabels);

    /**
     * @brief 最後のエラーメッセージを取得
     * @return エラーメッセージ
     */
    QString getLastError() const;

private:
    /**
     * @brief マスクから輪郭を抽出
     * @param mask 2Dマスク画像
     * @param labelValue 抽出する臓器ラベル値
     * @return 輪郭点のリスト（ピクセル座標）
     */
    std::vector<std::vector<cv::Point>> extractContours(const cv::Mat& mask,
                                                        int labelValue);

    /**
     * @brief ピクセル座標を患者座標に変換
     * @param pixelPoint ピクセル座標
     * @param sliceIndex スライスインデックス
     * @param volume 参照ボリューム
     * @return 患者座標系の3D点
     */
    QVector3D pixelToPatient(const cv::Point& pixelPoint,
                            int sliceIndex,
                            const DicomVolume& volume);

    QString m_lastError;
};

#endif // SEGMENTATION_PIPELINE_H
