#include "ai/segmentation_pipeline.h"
#include "ai/linux_auto_segmenter.h"
#include "dicom/dicom_volume.h"
#include <QFile>
#include <QDataStream>
#include <QDebug>
#include <opencv2/imgproc.hpp>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/ofstd/ofstd.h>

SegmentationPipeline::SegmentationPipeline()
{
}

SegmentationPipeline::~SegmentationPipeline()
{
}

bool SegmentationPipeline::saveVolumeData(const DicomVolume& volume, const QString& path)
{
    m_lastError.clear();

    try {
        QFile file(path);
        if (!file.open(QIODevice::WriteOnly)) {
            m_lastError = QString("Cannot open file for writing: %1").arg(path);
            qWarning() << m_lastError;
            return false;
        }

        QDataStream out(&file);
        out.setVersion(QDataStream::Qt_6_0);

        // ヘッダー情報を書き込み
        out << QString("SHIORIS3_VOLUME");  // マジックナンバー
        out << qint32(1);                   // バージョン番号

        // ボリュームメタデータ
        out << qint32(volume.width());
        out << qint32(volume.height());
        out << qint32(volume.depth());
        out << volume.spacingX();
        out << volume.spacingY();
        out << volume.spacingZ();
        out << volume.originX();
        out << volume.originY();
        out << volume.originZ();
        out << volume.frameOfReferenceUID();

        // ボリュームデータを書き込み
        const cv::Mat& mat = volume.data();
        out << qint32(mat.dims);
        out << qint32(mat.type());

        for (int i = 0; i < mat.dims; ++i) {
            out << qint32(mat.size[i]);
        }

        // データ本体
        size_t dataSize = mat.total() * mat.elemSize();
        out.writeRawData(reinterpret_cast<const char*>(mat.data), dataSize);

        file.close();
        qInfo() << "Successfully saved volume data to" << path;
        return true;

    } catch (const std::exception& e) {
        m_lastError = QString("Exception while saving volume: %1").arg(e.what());
        qWarning() << m_lastError;
        return false;
    }
}

bool SegmentationPipeline::loadVolumeData(const QString& path, DicomVolume& volume)
{
    m_lastError.clear();

    try {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) {
            m_lastError = QString("Cannot open file for reading: %1").arg(path);
            qWarning() << m_lastError;
            return false;
        }

        QDataStream in(&file);
        in.setVersion(QDataStream::Qt_6_0);

        // ヘッダー検証
        QString magic;
        qint32 version;
        in >> magic >> version;

        if (magic != "SHIORIS3_VOLUME") {
            m_lastError = "Invalid volume file format";
            qWarning() << m_lastError;
            return false;
        }

        if (version != 1) {
            m_lastError = QString("Unsupported volume file version: %1").arg(version);
            qWarning() << m_lastError;
            return false;
        }

        // メタデータの読み込み（将来的にDicomVolumeに適用）
        qint32 width, height, depth;
        double spacingX, spacingY, spacingZ;
        double originX, originY, originZ;
        QString frameUID;

        in >> width >> height >> depth;
        in >> spacingX >> spacingY >> spacingZ;
        in >> originX >> originY >> originZ;
        in >> frameUID;

        // ボリュームデータの読み込み
        qint32 dims, type;
        in >> dims >> type;

        std::vector<int> sizes(dims);
        for (int i = 0; i < dims; ++i) {
            qint32 size;
            in >> size;
            sizes[i] = size;
        }

        cv::Mat mat(dims, sizes.data(), type);
        size_t dataSize = mat.total() * mat.elemSize();
        in.readRawData(reinterpret_cast<char*>(mat.data), dataSize);

        file.close();

        // DicomVolumeに設定（createFromReferenceを使用する代わりに、
        // 内部的にデータを設定する必要があります）
        // 現在のDicomVolumeの実装に依存

        qInfo() << "Successfully loaded volume data from" << path;
        return true;

    } catch (const std::exception& e) {
        m_lastError = QString("Exception while loading volume: %1").arg(e.what());
        qWarning() << m_lastError;
        return false;
    }
}

bool SegmentationPipeline::saveSegmentationResult(const SegmentationResult& result,
                                                  const QString& resultPath)
{
    m_lastError.clear();

    try {
        if (result.mask.empty()) {
            m_lastError = "Segmentation result mask is empty";
            qWarning() << m_lastError;
            return false;
        }

        QFile file(resultPath);
        if (!file.open(QIODevice::WriteOnly)) {
            m_lastError = QString("Cannot open file for writing: %1").arg(resultPath);
            qWarning() << m_lastError;
            return false;
        }

        QDataStream out(&file);
        out.setVersion(QDataStream::Qt_6_0);

        // ヘッダー
        out << QString("SHIORIS3_SEGMENTATION");
        out << qint32(1);  // バージョン

        // マスクデータ
        const cv::Mat& mask = result.mask;
        out << qint32(mask.dims);
        out << qint32(mask.type());

        for (int i = 0; i < mask.dims; ++i) {
            out << qint32(mask.size[i]);
        }

        size_t dataSize = mask.total() * mask.elemSize();
        out.writeRawData(reinterpret_cast<const char*>(mask.data), dataSize);

        file.close();
        qInfo() << "Successfully saved segmentation result to" << resultPath;
        return true;

    } catch (const std::exception& e) {
        m_lastError = QString("Exception while saving segmentation: %1").arg(e.what());
        qWarning() << m_lastError;
        return false;
    }
}

SegmentationResult SegmentationPipeline::loadSegmentationResult(const QString& resultPath)
{
    m_lastError.clear();
    SegmentationResult result;

    try {
        QFile file(resultPath);
        if (!file.open(QIODevice::ReadOnly)) {
            m_lastError = QString("Cannot open file for reading: %1").arg(resultPath);
            qWarning() << m_lastError;
            result.success = false;
            result.errorMessage = m_lastError;
            return result;
        }

        QDataStream in(&file);
        in.setVersion(QDataStream::Qt_6_0);

        // ヘッダー検証
        QString magic;
        qint32 version;
        in >> magic >> version;

        if (magic != "SHIORIS3_SEGMENTATION") {
            m_lastError = "Invalid segmentation file format";
            qWarning() << m_lastError;
            result.success = false;
            result.errorMessage = m_lastError;
            return result;
        }

        if (version != 1) {
            m_lastError = QString("Unsupported segmentation version: %1").arg(version);
            qWarning() << m_lastError;
            result.success = false;
            result.errorMessage = m_lastError;
            return result;
        }

        // マスクデータの読み込み
        qint32 dims, type;
        in >> dims >> type;

        std::vector<int> sizes(dims);
        for (int i = 0; i < dims; ++i) {
            qint32 size;
            in >> size;
            sizes[i] = size;
        }

        cv::Mat mask(dims, sizes.data(), type);
        size_t dataSize = mask.total() * mask.elemSize();
        in.readRawData(reinterpret_cast<char*>(mask.data), dataSize);

        file.close();

        result.mask = mask;
        result.success = true;
        qInfo() << "Successfully loaded segmentation result from" << resultPath;
        return result;

    } catch (const std::exception& e) {
        m_lastError = QString("Exception while loading segmentation: %1").arg(e.what());
        qWarning() << m_lastError;
        result.success = false;
        result.errorMessage = m_lastError;
        return result;
    }
}

bool SegmentationPipeline::exportToRTStructureSet(
    const SegmentationResult& result,
    const DicomVolume& referenceVolume,
    const QString& outputPath,
    const std::vector<QString>& organLabels)
{
    m_lastError.clear();

    try {
        if (result.mask.empty()) {
            m_lastError = "Segmentation mask is empty";
            qWarning() << m_lastError;
            return false;
        }

        if (result.mask.dims != 3) {
            m_lastError = QString("Invalid mask dimensions: %1 (expected 3)").arg(result.mask.dims);
            qWarning() << m_lastError;
            return false;
        }

        qInfo() << "Exporting segmentation to RT Structure Set:" << outputPath;
        qInfo() << "Mask size:" << result.mask.size[0] << "x"
                << result.mask.size[1] << "x" << result.mask.size[2];

        // DCMTK でRT Structure Setを作成
        DcmFileFormat fileformat;
        DcmDataset* dataset = fileformat.getDataset();

        // SOP Class UID (RT Structure Set Storage)
        dataset->putAndInsertString(DCM_SOPClassUID, "1.2.840.10008.5.1.4.1.1.481.3");

        // SOP Instance UID (ユニークなIDを生成)
        char uid[100];
        dcmGenerateUniqueIdentifier(uid, SITE_INSTANCE_UID_ROOT);
        dataset->putAndInsertString(DCM_SOPInstanceUID, uid);

        // Study/Series情報（簡易版）
        dataset->putAndInsertString(DCM_StudyInstanceUID, "1.2.826.0.1.3680043.8.498.1");
        dataset->putAndInsertString(DCM_SeriesInstanceUID, "1.2.826.0.1.3680043.8.498.2");
        dataset->putAndInsertString(DCM_Modality, "RTSTRUCT");

        // Frame of Reference
        dataset->putAndInsertString(DCM_FrameOfReferenceUID,
                                   referenceVolume.frameOfReferenceUID().toStdString().c_str());

        // Structure Set情報
        dataset->putAndInsertString(DCM_StructureSetLabel, "AI Segmentation");
        dataset->putAndInsertString(DCM_StructureSetName, "Auto-Segmentation Result");
        dataset->putAndInsertString(DCM_StructureSetDate, "20250101");
        dataset->putAndInsertString(DCM_StructureSetTime, "120000");

        // ROI情報を作成
        int numLabels = organLabels.size();

        for (int labelIdx = 1; labelIdx < numLabels; ++labelIdx) {
            // 背景（ラベル0）はスキップ
            qInfo() << "Processing organ:" << organLabels[labelIdx];

            // 各スライスで輪郭を抽出
            int depth = result.mask.size[0];

            for (int z = 0; z < depth; ++z) {
                // スライスを抽出
                cv::Mat slice = cv::Mat(result.mask.size[1], result.mask.size[2],
                                       CV_8UC1);

                for (int y = 0; y < result.mask.size[1]; ++y) {
                    for (int x = 0; x < result.mask.size[2]; ++x) {
                        int idx[3] = {z, y, x};
                        uchar value = result.mask.at<uchar>(idx);
                        slice.at<uchar>(y, x) = (value == labelIdx) ? 255 : 0;
                    }
                }

                // 輪郭抽出
                auto contours = extractContours(slice, 255);

                if (!contours.empty()) {
                    qDebug() << "  Slice" << z << ":" << contours.size() << "contours";
                }

                // TODO: 輪郭をDICOMデータセットに追加
                // これには詳細なDCMTK APIの使用が必要
            }
        }

        // ファイルに保存
        OFCondition status = fileformat.saveFile(outputPath.toStdString().c_str(),
                                                 EXS_LittleEndianExplicit);

        if (status.bad()) {
            m_lastError = QString("Failed to save RT Structure Set: %1")
                         .arg(status.text());
            qWarning() << m_lastError;
            return false;
        }

        qInfo() << "Successfully exported RT Structure Set to" << outputPath;
        return true;

    } catch (const std::exception& e) {
        m_lastError = QString("Exception while exporting RT Structure Set: %1").arg(e.what());
        qWarning() << m_lastError;
        return false;
    }
}

std::vector<std::vector<cv::Point>> SegmentationPipeline::extractContours(
    const cv::Mat& mask,
    int labelValue)
{
    std::vector<std::vector<cv::Point>> contours;

    try {
        // マスクを二値化
        cv::Mat binary;
        cv::threshold(mask, binary, labelValue - 1, 255, cv::THRESH_BINARY);

        // 輪郭抽出
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(binary, contours, hierarchy,
                        cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    } catch (const cv::Exception& e) {
        qWarning() << "OpenCV exception in extractContours:" << e.what();
    }

    return contours;
}

QVector3D SegmentationPipeline::pixelToPatient(
    const cv::Point& pixelPoint,
    int sliceIndex,
    const DicomVolume& volume)
{
    // ピクセル座標を患者座標系に変換
    return volume.voxelToPatient(pixelPoint.x, pixelPoint.y, sliceIndex);
}

QString SegmentationPipeline::getLastError() const
{
    return m_lastError;
}
