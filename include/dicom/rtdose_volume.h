#ifndef RTDOSE_VOLUME_H
#define RTDOSE_VOLUME_H

#include <QString>
#include <QImage>
#include <vector>
#include <opencv2/core.hpp>
#include "dicom/dicom_volume.h"
#include <QVector3D>
#include <functional>
#include <QMatrix4x4>

class RTDoseVolume
{
public:
    RTDoseVolume();

    bool loadFromFile(const QString& filename,
                     std::function<void(int, int)> progress = {});

    bool saveToFile(const QString& filename,
                   std::function<void(int, int)> progress = {}) const;

    int width() const { return m_width; }
    int height() const { return m_height; }
    int depth() const { return m_depth; }
    double spacingX() const { return m_spacingX; }
    double spacingY() const { return m_spacingY; }
    double spacingZ() const { return m_spacingZ; }
    double maxDose() const { return m_maxDose; }
    double originX() const { return m_originX; }
    double originY() const { return m_originY; }
    double originZ() const { return m_originZ; }
    QString frameOfReferenceUID() const { return m_frameUID; }
    bool hasIPP() const { return m_hasIPP; }

    QVector3D voxelToPatient(double x, double y, double z) const;
    // Native (no patientShift) patient coordinate for diagnostics/UI
    QVector3D voxelToPatientNative(double x, double y, double z) const;
    // Compute native patient-space extents (mm) from 8 volume corners
    bool nativeExtents(double &minX, double &maxX,
                       double &minY, double &maxY,
                       double &minZ, double &maxZ) const;
    QVector3D voxelToPatient(int x, int y, int z) const {
        return voxelToPatient(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z));
    }
    QVector3D patientToVoxel(const QVector3D& p) const;
    QVector3D patientToVoxelContinuous(const QVector3D& p) const;
    QVector3D patientToVoxelContinuousNative(const QVector3D& p) const;

    // Continuous voxel coordinate sampling
    float sampleDose(double x, double y, double z) const;

    // 任意の患者座標から線量(Gy)を取得（トリリニア補間）。
    // inside!=nullptr の場合、ボリューム内かどうかを返す。
    float doseAtPatient(const QVector3D& patient, bool* inside = nullptr) const;
    float doseAtPatientNative(const QVector3D& patient, bool* inside = nullptr) const;

    // 任意のDICOM患者座標(mm)での線量(Gy)を直接取得する薄いラッパ
    // 例: doseAtDicom(-125.0, -125.0, 50.0)
    float doseAtDicom(double x_mm, double y_mm, double z_mm, bool* inside = nullptr) const {
        return doseAtPatient(QVector3D(x_mm, y_mm, z_mm), inside);
    }

    // パティエントシフト（アライメント用）
    void setPatientShift(const QVector3D& shift) { m_patientShift = shift; }
    QVector3D patientShift() const { return m_patientShift; }

    // CT患者座標 -> Dose患者座標 へのアフィン（回転・拡大縮小なしの前提で回転＋並進を想定）
    void setCtToDoseTransform(const QMatrix4x4 &m) { m_ctToDose = m; }
    const QMatrix4x4 &ctToDoseTransform() const { return m_ctToDose; }
    bool hasIOP() const { return m_hasIOP; }

    const cv::Mat &data() const { return m_volume; }
    cv::Mat &data() { return m_volume; }

    // CTボリュームの向きを採用（IOP未定義のRTDOSE対策）
    void adoptOrientationFrom(const DicomVolume& ct) {
        // 行列の正規化を維持しつつ、CTの行・列・スライス方向余弦をコピー
        QVector3D row(ct.voxelToPatient(1,0,0) - ct.voxelToPatient(0,0,0));
        QVector3D col(ct.voxelToPatient(0,1,0) - ct.voxelToPatient(0,0,0));
        QVector3D slice(ct.voxelToPatient(0,0,1) - ct.voxelToPatient(0,0,0));
        if (row.length() > 0 && col.length() > 0 && slice.length() > 0) {
            row.normalize(); col.normalize(); slice.normalize();
            m_rowDir[0]=row.x(); m_rowDir[1]=row.y(); m_rowDir[2]=row.z();
            m_colDir[0]=col.x(); m_colDir[1]=col.y(); m_colDir[2]=col.z();
            m_sliceDir[0]=slice.x(); m_sliceDir[1]=slice.y(); m_sliceDir[2]=slice.z();
        }
    }

    void setFromMatAndGeometry(const cv::Mat &vol, const DicomVolume &ctVolume);

    // Setters for programmatic dose volume creation (e.g., from dose calculators)
    void setVolume(const cv::Mat &vol) {
        m_volume = vol;
        if (vol.dims == 3) {
            // 3D array: size[0]=depth, size[1]=height, size[2]=width
            m_depth = vol.size[0];
            m_height = vol.size[1];
            m_width = vol.size[2];
        } else if (vol.dims == 2) {
            // 2D array: rows=height, cols=width
            m_depth = 1;
            m_height = vol.rows;
            m_width = vol.cols;
        } else {
            // Invalid
            m_width = m_height = m_depth = 0;
        }
    }
    void setSpacing(double sx, double sy, double sz) { m_spacingX = sx; m_spacingY = sy; m_spacingZ = sz; }
    void setOrigin(double ox, double oy, double oz) { m_originX = ox; m_originY = oy; m_originZ = oz; }
    void setDirectionCosines(const double rowDir[3], const double colDir[3], const double sliceDir[3]) {
        for (int i = 0; i < 3; ++i) {
            m_rowDir[i] = rowDir[i];
            m_colDir[i] = colDir[i];
            m_sliceDir[i] = sliceDir[i];
        }
        m_hasIOP = true;
    }
    void computeMaxDose() {
        if (m_volume.empty()) {
            m_maxDose = 0.0;
            return;
        }

        // For 3D volumes, manually find max value
        m_maxDose = 0.0;
        if (m_volume.isContinuous()) {
            // Fast path: treat as 1D array
            const float* data = m_volume.ptr<float>();
            size_t total = m_volume.total();
            for (size_t i = 0; i < total; ++i) {
                if (data[i] > m_maxDose) {
                    m_maxDose = data[i];
                }
            }
        } else {
            // Slow path: use at() accessor for 3D array
            for (int z = 0; z < m_depth; ++z) {
                for (int y = 0; y < m_height; ++y) {
                    for (int x = 0; x < m_width; ++x) {
                        float val = m_volume.at<float>(z, y, x);
                        if (val > m_maxDose) {
                            m_maxDose = val;
                        }
                    }
                }
            }
        }
    }

    // 改良版リサンプリングメソッド
    QImage resampleSlice(const DicomVolume& refVol, int index, DicomVolume::Orientation ori) const;

    // デバッグ用カバレッジ表示
    QImage coverageSlice(const DicomVolume& refVol, int index,
                         DicomVolume::Orientation ori) const;

    // 従来の方法（互換性のため残す）
    QImage getOverlaySlice(int index, DicomVolume::Orientation ori,
                           const QSize& size) const;

    // デバッグ・テスト用メソッド（publicに移動）
    void testCoordinateTransforms(const DicomVolume& refVol) const;
    QImage createSimpleOverlay(const DicomVolume& refVol, int index, DicomVolume::Orientation ori) const;
    void getDoseStatistics(double& min, double& max, double& mean) const;
    void diagnoseDoseData() const;
    bool loadFromFileWithDiagnostics(const QString& filename);
    void analyzeRTDoseStructure(const QString& filename) const;
    void testSpecificCoordinate(const DicomVolume& refVol, int ctX, int ctY, int ctZ) const;
    QImage createDebugGridOverlay(const DicomVolume& refVol, int index,DicomVolume::Orientation ori) const;

private:
    // トリリニア補間ヘルパー
    float interpolateTrilinear(double x, double y, double z) const;

    cv::Mat m_volume; // float32 3D: depth x height x width
    int m_width{0};
    int m_height{0};
    int m_depth{0};
    double m_spacingX{1.0};
    double m_spacingY{1.0};
    double m_spacingZ{1.0};
    double m_maxDose{1.0};
    double m_originX{0.0};
    double m_originY{0.0};
    double m_originZ{0.0};
    QString m_frameUID;
    double m_rowDir[3]{1.0,0.0,0.0};
    double m_colDir[3]{0.0,1.0,0.0};
    double m_sliceDir[3]{0.0,0.0,1.0};
    std::vector<double> m_zOffsets;
    QVector3D m_patientShift{0.0, 0.0, 0.0};
    QMatrix4x4 m_ctToDose; // CT patient -> Dose patient (affine), identity by default
    bool m_hasIOP{false};
    bool m_hasIPP{false};
};

#endif // RTDOSE_VOLUME_H
