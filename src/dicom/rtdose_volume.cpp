#include "dicom/rtdose_volume.h"
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#include <QDebug>
#include <QDate>
#include <QTime>
#include <opencv2/imgproc.hpp>
#include <QVector3D>
#include <QPainter>  // 追加
#include <QPen>      // 追加
#include <QFont>     // 追加
#include <algorithm>
#include <limits>
#include <cmath>
#include <map>
#include <QRegularExpression>
#include <QtConcurrent>
#include <atomic>
#include <numeric>

RTDoseVolume::RTDoseVolume()
    : m_frameUID()
{
    m_ctToDose.setToIdentity();
}

QVector3D RTDoseVolume::voxelToPatient(double x, double y, double z) const
{
    QVector3D origin(m_originX, m_originY, m_originZ);
    QVector3D row(m_rowDir[0], m_rowDir[1], m_rowDir[2]);
    QVector3D col(m_colDir[0], m_colDir[1], m_colDir[2]);
    QVector3D slice(m_sliceDir[0], m_sliceDir[1], m_sliceDir[2]);
    // Interpolate Z offset for fractional indices when GFOV is available
    double zOffset = z * m_spacingZ;
    if (!m_zOffsets.empty()) {
        const int n = static_cast<int>(m_zOffsets.size());
        if (n == 1) {
            zOffset = m_zOffsets[0];
        } else {
            if (z <= 0.0) {
                double dz = m_zOffsets[1] - m_zOffsets[0];
                zOffset = m_zOffsets[0] + z * dz;
            } else if (z >= n - 1) {
                double dz = m_zOffsets[n - 1] - m_zOffsets[n - 2];
                zOffset = m_zOffsets[n - 1] + (z - (n - 1)) * dz;
            } else {
                int zi0 = static_cast<int>(std::floor(z));
                int zi1 = zi0 + 1;
                double t = z - zi0;
                zOffset = (1.0 - t) * m_zOffsets[zi0] + t * m_zOffsets[zi1];
            }
        }
    }
    
    // デバッグ出力（一時的）
    if (z == 0) { // 最初のスライスのみログ出力
        qDebug() << QString("voxelToPatient debug: z=%1, zOffset=%2").arg(z).arg(zOffset);
        qDebug() << QString("  origin: (%1,%2,%3)").arg(m_originX).arg(m_originY).arg(m_originZ);
        qDebug() << QString("  zOffsets available: %1").arg(!m_zOffsets.empty());
        if (!m_zOffsets.empty() && !m_zOffsets.empty()) {
            qDebug() << QString("  zOffsets[0]: %1").arg(m_zOffsets[0]);
        }
    }
    
    // DICOM座標系に従った変換
    QVector3D result = origin + 
                      row * (x * m_spacingX) + 
                      col * (y * m_spacingY) + 
                      slice * zOffset;
    
    // パティエントシフトを適用
    return result + m_patientShift;
}

QVector3D RTDoseVolume::voxelToPatientNative(double x, double y, double z) const
{
    QVector3D origin(m_originX, m_originY, m_originZ);
    QVector3D row(m_rowDir[0], m_rowDir[1], m_rowDir[2]);
    QVector3D col(m_colDir[0], m_colDir[1], m_colDir[2]);
    QVector3D slice(m_sliceDir[0], m_sliceDir[1], m_sliceDir[2]);

    // Interpolate Z offset for fractional indices when GFOV is available (native, no shift)
    double zOffset = z * m_spacingZ;
    if (!m_zOffsets.empty()) {
        const int n = static_cast<int>(m_zOffsets.size());
        if (n == 1) {
            zOffset = m_zOffsets[0];
        } else {
            if (z <= 0.0) {
                double dz = m_zOffsets[1] - m_zOffsets[0];
                zOffset = m_zOffsets[0] + z * dz;
            } else if (z >= n - 1) {
                double dz = m_zOffsets[n - 1] - m_zOffsets[n - 2];
                zOffset = m_zOffsets[n - 1] + (z - (n - 1)) * dz;
            } else {
                int zi0 = static_cast<int>(std::floor(z));
                int zi1 = zi0 + 1;
                double t = z - zi0;
                zOffset = (1.0 - t) * m_zOffsets[zi0] + t * m_zOffsets[zi1];
            }
        }
    }

    return origin + row * (x * m_spacingX) + col * (y * m_spacingY) + slice * zOffset;
}

QVector3D RTDoseVolume::patientToVoxel(const QVector3D& p) const
{
    // CT患者座標からDose患者座標へアフィンを適用
    QVector3D pDose = (m_ctToDose * QVector4D(p, 1.0f)).toVector3D();
    QVector3D origin(m_originX, m_originY, m_originZ);
    QVector3D rel = pDose - origin - m_patientShift;
    QVector3D row(m_rowDir[0], m_rowDir[1], m_rowDir[2]);
    QVector3D col(m_colDir[0], m_colDir[1], m_colDir[2]);
    QVector3D slice(m_sliceDir[0], m_sliceDir[1], m_sliceDir[2]);
    // Inverse of voxelToPatient: project the point onto the row and column
    // directions associated with the X (column) and Y (row) axes respectively.
    double x = QVector3D::dotProduct(row, rel) / m_spacingX;
    double y = QVector3D::dotProduct(col, rel) / m_spacingY;
    double z_mm = QVector3D::dotProduct(slice, rel);
    double z = 0.0;
    if (!m_zOffsets.empty()) {
        const auto &v = m_zOffsets;
        const int n = static_cast<int>(v.size());
        if (n == 1) {
            z = 0.0;
        } else {
            const bool asc = (v.back() >= v.front());
            if (asc) {
                auto it = std::lower_bound(v.begin(), v.end(), z_mm);
                if (it == v.begin()) {
                    z = 0.0;
                } else if (it == v.end()) {
                    z = static_cast<double>(n - 1);
                } else {
                    int i = static_cast<int>(it - v.begin()); // v[i-1] < z_mm <= v[i]
                    double dl = std::abs(z_mm - v[static_cast<size_t>(i - 1)]);
                    double dr = std::abs(v[static_cast<size_t>(i)] - z_mm);
                    z = (dl <= dr) ? static_cast<double>(i - 1) : static_cast<double>(i);
                }
            } else {
                // v is strictly descending
                // find i such that v[i] >= z_mm > v[i+1]
                int idx = 0;
                if (z_mm >= v.front()) {
                    idx = 0;
                } else if (z_mm <= v.back()) {
                    idx = n - 1;
                } else {
                    int lo = 0, hi = n - 1;
                    while (hi - lo > 1) {
                        int mid = (lo + hi) / 2;
                        if (v[mid] >= z_mm) {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    // choose closer of lo or hi
                    double dl = std::abs(z_mm - v[lo]);
                    double dr = std::abs(z_mm - v[hi]);
                    idx = (dl <= dr) ? lo : hi;
                }
                z = static_cast<double>(idx);
            }
        }
    } else {
        z = z_mm / m_spacingZ;
    }
    return QVector3D(x, y, z);
}

QVector3D RTDoseVolume::patientToVoxelContinuous(const QVector3D& p) const
{
    QVector3D pDose = (m_ctToDose * QVector4D(p, 1.0f)).toVector3D();
    QVector3D origin(m_originX, m_originY, m_originZ);
    QVector3D rel = pDose - origin - m_patientShift;
    
    QVector3D row(m_rowDir[0], m_rowDir[1], m_rowDir[2]);
    QVector3D col(m_colDir[0], m_colDir[1], m_colDir[2]);
    QVector3D slice(m_sliceDir[0], m_sliceDir[1], m_sliceDir[2]);
    
    double x = QVector3D::dotProduct(rel, row) / m_spacingX;
    double y = QVector3D::dotProduct(rel, col) / m_spacingY;
    double z_mm = QVector3D::dotProduct(rel, slice);
    
    double z = 0.0;
    if (!m_zOffsets.empty()) {
        const auto &v = m_zOffsets;
        const size_t n = v.size();
        if (n == 1) {
            z = 0.0;
        } else {
            const bool asc = (v.back() >= v.front());
            if (asc) {
                if (z_mm <= v.front()) {
                    double dz = (n > 1) ? (v[1] - v[0]) : m_spacingZ;
                    z = (dz != 0.0) ? (z_mm - v.front()) / dz : 0.0;
                } else if (z_mm >= v.back()) {
                    double dz = (n > 1) ? (v[n - 1] - v[n - 2]) : m_spacingZ;
                    z = (n - 1) + ((dz != 0.0) ? (z_mm - v[n - 1]) / dz : 0.0);
                } else {
                    auto it = std::lower_bound(v.begin() + 1, v.end(), z_mm);
                    size_t i = static_cast<size_t>(it - v.begin());
                    double denom = v[i] - v[i - 1];
                    double t = (denom != 0.0) ? (z_mm - v[i - 1]) / denom : 0.0;
                    z = static_cast<double>(i - 1) + t;
                }
            } else {
                // strictly descending
                if (z_mm >= v.front()) {
                    double dz = (n > 1) ? (v[0] - v[1]) : m_spacingZ;
                    z = (dz != 0.0) ? (v.front() - z_mm) / dz : 0.0;
                } else if (z_mm <= v.back()) {
                    double dz = (n > 1) ? (v[n - 2] - v[n - 1]) : m_spacingZ;
                    z = (n - 1) + ((dz != 0.0) ? (v.back() - z_mm) / dz : 0.0);
                } else {
                    // find i such that v[i] >= z_mm >= v[i+1]
                    size_t lo = 0, hi = n - 1;
                    while (hi - lo > 1) {
                        size_t mid = (lo + hi) / 2;
                        if (v[mid] >= z_mm) {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    double denom = v[lo] - v[hi];
                    double t = (denom != 0.0) ? (v[lo] - z_mm) / denom : 0.0;
                    z = static_cast<double>(lo) + t;
                }
            }
        }
    } else {
        z = z_mm / m_spacingZ;
    }
    
    return QVector3D(x, y, z);
}

QVector3D RTDoseVolume::patientToVoxelContinuousNative(const QVector3D& p) const
{
    // Do not apply patientShift (native geometry). Keep ctToDose in case it is used.
    QVector3D pDose = (m_ctToDose * QVector4D(p, 1.0f)).toVector3D();
    QVector3D origin(m_originX, m_originY, m_originZ);
    QVector3D rel = pDose - origin; // no shift subtraction

    QVector3D row(m_rowDir[0], m_rowDir[1], m_rowDir[2]);
    QVector3D col(m_colDir[0], m_colDir[1], m_colDir[2]);
    QVector3D slice(m_sliceDir[0], m_sliceDir[1], m_sliceDir[2]);

    double x = QVector3D::dotProduct(rel, row) / m_spacingX;
    double y = QVector3D::dotProduct(rel, col) / m_spacingY;
    double z_mm = QVector3D::dotProduct(rel, slice);

    double z = 0.0;
    if (!m_zOffsets.empty()) {
        const auto &v = m_zOffsets;
        const size_t n = v.size();
        if (n == 1) {
            z = 0.0;
        } else {
            const bool asc = (v.back() >= v.front());
            if (asc) {
                if (z_mm <= v.front()) {
                    double dz = (n > 1) ? (v[1] - v[0]) : m_spacingZ;
                    z = (dz != 0.0) ? (z_mm - v.front()) / dz : 0.0;
                } else if (z_mm >= v.back()) {
                    double dz = (n > 1) ? (v[n - 1] - v[n - 2]) : m_spacingZ;
                    z = (n - 1) + ((dz != 0.0) ? (z_mm - v[n - 1]) / dz : 0.0);
                } else {
                    auto it = std::lower_bound(v.begin() + 1, v.end(), z_mm);
                    size_t i = static_cast<size_t>(it - v.begin());
                    double denom = v[i] - v[i - 1];
                    double t = (denom != 0.0) ? (z_mm - v[i - 1]) / denom : 0.0;
                    z = static_cast<double>(i - 1) + t;
                }
            } else {
                if (z_mm >= v.front()) {
                    double dz = (n > 1) ? (v[0] - v[1]) : m_spacingZ;
                    z = (dz != 0.0) ? (v.front() - z_mm) / dz : 0.0;
                } else if (z_mm <= v.back()) {
                    double dz = (n > 1) ? (v[n - 2] - v[n - 1]) : m_spacingZ;
                    z = (n - 1) + ((dz != 0.0) ? (v.back() - z_mm) / dz : 0.0);
                } else {
                    size_t lo = 0, hi = n - 1;
                    while (hi - lo > 1) {
                        size_t mid = (lo + hi) / 2;
                        if (v[mid] >= z_mm) {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    double denom = v[lo] - v[hi];
                    double t = (denom != 0.0) ? (v[lo] - z_mm) / denom : 0.0;
                    z = static_cast<double>(lo) + t;
                }
            }
        }
    } else {
        z = z_mm / m_spacingZ;
    }

    return QVector3D(x, y, z);
}

float RTDoseVolume::doseAtPatientNative(const QVector3D& patient, bool* inside) const
{
    QVector3D vox = patientToVoxelContinuousNative(patient);
    bool in = (vox.x() >= 0 && vox.x() < m_width &&
               vox.y() >= 0 && vox.y() < m_height &&
               vox.z() >= 0 && vox.z() < m_depth);
    if (inside) *inside = in;
    if (!in) return 0.0f;
    return sampleDose(vox.x(), vox.y(), vox.z());
}

bool RTDoseVolume::nativeExtents(double &minX, double &maxX,
                                 double &minY, double &maxY,
                                 double &minZ, double &maxZ) const
{
    if (m_width <= 0 || m_height <= 0 || m_depth <= 0) return false;
    auto upd = [](double &mn, double &mx, double v){ mn = std::min(mn, v); mx = std::max(mx, v); };
    minX = minY = minZ = std::numeric_limits<double>::infinity();
    maxX = maxY = maxZ = -std::numeric_limits<double>::infinity();
    int xs[2] = {0, m_width - 1};
    int ys[2] = {0, m_height - 1};
    int zs[2] = {0, m_depth - 1};
    for (int ix : xs) for (int iy : ys) for (int iz : zs) {
        // Origin is now at voxel center, so use integer indices directly
        QVector3D p = voxelToPatientNative(ix, iy, iz);
        upd(minX, maxX, p.x());
        upd(minY, maxY, p.y());
        upd(minZ, maxZ, p.z());
    }
    return std::isfinite(minX) && std::isfinite(maxX) &&
           std::isfinite(minY) && std::isfinite(maxY) &&
           std::isfinite(minZ) && std::isfinite(maxZ);
}

float RTDoseVolume::sampleDose(double x, double y, double z) const
{
    return interpolateTrilinear(x, y, z);
}

float RTDoseVolume::doseAtPatient(const QVector3D& patient, bool* inside) const
{
    QVector3D vox = patientToVoxelContinuous(patient);
    bool in = (vox.x() >= 0 && vox.x() < m_width &&
               vox.y() >= 0 && vox.y() < m_height &&
               vox.z() >= 0 && vox.z() < m_depth);
    if (inside) *inside = in;
    if (!in) return 0.0f;
    return sampleDose(vox.x(), vox.y(), vox.z());
}

static cv::Mat getSliceAxialFloat(const cv::Mat& vol, int index)
{
    int depth = vol.size[0];
    int height = vol.size[1];
    int width = vol.size[2];
    if (index < 0 || index >= depth) return cv::Mat();
    return cv::Mat(height, width, CV_32F,
                   const_cast<float*>(vol.ptr<float>(index))).clone();
}

static cv::Mat getSliceSagittalFloat(const cv::Mat& vol, int index)
{
    int depth = vol.size[0];
    int height = vol.size[1];
    int width = vol.size[2];
    if (index < 0 || index >= width) return cv::Mat();
    cv::Mat slice(height, depth, CV_32F);
    for (int z = 0; z < depth; ++z) {
        const float* srcRow = vol.ptr<float>(z);
        for (int y = 0; y < height; ++y) {
            slice.at<float>(y, z) = srcRow[y * width + index];
        }
    }
    // Transpose then flip vertically so head-foot orientation is up.
    cv::transpose(slice, slice);
    cv::flip(slice, slice, 0);
    return slice;
}

static cv::Mat getSliceCoronalFloat(const cv::Mat& vol, int index)
{
    int depth = vol.size[0];
    int height = vol.size[1];
    int width = vol.size[2];
    if (index < 0 || index >= height) return cv::Mat();
    cv::Mat slice(width, depth, CV_32F);
    for (int z = 0; z < depth; ++z) {
        const float* srcRow = vol.ptr<float>(z);
        for (int x = 0; x < width; ++x) {
            slice.at<float>(x, z) = srcRow[index * width + x];
        }
    }
    // After transposing, flip vertically so superior is up.
    cv::transpose(slice, slice);
    cv::flip(slice, slice, 0);
    return slice;
}

bool RTDoseVolume::loadFromFile(const QString& filename,
                                std::function<void(int, int)> progress)
{
    qDebug() << "=== Loading RT-Dose ===" << filename;
    
    m_patientShift = QVector3D(0.0, 0.0, 0.0);

    DcmFileFormat file;
    if (file.loadFile(filename.toLocal8Bit().data()).bad()) {
        qWarning() << "Failed to load RTDOSE" << filename;
        return false;
    }
    
    DcmDataset* ds = file.getDataset();
    
    // 基本画像情報
    Uint16 rows = 0, cols = 0;
    ds->findAndGetUint16(DCM_Rows, rows);
    ds->findAndGetUint16(DCM_Columns, cols);
    qDebug() << QString("Image size: %1 x %2").arg(cols).arg(rows);
    
    // フレーム数
    int frames = 1; // デフォルトは1
    OFString framesStr;
    if (ds->findAndGetOFString(DCM_NumberOfFrames, framesStr).good()) {
        bool ok = false;
        int tmp = QString::fromLatin1(framesStr.c_str()).toInt(&ok);
        if (ok && tmp > 0) {
            frames = tmp;
        }
    }
    qDebug() << QString("Frames: %1").arg(frames);
    
    // Dose grid scaling and optional rescale
    double doseGridScaling = 1.0;
    ds->findAndGetFloat64(DCM_DoseGridScaling, doseGridScaling);
    double rescaleSlope = 1.0;
    double rescaleIntercept = 0.0;
    ds->findAndGetFloat64(DCM_RescaleSlope, rescaleSlope);
    ds->findAndGetFloat64(DCM_RescaleIntercept, rescaleIntercept);
    // RTDOSE は基本的に PixelData * DoseGridScaling が物理線量(Gy)
    // 一部データで RescaleSlope=0 が入っているケースがあるため無視する
    OFString modalityStr;
    if (ds->findAndGetOFString(DCM_Modality, modalityStr).good()) {
        QString modality = QString::fromLatin1(modalityStr.c_str()).trimmed();
        if (modality == "RTDOSE") {
            rescaleSlope = 1.0;
            rescaleIntercept = 0.0;
        }
    }
    // slope が 0 の場合は恒等に矯正（安全策）
    if (rescaleSlope == 0.0) rescaleSlope = 1.0;
    // DoseUnits (3004,0002): if centigray, convert to Gy later by 0.01
    double unitsScaleToGy = 1.0;
    OFString doseUnitsStr;
    if (ds->findAndGetOFString(DcmTagKey(0x3004, 0x0002), doseUnitsStr).good()) {
        QString du = QString::fromLatin1(doseUnitsStr.c_str()).trimmed().toLower();
        if (du.contains("cgy") || du.contains("centi")) {
            unitsScaleToGy = 0.01; // convert cGy -> Gy
        }
        qDebug() << "DoseUnits:" << QString::fromLatin1(doseUnitsStr.c_str()) << ", unitsScaleToGy:" << unitsScaleToGy;
    }
    qDebug() << QString("DoseGridScaling: %1, RescaleSlope: %2, Intercept: %3")
                    .arg(doseGridScaling, 0, 'e', 6)
                    .arg(rescaleSlope, 0, 'f', 6)
                    .arg(rescaleIntercept, 0, 'f', 6);
    
    // Frame of Reference UID
    OFString frameUIDStr;
    if (ds->findAndGetOFString(DCM_FrameOfReferenceUID, frameUIDStr).good()) {
        m_frameUID = QString::fromLatin1(frameUIDStr.c_str());
    }
    qDebug() << "RTDOSE FrameOfReferenceUID:" << m_frameUID;
    
    // PatientPosition (0018,5100)
    {
        OFString pos;
        if (ds->findAndGetOFString(DCM_PatientPosition, pos).good()) {
            qDebug() << "RTDOSE PatientPosition:" << QString::fromLatin1(pos.c_str());
        } else {
            qDebug() << "RTDOSE PatientPosition: NOT FOUND";
        }
    }
    
    // ===== 座標情報の取得 =====
    double finalOrigin[3] = {0.0, 0.0, 0.0};
    double rowSpacing = 1.0, colSpacing = 1.0;
    double finalOrientation[6] = {1,0,0, 0,1,0};

    // ImagePositionPatient (0020,0032)
    const Float64* ippArr = nullptr; unsigned long ippCount = 0;
    if (ds->findAndGetFloat64Array(DCM_ImagePositionPatient, ippArr, &ippCount).good() && ippArr && ippCount >= 3) {
        finalOrigin[0] = ippArr[0];
        finalOrigin[1] = ippArr[1];
        finalOrigin[2] = ippArr[2];
        qDebug() << "Origin from top-level IPP (float64)";
        m_hasIPP = true;
    } else {
        // Fallback: DS multi-valued components individually
        OFString s0, s1, s2;
        bool ok0 = ds->findAndGetOFString(DCM_ImagePositionPatient, s0, 0).good();
        bool ok1 = ds->findAndGetOFString(DCM_ImagePositionPatient, s1, 1).good();
        bool ok2 = ds->findAndGetOFString(DCM_ImagePositionPatient, s2, 2).good();
        if (ok0 && ok1 && ok2) {
            bool p0=false,p1=false,p2=false;
            double v0 = QString::fromLatin1(s0.c_str()).toDouble(&p0);
            double v1 = QString::fromLatin1(s1.c_str()).toDouble(&p1);
            double v2 = QString::fromLatin1(s2.c_str()).toDouble(&p2);
            if (p0 && p1 && p2) {
                finalOrigin[0]=v0; finalOrigin[1]=v1; finalOrigin[2]=v2;
                qDebug() << "Origin from top-level IPP (DS components)";
                m_hasIPP = true;
            }
        }
        if (finalOrigin[0]==0.0 && finalOrigin[1]==0.0 && finalOrigin[2]==0.0) {
            qDebug() << "ImagePositionPatient not found at top-level, trying FG sequences...";
        }
    }
    qDebug() << QString("Origin: (%1, %2, %3)")
                    .arg(finalOrigin[0], 0, 'f', 3)
                    .arg(finalOrigin[1], 0, 'f', 3)
                    .arg(finalOrigin[2], 0, 'f', 3);

    // PixelSpacing (0028,0030)
    const Float64* psArr = nullptr; unsigned long psCount = 0;
    OFCondition psCond = ds->findAndGetFloat64Array(DCM_PixelSpacing, psArr, &psCount);
    if (psCond.good() && psArr && psCount >= 2) {
        // PixelSpacing = [row spacing, column spacing]
        rowSpacing = psArr[0];
        colSpacing = psArr[1];
    } else {
        // Try DS string
        OFString psStr;
        if (ds->findAndGetOFString(DCM_PixelSpacing, psStr).good()) {
            QString s = QString::fromLatin1(psStr.c_str()).trimmed();
            QStringList parts = s.split("\\", Qt::SkipEmptyParts);
            if (parts.size() >= 2) {
                bool ok1=false, ok2=false; double r = parts[0].toDouble(&ok1); double c = parts[1].toDouble(&ok2);
                if (ok1 && ok2) { rowSpacing = r; colSpacing = c; }
            } else if (parts.size() == 1) {
                bool ok=false; double v = parts[0].toDouble(&ok);
                if (ok) { rowSpacing = v; colSpacing = v; }
            }
        }
        if (rowSpacing <= 0.0 || colSpacing <= 0.0) {
            // Try Shared Functional Groups: Pixel Measures Sequence (0028,9110)
            DcmSequenceOfItems* sharedFG = nullptr;
            if (ds->findAndGetSequence(DCM_SharedFunctionalGroupsSequence, sharedFG).good() && sharedFG && sharedFG->card() > 0) {
                DcmItem* fgItem = sharedFG->getItem(0);
                if (fgItem) {
                    DcmSequenceOfItems* pixMeas = nullptr;
                    if (fgItem->findAndGetSequence(DcmTagKey(0x0028,0x9110), pixMeas).good() && pixMeas && pixMeas->card() > 0) {
                        DcmItem* pmItem = pixMeas->getItem(0);
                        if (pmItem) {
                            const Float64* ps2 = nullptr; unsigned long cnt2 = 0;
                            if (pmItem->findAndGetFloat64Array(DCM_PixelSpacing, ps2, &cnt2).good() && ps2 && cnt2 >= 2) {
                                rowSpacing = ps2[0]; colSpacing = ps2[1];
                            } else {
                                OFString s2;
                                if (pmItem->findAndGetOFString(DCM_PixelSpacing, s2).good()) {
                                    QString ss = QString::fromLatin1(s2.c_str()).trimmed();
                                    QStringList p = ss.split("\\", Qt::SkipEmptyParts);
                                    if (p.size() >= 2) {
                                        bool ok1=false, ok2=false; double r = p[0].toDouble(&ok1); double c = p[1].toDouble(&ok2);
                                        if (ok1 && ok2) { rowSpacing = r; colSpacing = c; }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (rowSpacing <= 0.0 || colSpacing <= 0.0) {
            // Try Per-frame Functional Groups: Pixel Measures (first frame)
            DcmSequenceOfItems* perFrame = nullptr;
            if (ds->findAndGetSequence(DCM_PerFrameFunctionalGroupsSequence, perFrame).good() && perFrame && perFrame->card() > 0) {
                DcmItem* pf0 = perFrame->getItem(0);
                if (pf0) {
                    DcmSequenceOfItems* pixMeas = nullptr;
                    if (pf0->findAndGetSequence(DcmTagKey(0x0028,0x9110), pixMeas).good() && pixMeas && pixMeas->card() > 0) {
                        DcmItem* pmItem = pixMeas->getItem(0);
                        if (pmItem) {
                            const Float64* ps2 = nullptr; unsigned long cnt2 = 0;
                            if (pmItem->findAndGetFloat64Array(DCM_PixelSpacing, ps2, &cnt2).good() && ps2 && cnt2 >= 2) {
                                rowSpacing = ps2[0]; colSpacing = ps2[1];
                            } else {
                                OFString s2;
                                if (pmItem->findAndGetOFString(DCM_PixelSpacing, s2).good()) {
                                    QString ss = QString::fromLatin1(s2.c_str()).trimmed();
                                    QStringList p = ss.split("\\", Qt::SkipEmptyParts);
                                    if (p.size() >= 2) {
                                        bool ok1=false, ok2=false; double r = p[0].toDouble(&ok1); double c = p[1].toDouble(&ok2);
                                        if (ok1 && ok2) { rowSpacing = r; colSpacing = c; }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (rowSpacing <= 0.0 || colSpacing <= 0.0) {
            qDebug() << "PixelSpacing not found, defaulting to 1mm";
            rowSpacing = colSpacing = 1.0;
        }
    }
    qDebug() << QString("PixelSpacing: row=%1 mm, col=%2 mm")
                    .arg(rowSpacing, 0, 'f', 3)
                    .arg(colSpacing, 0, 'f', 3);

    // ImageOrientationPatient (0020,0037)
    m_hasIOP = false;
    const Float64* iopArr = nullptr; unsigned long iopCount = 0;
    if (ds->findAndGetFloat64Array(DCM_ImageOrientationPatient, iopArr, &iopCount).good() && iopArr && iopCount >= 6) {
        for (int i = 0; i < 6; ++i) finalOrientation[i] = iopArr[i];
        m_hasIOP = true;
        qDebug() << "IOP from top-level (float64)";
    } else {
        // Try DS multi-valued at top-level
        OFString s[6]; bool okComp=true; bool okNum[6]={false,false,false,false,false,false}; double v[6]={0};
        for (int i=0;i<6;++i) {
            if (!ds->findAndGetOFString(DCM_ImageOrientationPatient, s[i], i).good()) { okComp=false; break; }
            v[i] = QString::fromLatin1(s[i].c_str()).toDouble(&okNum[i]);
            okComp = okComp && okNum[i];
        }
        if (okComp) {
            for (int i=0;i<6;++i) finalOrientation[i]=v[i];
            m_hasIOP = true;
            qDebug() << "IOP from top-level (DS components)";
        }
        // Try Shared Functional Groups: Plane Orientation (0020,9116)
        DcmSequenceOfItems* sharedFG = nullptr;
        if (ds->findAndGetSequence(DCM_SharedFunctionalGroupsSequence, sharedFG).good() && sharedFG && sharedFG->card() > 0) {
            DcmItem* fgItem = sharedFG->getItem(0);
            if (fgItem) {
                DcmSequenceOfItems* planeOri = nullptr;
                if (fgItem->findAndGetSequence(DcmTagKey(0x0020,0x9116), planeOri).good() && planeOri && planeOri->card() > 0) {
                    DcmItem* poItem = planeOri->getItem(0);
                    if (poItem) {
                        const Float64* arr = nullptr; unsigned long cnt = 0;
                        if (poItem->findAndGetFloat64Array(DCM_ImageOrientationPatient, arr, &cnt).good() && arr && cnt >= 6) {
                            for (int i = 0; i < 6; ++i) finalOrientation[i] = arr[i];
                            m_hasIOP = true;
                        } else {
                            OFString s;
                            if (poItem->findAndGetOFString(DCM_ImageOrientationPatient, s).good()) {
                                QString so = QString::fromLatin1(s.c_str()).trimmed();
                                QStringList p = so.split("\\", Qt::SkipEmptyParts);
                                if (p.size() >= 6) {
                                    bool ok[6]; double v[6];
                                    for (int i=0;i<6;++i){ v[i]=p[i].toDouble(&ok[i]); }
                                    bool all=true; for(int i=0;i<6;++i) all=all&&ok[i];
                                    if (all) { for (int i=0;i<6;++i) finalOrientation[i]=v[i]; m_hasIOP = true; }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (finalOrientation[0]==0 && finalOrientation[4]==0) {
            qDebug() << "ImageOrientationPatient not found, assuming identity";
        }
    }

    // GridFrameOffsetVector (3004,000C) provides Z offsets for each frame
    // Read as DS (Decimal String) - matching how we write it
    m_zOffsets.clear();

    // First try reading as DS element (correct VR)
    DcmElement *gfovElem = nullptr;
    if (ds->findAndGetElement(DcmTagKey(0x3004, 0x000C), gfovElem).good() && gfovElem) {
        unsigned long vm = gfovElem->getVM();
        qDebug() << "=== GFOV READ FROM DICOM (DS Element) ===";
        qDebug() << QString("  Found GFOV element with VM=%1 (value multiplicity)").arg(vm);

        // Read each value individually using getFloat64(value, pos)
        m_zOffsets.reserve(vm);
        for (unsigned long i = 0; i < vm; ++i) {
            Float64 val;
            if (gfovElem->getFloat64(val, i).good()) {
                m_zOffsets.push_back(val);
            }
        }

        qDebug() << QString("  Successfully loaded %1 GFOV values").arg(m_zOffsets.size());
        if (m_zOffsets.size() >= 3) {
            qDebug() << QString("  m_zOffsets[0]=%1, [1]=%2, [2]=%3")
                .arg(m_zOffsets[0], 0, 'f', 6).arg(m_zOffsets[1], 0, 'f', 6).arg(m_zOffsets[2], 0, 'f', 6);
        }
        frames = static_cast<int>(m_zOffsets.size());
    }
    // Fallback: try Float64Array (for backwards compatibility with old files)
    else {
        const Float64* gfo = nullptr;
        unsigned long gfoCount = 0;
        if (ds->findAndGetFloat64Array(DcmTagKey(0x3004, 0x000C), gfo, &gfoCount).good() && gfo && gfoCount > 0) {
            qDebug() << "=== GFOV READ FROM DICOM (Float64Array - legacy) ===";
            m_zOffsets.resize(gfoCount);
            for (unsigned long i = 0; i < gfoCount; ++i) {
                m_zOffsets[i] = gfo[i];
            }
            frames = static_cast<int>(gfoCount);
        }
    }

    // Final fallback if GFOV not found
    if (m_zOffsets.empty()) {
        // Fallback: use SliceThickness or SpacingBetweenSlices if present
        double sliceThickness = 0.0;
        double spacingBetweenSlices = 0.0;
        ds->findAndGetFloat64(DCM_SliceThickness, sliceThickness);
        ds->findAndGetFloat64(DCM_SpacingBetweenSlices, spacingBetweenSlices);
        double dz = spacingBetweenSlices > 0.0 ? spacingBetweenSlices : (sliceThickness > 0.0 ? sliceThickness : 1.0);
        m_zOffsets.resize(std::max(frames, 1));
        for (int i = 0; i < frames; ++i) m_zOffsets[static_cast<size_t>(i)] = i * dz;
    }
    if (!m_zOffsets.empty()) {
        double first = m_zOffsets.front();
        double last  = m_zOffsets.back();
        bool asc = (last >= first);
        qDebug() << QString("Z Offsets (GFOV): %1 frames, first=%2, last=%3, ascending=%4")
                        .arg(m_zOffsets.size())
                        .arg(first, 0, 'f', 3)
                        .arg(last, 0, 'f', 3)
                        .arg(asc);
    } else {
        qDebug() << QString("Z Offsets: %1 frames").arg(m_zOffsets.size());
    }

    // If origin is missing, try Shared Functional Groups: Plane Position (Patient) (0020,9113)
    if (finalOrigin[0]==0.0 && finalOrigin[1]==0.0 && finalOrigin[2]==0.0) {
        DcmSequenceOfItems* sharedFG = nullptr;
        if (ds->findAndGetSequence(DCM_SharedFunctionalGroupsSequence, sharedFG).good() && sharedFG && sharedFG->card() > 0) {
            DcmItem* fgItem = sharedFG->getItem(0);
            if (fgItem) {
                DcmSequenceOfItems* planePos = nullptr;
                if (fgItem->findAndGetSequence(DcmTagKey(0x0020,0x9113), planePos).good() && planePos && planePos->card() > 0) {
                    DcmItem* ppItem = planePos->getItem(0);
                    if (ppItem) {
                        const Float64* ipp2 = nullptr; unsigned long c2 = 0;
                        if (ppItem->findAndGetFloat64Array(DCM_ImagePositionPatient, ipp2, &c2).good() && ipp2 && c2 >= 3) {
                            finalOrigin[0]=ipp2[0]; finalOrigin[1]=ipp2[1]; finalOrigin[2]=ipp2[2];
                            qDebug() << "Origin from Shared FG PlanePosition(Patient)";
                        } else {
                            OFString s;
                            if (ppItem->findAndGetOFString(DCM_ImagePositionPatient, s).good()) {
                                QString so = QString::fromLatin1(s.c_str()).trimmed();
                                QStringList p = so.split("\\", Qt::SkipEmptyParts);
                                if (p.size() >= 3) {
                                    bool ok[3]; double v[3];
                                    for(int i=0;i<3;++i){ v[i]=p[i].toDouble(&ok[i]); }
                                    if (ok[0]&&ok[1]&&ok[2]) {
                                        finalOrigin[0]=v[0]; finalOrigin[1]=v[1]; finalOrigin[2]=v[2];
                                        qDebug() << "Origin from Shared FG PlanePosition(Patient) (DS)";
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // If origin is still missing, try Per-frame Functional Groups for frame 0: Plane Position (Patient) (0020,9113)
    if (finalOrigin[0]==0.0 && finalOrigin[1]==0.0 && finalOrigin[2]==0.0) {
        DcmSequenceOfItems* perFrame = nullptr;
        if (ds->findAndGetSequence(DCM_PerFrameFunctionalGroupsSequence, perFrame).good() && perFrame && perFrame->card() > 0) {
            DcmItem* pfItem0 = perFrame->getItem(0);
            if (pfItem0) {
                DcmSequenceOfItems* planePos = nullptr;
                if (pfItem0->findAndGetSequence(DcmTagKey(0x0020,0x9113), planePos).good() && planePos && planePos->card() > 0) {
                    DcmItem* ppItem = planePos->getItem(0);
                    if (ppItem) {
                        const Float64* ipp2 = nullptr; unsigned long c2=0;
                        if (ppItem->findAndGetFloat64Array(DCM_ImagePositionPatient, ipp2, &c2).good() && ipp2 && c2 >= 3) {
                            finalOrigin[0]=ipp2[0]; finalOrigin[1]=ipp2[1]; finalOrigin[2]=ipp2[2];
                            m_hasIPP = true;
                        } else {
                            OFString s;
                            if (ppItem->findAndGetOFString(DCM_ImagePositionPatient, s).good()) {
                                QString so = QString::fromLatin1(s.c_str()).trimmed();
                                QStringList p = so.split("\\", Qt::SkipEmptyParts);
                                if (p.size() >= 3) {
                                    bool ok[3]; double v[3];
                                    for(int i=0;i<3;++i){ v[i]=p[i].toDouble(&ok[i]); }
                                    if (ok[0]&&ok[1]&&ok[2]) { finalOrigin[0]=v[0]; finalOrigin[1]=v[1]; finalOrigin[2]=v[2]; m_hasIPP = true; }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // ===== ボリューム構造の設定 =====
    
    m_width = cols;
    m_height = rows;
    m_depth = frames;
    
    m_spacingX = colSpacing;  // Column spacing -> X
    m_spacingY = rowSpacing;  // Row spacing   -> Y

    // spacingZ: average delta of zOffsets if available
    qDebug() << "=== SPACING Z CALCULATION ===";
    if (m_zOffsets.size() >= 2) {
        if (m_zOffsets.size() >= 3) {
            qDebug() << QString("  m_zOffsets[0]=%1, [1]=%2, [2]=%3")
                .arg(m_zOffsets[0]).arg(m_zOffsets[1]).arg(m_zOffsets[2]);
        }
        double sumdz = 0.0; int cnt = 0;
        for (size_t i = 1; i < m_zOffsets.size(); ++i) {
            sumdz += std::abs(m_zOffsets[i] - m_zOffsets[i-1]);
            ++cnt;
        }
        m_spacingZ = (cnt > 0) ? sumdz / cnt : 1.0;
        qDebug() << QString("  Calculated spacingZ from m_zOffsets: %1 (sumdz=%2, cnt=%3)")
            .arg(m_spacingZ).arg(sumdz).arg(cnt);
    } else {
        m_spacingZ = 1.0;
        qDebug() << "  Using default spacingZ: 1.0 (m_zOffsets.size < 2)";
    }

    m_originX = finalOrigin[0];
    m_originY = finalOrigin[1];
    m_originZ = finalOrigin[2];

    // デバッグ: 読み込み時の値を出力
    qDebug() << "=== RTDOSE LOAD DEBUG ===";
    qDebug() << QString("  Final spacing: %1 x %2 x %3").arg(m_spacingX).arg(m_spacingY).arg(m_spacingZ);
    qDebug() << QString("  ImagePositionPatient loaded: (%1, %2, %3)").arg(m_originX).arg(m_originY).arg(m_originZ);
    if (!m_zOffsets.empty()) {
        qDebug() << QString("  GridFrameOffsetVector[0] loaded: %1").arg(m_zOffsets[0]);
        qDebug() << QString("  GridFrameOffsetVector size loaded: %1").arg(m_zOffsets.size());
    }

    // 方向余弦 - Normalize orientation vectors and derive slice direction
    QVector3D row(finalOrientation[0], finalOrientation[1], finalOrientation[2]);
    QVector3D col(finalOrientation[3], finalOrientation[4], finalOrientation[5]);
    if (row.length() == 0.0 || col.length() == 0.0) {
        row = QVector3D(1,0,0);
        col = QVector3D(0,1,0);
    }
    row.normalize();
    col.normalize();
    QVector3D slice = QVector3D::crossProduct(row, col).normalized();
    m_rowDir[0] = row.x(); m_rowDir[1] = row.y(); m_rowDir[2] = row.z();
    m_colDir[0] = col.x(); m_colDir[1] = col.y(); m_colDir[2] = col.z();
    m_sliceDir[0] = slice.x(); m_sliceDir[1] = slice.y(); m_sliceDir[2] = slice.z();
    qDebug() << QString("RTDOSE IOP adopted: hasIOP=%1, row=(%2,%3,%4) col=(%5,%6,%7) slice=(%8,%9,%10)")
                    .arg(m_hasIOP)
                    .arg(row.x(), 0, 'f', 6).arg(row.y(), 0, 'f', 6).arg(row.z(), 0, 'f', 6)
                    .arg(col.x(), 0, 'f', 6).arg(col.y(), 0, 'f', 6).arg(col.z(), 0, 'f', 6)
                    .arg(slice.x(), 0, 'f', 6).arg(slice.y(), 0, 'f', 6).arg(slice.z(), 0, 'f', 6);
    
    qDebug() << QString("=== Final RT-Dose Coordinate System ===");
    qDebug() << QString("Size: %1 x %2 x %3").arg(m_width).arg(m_height).arg(m_depth);
    qDebug() << QString("Origin: (%1, %2, %3)")
                .arg(m_originX, 0, 'f', 1).arg(m_originY, 0, 'f', 1).arg(m_originZ, 0, 'f', 1);
    qDebug() << QString("Spacing: (%1, %2, %3)")
                .arg(m_spacingX, 0, 'f', 3).arg(m_spacingY, 0, 'f', 3).arg(m_spacingZ, 0, 'f', 3);

    // Report patient-space extents (mm) for the whole RTDOSE volume
    if (m_width > 0 && m_height > 0 && m_depth > 0) {
        auto upd = [](double &mn, double &mx, double v){ mn = std::min(mn, v); mx = std::max(mx, v); };
        double minX = std::numeric_limits<double>::infinity();
        double minY = std::numeric_limits<double>::infinity();
        double minZ = std::numeric_limits<double>::infinity();
        double maxX = -std::numeric_limits<double>::infinity();
        double maxY = -std::numeric_limits<double>::infinity();
        double maxZ = -std::numeric_limits<double>::infinity();

        int xs[2] = {0, m_width - 1};
        int ys[2] = {0, m_height - 1};
        int zs[2] = {0, m_depth - 1};
        for (int ix : xs) for (int iy : ys) for (int iz : zs) {
            // Origin is now at voxel center, so use integer indices directly
            QVector3D p = voxelToPatient(ix, iy, iz);
            upd(minX, maxX, p.x());
            upd(minY, maxY, p.y());
            upd(minZ, maxZ, p.z());
        }
        qDebug() << QString("RTDOSE Extents (patient mm): X:[%1, %2] Y:[%3, %4] Z:[%5, %6]")
                        .arg(minX, 0, 'f', 3).arg(maxX, 0, 'f', 3)
                        .arg(minY, 0, 'f', 3).arg(maxY, 0, 'f', 3)
                        .arg(minZ, 0, 'f', 3).arg(maxZ, 0, 'f', 3);
    }
    
    // ===== ピクセルデータ読み込み =====

    ds->chooseRepresentation(EXS_LittleEndianExplicit, nullptr);

    // Check pixel representation (signed/unsigned) and basic pixel attributes
    Uint16 pixelRepresentation = 0; // 0=unsigned, 1=signed
    ds->findAndGetUint16(DCM_PixelRepresentation, pixelRepresentation);
    Uint16 bitsAllocated = 0, bitsStored = 0, highBit = 0, samplesPerPixel = 1;
    ds->findAndGetUint16(DCM_BitsAllocated, bitsAllocated);
    ds->findAndGetUint16(DCM_BitsStored, bitsStored);
    ds->findAndGetUint16(DCM_HighBit, highBit);
    ds->findAndGetUint16(DCM_SamplesPerPixel, samplesPerPixel);
    // Transfer syntax
    DcmXfer xfer(ds->getOriginalXfer());
    qDebug() << QString("Pixel info: BitsAllocated=%1 BitsStored=%2 HighBit=%3 PixelRep=%4 SamplesPerPixel=%5 Xfer=%6")
                    .arg(bitsAllocated).arg(bitsStored).arg(highBit).arg(pixelRepresentation).arg(samplesPerPixel)
                    .arg(xfer.getXferName());

    // Validate expected element count vs actual
    size_t expectedCount = static_cast<size_t>(cols) * rows * frames;
    DcmElement *pixElem = nullptr;
    size_t availableCount = 0;
    size_t availableWords = 0;
    if (ds->findAndGetElement(DCM_PixelData, pixElem).good() && pixElem) {
        // Length in bytes may be undefined for compressed; fallback if cannot get
        Uint32 len = pixElem->getLength();
        if (len > 0) {
            // compute bytes per sample
            unsigned bytesPerSample = (bitsAllocated > 0 ? (bitsAllocated + 7) / 8 : 2) * (samplesPerPixel > 0 ? samplesPerPixel : 1);
            if (bytesPerSample == 0) bytesPerSample = 2;
            availableCount = static_cast<size_t>(len / bytesPerSample);
            availableWords = static_cast<size_t>(len / 2);
        }
    }
    if (availableCount > 0 && availableCount < expectedCount) {
        // Adjust frames to match available pixel samples
        size_t planeSize = static_cast<size_t>(cols) * rows;
        if (planeSize > 0) {
            frames = static_cast<int>(availableCount / planeSize);
            m_depth = frames;
            qDebug() << QString("Adjusted frames based on pixel length: %1").arg(frames);
            // If zOffsets size mismatches, resize accordingly
            if (m_zOffsets.size() != static_cast<size_t>(frames)) {
                m_zOffsets.resize(static_cast<size_t>(frames));
                for (int i = 0; i < frames; ++i) m_zOffsets[static_cast<size_t>(i)] = i * m_spacingZ;
            }
        }
    }

    const Uint16* data16 = nullptr;
    const Sint16* dataS16 = nullptr;
    const Uint32* dataU32 = nullptr;
    const Sint32* dataS32 = nullptr;
    const Float32* dataF32 = nullptr;
    const Float64* dataF64 = nullptr;
    bool haveU16 = ds->findAndGetUint16Array(DCM_PixelData, data16).good() && data16 != nullptr;
    bool haveS16 = ds->findAndGetSint16Array(DCM_PixelData, dataS16).good() && dataS16 != nullptr;
    bool haveU32 = ds->findAndGetUint32Array(DCM_PixelData, dataU32).good() && dataU32 != nullptr;
    bool haveS32 = ds->findAndGetSint32Array(DCM_PixelData, dataS32).good() && dataS32 != nullptr;
    // Float Pixel Data (7FE0,0008) and Double Float Pixel Data (7FE0,0009)
    bool haveF32 = ds->findAndGetFloat32Array(DcmTagKey(0x7FE0,0x0008), dataF32).good() && dataF32 != nullptr;
    bool haveF64 = ds->findAndGetFloat64Array(DcmTagKey(0x7FE0,0x0009), dataF64).good() && dataF64 != nullptr;

    bool canDecode = (haveF64 || haveF32 ||
                      (bitsAllocated == 32 && ((pixelRepresentation == 0 && haveU32) || (pixelRepresentation == 1 && haveS32) || haveU32 || haveS32 || haveU16)) ||
                      (bitsAllocated <= 16 && ((pixelRepresentation == 0 && haveU16) || (pixelRepresentation == 1 && haveS16) || haveU16 || haveS16)));
    if (canDecode) {
        QString decodePath = QString("Unknown");
        if (haveF64) decodePath = "Float64";
        else if (haveF32) decodePath = "Float32";
        else if (bitsAllocated == 32 && (haveU32 || haveS32)) decodePath = (pixelRepresentation == 1 && haveS32) ? "Sint32" : (haveU32 ? "Uint32" : "Sint32");
        else if (bitsAllocated == 32 && haveU16) decodePath = "Packed32FromUint16";
        else if (haveS16 || haveU16) decodePath = (pixelRepresentation == 1 && haveS16) ? "Sint16" : "Uint16";
        qDebug() << "Dose pixel decode path:" << decodePath;
        int sizes[3] = {m_depth, m_height, m_width};
        m_volume.create(3, sizes, CV_32F);

        QVector<float> sliceMax(m_depth, 0.0f);
        size_t totalElements = static_cast<size_t>(m_width) * m_height * m_depth;

        if (progress)
            progress(0, m_depth);

        QVector<int> zIndices(m_depth);
        std::iota(zIndices.begin(), zIndices.end(), 0);
        std::atomic<int> done{0};

        const double intFactor = (rescaleSlope)*doseGridScaling*unitsScaleToGy;
        const double intOffset = (rescaleIntercept)*doseGridScaling*unitsScaleToGy;
        const double floatFactor = doseGridScaling*unitsScaleToGy;

        if (haveF64) {
            QtConcurrent::blockingMap(zIndices, [&](int z) {
                float *dst = m_volume.ptr<float>(z);
                size_t idx = static_cast<size_t>(z) * m_width * m_height;
                float localMax = 0.0f;
                for (int y = 0; y < m_height; ++y) {
                    for (int x = 0; x < m_width; ++x, ++idx) {
                        float val = (idx < totalElements)
                            ? static_cast<float>(dataF64[idx] * floatFactor)
                            : 0.0f;
                        dst[y * m_width + x] = val;
                        if (val > localMax) localMax = val;
                    }
                }
                sliceMax[z] = localMax;
                int current = done.fetch_add(1) + 1;
                if (progress) progress(current, m_depth);
            });
        } else if (haveF32) {
            QtConcurrent::blockingMap(zIndices, [&](int z) {
                float *dst = m_volume.ptr<float>(z);
                size_t idx = static_cast<size_t>(z) * m_width * m_height;
                float localMax = 0.0f;
                for (int y = 0; y < m_height; ++y) {
                    for (int x = 0; x < m_width; ++x, ++idx) {
                        float val = (idx < totalElements)
                            ? static_cast<float>(static_cast<double>(dataF32[idx]) * floatFactor)
                            : 0.0f;
                        dst[y * m_width + x] = val;
                        if (val > localMax) localMax = val;
                    }
                }
                sliceMax[z] = localMax;
                int current = done.fetch_add(1) + 1;
                if (progress) progress(current, m_depth);
            });
        } else if (bitsAllocated == 32 && (haveU32 || haveS32)) {
            const bool useS32 = (pixelRepresentation == 1 && haveS32);
            QtConcurrent::blockingMap(zIndices, [&](int z) {
                float *dst = m_volume.ptr<float>(z);
                size_t idx = static_cast<size_t>(z) * m_width * m_height;
                float localMax = 0.0f;
                for (int y = 0; y < m_height; ++y) {
                    for (int x = 0; x < m_width; ++x, ++idx) {
                        float val = 0.0f;
                        if (idx < totalElements) {
                            double raw = useS32 ? static_cast<double>(dataS32[idx])
                                                : static_cast<double>(dataU32[idx]);
                            val = static_cast<float>(raw * intFactor + intOffset);
                        }
                        dst[y * m_width + x] = val;
                        if (val > localMax) localMax = val;
                    }
                }
                sliceMax[z] = localMax;
                int current = done.fetch_add(1) + 1;
                if (progress) progress(current, m_depth);
            });
        } else if (bitsAllocated == 32 && haveU16) {
            QtConcurrent::blockingMap(zIndices, [&](int z) {
                float *dst = m_volume.ptr<float>(z);
                size_t idx = static_cast<size_t>(z) * m_width * m_height;
                float localMax = 0.0f;
                for (int y = 0; y < m_height; ++y) {
                    for (int x = 0; x < m_width; ++x, ++idx) {
                        float val = 0.0f;
                        if (idx < totalElements) {
                            size_t wIndex = idx * 2;
                            if (availableWords >= wIndex + 2) {
                                uint32_t lo = static_cast<uint32_t>(data16[wIndex]);
                                uint32_t hi = static_cast<uint32_t>(data16[wIndex + 1]);
                                uint32_t u32 = (hi << 16) | lo;
                                double raw = (pixelRepresentation == 1)
                                               ? static_cast<double>(static_cast<int32_t>(u32))
                                               : static_cast<double>(u32);
                                val = static_cast<float>(raw * intFactor + intOffset);
                            }
                        }
                        dst[y * m_width + x] = val;
                        if (val > localMax) localMax = val;
                    }
                }
                sliceMax[z] = localMax;
                int current = done.fetch_add(1) + 1;
                if (progress) progress(current, m_depth);
            });
        } else {
            const bool useS16 = (pixelRepresentation == 1 && haveS16);
            QtConcurrent::blockingMap(zIndices, [&](int z) {
                float *dst = m_volume.ptr<float>(z);
                size_t idx = static_cast<size_t>(z) * m_width * m_height;
                float localMax = 0.0f;
                for (int y = 0; y < m_height; ++y) {
                    for (int x = 0; x < m_width; ++x, ++idx) {
                        float val = 0.0f;
                        if (idx < totalElements) {
                            double raw = useS16 ? static_cast<double>(dataS16[idx])
                                                : static_cast<double>(data16[idx]);
                            val = static_cast<float>(raw * intFactor + intOffset);
                        }
                        dst[y * m_width + x] = val;
                        if (val > localMax) localMax = val;
                    }
                }
                sliceMax[z] = localMax;
                int current = done.fetch_add(1) + 1;
                if (progress) progress(current, m_depth);
            });
        }

        m_maxDose = *std::max_element(sliceMax.begin(), sliceMax.end());

        qDebug() << QString("Max dose: %1 Gy").arg(m_maxDose, 0, 'f', 3);

        // 座標変換テスト
        QVector3D testOrigin = voxelToPatient(0, 0, 0);
        QVector3D testCenter = voxelToPatient(m_width/2, m_height/2, m_depth/2);
        QVector3D testMax = voxelToPatient(m_width-1, m_height-1, m_depth-1);

        qDebug() << QString("Coordinate validation:");
        qDebug() << QString("  Origin [0,0,0] -> (%1,%2,%3)")
                    .arg(testOrigin.x(), 0, 'f', 1).arg(testOrigin.y(), 0, 'f', 1).arg(testOrigin.z(), 0, 'f', 1);
        qDebug() << QString("  Center [%1,%2,%3] -> (%4,%5,%6)")
                    .arg(m_width/2).arg(m_height/2).arg(m_depth/2)
                    .arg(testCenter.x(), 0, 'f', 1).arg(testCenter.y(), 0, 'f', 1).arg(testCenter.z(), 0, 'f', 1);
        qDebug() << QString("  Max [%1,%2,%3] -> (%4,%5,%6)")
                    .arg(m_width-1).arg(m_height-1).arg(m_depth-1)
                    .arg(testMax.x(), 0, 'f', 1).arg(testMax.y(), 0, 'f', 1).arg(testMax.z(), 0, 'f', 1);

        return true;
    }

    qWarning() << "Failed to read pixel data";
    return false;
}

void RTDoseVolume::setFromMatAndGeometry(const cv::Mat &vol,
                                         const DicomVolume &ctVolume)
{
    if (vol.dims != 3 || vol.type() != CV_32F) {
        m_volume.release();
        m_width = 0;
        m_height = 0;
        m_depth = 0;
        m_maxDose = 0.0;
        return;
    }

    m_volume = vol.clone();
    m_depth = vol.size[0];
    m_height = vol.size[1];
    m_width = vol.size[2];

    m_spacingX = ctVolume.spacingX();
    m_spacingY = ctVolume.spacingY();
    m_spacingZ = ctVolume.spacingZ();

    QVector3D origin = ctVolume.voxelToPatient(0, 0, 0);
    m_originX = origin.x();
    m_originY = origin.y();
    m_originZ = origin.z();

    QVector3D rowVector = ctVolume.voxelToPatient(1, 0, 0) - origin;
    QVector3D colVector = ctVolume.voxelToPatient(0, 1, 0) - origin;
    QVector3D sliceVector = ctVolume.voxelToPatient(0, 0, 1) - origin;

    if (!rowVector.isNull()) {
        rowVector.normalize();
    }
    if (!colVector.isNull()) {
        colVector.normalize();
    }
    if (!sliceVector.isNull()) {
        sliceVector.normalize();
    }

    m_rowDir[0] = rowVector.x();
    m_rowDir[1] = rowVector.y();
    m_rowDir[2] = rowVector.z();
    m_colDir[0] = colVector.x();
    m_colDir[1] = colVector.y();
    m_colDir[2] = colVector.z();
    m_sliceDir[0] = sliceVector.x();
    m_sliceDir[1] = sliceVector.y();
    m_sliceDir[2] = sliceVector.z();

    m_frameUID = ctVolume.frameOfReferenceUID();

    m_zOffsets.resize(static_cast<size_t>(m_depth));
    for (int z = 0; z < m_depth; ++z) {
        QVector3D point = ctVolume.voxelToPatient(0, 0, z);
        QVector3D diff = point - origin;
        m_zOffsets[static_cast<size_t>(z)] = QVector3D::dotProduct(sliceVector, diff);
    }

    double minDose = 0.0;
    double maxDose = 0.0;
    cv::minMaxIdx(m_volume, &minDose, &maxDose);
    m_maxDose = maxDose;

    m_patientShift = QVector3D(0.0, 0.0, 0.0);
    m_ctToDose.setToIdentity();
    m_hasIOP = true;
    m_hasIPP = true;
}

QImage RTDoseVolume::getOverlaySlice(int index, DicomVolume::Orientation ori,
                                     const QSize& size) const
{
    if (m_volume.empty()) return QImage();
    cv::Mat slice;
    switch (ori) {
    case DicomVolume::Orientation::Axial:
        slice = getSliceAxialFloat(m_volume, index);
        break;
    case DicomVolume::Orientation::Sagittal:
        slice = getSliceSagittalFloat(m_volume, index);
        break;
    case DicomVolume::Orientation::Coronal:
        slice = getSliceCoronalFloat(m_volume, index);
        break;
    }
    if (slice.empty()) return QImage();
    cv::Mat norm;
    slice.convertTo(norm, CV_8U, 255.0 / m_maxDose);
    QImage gray(norm.data, norm.cols, norm.rows, norm.step, QImage::Format_Grayscale8);
    QImage scaled = gray.copy().scaled(size, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    QImage color(scaled.size(), QImage::Format_ARGB32);
    for (int y = 0; y < scaled.height(); ++y) {
        const uchar* src = scaled.constScanLine(y);
        QRgb* dst = reinterpret_cast<QRgb*>(color.scanLine(y));
        for (int x = 0; x < scaled.width(); ++x) {
            int alpha = src[x];
            dst[x] = qRgba(255, 0, 0, alpha);
        }
    }
    return color;
}

QImage RTDoseVolume::resampleSlice(const DicomVolume& refVol, int index,
                                   DicomVolume::Orientation ori) const
{
    if (m_volume.empty()) return QImage();
    
    // CT画像のスライスサイズを取得
    int ctWidth = 0, ctHeight = 0;
    switch(ori) {
    case DicomVolume::Orientation::Axial:
        ctWidth = refVol.width();
        ctHeight = refVol.height();
        break;
    case DicomVolume::Orientation::Sagittal:
        ctWidth = refVol.height();
        ctHeight = refVol.depth();
        break;
    case DicomVolume::Orientation::Coronal:
        ctWidth = refVol.width();
        ctHeight = refVol.depth();
        break;
    }
    
    qDebug() << QString("Resampling dose for %1 slice %2, CT size: %3x%4")
                .arg(ori == DicomVolume::Orientation::Axial ? "Axial" :
                     ori == DicomVolume::Orientation::Sagittal ? "Sagittal" : "Coronal")
                .arg(index).arg(ctWidth).arg(ctHeight);
    
    // 出力画像を作成
    QImage result(ctWidth, ctHeight, QImage::Format_ARGB32);
    result.fill(Qt::transparent);
    
    // 効率的な処理のため、事前に変換行列を計算
    QVector3D ctRowDir, ctColDir, ctSliceDir;
    QVector3D ctOrigin;
    double ctSpacingX, ctSpacingY;
    
    // CT画像の座標系情報を取得（スライス中心座標を使用 +0.5）
    switch(ori) {
    case DicomVolume::Orientation::Axial:
        ctOrigin = refVol.voxelToPatient(0.0, 0.0, index + 0.5);
        ctRowDir = refVol.voxelToPatient(1.0, 0.0, index + 0.5) - ctOrigin;
        ctColDir = refVol.voxelToPatient(0.0, 1.0, index + 0.5) - ctOrigin;
        ctSpacingX = refVol.spacingX();
        ctSpacingY = refVol.spacingY();
        break;
    case DicomVolume::Orientation::Sagittal:
        ctOrigin = refVol.voxelToPatient(index + 0.5, 0.0, 0.0);
        ctRowDir = refVol.voxelToPatient(index + 0.5, 1.0, 0.0) - ctOrigin;
        ctColDir = refVol.voxelToPatient(index + 0.5, 0.0, 1.0) - ctOrigin;
        ctSpacingX = refVol.spacingY();
        ctSpacingY = refVol.spacingZ();
        break;
    case DicomVolume::Orientation::Coronal:
        ctOrigin = refVol.voxelToPatient(0.0, index + 0.5, 0.0);
        ctRowDir = refVol.voxelToPatient(1.0, index + 0.5, 0.0) - ctOrigin;
        ctColDir = refVol.voxelToPatient(0.0, index + 0.5, 1.0) - ctOrigin;
        ctSpacingX = refVol.spacingX();
        ctSpacingY = refVol.spacingZ();
        break;
    }
    
    // 正規化
    ctRowDir.normalize();
    ctColDir.normalize();
    
    int validPixels = 0;
    double minDose = std::numeric_limits<double>::max();
    double maxDose = 0.0;
    
    // 各ピクセルについてRT-Dose値を計算
    for (int y = 0; y < ctHeight; ++y) {
        for (int x = 0; x < ctWidth; ++x) {
            // CT画像ピクセルの患者座標を計算（中心座標 +0.5）
            QVector3D patientPos = ctOrigin +
                                   ctRowDir * ((x + 0.5) * ctSpacingX) +
                                   ctColDir * ((y + 0.5) * ctSpacingY);
            
            // RT-Dose ボリューム座標に変換
            QVector3D doseVoxel = patientToVoxelContinuous(patientPos);
            
            // ボリューム境界チェック（マージンを設ける）
            if (doseVoxel.x() >= -0.5 && doseVoxel.x() < m_width - 0.5 &&
                doseVoxel.y() >= -0.5 && doseVoxel.y() < m_height - 0.5 &&
                doseVoxel.z() >= -0.5 && doseVoxel.z() < m_depth - 0.5) {
                
                // トリリニア補間でDose値を取得
                float doseValue = interpolateTrilinear(doseVoxel.x(), doseVoxel.y(), doseVoxel.z());
                
                if (doseValue > 0.0f) {
                    validPixels++;
                    minDose = std::min(minDose, static_cast<double>(doseValue));
                    maxDose = std::max(maxDose, static_cast<double>(doseValue));
                    
                    // カラーマッピング（最大線量に対する比率）
                    int alpha = static_cast<int>(255.0 * doseValue / m_maxDose);
                    alpha = std::clamp(alpha, 0, 255);
                    
                    // 線量レベルに応じた色分け（オプション）
                    QRgb color;
                    if (doseValue / m_maxDose > 0.9) {
                        color = qRgba(255, 0, 0, alpha);     // 高線量：赤
                    } else if (doseValue / m_maxDose > 0.7) {
                        color = qRgba(255, 128, 0, alpha);   // 中高線量：オレンジ
                    } else if (doseValue / m_maxDose > 0.5) {
                        color = qRgba(255, 255, 0, alpha);   // 中線量：黄
                    } else {
                        color = qRgba(0, 255, 0, alpha);     // 低線量：緑
                    }
                    
                    result.setPixel(x, y, color);
                }
            }
        }
    }
    
    qDebug() << QString("Dose resampling complete: %1 valid pixels, dose range: %2 - %3")
                .arg(validPixels).arg(minDose, 0, 'f', 2).arg(maxDose, 0, 'f', 2);
    
    return result;
}

QImage RTDoseVolume::coverageSlice(const DicomVolume& refVol, int index,
                                   DicomVolume::Orientation ori) const
{
    if (m_volume.empty()) return QImage();
    
    int ctWidth = 0, ctHeight = 0;
    switch (ori) {
    case DicomVolume::Orientation::Axial:
        ctWidth = refVol.width();
        ctHeight = refVol.height();
        break;
    case DicomVolume::Orientation::Sagittal:
        ctWidth = refVol.height();
        ctHeight = refVol.depth();
        break;
    case DicomVolume::Orientation::Coronal:
        ctWidth = refVol.width();
        ctHeight = refVol.depth();
        break;
    }
    
    QImage result(ctWidth, ctHeight, QImage::Format_ARGB32);
    result.fill(Qt::transparent);
    
    int coverageCount = 0;
    int totalPixels = ctWidth * ctHeight;
    
    for (int y = 0; y < ctHeight; ++y) {
        for (int x = 0; x < ctWidth; ++x) {
            // CT画像ピクセルの患者座標を取得
            QVector3D patientPos;
            switch (ori) {
            case DicomVolume::Orientation::Axial:
                patientPos = refVol.voxelToPatient(x, y, index);
                break;
            case DicomVolume::Orientation::Sagittal:
                patientPos = refVol.voxelToPatient(index, x, y);
                break;
            case DicomVolume::Orientation::Coronal:
                patientPos = refVol.voxelToPatient(x, index, y);
                break;
            }
            
            // RT-Dose座標に変換
            QVector3D doseVoxel = patientToVoxelContinuous(patientPos);
            
            // カバレッジチェック
            bool isInside = (doseVoxel.x() >= 0 && doseVoxel.x() < m_width &&
                           doseVoxel.y() >= 0 && doseVoxel.y() < m_height &&
                           doseVoxel.z() >= 0 && doseVoxel.z() < m_depth);
            
            if (isInside) {
                coverageCount++;
                // カバレッジ領域を青で表示（デバッグ用）
                result.setPixel(x, y, qRgba(0, 0, 255, 100));
            }
        }
    }
    
    double coverageRatio = static_cast<double>(coverageCount) / totalPixels * 100.0;
    qDebug() << QString("Coverage for %1 slice %2: %3/%4 pixels (%5%)")
                .arg(ori == DicomVolume::Orientation::Axial ? "Axial" :
                     ori == DicomVolume::Orientation::Sagittal ? "Sagittal" : "Coronal")
                .arg(index).arg(coverageCount).arg(totalPixels).arg(coverageRatio, 0, 'f', 1);
    
    return result;
}

float RTDoseVolume::interpolateTrilinear(double x, double y, double z) const
{
    // **重要**: 範囲チェックを厳密に実行
    if (x < -0.5 || y < -0.5 || z < -0.5 || 
        x >= m_width - 0.5 || y >= m_height - 0.5 || z >= m_depth - 0.5) {
        return 0.0f;  // 範囲外は0を返す
    }
    
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int z0 = static_cast<int>(std::floor(z));
    
    int x1 = std::min(x0 + 1, m_width - 1);
    int y1 = std::min(y0 + 1, m_height - 1);
    int z1 = std::min(z0 + 1, m_depth - 1);
    
    // さらに厳密な境界チェック
    if (x0 < 0 || y0 < 0 || z0 < 0 || 
        x1 >= m_width || y1 >= m_height || z1 >= m_depth) {
        return 0.0f;
    }
    
    double fx = x - x0;
    double fy = y - y0;
    double fz = z - z0;
    
    // 8つの隣接ボクセル値を取得
    auto getValue = [&](int zz, int yy, int xx) -> float {
        if (zz < 0 || yy < 0 || xx < 0 || zz >= m_depth || yy >= m_height || xx >= m_width)
            return 0.0f;
        return m_volume.ptr<float>(zz)[yy * m_width + xx];
    };
    
    float v000 = getValue(z0, y0, x0);
    float v001 = getValue(z0, y0, x1);
    float v010 = getValue(z0, y1, x0);
    float v011 = getValue(z0, y1, x1);
    float v100 = getValue(z1, y0, x0);
    float v101 = getValue(z1, y0, x1);
    float v110 = getValue(z1, y1, x0);
    float v111 = getValue(z1, y1, x1);
    
    // トリリニア補間
    float c00 = v000 * (1 - fx) + v001 * fx;
    float c01 = v010 * (1 - fx) + v011 * fx;
    float c10 = v100 * (1 - fx) + v101 * fx;
    float c11 = v110 * (1 - fx) + v111 * fx;
    
    float c0 = c00 * (1 - fy) + c01 * fy;
    float c1 = c10 * (1 - fy) + c11 * fy;
    
    return c0 * (1 - fz) + c1 * fz;
}

void RTDoseVolume::testCoordinateTransforms(const DicomVolume& refVol) const
{
    qDebug() << "=== Coordinate Transform Test ===";
    
    // CT volume 情報
    qDebug() << "CT Volume Info:";
    qDebug() << QString("  Size: %1x%2x%3").arg(refVol.width()).arg(refVol.height()).arg(refVol.depth());
    qDebug() << QString("  Spacing: %1 x %2 x %3").arg(refVol.spacingX()).arg(refVol.spacingY()).arg(refVol.spacingZ());
    qDebug() << QString("  Origin: (%1, %2, %3)").arg(refVol.originX()).arg(refVol.originY()).arg(refVol.originZ());
    
    // RT-Dose volume 情報
    qDebug() << "RT-Dose Volume Info:";
    qDebug() << QString("  Size: %1x%2x%3").arg(m_width).arg(m_height).arg(m_depth);
    qDebug() << QString("  Spacing: %1 x %2 x %3").arg(m_spacingX).arg(m_spacingY).arg(m_spacingZ);
    qDebug() << QString("  Origin: (%1, %2, %3)").arg(m_originX).arg(m_originY).arg(m_originZ);
    qDebug() << QString("  Patient Shift: (%1, %2, %3)").arg(m_patientShift.x()).arg(m_patientShift.y()).arg(m_patientShift.z());
    
    // 座標変換テスト
    qDebug() << "Coordinate Transform Tests:";
    
    // テスト1: CT volume の中心点
    int ctCenterX = refVol.width() / 2;
    int ctCenterY = refVol.height() / 2;
    int ctCenterZ = refVol.depth() / 2;
    
    QVector3D ctCenterPatient = refVol.voxelToPatient(ctCenterX, ctCenterY, ctCenterZ);
    QVector3D doseCenterVoxel = patientToVoxelContinuous(ctCenterPatient);
    
    qDebug() << QString("CT center (%1,%2,%3) -> Patient (%4,%5,%6) -> Dose (%7,%8,%9)")
                .arg(ctCenterX).arg(ctCenterY).arg(ctCenterZ)
                .arg(ctCenterPatient.x(), 0, 'f', 1).arg(ctCenterPatient.y(), 0, 'f', 1).arg(ctCenterPatient.z(), 0, 'f', 1)
                .arg(doseCenterVoxel.x(), 0, 'f', 2).arg(doseCenterVoxel.y(), 0, 'f', 2).arg(doseCenterVoxel.z(), 0, 'f', 2);
    
    // テスト2: RT-Dose volume の中心点
    int doseCenterX = m_width / 2;
    int doseCenterY = m_height / 2;
    int doseCenterZ = m_depth / 2;
    
    QVector3D doseCenterPatient = voxelToPatient(doseCenterX, doseCenterY, doseCenterZ);
    qDebug() << QString("Dose center (%1,%2,%3) -> Patient (%4,%5,%6)")
                .arg(doseCenterX).arg(doseCenterY).arg(doseCenterZ)
                .arg(doseCenterPatient.x(), 0, 'f', 1).arg(doseCenterPatient.y(), 0, 'f', 1).arg(doseCenterPatient.z(), 0, 'f', 1);
    
    // テスト3: オリジン同士の比較
    QVector3D ctOriginPatient = refVol.voxelToPatient(0, 0, 0);
    QVector3D doseOriginPatient = voxelToPatient(0, 0, 0);
    QVector3D originDiff = ctOriginPatient - doseOriginPatient;
    
    qDebug() << QString("CT origin in patient: (%1,%2,%3)")
                .arg(ctOriginPatient.x(), 0, 'f', 1).arg(ctOriginPatient.y(), 0, 'f', 1).arg(ctOriginPatient.z(), 0, 'f', 1);
    qDebug() << QString("Dose origin in patient: (%1,%2,%3)")
                .arg(doseOriginPatient.x(), 0, 'f', 1).arg(doseOriginPatient.y(), 0, 'f', 1).arg(doseOriginPatient.z(), 0, 'f', 1);
    qDebug() << QString("Origin difference: (%1,%2,%3)")
                .arg(originDiff.x(), 0, 'f', 1).arg(originDiff.y(), 0, 'f', 1).arg(originDiff.z(), 0, 'f', 1);
    
    // テスト4: 実際の線量値チェック
    float maxFoundDose = 0.0f;
    int maxDoseX = -1, maxDoseY = -1, maxDoseZ = -1;
    
    for (int z = 0; z < m_depth; ++z) {
        const float* slice = m_volume.ptr<float>(z);
        for (int y = 0; y < m_height; ++y) {
            for (int x = 0; x < m_width; ++x) {
                float dose = slice[y * m_width + x];
                if (dose > maxFoundDose) {
                    maxFoundDose = dose;
                    maxDoseX = x;
                    maxDoseY = y;
                    maxDoseZ = z;
                }
            }
        }
    }
    
    if (maxFoundDose > 0) {
        QVector3D maxDosePatient = voxelToPatient(maxDoseX, maxDoseY, maxDoseZ);
        qDebug() << QString("Max dose %1 at dose voxel (%2,%3,%4) -> patient (%5,%6,%7)")
                    .arg(maxFoundDose, 0, 'f', 2)
                    .arg(maxDoseX).arg(maxDoseY).arg(maxDoseZ)
                    .arg(maxDosePatient.x(), 0, 'f', 1).arg(maxDosePatient.y(), 0, 'f', 1).arg(maxDosePatient.z(), 0, 'f', 1);
    } else {
        qDebug() << "Warning: No positive dose values found in RT-Dose volume!";
    }
}

QImage RTDoseVolume::createSimpleOverlay(const DicomVolume& refVol, int index,
                                         DicomVolume::Orientation ori) const
{
    int ctWidth = 0, ctHeight = 0;
    switch(ori) {
    case DicomVolume::Orientation::Axial:
        ctWidth = refVol.width();
        ctHeight = refVol.height();
        break;
    case DicomVolume::Orientation::Sagittal:
        ctWidth = refVol.height();
        ctHeight = refVol.depth();
        break;
    case DicomVolume::Orientation::Coronal:
        ctWidth = refVol.width();
        ctHeight = refVol.depth();
        break;
    }
    
    QImage result(ctWidth, ctHeight, QImage::Format_ARGB32);
    result.fill(Qt::transparent);
    
    qDebug() << QString("Creating simple overlay for %1 slice %2")
                .arg(ori == DicomVolume::Orientation::Axial ? "Axial" :
                     ori == DicomVolume::Orientation::Sagittal ? "Sagittal" : "Coronal")
                .arg(index);
    
    // デバッグ：ボリューム寸法の確認
    qDebug() << QString("CT dimensions: %1x%2, RT-Dose dimensions: %3x%4")
                .arg(ctWidth).arg(ctHeight).arg(m_width).arg(m_height);
    
    // デバッグ：RT-Doseの各スライスのZ位置を確認
    qDebug() << "RT-Dose slice positions:";
    for (int z = 0; z < std::min(m_depth, 3); ++z) { // 最初の3スライスまで
        QVector3D dosePos = voxelToPatient(m_width/2, m_height/2, z);
        qDebug() << QString("  Slice %1: Z = %2 mm").arg(z).arg(dosePos.z(), 0, 'f', 1);
    }
    
    // CTスライスのZ位置を確認
    QVector3D ctSliceCenter;
    switch (ori) {
    case DicomVolume::Orientation::Axial:
        ctSliceCenter = refVol.voxelToPatient(refVol.width()/2, refVol.height()/2, index);
        break;
    case DicomVolume::Orientation::Sagittal:
        ctSliceCenter = refVol.voxelToPatient(index, refVol.width()/2, refVol.height()/2);
        break;
    case DicomVolume::Orientation::Coronal:
        ctSliceCenter = refVol.voxelToPatient(refVol.width()/2, index, refVol.height()/2);
        break;
    }
    
    qDebug() << QString("CT slice %1 center Z position: %2 mm").arg(index).arg(ctSliceCenter.z(), 0, 'f', 1);
    
    // 1スライスのRT-Doseの場合の特別処理
    if (m_depth == 1) {
        qDebug() << "Single-slice RT-Dose detected, using patient coordinate mapping";
        
        // RT-Dose平面のZ位置（修正版）
        QVector3D doseCenter = voxelToPatient(m_width/2, m_height/2, 0);
        qDebug() << QString("RT-Dose plane Z position: %1 mm").arg(doseCenter.z(), 0, 'f', 1);
        
        double zDistance = std::abs(ctSliceCenter.z() - doseCenter.z());
        qDebug() << QString("Distance between CT slice and RT-Dose plane: %1 mm").arg(zDistance, 0, 'f', 1);
        
        // 距離が大きすぎる場合はスキップ
        if (zDistance > 50.0) {
            qDebug() << "Distance too large, skipping overlay";
            return result;
        }
        
        int overlayPixels = 0;
        const float* doseSlice = m_volume.ptr<float>(0);
        
        // 患者座標系を経由した正確な座標変換
        for (int ctY = 0; ctY < ctHeight; ++ctY) {
            for (int ctX = 0; ctX < ctWidth; ++ctX) {
                // CT画像ピクセルの患者座標を取得
                QVector3D ctPatientPos;
                switch (ori) {
                case DicomVolume::Orientation::Axial:
                    ctPatientPos = refVol.voxelToPatient(ctX, ctY, index);
                    break;
                case DicomVolume::Orientation::Sagittal:
                    ctPatientPos = refVol.voxelToPatient(index, ctX, ctY);
                    break;
                case DicomVolume::Orientation::Coronal:
                    ctPatientPos = refVol.voxelToPatient(ctX, index, ctY);
                    break;
                }
                
                // 患者座標からRT-Dose座標に変換（Z=0に投影）
                QVector3D doseVoxel = patientToVoxelContinuous(ctPatientPos);
                
                // RT-Doseボリューム内かチェック（Z方向は無視）
                if (doseVoxel.x() >= 0 && doseVoxel.x() < m_width &&
                    doseVoxel.y() >= 0 && doseVoxel.y() < m_height) {
                    
                    // バイリニア補間でDose値を取得
                    int x0 = static_cast<int>(std::floor(doseVoxel.x()));
                    int y0 = static_cast<int>(std::floor(doseVoxel.y()));
                    int x1 = std::min(x0 + 1, m_width - 1);
                    int y1 = std::min(y0 + 1, m_height - 1);
                    
                    double fx = doseVoxel.x() - x0;
                    double fy = doseVoxel.y() - y0;
                    
                    // 境界チェック
                    if (x0 >= 0 && x1 < m_width && y0 >= 0 && y1 < m_height) {
                        float v00 = doseSlice[y0 * m_width + x0];
                        float v01 = doseSlice[y0 * m_width + x1];
                        float v10 = doseSlice[y1 * m_width + x0];
                        float v11 = doseSlice[y1 * m_width + x1];
                        
                        float v0 = v00 * (1 - fx) + v01 * fx;
                        float v1 = v10 * (1 - fx) + v11 * fx;
                        float doseValue = v0 * (1 - fy) + v1 * fy;
                        
                        if (doseValue > 0.001f) {
                            overlayPixels++;
                            
                            // 線量レベルに応じた透明度（距離による減衰なし）
                            int alpha = static_cast<int>((doseValue / m_maxDose) * 255);
                            alpha = std::clamp(alpha, 20, 255);
                            
                            // 線量レベルに応じた色分け
                            QRgb color;
                            double ratio = doseValue / m_maxDose;
                            if (ratio > 0.8) {
                                color = qRgba(255, 0, 0, alpha);     // 高線量：赤
                            } else if (ratio > 0.6) {
                                color = qRgba(255, 128, 0, alpha);   // 中高線量：オレンジ
                            } else if (ratio > 0.4) {
                                color = qRgba(255, 255, 0, alpha);   // 中線量：黄
                            } else if (ratio > 0.2) {
                                color = qRgba(128, 255, 0, alpha);   // 中低線量：緑黄
                            } else {
                                color = qRgba(0, 255, 0, alpha);     // 低線量：緑
                            }
                            
                            result.setPixel(ctX, ctY, color);
                        }
                    }
                }
            }
        }
        
        qDebug() << QString("Overlay complete: %1 pixels with dose").arg(overlayPixels);
        
        // デバッグ：座標変換のサンプルテスト
        if (overlayPixels == 0) {
            qDebug() << "=== Debug coordinate mapping (no overlay pixels found) ===";
            for (int testY = 0; testY < ctHeight; testY += ctHeight/4) {
                for (int testX = 0; testX < ctWidth; testX += ctWidth/4) {
                    QVector3D ctPatientPos = refVol.voxelToPatient(testX, testY, index);
                    QVector3D doseVoxel = patientToVoxelContinuous(ctPatientPos);
                    
                    qDebug() << QString("CT[%1,%2] -> Patient(%3,%4,%5) -> Dose(%6,%7,%8)")
                                .arg(testX).arg(testY)
                                .arg(ctPatientPos.x(), 0, 'f', 1).arg(ctPatientPos.y(), 0, 'f', 1).arg(ctPatientPos.z(), 0, 'f', 1)
                                .arg(doseVoxel.x(), 0, 'f', 2).arg(doseVoxel.y(), 0, 'f', 2).arg(doseVoxel.z(), 0, 'f', 2);
                    
                    if (doseVoxel.x() >= 0 && doseVoxel.x() < m_width &&
                        doseVoxel.y() >= 0 && doseVoxel.y() < m_height) {
                        int dx = static_cast<int>(doseVoxel.x());
                        int dy = static_cast<int>(doseVoxel.y());
                        float doseVal = doseSlice[dy * m_width + dx];
                        qDebug() << QString("  -> Dose value: %1").arg(doseVal, 0, 'f', 6);
                    }
                }
            }
        }
        
        return result;
    }
    
    // 従来の3Dボリューム処理（複数スライスの場合）
    qDebug() << "Multi-slice RT-Dose processing";
    
    // 最も近いRT-DoseスライスZ位置を探す
    double minDistance = std::numeric_limits<double>::max();
    int bestSliceIndex = 0;
    
    for (int z = 0; z < m_depth; ++z) {
        QVector3D doseSlicePos = voxelToPatient(m_width/2, m_height/2, z);
        double distance = std::abs(ctSliceCenter.z() - doseSlicePos.z());
        if (distance < minDistance) {
            minDistance = distance;
            bestSliceIndex = z;
        }
    }
    
    qDebug() << QString("Best matching RT-Dose slice: %1, distance: %2 mm")
                .arg(bestSliceIndex).arg(minDistance, 0, 'f', 1);
    
    // 距離が大きすぎる場合はスキップ
    if (minDistance > 10.0) {
        qDebug() << "Distance too large for multi-slice RT-Dose, skipping overlay";
        return result;
    }
    
    // 選択されたスライスでオーバーレイを作成
    const float* doseSlice = m_volume.ptr<float>(bestSliceIndex);
    int overlayPixels = 0;
    
    for (int ctY = 0; ctY < ctHeight; ++ctY) {
        for (int ctX = 0; ctX < ctWidth; ++ctX) {
            QVector3D ctPatientPos;
            switch (ori) {
            case DicomVolume::Orientation::Axial:
                ctPatientPos = refVol.voxelToPatient(ctX, ctY, index);
                break;
            case DicomVolume::Orientation::Sagittal:
                ctPatientPos = refVol.voxelToPatient(index, ctX, ctY);
                break;
            case DicomVolume::Orientation::Coronal:
                ctPatientPos = refVol.voxelToPatient(ctX, index, ctY);
                break;
            }
            
            QVector3D doseVoxel = patientToVoxelContinuous(ctPatientPos);
            
            if (doseVoxel.x() >= 0 && doseVoxel.x() < m_width &&
                doseVoxel.y() >= 0 && doseVoxel.y() < m_height) {
                
                int x0 = static_cast<int>(std::floor(doseVoxel.x()));
                int y0 = static_cast<int>(std::floor(doseVoxel.y()));
                int x1 = std::min(x0 + 1, m_width - 1);
                int y1 = std::min(y0 + 1, m_height - 1);
                
                if (x0 >= 0 && x1 < m_width && y0 >= 0 && y1 < m_height) {
                    double fx = doseVoxel.x() - x0;
                    double fy = doseVoxel.y() - y0;
                    
                    float v00 = doseSlice[y0 * m_width + x0];
                    float v01 = doseSlice[y0 * m_width + x1];
                    float v10 = doseSlice[y1 * m_width + x0];
                    float v11 = doseSlice[y1 * m_width + x1];
                    
                    float v0 = v00 * (1 - fx) + v01 * fx;
                    float v1 = v10 * (1 - fx) + v11 * fx;
                    float doseValue = v0 * (1 - fy) + v1 * fy;
                    
                    if (doseValue > 0.001f) {
                        overlayPixels++;
                        
                        int alpha = static_cast<int>((doseValue / m_maxDose) * 255);
                        alpha = std::clamp(alpha, 20, 255);
                        
                        QRgb color;
                        double ratio = doseValue / m_maxDose;
                        if (ratio > 0.8) {
                            color = qRgba(255, 0, 0, alpha);
                        } else if (ratio > 0.6) {
                            color = qRgba(255, 128, 0, alpha);
                        } else if (ratio > 0.4) {
                            color = qRgba(255, 255, 0, alpha);
                        } else if (ratio > 0.2) {
                            color = qRgba(128, 255, 0, alpha);
                        } else {
                            color = qRgba(0, 255, 0, alpha);
                        }
                        
                        result.setPixel(ctX, ctY, color);
                    }
                }
            }
        }
    }
    
    qDebug() << QString("Multi-slice overlay complete: %1 pixels with dose").arg(overlayPixels);
    
    return result;
}

void RTDoseVolume::diagnoseDoseData() const
{
    if (m_volume.empty()) {
        qDebug() << "ERROR: RT-Dose volume is empty!";
        return;
    }
    
    // 簡潔な統計情報のみ
    int totalVoxels = m_width * m_height * m_depth;
    int nonZeroCount = 0;
    double sum = 0.0;
    
    for (int z = 0; z < m_depth; ++z) {
        const float* slice = m_volume.ptr<float>(z);
        for (int i = 0; i < m_width * m_height; ++i) {
            float val = slice[i];
            if (val > 0.0f) {
                nonZeroCount++;
                sum += val;
            }
        }
    }
    
    double coverage = 100.0 * nonZeroCount / totalVoxels;
    double mean = (nonZeroCount > 0) ? sum / nonZeroCount : 0.0;
    
    qDebug() << QString("Dose statistics: %1/%2 voxels (%3%), mean dose: %4 Gy")
                .arg(nonZeroCount).arg(totalVoxels).arg(coverage, 0, 'f', 1).arg(mean, 0, 'f', 4);
}

bool RTDoseVolume::loadFromFileWithDiagnostics(const QString& filename)
{
    qDebug() << "=== Loading RT-Dose with detailed diagnostics ===";
    qDebug() << "File:" << filename;
    
    bool result = loadFromFile(filename);
    
    if (result) {
        qDebug() << "RT-Dose loaded successfully, running diagnostics...";
        diagnoseDoseData();
    } else {
        qDebug() << "Failed to load RT-Dose file";
    }
    
    return result;
}

void RTDoseVolume::analyzeRTDoseStructure(const QString& filename) const
{
    // 構造解析も簡潔に
    qDebug() << "RT-Dose structure analysis:" << filename;
    
    DcmFileFormat file;
    if (file.loadFile(filename.toLocal8Bit().data()).bad()) {
        qWarning() << "Failed to load file for analysis";
        return;
    }
    
    DcmDataset* ds = file.getDataset();
    
    // 重要な情報のみ表示
    OFString value;
    if (ds->findAndGetOFString(DCM_DoseUnits, value).good()) {
        qDebug() << "Dose units:" << QString::fromLatin1(value.c_str());
    }
    if (ds->findAndGetOFString(DCM_DoseType, value).good()) {
        qDebug() << "Dose type:" << QString::fromLatin1(value.c_str());
    }
    
    Float64 scaling;
    if (ds->findAndGetFloat64(DCM_DoseGridScaling, scaling).good()) {
        qDebug() << "Dose scaling:" << scaling;
    }
}

void RTDoseVolume::testSpecificCoordinate(const DicomVolume& refVol, int ctX, int ctY, int ctZ) const
{
    qDebug() << QString("=== Testing specific coordinate CT[%1,%2,%3] ===").arg(ctX).arg(ctY).arg(ctZ);
    
    // CT座標から患者座標
    QVector3D ctPatient = refVol.voxelToPatient(ctX, ctY, ctZ);
    qDebug() << QString("CT[%1,%2,%3] -> Patient (%4,%5,%6)")
                .arg(ctX).arg(ctY).arg(ctZ)
                .arg(ctPatient.x(), 0, 'f', 1)
                .arg(ctPatient.y(), 0, 'f', 1)
                .arg(ctPatient.z(), 0, 'f', 1);
    
    // 患者座標からRT-Dose座標
    QVector3D doseVoxel = patientToVoxelContinuous(ctPatient);
    qDebug() << QString("Patient (%1,%2,%3) -> Dose (%4,%5,%6)")
                .arg(ctPatient.x(), 0, 'f', 1)
                .arg(ctPatient.y(), 0, 'f', 1)
                .arg(ctPatient.z(), 0, 'f', 1)
                .arg(doseVoxel.x(), 0, 'f', 2)
                .arg(doseVoxel.y(), 0, 'f', 2)
                .arg(doseVoxel.z(), 0, 'f', 2);
    
    // RT-Dose値の取得
    if (doseVoxel.x() >= 0 && doseVoxel.x() < m_width &&
        doseVoxel.y() >= 0 && doseVoxel.y() < m_height &&
        doseVoxel.z() >= 0 && doseVoxel.z() < m_depth) {
        
        float doseValue = interpolateTrilinear(doseVoxel.x(), doseVoxel.y(), doseVoxel.z());
        qDebug() << QString("Dose value at this location: %1").arg(doseValue, 0, 'f', 6);
    } else {
        qDebug() << "Dose coordinate is outside volume bounds";
    }
}

QImage RTDoseVolume::createDebugGridOverlay(const DicomVolume& refVol, int index,
                                           DicomVolume::Orientation ori) const
{
    int ctWidth = 0, ctHeight = 0;
    switch(ori) {
    case DicomVolume::Orientation::Axial:
        ctWidth = refVol.width();
        ctHeight = refVol.height();
        break;
    case DicomVolume::Orientation::Sagittal:
        ctWidth = refVol.height();
        ctHeight = refVol.depth();
        break;
    case DicomVolume::Orientation::Coronal:
        ctWidth = refVol.width();
        ctHeight = refVol.depth();
        break;
    }
    
    QImage result(ctWidth, ctHeight, QImage::Format_ARGB32);
    result.fill(Qt::transparent);
    
    // 簡単な実装：デバッグ情報をログ出力のみ
    QVector3D ctSliceCenter;
    switch (ori) {
    case DicomVolume::Orientation::Axial:
        ctSliceCenter = refVol.voxelToPatient(ctWidth/2, ctHeight/2, index);
        break;
    case DicomVolume::Orientation::Sagittal:
        ctSliceCenter = refVol.voxelToPatient(index, ctWidth/2, ctHeight/2);
        break;
    case DicomVolume::Orientation::Coronal:
        ctSliceCenter = refVol.voxelToPatient(ctWidth/2, index, ctHeight/2);
        break;
    }
    
    QVector3D doseCenter = voxelToPatient(m_width/2, m_height/2, m_depth/2);
    
    qDebug() << QString("Debug Grid - CT Center: (%1,%2,%3), Dose Center: (%4,%5,%6)")
                .arg(ctSliceCenter.x(), 0, 'f', 1).arg(ctSliceCenter.y(), 0, 'f', 1).arg(ctSliceCenter.z(), 0, 'f', 1)
                .arg(doseCenter.x(), 0, 'f', 1).arg(doseCenter.y(), 0, 'f', 1).arg(doseCenter.z(), 0, 'f', 1);
    
    double distance = (ctSliceCenter - doseCenter).length();
    qDebug() << QString("Distance: %1 mm, Dose Volume: %2x%3x%4")
                .arg(distance, 0, 'f', 1).arg(m_width).arg(m_height).arg(m_depth);
    
    return result;
}

// getDoseStatistics メソッド（既に実装済みの場合は重複を避ける）
void RTDoseVolume::getDoseStatistics(double& min, double& max, double& mean) const
{
    if (m_volume.empty()) {
        min = max = mean = 0.0;
        return;
    }
    
    double sum = 0.0;
    int count = 0;
    min = std::numeric_limits<double>::max();
    max = 0.0;
    
    for (int z = 0; z < m_depth; ++z) {
        const float* slice = m_volume.ptr<float>(z);
        for (int i = 0; i < m_width * m_height; ++i) {
            float val = slice[i];
            if (val > 0.0f) {  // 正の線量値のみ統計
                min = std::min(min, static_cast<double>(val));
                max = std::max(max, static_cast<double>(val));
                sum += val;
                count++;
            }
        }
    }
    
    mean = (count > 0) ? sum / count : 0.0;
    if (count == 0) min = 0.0;
}

bool RTDoseVolume::saveToFile(const QString& filename,
                              std::function<void(int, int)> progress) const
{
    qDebug() << "=== Saving RT-Dose ===" << filename;

    if (m_volume.empty() || m_width == 0 || m_height == 0 || m_depth == 0) {
        qWarning() << "Cannot save empty dose volume";
        return false;
    }

    DcmFileFormat fileformat;
    DcmDataset* dataset = fileformat.getDataset();

    // Generate UIDs
    char uidBuf[100];
    dcmGenerateUniqueIdentifier(uidBuf, SITE_INSTANCE_UID_ROOT);
    QString sopInstanceUID = QString::fromLatin1(uidBuf);
    dcmGenerateUniqueIdentifier(uidBuf, SITE_SERIES_UID_ROOT);
    QString seriesInstanceUID = QString::fromLatin1(uidBuf);

    // SOP Common Module
    dataset->putAndInsertString(DCM_SOPClassUID, UID_RTDoseStorage);
    dataset->putAndInsertString(DCM_SOPInstanceUID, sopInstanceUID.toLatin1().data());

    // Patient Module (minimal required fields)
    dataset->putAndInsertString(DCM_PatientName, "Anonymous");
    dataset->putAndInsertString(DCM_PatientID, "00000000");
    dataset->putAndInsertString(DCM_PatientBirthDate, "");
    dataset->putAndInsertString(DCM_PatientSex, "O");

    // General Study Module
    dcmGenerateUniqueIdentifier(uidBuf, SITE_STUDY_UID_ROOT);
    dataset->putAndInsertString(DCM_StudyInstanceUID, uidBuf);
    QString currentDate = QDate::currentDate().toString("yyyyMMdd");
    QString currentTime = QTime::currentTime().toString("HHmmss");
    dataset->putAndInsertString(DCM_StudyDate, currentDate.toLatin1().data());
    dataset->putAndInsertString(DCM_StudyTime, currentTime.toLatin1().data());
    dataset->putAndInsertString(DCM_ReferringPhysicianName, "");
    dataset->putAndInsertString(DCM_StudyID, "1");
    dataset->putAndInsertString(DCM_AccessionNumber, "");

    // General Series Module
    dataset->putAndInsertString(DCM_Modality, "RTDOSE");
    dataset->putAndInsertString(DCM_SeriesInstanceUID, seriesInstanceUID.toLatin1().data());
    dataset->putAndInsertString(DCM_SeriesNumber, "1");

    // Frame of Reference Module
    if (!m_frameUID.isEmpty()) {
        dataset->putAndInsertString(DCM_FrameOfReferenceUID, m_frameUID.toLatin1().data());
    } else {
        dcmGenerateUniqueIdentifier(uidBuf, SITE_INSTANCE_UID_ROOT);
        dataset->putAndInsertString(DCM_FrameOfReferenceUID, uidBuf);
    }
    dataset->putAndInsertString(DCM_PositionReferenceIndicator, "");

    // General Equipment Module (optional but recommended)
    dataset->putAndInsertString(DCM_Manufacturer, "ShioRIS3");
    dataset->putAndInsertString(DCM_ManufacturerModelName, "ShioRIS3 Dose Calculator");
    dataset->putAndInsertString(DCM_SoftwareVersions, "1.0");

    // General Image Module
    dataset->putAndInsertString(DCM_InstanceNumber, "1");
    dataset->putAndInsertString(DCM_ContentDate, currentDate.toLatin1().data());
    dataset->putAndInsertString(DCM_ContentTime, currentTime.toLatin1().data());

    // Image Plane Module
    dataset->putAndInsertUint16(DCM_Rows, m_height);
    dataset->putAndInsertUint16(DCM_Columns, m_width);

    // Pixel spacing
    QString pixelSpacing = QString("%1\\%2").arg(m_spacingY, 0, 'f', 6).arg(m_spacingX, 0, 'f', 6);
    dataset->putAndInsertString(DCM_PixelSpacing, pixelSpacing.toLatin1().data());

    // Image Position Patient (origin)
    QString ipp = QString("%1\\%2\\%3").arg(m_originX, 0, 'f', 6).arg(m_originY, 0, 'f', 6).arg(m_originZ, 0, 'f', 6);
    dataset->putAndInsertString(DCM_ImagePositionPatient, ipp.toLatin1().data());

    // デバッグ: 保存時の値を出力
    qDebug() << "=== RTDOSE SAVE DEBUG ===";
    qDebug() << QString("  ImagePositionPatient: (%1, %2, %3)").arg(m_originX).arg(m_originY).arg(m_originZ);
    if (!m_zOffsets.empty()) {
        qDebug() << QString("  GridFrameOffsetVector[0]: %1").arg(m_zOffsets[0]);
        qDebug() << QString("  GridFrameOffsetVector size: %1").arg(m_zOffsets.size());
    }

    // Image Orientation Patient
    QString iop = QString("%1\\%2\\%3\\%4\\%5\\%6")
        .arg(m_rowDir[0], 0, 'f', 6).arg(m_rowDir[1], 0, 'f', 6).arg(m_rowDir[2], 0, 'f', 6)
        .arg(m_colDir[0], 0, 'f', 6).arg(m_colDir[1], 0, 'f', 6).arg(m_colDir[2], 0, 'f', 6);
    dataset->putAndInsertString(DCM_ImageOrientationPatient, iop.toLatin1().data());

    // Multi-frame Image Module
    dataset->putAndInsertString(DCM_NumberOfFrames, QString::number(m_depth).toLatin1().data());

    // RT Dose Module
    dataset->putAndInsertString(DCM_DoseUnits, "GY");
    dataset->putAndInsertString(DCM_DoseType, "PHYSICAL");
    dataset->putAndInsertString(DCM_DoseSummationType, "PLAN");

    // Calculate DoseGridScaling (max value that fits in uint16)
    double maxDoseValue = m_maxDose;
    if (maxDoseValue <= 0.0) {
        double minVal, maxVal;
        cv::minMaxIdx(m_volume, &minVal, &maxVal);
        maxDoseValue = maxVal;
    }

    // DoseGridScaling: physical dose = pixel value * scaling
    // Max pixel value for uint16 is 65535
    double doseGridScaling = (maxDoseValue > 0.0) ? (maxDoseValue / 65535.0) : 1e-6;
    dataset->putAndInsertFloat64(DCM_DoseGridScaling, doseGridScaling);

    // Grid Frame Offset Vector (z positions)
    // Use the actual m_zOffsets computed during dose calculation
    // to preserve non-uniform frame spacing and correct positioning
    std::vector<double> frameOffsets;
    if (!m_zOffsets.empty() && static_cast<int>(m_zOffsets.size()) == m_depth) {
        // Use the actual computed offsets to preserve accurate positioning
        frameOffsets = m_zOffsets;
        qDebug() << "GFOV: Using m_zOffsets (size=" << m_zOffsets.size() << ")";
        if (m_zOffsets.size() >= 3) {
            qDebug() << QString("  m_zOffsets[0]=%1, [1]=%2, [2]=%3")
                .arg(m_zOffsets[0]).arg(m_zOffsets[1]).arg(m_zOffsets[2]);
        }
    } else {
        // Fallback to uniform spacing if m_zOffsets not available
        qDebug() << "GFOV: Using fallback (m_spacingZ=" << m_spacingZ << ")";
        for (int z = 0; z < m_depth; ++z) {
            double offset = z * m_spacingZ;
            frameOffsets.push_back(offset);
        }
        if (frameOffsets.size() >= 3) {
            qDebug() << QString("  frameOffsets[0]=%1, [1]=%2, [2]=%3")
                .arg(frameOffsets[0]).arg(frameOffsets[1]).arg(frameOffsets[2]);
        }
    }
    qDebug() << "=== GFOV WRITE TO DICOM ===";
    qDebug() << QString("  Writing %1 values to DCM_GridFrameOffsetVector").arg(frameOffsets.size());
    if (frameOffsets.size() >= 3) {
        qDebug() << QString("  frameOffsets[0]=%1, [1]=%2, [2]=%3")
            .arg(frameOffsets[0], 0, 'f', 6).arg(frameOffsets[1], 0, 'f', 6).arg(frameOffsets[2], 0, 'f', 6);
    }

    // Write GridFrameOffsetVector as DS (Decimal String) - CORRECT VR per DICOM standard
    qDebug() << QString("  Writing %1 values as DcmDecimalString (DS)").arg(frameOffsets.size());

    // Create DcmDecimalString element with correct VR
    DcmDecimalString *gfovElement = new DcmDecimalString(DcmTagKey(0x3004, 0x000C));

    // Add each value using putFloat64 with index
    for (int i = 0; i < frameOffsets.size(); ++i) {
        OFCondition putStatus = gfovElement->putFloat64(frameOffsets[i], static_cast<unsigned long>(i));
        if (putStatus.bad()) {
            qDebug() << QString("  ERROR: Failed to put value[%1]: %2").arg(i).arg(putStatus.text());
        }
    }

    // Insert into dataset (OFTrue means replace if exists)
    OFCondition gfovStatus = dataset->insert(gfovElement, OFTrue);

    if (gfovStatus.good()) {
        qDebug() << "  Successfully inserted GridFrameOffsetVector as DS";

        // Verify by reading back as DS - get the element to access all values
        DcmElement *elem = nullptr;
        if (dataset->findAndGetElement(DcmTagKey(0x3004, 0x000C), elem).good() && elem) {
            // Get value multiplicity (number of values)
            unsigned long vm = elem->getVM();
            qDebug() << QString("  Verification: GFOV has %1 values (VM=%2)").arg(vm).arg(vm);

            // Read each value individually
            if (vm >= 3) {
                Float64 v0, v1, v2;
                if (elem->getFloat64(v0, 0).good() && elem->getFloat64(v1, 1).good() && elem->getFloat64(v2, 2).good()) {
                    qDebug() << QString("  Verified values[0]=%1, [1]=%2, [2]=%3")
                        .arg(v0, 0, 'f', 6).arg(v1, 0, 'f', 6).arg(v2, 0, 'f', 6);
                }
            }
        } else {
            qDebug() << "  WARNING: Could not find GFOV element!";
        }
    } else {
        qDebug() << "  ERROR inserting GridFrameOffsetVector:" << gfovStatus.text();
        delete gfovElement;  // Clean up on failure
    }

    // Image Pixel Module
    dataset->putAndInsertUint16(DCM_SamplesPerPixel, 1);
    dataset->putAndInsertString(DCM_PhotometricInterpretation, "MONOCHROME2");
    dataset->putAndInsertUint16(DCM_BitsAllocated, 16);
    dataset->putAndInsertUint16(DCM_BitsStored, 16);
    dataset->putAndInsertUint16(DCM_HighBit, 15);
    dataset->putAndInsertUint16(DCM_PixelRepresentation, 0); // unsigned

    // Convert dose data to uint16 array
    int totalVoxels = m_width * m_height * m_depth;
    std::vector<Uint16> pixelData(totalVoxels);

    int idx = 0;
    for (int z = 0; z < m_depth; ++z) {
        if (progress && z % 10 == 0) {
            progress(z, m_depth);
        }

        const float* slice = m_volume.ptr<float>(z);
        for (int y = 0; y < m_height; ++y) {
            for (int x = 0; x < m_width; ++x) {
                float doseValue = slice[y * m_width + x];
                // Convert to uint16 using DoseGridScaling
                Uint16 pixelValue = static_cast<Uint16>(std::min(65535.0, doseValue / doseGridScaling));
                pixelData[idx++] = pixelValue;
            }
        }
    }

    // Insert pixel data
    dataset->putAndInsertUint16Array(DCM_PixelData, pixelData.data(), pixelData.size());

    // CRITICAL: Verify GFOV is still in dataset right before save
    qDebug() << "=== FINAL VERIFICATION BEFORE SAVE ===";
    DcmElement *checkElem = nullptr;
    if (dataset->findAndGetElement(DcmTagKey(0x3004, 0x000C), checkElem).good() && checkElem) {
        unsigned long checkVM = checkElem->getVM();
        qDebug() << QString("  GFOV element present in dataset: VM=%1 values").arg(checkVM);
        if (checkVM >= 3) {
            Float64 c0, c1, c2;
            if (checkElem->getFloat64(c0, 0).good() && checkElem->getFloat64(c1, 1).good() && checkElem->getFloat64(c2, 2).good()) {
                qDebug() << QString("  Values[0]=%1, [1]=%2, [2]=%3")
                    .arg(c0, 0, 'f', 6).arg(c1, 0, 'f', 6).arg(c2, 0, 'f', 6);
            }
        }
    } else {
        qDebug() << "  WARNING: GFOV element NOT FOUND in dataset before save!";
    }

    // Save file
    OFCondition status = fileformat.saveFile(filename.toLocal8Bit().data(), EXS_LittleEndianExplicit);

    if (status.bad()) {
        qWarning() << "Failed to save RTDOSE file:" << status.text();
        return false;
    }

    // CRITICAL: Verify GFOV by reading the saved file back
    qDebug() << "=== VERIFICATION BY READING SAVED FILE ===";
    DcmFileFormat verifyFormat;
    OFCondition loadStatus = verifyFormat.loadFile(filename.toLocal8Bit().data());
    if (loadStatus.good()) {
        DcmDataset* verifyDataset = verifyFormat.getDataset();

        // Check if tag exists using element
        DcmElement *savedElem = nullptr;
        if (verifyDataset->findAndGetElement(DcmTagKey(0x3004, 0x000C), savedElem).good() && savedElem) {
            unsigned long savedVM = savedElem->getVM();
            qDebug() << QString("  SUCCESS: GFOV (3004,000C) found in saved file!");
            qDebug() << QString("  VM=%1 values").arg(savedVM);

            if (savedVM >= 3) {
                Float64 s0, s1, s2;
                if (savedElem->getFloat64(s0, 0).good() && savedElem->getFloat64(s1, 1).good() && savedElem->getFloat64(s2, 2).good()) {
                    qDebug() << QString("  Saved values[0]=%1, [1]=%2, [2]=%3")
                        .arg(s0, 0, 'f', 6).arg(s1, 0, 'f', 6).arg(s2, 0, 'f', 6);
                }
            }
        } else {
            qDebug() << "  ERROR: GFOV (3004,000C) NOT FOUND in saved file!";
        }
    } else {
        qDebug() << "  ERROR: Could not reload file for verification:" << loadStatus.text();
    }

    if (progress) {
        progress(m_depth, m_depth);
    }

    qDebug() << "RT-Dose saved successfully:" << filename;
    qDebug() << "  Size:" << m_width << "x" << m_height << "x" << m_depth;
    qDebug() << "  Max dose:" << maxDoseValue << "Gy";
    qDebug() << "  DoseGridScaling:" << doseGridScaling;

    return true;
}
