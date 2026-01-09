#include "dicom/dicom_volume.h"
#include "dicom/dicom_reader.h"
#include <QDataStream>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QVector3D>
#include <algorithm>
#include <limits>
#include <cmath>

// DCMTK includes for direct pixel access
#include <dcmtk/dcmimgle/dipixel.h>
#include <dcmtk/dcmimgle/diutils.h>
#include <dcmtk/dcmdata/dctk.h>

DicomVolume::DicomVolume()
    : m_width(0), m_height(0), m_depth(0)
    , m_spacingX(1.0)
    , m_spacingY(1.0)
    , m_spacingZ(1.0)
{
}

void DicomVolume::clear()
{
    m_volume.release();
    m_width = m_height = m_depth = 0;
    m_spacingX = m_spacingY = m_spacingZ = 1.0;
    m_originX = m_originY = m_originZ = 0.0;
    m_frameUID.clear();
    m_rowDir[0]=1.0; m_rowDir[1]=0.0; m_rowDir[2]=0.0;
    m_colDir[0]=0.0; m_colDir[1]=1.0; m_colDir[2]=0.0;
    m_sliceDir[0]=0.0; m_sliceDir[1]=0.0; m_sliceDir[2]=1.0;
    m_zOffsets.clear();
}

bool DicomVolume::loadFromDirectory(const QString& directory)
{
    qDebug() << "★★★ DicomVolume::loadFromDirectory() called with:" << directory;
    clear();
    QDir dir(directory);
    QFileInfoList fileInfos = dir.entryInfoList(QDir::Files, QDir::Name);

    struct Entry {
        double fallbackPos;     // slice position if no orientation available
        QString path;
        QVector3D pos3;         // ImagePositionPatient
        double loc;             // SliceLocation
        double sortKey;         // projection onto slice direction
    };

    std::vector<Entry> entries;

    qDebug() << "=== CT Volume Coordinate Loading ===";  // ← この行以降が実行されていない

    QVector3D rowDir, colDir; // orientation vectors (normalized)
    bool orientationSet = false;

    for (const QFileInfo& info : fileInfos) {
        QString path = info.absoluteFilePath();

        DcmFileFormat ff;
        if (ff.loadFile(path.toLocal8Bit().data()).bad()) {
            continue;
        }

        OFString modality;
        if (ff.getDataset()->findAndGetOFString(DCM_Modality, modality).good()) {
            QString mod = QString::fromLatin1(modality.c_str());
            if (mod == "RTDOSE" || mod == "RTSTRUCT") {
                continue;
            }
        }

        double pos = DicomReader::getSlicePosition(path);
        QVector3D pos3;
        double locVal = std::numeric_limits<double>::quiet_NaN();

        DicomReader r;
        if (!r.loadDicomFile(path)) {
            continue;
        }

        double x,y,z;
        if (r.getImagePositionPatient(x,y,z)) {
            pos3 = QVector3D(x,y,z);
            qDebug() << QString("File %1: Position (%2,%3,%4)")
                        .arg(info.fileName()).arg(x, 0, 'f', 1).arg(y, 0, 'f', 1).arg(z, 0, 'f', 1);
        } else {
            qDebug() << QString("File %1: No position data").arg(info.fileName());
        }

        double lv;
        if (r.getImageLocation(lv)) {
            locVal = lv;
        }

        if (!orientationSet) {
            double r1,r2,r3,c1,c2,c3;
            if (r.getImageOrientationPatient(r1,r2,r3,c1,c2,c3)) {
                rowDir = QVector3D(r1,r2,r3).normalized();
                colDir = QVector3D(c1,c2,c3).normalized();
                orientationSet = true;
            }
        }

        entries.push_back({pos, path, pos3, locVal, 0.0});
    }

    QVector3D sliceDir;
    if (orientationSet) {
        sliceDir = QVector3D::crossProduct(rowDir, colDir).normalized();
    }
    for (auto& e : entries) {
        if (orientationSet && !e.pos3.isNull()) {
            e.sortKey = QVector3D::dotProduct(e.pos3, sliceDir);
        } else if (!std::isnan(e.loc)) {
            e.sortKey = e.loc;
        } else {
            e.sortKey = e.fallbackPos;
        }
    }

    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b){ return a.sortKey < b.sortKey; });
    
    if (entries.empty()) return false;

    std::vector<cv::Mat> mats;
    std::vector<QVector3D> positions;
    std::vector<double> locations;
    
    for (size_t idx = 0; idx < entries.size(); ++idx) {
        const Entry& e = entries[idx];
        DicomReader reader;
        if (!reader.loadDicomFile(e.path)) return false;
        int w = reader.getWidth();
        int h = reader.getHeight();
        if (m_width == 0) {
            m_width = w;
            m_height = h;
            m_frameUID = reader.getFrameOfReferenceUID();
            double row, col;
            if (reader.getPixelSpacing(row, col)) {
                m_spacingY = row;
                m_spacingX = col;
            }
            m_spacingZ = reader.getSliceThickness();
            double r1,r2,r3,c1,c2,c3;
            if (reader.getImageOrientationPatient(r1,r2,r3,c1,c2,c3)) {
                QVector3D row(r1,r2,r3);
                QVector3D col(c1,c2,c3);
                row.normalize();
                col.normalize();
                QVector3D slice = QVector3D::crossProduct(row, col).normalized();
                m_rowDir[0]=row.x(); m_rowDir[1]=row.y(); m_rowDir[2]=row.z();
                m_colDir[0]=col.x(); m_colDir[1]=col.y(); m_colDir[2]=col.z();
                m_sliceDir[0]=slice.x(); m_sliceDir[1]=slice.y(); m_sliceDir[2]=slice.z();
            }
        }
        if (w != m_width || h != m_height) return false;
        DicomImage* di = reader.dicomImage();
        if (!di) return false;

        // Use the intermediate pixel data so that modality LUT
        // (e.g. RescaleSlope/Intercept for CT) is applied and we get
        // the true CT values rather than windowed 0-65535 output data.
        const DiPixel* pixel = di->getInterData();
        if (!pixel) return false;
        const void* data = pixel->getData();
        if (!data) return false;

        cv::Mat img;
        const auto representation = pixel->getRepresentation();
        auto convertTo16S = [&](int cvType) {
            cv::Mat tmp(h, w, cvType, const_cast<void*>(data));
            cv::Mat converted;
            tmp.convertTo(converted, CV_16SC1);
            return converted;
        };

        switch (representation) {
        case EPR_Sint16:
            img = cv::Mat(h, w, CV_16SC1, const_cast<void*>(data)).clone();
            break;
        case EPR_Uint16:
            img = convertTo16S(CV_16UC1);
            break;
        case EPR_Sint8:
            img = convertTo16S(CV_8SC1);
            break;
        case EPR_Uint8:
            img = convertTo16S(CV_8UC1);
            break;
        default:
            qDebug() << "Unsupported pixel representation:" << representation;
            return false;
        }

        mats.push_back(img);
        
        double x,y,z;
        double loc = std::numeric_limits<double>::quiet_NaN();
        if (reader.getImagePositionPatient(x,y,z)) {
            positions.push_back(QVector3D(x,y,z));
        } else {
            positions.push_back(QVector3D());
        }
        reader.getImageLocation(loc);
        locations.push_back(loc);
    }
    
    m_depth = static_cast<int>(mats.size());
    
    // **重要**: 座標系の設定をデバッグ付きで修正
    qDebug() << "=== Setting CT coordinate system ===";
    
    if (!positions.empty() && !positions.front().isNull()) {
        m_originX = positions.front().x();
        m_originY = positions.front().y();
        m_originZ = positions.front().z();
        qDebug() << QString("Using first slice position: (%1, %2, %3)")
                    .arg(m_originX, 0, 'f', 1).arg(m_originY, 0, 'f', 1).arg(m_originZ, 0, 'f', 1);
    } else if (!locations.empty() && !std::isnan(locations.front())) {
        m_originX = 0.0;
        m_originY = 0.0;
        m_originZ = locations.front();
        qDebug() << QString("Using slice location: (0, 0, %1)").arg(m_originZ, 0, 'f', 1);
    } else {
        m_originX = m_originY = m_originZ = 0.0;
        qDebug() << "No coordinate information found, using (0,0,0)";
    }
    
    // Z-offsetsとスペーシングの計算
    if (positions.size() >= 2) {
        QVector3D row(m_rowDir[0], m_rowDir[1], m_rowDir[2]);
        QVector3D col(m_colDir[0], m_colDir[1], m_colDir[2]);
        QVector3D slice(m_sliceDir[0], m_sliceDir[1], m_sliceDir[2]);
        QVector3D diff;
        if (!positions.front().isNull() && !positions.back().isNull()) {
            diff = positions.back() - positions.front();
        } else if (!std::isnan(locations.front()) && !std::isnan(locations.back())) {
            diff = slice * (locations.back() - locations.front());
        }
        if (!diff.isNull() && QVector3D::dotProduct(slice, diff) < 0) {
            slice *= -1.0;
        }
        slice.normalize();
        m_sliceDir[0]=slice.x(); m_sliceDir[1]=slice.y(); m_sliceDir[2]=slice.z();

        double sum = 0.0;
        int count = 0;
        for (size_t i = 1; i < positions.size(); ++i) {
            if (!positions[i].isNull() && !positions[i-1].isNull()) {
                QVector3D d = positions[i] - positions[i-1];
                sum += std::fabs(QVector3D::dotProduct(slice, d));
                ++count;
            } else if (!std::isnan(locations[i]) && !std::isnan(locations[i-1])) {
                sum += std::fabs(locations[i] - locations[i-1]);
                ++count;
            }
        }
        double delta = (count > 0) ? sum / count : 0.0;
        if (delta > 0.0) m_spacingZ = delta;

        m_zOffsets.resize(positions.size());
        QVector3D origin(m_originX, m_originY, m_originZ);
        for (size_t i = 0; i < positions.size(); ++i) {
            if (!positions[i].isNull()) {
                QVector3D diff = positions[i] - origin;
                m_zOffsets[i] = QVector3D::dotProduct(slice, diff);
            } else if (!std::isnan(locations[i]) && !std::isnan(locations.front())) {
                m_zOffsets[i] = locations[i] - locations.front();
            } else {
                m_zOffsets[i] = static_cast<double>(i) * m_spacingZ;
            }
        }
    } else if (!positions.empty()) {
        m_zOffsets.resize(1);
        m_zOffsets[0] = 0.0;
    }
    
    int sizes[3] = {m_depth, m_height, m_width};
    m_volume.create(3, sizes, CV_16SC1);
    for (int z = 0; z < m_depth; ++z) {
        cv::Mat dst(m_height, m_width, CV_16SC1, m_volume.ptr<short>(z));
        mats[z].copyTo(dst);
    }
    
    qDebug() << QString("=== Final CT Volume Coordinate System ===");
    qDebug() << QString("Origin: (%1, %2, %3)")
                .arg(m_originX, 0, 'f', 1).arg(m_originY, 0, 'f', 1).arg(m_originZ, 0, 'f', 1);
    qDebug() << QString("Spacing: (%1, %2, %3)")
                .arg(m_spacingX, 0, 'f', 3).arg(m_spacingY, 0, 'f', 3).arg(m_spacingZ, 0, 'f', 1);

    // Report patient-space extents (mm) for the whole CT volume
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
            QVector3D p = voxelToPatient(ix + 0.5, iy + 0.5, iz + 0.5);
            upd(minX, maxX, p.x());
            upd(minY, maxY, p.y());
            upd(minZ, maxZ, p.z());
        }
        qDebug() << QString("CT Size (vox): %1 x %2 x %3").arg(m_width).arg(m_height).arg(m_depth);
        qDebug() << QString("CT Extents (patient mm): X:[%1, %2] Y:[%3, %4] Z:[%5, %6]")
                        .arg(minX, 0, 'f', 3).arg(maxX, 0, 'f', 3)
                        .arg(minY, 0, 'f', 3).arg(maxY, 0, 'f', 3)
                        .arg(minZ, 0, 'f', 3).arg(maxZ, 0, 'f', 3);
    }
    
    return true;
}

namespace {
template <typename T>
cv::Mat getSliceAxialImpl(const cv::Mat &vol, int index) {
    int sizes[3];
    for (int i = 0; i < 3; ++i) sizes[i] = vol.size[i];
    if (index < 0 || index >= sizes[0]) return cv::Mat();
    return cv::Mat(sizes[1], sizes[2], cv::DataType<T>::type,
                   const_cast<T *>(vol.ptr<T>(index)))
        .clone();
}

template <typename T>
cv::Mat getSliceSagittalImpl(const cv::Mat &vol, int index) {
    int depth = vol.size[0];
    int height = vol.size[1];
    int width = vol.size[2];
    if (index < 0 || index >= width) return cv::Mat();
    cv::Mat slice(height, depth, cv::DataType<T>::type);
    for (int z = 0; z < depth; ++z) {
        const T *srcRow = vol.ptr<T>(z);
        for (int y = 0; y < height; ++y) {
            slice.at<T>(y, z) = srcRow[y * width + index];
        }
    }
    // Historical orientation: transpose then flip vertically so head is up.
    cv::transpose(slice, slice);
    cv::flip(slice, slice, 0);
    return slice;
}

template <typename T>
cv::Mat getSliceCoronalImpl(const cv::Mat &vol, int index) {
    int depth = vol.size[0];
    int height = vol.size[1];
    int width = vol.size[2];
    if (index < 0 || index >= height) return cv::Mat();
    cv::Mat slice(width, depth, cv::DataType<T>::type);
    for (int z = 0; z < depth; ++z) {
        const T *srcRow = vol.ptr<T>(z);
        for (int x = 0; x < width; ++x) {
            slice.at<T>(x, z) = srcRow[index * width + x];
        }
    }
    // Historical orientation: transpose then flip vertically so head is up.
    cv::transpose(slice, slice);
    cv::flip(slice, slice, 0);
    return slice;
}
} // namespace

cv::Mat getSliceAxial(const cv::Mat &vol, int index) {
    switch (vol.type()) {
    case CV_8UC1:
        return getSliceAxialImpl<unsigned char>(vol, index);
    case CV_16SC1:
        return getSliceAxialImpl<short>(vol, index);
    case CV_32FC1:
        return getSliceAxialImpl<float>(vol, index);
    default:
        return cv::Mat();
    }
}

cv::Mat getSliceSagittal(const cv::Mat &vol, int index) {
    switch (vol.type()) {
    case CV_8UC1:
        return getSliceSagittalImpl<unsigned char>(vol, index);
    case CV_16SC1:
        return getSliceSagittalImpl<short>(vol, index);
    case CV_32FC1:
        return getSliceSagittalImpl<float>(vol, index);
    default:
        return cv::Mat();
    }
}

cv::Mat getSliceCoronal(const cv::Mat &vol, int index) {
    switch (vol.type()) {
    case CV_8UC1:
        return getSliceCoronalImpl<unsigned char>(vol, index);
    case CV_16SC1:
        return getSliceCoronalImpl<short>(vol, index);
    case CV_32FC1:
        return getSliceCoronalImpl<float>(vol, index);
    default:
        return cv::Mat();
    }
}

bool DicomVolume::createFromReference(const DicomVolume &reference,
                                      const cv::Mat &volumeData)
{
    if (volumeData.dims != 3)
        return false;

    const int depth = volumeData.size[0];
    const int height = volumeData.size[1];
    const int width = volumeData.size[2];

    if (width != reference.m_width || height != reference.m_height ||
        depth != reference.m_depth)
        return false;

    if (volumeData.type() != CV_16SC1 && volumeData.type() != CV_32FC1 &&
        volumeData.type() != CV_8UC1)
        return false;

    m_volume = volumeData.clone();
    m_width = reference.m_width;
    m_height = reference.m_height;
    m_depth = reference.m_depth;
    m_spacingX = reference.m_spacingX;
    m_spacingY = reference.m_spacingY;
    m_spacingZ = reference.m_spacingZ;
    m_originX = reference.m_originX;
    m_originY = reference.m_originY;
    m_originZ = reference.m_originZ;
    m_frameUID = reference.m_frameUID;
    m_rowDir[0] = reference.m_rowDir[0];
    m_rowDir[1] = reference.m_rowDir[1];
    m_rowDir[2] = reference.m_rowDir[2];
    m_colDir[0] = reference.m_colDir[0];
    m_colDir[1] = reference.m_colDir[1];
    m_colDir[2] = reference.m_colDir[2];
    m_sliceDir[0] = reference.m_sliceDir[0];
    m_sliceDir[1] = reference.m_sliceDir[1];
    m_sliceDir[2] = reference.m_sliceDir[2];
    m_zOffsets = reference.m_zOffsets;

    return true;
}

namespace {
    constexpr quint32 FUSION_VOLUME_MAGIC = 0x53335246; // 'F''R''3''S' little-endian
    constexpr quint32 FUSION_VOLUME_VERSION = 1;

    template <typename T>
    bool readArray(QDataStream &in, T *buffer, int count) {
        for (int i = 0; i < count; ++i) {
            in >> buffer[i];
            if (in.status() != QDataStream::Ok)
                return false;
        }
        return true;
    }
}

bool DicomVolume::saveToFile(const QString &filePath) const
{
    if (m_volume.empty() || m_volume.dims != 3)
        return false;

    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly))
        return false;

    QDataStream out(&file);
    out.setByteOrder(QDataStream::LittleEndian);
    out.setVersion(QDataStream::Qt_5_15);

    out << FUSION_VOLUME_MAGIC;
    out << FUSION_VOLUME_VERSION;
    out << static_cast<quint32>(m_width);
    out << static_cast<quint32>(m_height);
    out << static_cast<quint32>(m_depth);
    out << static_cast<quint32>(m_volume.type());
    out << m_spacingX << m_spacingY << m_spacingZ;
    out << m_originX << m_originY << m_originZ;
    out << m_rowDir[0] << m_rowDir[1] << m_rowDir[2];
    out << m_colDir[0] << m_colDir[1] << m_colDir[2];
    out << m_sliceDir[0] << m_sliceDir[1] << m_sliceDir[2];

    QByteArray frameBytes = m_frameUID.toUtf8();
    out << static_cast<quint32>(frameBytes.size());
    if (!frameBytes.isEmpty())
        out.writeRawData(frameBytes.constData(), frameBytes.size());

    out << static_cast<quint32>(m_zOffsets.size());
    for (double value : m_zOffsets)
        out << value;

    const size_t totalBytes = static_cast<size_t>(m_volume.total()) * m_volume.elemSize();
    if (totalBytes == 0)
        return false;

    if (!m_volume.isContinuous()) {
        cv::Mat contiguous = m_volume.clone();
        if (contiguous.empty())
            return false;
        out.writeRawData(reinterpret_cast<const char *>(contiguous.data),
                         static_cast<int>(totalBytes));
    } else {
        out.writeRawData(reinterpret_cast<const char *>(m_volume.data),
                         static_cast<int>(totalBytes));
    }

    return out.status() == QDataStream::Ok;
}

bool DicomVolume::loadFromFile(const QString &filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly))
        return false;

    QDataStream in(&file);
    in.setByteOrder(QDataStream::LittleEndian);
    in.setVersion(QDataStream::Qt_5_15);

    quint32 magic = 0;
    quint32 version = 0;
    in >> magic >> version;
    if (in.status() != QDataStream::Ok || magic != FUSION_VOLUME_MAGIC ||
        version != FUSION_VOLUME_VERSION)
        return false;

    quint32 width = 0, height = 0, depth = 0, type = 0;
    in >> width >> height >> depth >> type;
    double spacingX = 0.0, spacingY = 0.0, spacingZ = 0.0;
    double originX = 0.0, originY = 0.0, originZ = 0.0;
    in >> spacingX >> spacingY >> spacingZ;
    in >> originX >> originY >> originZ;
    if (in.status() != QDataStream::Ok)
        return false;

    double rowDir[3] = {0.0, 0.0, 0.0};
    double colDir[3] = {0.0, 0.0, 0.0};
    double sliceDir[3] = {0.0, 0.0, 0.0};
    if (!readArray(in, rowDir, 3) || !readArray(in, colDir, 3) ||
        !readArray(in, sliceDir, 3))
        return false;

    quint32 frameSize = 0;
    in >> frameSize;
    QByteArray frameBytes;
    if (frameSize > 0) {
        frameBytes.resize(static_cast<int>(frameSize));
        if (in.readRawData(frameBytes.data(), frameBytes.size()) != frameBytes.size())
            return false;
    }

    quint32 offsetCount = 0;
    in >> offsetCount;
    std::vector<double> offsets;
    offsets.reserve(offsetCount);
    for (quint32 i = 0; i < offsetCount; ++i) {
        double value = 0.0;
        in >> value;
        if (in.status() != QDataStream::Ok)
            return false;
        offsets.push_back(value);
    }

    if (width == 0 || height == 0 || depth == 0)
        return false;

    int sizes[3] = {static_cast<int>(depth), static_cast<int>(height),
                    static_cast<int>(width)};
    cv::Mat volume(3, sizes, static_cast<int>(type));
    const size_t totalBytes = static_cast<size_t>(volume.total()) * volume.elemSize();
    if (static_cast<size_t>(file.size()) - static_cast<size_t>(file.pos()) < totalBytes)
        return false;
    if (in.readRawData(reinterpret_cast<char *>(volume.data),
                       static_cast<int>(totalBytes)) != static_cast<int>(totalBytes))
        return false;

    if (!volume.isContinuous())
        volume = volume.clone();
    if (volume.empty())
        return false;

    m_volume = volume;
    m_width = static_cast<int>(width);
    m_height = static_cast<int>(height);
    m_depth = static_cast<int>(depth);
    m_spacingX = spacingX;
    m_spacingY = spacingY;
    m_spacingZ = spacingZ;
    m_originX = originX;
    m_originY = originY;
    m_originZ = originZ;
    m_rowDir[0] = rowDir[0];
    m_rowDir[1] = rowDir[1];
    m_rowDir[2] = rowDir[2];
    m_colDir[0] = colDir[0];
    m_colDir[1] = colDir[1];
    m_colDir[2] = colDir[2];
    m_sliceDir[0] = sliceDir[0];
    m_sliceDir[1] = sliceDir[1];
    m_sliceDir[2] = sliceDir[2];
    m_frameUID = QString::fromUtf8(frameBytes);
    m_zOffsets = std::move(offsets);

    return true;
}

QImage DicomVolume::matToImage(const cv::Mat& slice, double window, double level) const
{
    if (slice.empty()) return QImage();
    cv::Mat normalized(slice.rows, slice.cols, CV_8UC1);
    double minVal = level - window / 2.0;
    double maxVal = level + window / 2.0;
    slice.convertTo(normalized, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    QImage img(normalized.data, normalized.cols, normalized.rows, normalized.step, QImage::Format_Grayscale8);
    return img.copy();
}

QImage DicomVolume::getSlice(int index, Orientation ori, double window, double level) const
{
    cv::Mat slice;
    switch (ori) {
    case Orientation::Axial:
        slice = getSliceAxial(m_volume, index);
        break;
    case Orientation::Sagittal:
        slice = getSliceSagittal(m_volume, index);
        break;
    case Orientation::Coronal:
        slice = getSliceCoronal(m_volume, index);
        break;
    }
    QImage img = matToImage(slice, window, level);
    return adjustAspect(img, ori);
}

QImage DicomVolume::adjustAspect(const QImage& img, Orientation ori) const
{
    Q_UNUSED(ori);
    // OpenGLImageWidget で画素間隔を考慮した表示を行うため、
    // ここではリサイズを行わない。
    return img;
}

QVector3D DicomVolume::voxelToPatient(double x, double y, double z) const
{
    QVector3D origin(m_originX, m_originY, m_originZ);
    QVector3D row(m_rowDir[0], m_rowDir[1], m_rowDir[2]);
    QVector3D col(m_colDir[0], m_colDir[1], m_colDir[2]);
    QVector3D slice(m_sliceDir[0], m_sliceDir[1], m_sliceDir[2]);
    // Apply spacing along row/col and interpolate Z offset for fractional indices.
    double zOffset = z * m_spacingZ;
    if (!m_zOffsets.empty()) {
        const int n = static_cast<int>(m_zOffsets.size());
        if (n == 1) {
            zOffset = m_zOffsets[0];
        } else {
            if (z <= 0.0) {
                // extrapolate using first interval
                double dz = m_zOffsets[1] - m_zOffsets[0];
                zOffset = m_zOffsets[0] + z * dz;
            } else if (z >= n - 1) {
                // extrapolate using last interval
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
    return origin + row * (x * m_spacingX) + col * (y * m_spacingY)
           + slice * zOffset;
}

QVector3D DicomVolume::patientToVoxelContinuous(const QVector3D& p) const
{
    QVector3D origin(m_originX, m_originY, m_originZ);
    QVector3D rel = p - origin;
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
