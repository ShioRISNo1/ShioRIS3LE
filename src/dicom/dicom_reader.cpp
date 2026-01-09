#include "dicom/dicom_reader.h"
#include <QDebug>
#include <QApplication>
#include <QStringList>

DicomReader::DicomReader()
    : m_dicomImage(nullptr)
    , m_dataset(nullptr)
    , m_imageNeedsUpdate(true)
    , m_window(256.0)
    , m_level(128.0)
{
}

DicomReader::~DicomReader() = default;

bool DicomReader::loadDicomFile(const QString& filename)
{
    // DCMTKでDICOMファイルを読み込み
    DcmFileFormat fileformat;
    OFCondition status = fileformat.loadFile(filename.toLocal8Bit().data());
    
    if (status.bad()) {
        qDebug() << "Error: cannot read DICOM file" << filename;
        return false;
    }
    
    // データセットを取得
    m_dataset = std::make_unique<DcmDataset>(*fileformat.getDataset());

    // Pixel Data が存在する場合のみ DicomImage を生成する
    if (m_dataset->tagExistsWithValue(DCM_PixelData)) {
        m_dicomImage = std::make_unique<DicomImage>(filename.toLocal8Bit().data());
        if (m_dicomImage->getStatus() != EIS_Normal) {
            qDebug() << "Warning: cannot process DICOM image" << filename;
            m_dicomImage.reset();
        } else {
            // デフォルトのウィンドウ/レベルを設定
            if (m_dicomImage->isMonochrome()) {
                double windowCenter, windowWidth;
                if (m_dicomImage->getWindow(windowCenter, windowWidth)) {
                    m_level = windowCenter;
                    m_window = windowWidth;
                }
            }
            m_imageNeedsUpdate = true;
        }
    } else {
        m_dicomImage.reset();
    }
    
    // **修正**: デバッグ出力を条件付きにする（デバッグビルド時のみ）
#ifdef QT_DEBUG
    // デバッグビルド時のみ詳細な座標情報を出力
    static bool debugCoordinates = qEnvironmentVariableIsSet("SHIORIS3_DEBUG_COORDS");
    if (debugCoordinates) {
        qDebug() << "=== CT DICOM Coordinate Debug ===";
        qDebug() << "CT file:" << filename;
        
        // ImagePositionPatient
        OFString positionStr;
        if (m_dataset->findAndGetOFString(DCM_ImagePositionPatient, positionStr).good()) {
            qDebug() << "CT ImagePositionPatient:" << QString::fromLatin1(positionStr.c_str());
        }
        
        // ImageOrientationPatient
        OFString orientationStr;
        if (m_dataset->findAndGetOFString(DCM_ImageOrientationPatient, orientationStr).good()) {
            qDebug() << "CT ImageOrientationPatient:" << QString::fromLatin1(orientationStr.c_str());
        }
        
        // PatientPosition
        OFString pos;
        if (m_dataset->findAndGetOFString(DCM_PatientPosition, pos).good()) {
            qDebug() << "CT PatientPosition:" << QString::fromLatin1(pos.c_str());
        }
        
        // PixelSpacing
        OFString spacingStr;
        if (m_dataset->findAndGetOFString(DCM_PixelSpacing, spacingStr).good()) {
            qDebug() << "CT PixelSpacing:" << QString::fromLatin1(spacingStr.c_str());
        }
        
        // SliceLocation
        OFString sliceLocationStr;
        if (m_dataset->findAndGetOFString(DCM_SliceLocation, sliceLocationStr).good()) {
            qDebug() << "CT SliceLocation:" << QString::fromLatin1(sliceLocationStr.c_str());
        }
        
        qDebug() << "Successfully loaded DICOM file:" << filename;
        qDebug() << "Image size:" << getWidth() << "x" << getHeight();
        qDebug() << "Patient:" << getPatientName();
        qDebug() << "Modality:" << getModality();
    }
#endif
    
    return true;
}

QImage DicomReader::getImage() const
{
    if (m_imageNeedsUpdate) {
        updateQImage();
    }
    return m_qimage;
}

QPixmap DicomReader::getPixmap() const
{
    return QPixmap::fromImage(getImage());
}

QString DicomReader::getPatientName() const
{
    return getStringValue(DCM_PatientName);
}

QString DicomReader::getPatientID() const
{
    return getStringValue(DCM_PatientID);
}

QString DicomReader::getStudyDate() const
{
    return getStringValue(DCM_StudyDate);
}

QString DicomReader::getModality() const
{
    return getStringValue(DCM_Modality);
}

QString DicomReader::getStudyDescription() const
{
    return getStringValue(DCM_StudyDescription);
}

QString DicomReader::getFrameOfReferenceUID() const
{
    return getStringValue(DCM_FrameOfReferenceUID);
}

int DicomReader::getWidth() const
{
    if (!m_dicomImage) return 0;
    return static_cast<int>(m_dicomImage->getWidth());
}

int DicomReader::getHeight() const
{
    if (!m_dicomImage) return 0;
    return static_cast<int>(m_dicomImage->getHeight());
}

bool DicomReader::getPixelSpacing(double& row, double& col) const
{
    row = col = 1.0;
    if (!m_dataset) return false;
    
    // **修正**: デバッグ出力を削除または条件付きにする
#ifdef QT_DEBUG
    static bool debugSpacing = qEnvironmentVariableIsSet("SHIORIS3_DEBUG_SPACING");
    if (debugSpacing) {
        qDebug() << "=== getPixelSpacing Debug ===";
    }
#endif
    
    // 方法1: Float64配列として取得
    const Float64* spacingArray = nullptr;
    unsigned long spacingCount = 0;
    if (m_dataset->findAndGetFloat64Array(DCM_PixelSpacing, spacingArray, &spacingCount).good() 
        && spacingArray && spacingCount >= 2) {
        row = spacingArray[0];
        col = spacingArray[1];
#ifdef QT_DEBUG
        if (debugSpacing) {
            qDebug() << QString("Spacing array success: [%1, %2]").arg(row, 0, 'f', 3).arg(col, 0, 'f', 3);
        }
#endif
        return true;
    }
    
    // 方法2: 文字列として取得
    OFString value;
    if (m_dataset->findAndGetOFString(DCM_PixelSpacing, value).good()) {
        QString spacingStr = QString::fromLatin1(value.c_str());
#ifdef QT_DEBUG
        if (debugSpacing) {
            qDebug() << QString("Spacing string: '%1'").arg(spacingStr);
        }
#endif
        
        QStringList parts = spacingStr.split("\\", Qt::SkipEmptyParts);
        if (parts.size() >= 2) {
            bool ok1, ok2;
            row = parts[0].toDouble(&ok1);
            col = parts[1].toDouble(&ok2);
            if (ok1 && ok2) {
#ifdef QT_DEBUG
                if (debugSpacing) {
                    qDebug() << "Spacing parsing success";
                }
#endif
                return true;
            }
        }
        
        // 単一値の場合、両方向に同じ値を使用
        if (parts.size() == 1) {
            bool ok = false;
            double spacing = parts[0].toDouble(&ok);
            if (ok) {
                row = col = spacing;
#ifdef QT_DEBUG
                if (debugSpacing) {
                    qDebug() << QString("Using single spacing value: %1").arg(spacing, 0, 'f', 3);
                }
#endif
                return true;
            }
        }
    }
    
#ifdef QT_DEBUG
    if (debugSpacing) {
        qDebug() << "Spacing detection failed";
    }
#endif
    return false;
}

bool DicomReader::getImagePositionPatient(double& x, double& y, double& z) const
{
    x = y = z = 0.0;
    if (!m_dataset) return false;
    
#ifdef QT_DEBUG
    static bool debugPosition = qEnvironmentVariableIsSet("SHIORIS3_DEBUG_POSITION");
    if (debugPosition) {
        qDebug() << "=== getImagePositionPatient Debug ===";
    }
#endif
    
    // 方法1: 個別要素として取得
    Float64 pos[3] = {0.0, 0.0, 0.0};
    if (m_dataset->findAndGetFloat64(DCM_ImagePositionPatient, pos[0], 0).good() &&
        m_dataset->findAndGetFloat64(DCM_ImagePositionPatient, pos[1], 1).good() &&
        m_dataset->findAndGetFloat64(DCM_ImagePositionPatient, pos[2], 2).good()) {
        x = pos[0];
        y = pos[1];
        z = pos[2];
#ifdef QT_DEBUG
        if (debugPosition) {
            qDebug() << QString("Method 1 success: (%1, %2, %3)").arg(x, 0, 'f', 1).arg(y, 0, 'f', 1).arg(z, 0, 'f', 1);
        }
#endif
        return true;
    }
    
    // 方法2: Float64配列として取得
    const Float64* posArray = nullptr;
    unsigned long posCount = 0;
    if (m_dataset->findAndGetFloat64Array(DCM_ImagePositionPatient, posArray, &posCount).good() 
        && posArray && posCount >= 3) {
        x = posArray[0];
        y = posArray[1];
        z = posArray[2];
#ifdef QT_DEBUG
        if (debugPosition) {
            qDebug() << QString("Method 2 success: (%1, %2, %3)").arg(x, 0, 'f', 1).arg(y, 0, 'f', 1).arg(z, 0, 'f', 1);
        }
#endif
        return true;
    }
    
    // 方法3: 文字列として取得
    OFString value;
    if (m_dataset->findAndGetOFString(DCM_ImagePositionPatient, value).good()) {
        QString posStr = QString::fromLatin1(value.c_str());
        QStringList parts = posStr.split("\\", Qt::SkipEmptyParts);
#ifdef QT_DEBUG
        if (debugPosition) {
            qDebug() << QString("Method 3 raw string: '%1'").arg(posStr);
        }
#endif
        if (parts.size() >= 3) {
            bool ok[3];
            x = parts[0].toDouble(&ok[0]);
            y = parts[1].toDouble(&ok[1]);
            z = parts[2].toDouble(&ok[2]);
            if (ok[0] && ok[1] && ok[2]) {
#ifdef QT_DEBUG
                if (debugPosition) {
                    qDebug() << QString("Method 3 success: (%1, %2, %3)")
                                .arg(x, 0, 'f', 1).arg(y, 0, 'f', 1).arg(z, 0, 'f', 1);
                }
#endif
                return true;
            }
        }

        // 特殊ケース: 値が1つしかない場合の対処
        if (parts.size() == 1) {
            bool ok = false;
            double singleValue = parts[0].toDouble(&ok);
            if (ok) {
                x = singleValue;

                OFString yValue;
                if (m_dataset->findAndGetOFString(DcmTagKey(0x0020, 0x0032), yValue, 1).good()) {
                    y = QString::fromLatin1(yValue.c_str()).toDouble();
                } else {
                    y = singleValue; // 仮の値
                }

                OFString zValue;
                if (m_dataset->findAndGetOFString(DCM_SliceLocation, zValue).good()) {
                    z = QString::fromLatin1(zValue.c_str()).toDouble();
                } else if (m_dataset->findAndGetOFString(DcmTagKey(0x0020, 0x0032), zValue, 2).good()) {
                    z = QString::fromLatin1(zValue.c_str()).toDouble();
                } else {
                    OFString instanceStr;
                    if (m_dataset->findAndGetOFString(DCM_InstanceNumber, instanceStr).good()) {
                        bool okInst = false;
                        int instance = QString::fromLatin1(instanceStr.c_str()).toInt(&okInst);
                        if (okInst) {
                            z = (instance - 1) * 2.0; // 仮に2mm間隔とする
                        }
                    }
                }
#ifdef QT_DEBUG
                if (debugPosition) {
                    qDebug() << QString("Single value reconstruction: (%1, %2, %3)")
                                .arg(x, 0, 'f', 1).arg(y, 0, 'f', 1).arg(z, 0, 'f', 1);
                }
#endif
                return true;
            }
        }
    }

#ifdef QT_DEBUG
    if (debugPosition) {
        qDebug() << "Position detection failed";
    }
#endif
    return false;
}

double DicomReader::getSliceThickness() const
{
    if (!m_dataset) return 1.0;
    Float64 val = 1.0;
    if (m_dataset->findAndGetFloat64(DCM_SliceThickness, val).good()) {
        return static_cast<double>(val);
    }
    return 1.0;
}


bool DicomReader::getImageOrientationPatient(double& r1, double& r2, double& r3,
                                              double& c1, double& c2, double& c3) const
{
    r1 = r2 = r3 = c1 = c2 = c3 = 0.0;
    if (!m_dataset) return false;
    
    qDebug() << "=== getImageOrientationPatient Debug ===";
    
    // 方法1: Float64配列として取得
    const Float64* orientArray = nullptr;
    unsigned long orientCount = 0;
    if (m_dataset->findAndGetFloat64Array(DCM_ImageOrientationPatient, orientArray, &orientCount).good() 
        && orientArray && orientCount >= 6) {
        r1 = orientArray[0]; r2 = orientArray[1]; r3 = orientArray[2];
        c1 = orientArray[3]; c2 = orientArray[4]; c3 = orientArray[5];
        qDebug() << QString("Orientation array success: [%1,%2,%3,%4,%5,%6]")
                    .arg(r1, 0, 'f', 3).arg(r2, 0, 'f', 3).arg(r3, 0, 'f', 3)
                    .arg(c1, 0, 'f', 3).arg(c2, 0, 'f', 3).arg(c3, 0, 'f', 3);
        return true;
    }

    // 方法1b: DS多値を個別要素で取得（より堅牢）
    {
        OFString comp[6];
        bool okAll = true;
        double vals[6] = {0};
        for (int i = 0; i < 6; ++i) {
            if (!m_dataset->findAndGetOFString(DCM_ImageOrientationPatient, comp[i], i).good()) {
                okAll = false;
                break;
            }
            bool ok = false;
            vals[i] = QString::fromLatin1(comp[i].c_str()).toDouble(&ok);
            if (!ok) { okAll = false; break; }
        }
        if (okAll) {
            r1 = vals[0]; r2 = vals[1]; r3 = vals[2];
            c1 = vals[3]; c2 = vals[4]; c3 = vals[5];
            qDebug() << QString("Orientation DS components success: [%1,%2,%3,%4,%5,%6]")
                        .arg(r1, 0, 'f', 3).arg(r2, 0, 'f', 3).arg(r3, 0, 'f', 3)
                        .arg(c1, 0, 'f', 3).arg(c2, 0, 'f', 3).arg(c3, 0, 'f', 3);
            return true;
        }
    }
    
    // 方法2: 文字列として取得
    OFString value;
    if (m_dataset->findAndGetOFString(DCM_ImageOrientationPatient, value).good()) {
        QString orientStr = QString::fromLatin1(value.c_str());
        qDebug() << QString("Orientation string: '%1'").arg(orientStr);
        
        QStringList parts = orientStr.split("\\", Qt::SkipEmptyParts);
        if (parts.size() >= 6) {
            bool ok[6];
            double vals[6];
            for (int i = 0; i < 6; ++i) {
                vals[i] = parts[i].toDouble(&ok[i]);
            }
            if (ok[0] && ok[1] && ok[2] && ok[3] && ok[4] && ok[5]) {
                r1 = vals[0]; r2 = vals[1]; r3 = vals[2];
                c1 = vals[3]; c2 = vals[4]; c3 = vals[5];
                qDebug() << "Orientation parsing success";
                return true;
            }
        }
        
        // 単一値の場合、デフォルトの恒等行列を使用
        if (parts.size() == 1) {
            r1 = 1.0; r2 = 0.0; r3 = 0.0;
            c1 = 0.0; c2 = 1.0; c3 = 0.0;
            qDebug() << "Using identity matrix for single value";
            return true;
        }
    }
    
    qDebug() << "Orientation detection failed";
    return false;
}

bool DicomReader::getImageLocation(double& loc) const
{
    if (!m_dataset)
        return false;

    OFString value;
    if (m_dataset->findAndGetOFString(DCM_SliceLocation, value).good() ||
        m_dataset->findAndGetOFString(DcmTagKey(0x0020, 0x0050), value).good()) {
        bool ok = false;
        double tmp = QString::fromLatin1(value.c_str()).toDouble(&ok);
        if (ok) {
            loc = tmp;
            return true;
        }
    }
    return false;
}

void DicomReader::setWindowLevel(double window, double level)
{
    if (m_window != window || m_level != level) {
        m_window = window;
        m_level = level;
        m_imageNeedsUpdate = true;
    }
}

void DicomReader::getWindowLevel(double& window, double& level) const
{
    window = m_window;
    level = m_level;
}

double DicomReader::getSlicePosition(const QString& filename)
{
    DcmFileFormat fileformat;
    if (fileformat.loadFile(filename.toLocal8Bit().data()).bad()) {
        return 0.0;
    }

    DcmDataset* dataset = fileformat.getDataset();
    OFString value;

    if (dataset->findAndGetOFString(DCM_ImagePositionPatient, value).good()) {
        QStringList parts = QString::fromLatin1(value.c_str()).split("\\");
        if (parts.size() == 3) {
            bool ok = false;
            double pos = parts.at(2).toDouble(&ok);
            if (ok) return pos;
        }
    }

    if (dataset->findAndGetOFString(DCM_SliceLocation, value).good()) {
        bool ok = false;
        double pos = QString::fromLatin1(value.c_str()).toDouble(&ok);
        if (ok) return pos;
    }

    if (dataset->findAndGetOFString(DCM_InstanceNumber, value).good()) {
        bool ok = false;
        double pos = QString::fromLatin1(value.c_str()).toDouble(&ok);
        if (ok) return pos;
    }

    return 0.0;
}

void DicomReader::updateQImage() const
{
    if (!m_dicomImage) {
        m_qimage = QImage();
        return;
    }
    
    // ウィンドウ/レベルを適用
    if (m_dicomImage->isMonochrome()) {
        m_dicomImage->setWindow(m_level, m_window);
    }
    
    // 8ビットに変換してQImageに変換
    const DiPixel* pixel = m_dicomImage->getInterData();
    if (!pixel) {
        m_qimage = QImage();
        return;
    }
    
    int width = getWidth();
    int height = getHeight();
    
    // 8ビット画像として出力
    const void* data = m_dicomImage->getOutputData(8);
    if (!data) {
        m_qimage = QImage();
        return;
    }
    
    // QImageを作成（グレースケール）
    m_qimage = QImage(static_cast<const uchar*>(data), width, height, QImage::Format_Grayscale8);
    
    // 画像をコピーして独立させる
    m_qimage = m_qimage.copy();
    
    m_imageNeedsUpdate = false;
}

QString DicomReader::getStringValue(const DcmTagKey& tag) const
{
    if (!m_dataset) return QString();
    
    OFString value;
    if (m_dataset->findAndGetOFString(tag, value).good()) {
        return QString::fromLatin1(value.c_str());
    }
    return QString();
}
