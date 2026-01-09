#ifndef DICOM_READER_H
#define DICOM_READER_H

#include <QString>
#include <QPixmap>
#include <QImage>
#include <memory>

// DCMTK includes
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>

class DicomReader
{
public:
    DicomReader();
    ~DicomReader();

    // DICOM画像の読み込み
    bool loadDicomFile(const QString& filename);
    
    // QImageとして画像を取得
    QImage getImage() const;
    
    // QPixmapとして画像を取得
    QPixmap getPixmap() const;
    const DicomImage* dicomImage() const { return m_dicomImage.get(); }
    DicomImage* dicomImage() { return m_dicomImage.get(); }
    
    // 画像情報の取得
    QString getPatientName() const;
    QString getPatientID() const;
    QString getStudyDate() const;
    QString getModality() const;
    QString getStudyDescription() const;
    QString getFrameOfReferenceUID() const;
    
    // 画像サイズ
    int getWidth() const;
    int getHeight() const;

    // spacing information
    bool getPixelSpacing(double& row, double& col) const;
    double getSliceThickness() const;
    bool getImagePositionPatient(double& x, double& y, double& z) const;
    bool getImageOrientationPatient(double& r1, double& r2, double& r3,
                                    double& c1, double& c2, double& c3) const;
    bool getImageLocation(double& loc) const;
    
    // ウィンドウ/レベル調整
    void setWindowLevel(double window, double level);
    void getWindowLevel(double& window, double& level) const;
    // Slice position for sorting
    static double getSlicePosition(const QString& filename);
    
private:
    std::unique_ptr<DicomImage> m_dicomImage;
    std::unique_ptr<DcmDataset> m_dataset;
    mutable QImage m_qimage;
    mutable bool m_imageNeedsUpdate;
    
    double m_window;
    double m_level;
    
    void updateQImage() const;
    QString getStringValue(const DcmTagKey& tag) const;
};

#endif // DICOM_READER_H