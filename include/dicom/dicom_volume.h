#ifndef DICOM_VOLUME_H
#define DICOM_VOLUME_H

#include <QString>
#include <QImage>
#include <vector>
#include <opencv2/core.hpp>
#include <QVector3D>

class DicomVolume
{
public:
    enum class Orientation { Axial, Sagittal, Coronal };

    DicomVolume();

    bool loadFromDirectory(const QString& directory);

    int width() const { return m_width; }
    int height() const { return m_height; }
    int depth() const { return m_depth; }
    double spacingX() const { return m_spacingX; }
    double spacingY() const { return m_spacingY; }
    double spacingZ() const { return m_spacingZ; }
    double originX() const { return m_originX; }
    double originY() const { return m_originY; }
    double originZ() const { return m_originZ; }
    const double* rowDirection() const { return m_rowDir; }
    const double* colDirection() const { return m_colDir; }
    const double* sliceDirection() const { return m_sliceDir; }
    QString frameOfReferenceUID() const { return m_frameUID; }

    QVector3D voxelToPatient(double x, double y, double z) const;
    QVector3D voxelToPatient(int x, int y, int z) const {
        return voxelToPatient(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z));
    }
    QVector3D patientToVoxelContinuous(const QVector3D& p) const;

    QImage getSlice(int index, Orientation ori, double window, double level) const;

    // Raw volume access (for segmentation and processing)
    const cv::Mat& data() const { return m_volume; }

    bool createFromReference(const DicomVolume& reference,
                             const cv::Mat& volumeData);

    bool saveToFile(const QString& filePath) const;
    bool loadFromFile(const QString& filePath);

private:
    void clear();
    QImage matToImage(const cv::Mat& slice, double window, double level) const;
    QImage adjustAspect(const QImage& img, Orientation ori) const;

    cv::Mat m_volume; // 3D volume: depth x height x width
    int m_width;
    int m_height;
    int m_depth;
    double m_spacingX;
    double m_spacingY;
    double m_spacingZ;
    double m_originX{0.0};
    double m_originY{0.0};
    double m_originZ{0.0};
    QString m_frameUID;
    double m_rowDir[3]{1.0, 0.0, 0.0};
    double m_colDir[3]{0.0, 1.0, 0.0};
    double m_sliceDir[3]{0.0, 0.0, 1.0};
    std::vector<double> m_zOffsets;
};

// Helper functions to extract raw slices from a 3D cv::Mat volume
cv::Mat getSliceAxial(const cv::Mat& vol, int index);
cv::Mat getSliceSagittal(const cv::Mat& vol, int index);
cv::Mat getSliceCoronal(const cv::Mat& vol, int index);

#endif // DICOM_VOLUME_H
