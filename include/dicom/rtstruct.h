#ifndef RTSTRUCT_H
#define RTSTRUCT_H

#include <QString>
#include <QImage>
#include <QVector3D>
#include <QPointF>
#include <QColor>
#include <QVector>
#include <vector>
#include <unordered_map>
#include <functional>
#include "dicom/dicom_volume.h"

struct StructureLine {
    QVector<QPointF> points;  // mm座標系での線分または折れ線
    QColor color;             // 描画色
};
using StructureLineList = QVector<StructureLine>;

struct StructurePoint {
    QPointF point;  // mm座標系での点
    QColor color;   // 描画色
};
using StructurePointList = QVector<StructurePoint>;

struct StructureLine3D {
    QVector<QVector3D> points; // mm座標系での線分または折れ線
    QColor color;              // 描画色
};
using StructureLine3DList = QVector<StructureLine3D>;

// 3D点の描画情報（位置と色）
struct StructurePoint3D {
    QVector3D point; // mm座標系での点
    QColor color;    // 描画色
};
using StructurePoint3DList = QVector<StructurePoint3D>;

class RTStructureSet
{
public:
    bool loadFromFile(const QString& filename, std::function<void(int, int)> progress = {});
    QImage axialOverlay(const DicomVolume& volume, int sliceIndex) const;
    QImage sagittalOverlay(const DicomVolume& volume, int sliceIndex, int stride = 2) const;
    QImage coronalOverlay(const DicomVolume& volume, int sliceIndex, int stride = 2) const;
    StructureLineList axialContours(const DicomVolume& volume, int sliceIndex) const;
    StructureLineList sagittalContours(const DicomVolume& volume, int sliceIndex, int stride = 2) const;
    StructureLineList coronalContours(const DicomVolume& volume, int sliceIndex, int stride = 2) const;
    StructureLine3DList allContours3D(const DicomVolume& volume) const;
    StructureLine3DList roiContours3D(int roiIndex, const DicomVolume& volume) const;
    StructurePointList sagittalVertices(const DicomVolume& volume, int sliceIndex, int stride = 2) const;
    StructurePointList coronalVertices(const DicomVolume& volume, int sliceIndex, int stride = 2) const;
    int roiCount() const { return static_cast<int>(m_rois.size()); }
    QString roiName(int index) const;
    void setROIVisible(int index, bool visible);
    bool isROIVisible(int index) const;
    // 指定したROIに点が含まれるか判定
    bool isPointInsideROI(const QVector3D& point, int roiIndex) const;

    // ROIの境界ボックスを取得（患者座標系）
    bool roiBoundingBox(int index, QVector3D& minPoint, QVector3D& maxPoint) const;

    // ROIの輪郭を患者座標系で取得（Surface生成用）
    QVector<QVector<QVector3D>> roiContoursPatient(int roiIndex) const;

private:
    struct Contour {
        std::vector<QVector3D> points;
    };
    struct ROI {
        QString name;
        std::vector<Contour> contours;
        bool visible{true};
        QVector3D minPoint;
        QVector3D maxPoint;
        bool hasBoundingBox{false};
        std::unordered_map<int, std::vector<size_t>> contoursBySlice;
    };
    std::vector<ROI> m_rois;

    // Ray Casting Algorithm を用いて輪郭内判定
    bool isPointInsideContour(const QVector3D& point, const Contour& contour) const;
};

#endif // RTSTRUCT_H
