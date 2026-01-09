#ifndef USDZ_EXPORTER_H
#define USDZ_EXPORTER_H

#include <QString>
#include <QVector>
#include <QVector2D>
#include <QVector3D>
#include <QColor>
#include "dicom/rtstruct.h"
#include "dicom/dose_isosurface.h"
#include "dicom/dicom_volume.h"

/**
 * @brief Export 3D visualization data to USDZ format for Vision Pro
 *
 * This exporter creates USDZ files containing:
 * - RT Structure contours as 3D lines
 * - Dose isosurfaces as 3D meshes
 * - CT slice planes (optional)
 */
class USDZExporter
{
public:
    USDZExporter();
    ~USDZExporter();

    /**
     * @brief Export 3D data to USDZ file
     * @param filename Output USDZ file path
     * @param structureLines RT Structure contours
     * @param isosurfaces Dose isosurface meshes
     * @param ctVolume CT volume data (can be nullptr if not available)
     * @param ctWindow Window width for CT display
     * @param ctLevel Window level for CT display
     * @param axialSliceIndex Current axial slice index (-1 to skip)
     * @param sagittalSliceIndex Current sagittal slice index (-1 to skip)
     * @param coronalSliceIndex Current coronal slice index (-1 to skip)
     * @return true if export successful, false otherwise
     */
    bool exportToUSDZ(const QString& filename,
                      const StructureLine3DList& structureLines,
                      const QVector<DoseIsosurface>& isosurfaces,
                      const DicomVolume* ctVolume = nullptr,
                      double ctWindow = 400.0,
                      double ctLevel = 40.0,
                      int axialSliceIndex = -1,
                      int sagittalSliceIndex = -1,
                      int coronalSliceIndex = -1);

private:
    /**
     * @brief Generate USDA (USD ASCII) content
     */
    QString generateUSDA(const StructureLine3DList& structureLines,
                         const QVector<DoseIsosurface>& isosurfaces,
                         const DicomVolume* ctVolume,
                         double ctWindow,
                         double ctLevel,
                         int axialSliceIndex,
                         int sagittalSliceIndex,
                         int coronalSliceIndex,
                         const QString& outputDir,
                         QStringList& textureFiles);

    /**
     * @brief Write USDA mesh geometry
     */
    QString generateMeshGeometry(const QString& meshName,
                                 const QVector<QVector3D>& vertices,
                                 const QVector<int>& indices,
                                 const QVector<QVector3D>& normals,
                                 const QVector3D& color,
                                 float opacity);

    /**
     * @brief Write USDA mesh geometry with texture
     */
    QString generateTexturedMeshGeometry(const QString& meshName,
                                         const QVector<QVector3D>& vertices,
                                         const QVector<int>& indices,
                                         const QVector<QVector3D>& normals,
                                         const QVector<QVector2D>& uvs,
                                         const QString& texturePath,
                                         float opacity);

    /**
     * @brief Write USDA line geometry for structures (deprecated - use tube mesh instead)
     */
    QString generateLineGeometry(const QString& lineName,
                                 const StructureLine3D& line);

    /**
     * @brief Generate tube mesh geometry for structure contours (better visibility in VR)
     */
    QString generateTubeMeshGeometry(const QString& tubeName,
                                     const StructureLine3D& line,
                                     float radius = 1.0f,
                                     int segments = 8);

    /**
     * @brief Generate CT slice planes as textured meshes (Axial, Sagittal, Coronal)
     */
    QString generateCTSliceMeshes(const DicomVolume* ctVolume,
                                  double window,
                                  double level,
                                  int axialSliceIndex,
                                  int sagittalSliceIndex,
                                  int coronalSliceIndex,
                                  const QString& outputDir,
                                  QStringList& textureFiles);

    /**
     * @brief Create USDZ archive from USDA file and texture files
     */
    bool createUSDZArchive(const QString& usdaPath, const QString& usdzPath, const QStringList& textureFiles = QStringList());
};

#endif // USDZ_EXPORTER_H
