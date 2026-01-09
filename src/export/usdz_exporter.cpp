#include "export/usdz_exporter.h"
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QProcess>
#include <QTextStream>
#include <QDebug>
#include <QMap>
#include <QtMath>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

USDZExporter::USDZExporter()
{
}

USDZExporter::~USDZExporter()
{
}

bool USDZExporter::exportToUSDZ(const QString& filename,
                                 const StructureLine3DList& structureLines,
                                 const QVector<DoseIsosurface>& isosurfaces,
                                 const DicomVolume* ctVolume,
                                 double ctWindow,
                                 double ctLevel,
                                 int axialSliceIndex,
                                 int sagittalSliceIndex,
                                 int coronalSliceIndex)
{
    // Determine file paths
    QFileInfo fileInfo(filename);
    QString basePath = fileInfo.absolutePath();
    QString baseName = fileInfo.completeBaseName();

    // Ensure output directory exists
    QDir dir(basePath);
    if (!dir.exists()) {
        dir.mkpath(".");
    }

    // Track texture files for archiving
    QStringList textureFiles;

    // Generate USDA content (this may create texture PNG files)
    QString usdaContent = generateUSDA(structureLines, isosurfaces, ctVolume, ctWindow, ctLevel,
                                       axialSliceIndex, sagittalSliceIndex, coronalSliceIndex,
                                       basePath, textureFiles);

    if (usdaContent.isEmpty()) {
        qWarning() << "Failed to generate USDA content";
        return false;
    }

    // Write USDA file
    QString usdaPath = basePath + "/" + baseName + ".usda";
    QFile usdaFile(usdaPath);
    if (!usdaFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Failed to open USDA file for writing:" << usdaPath;
        return false;
    }

    QTextStream out(&usdaFile);
    out << usdaContent;
    usdaFile.close();

    qInfo() << "USDA file written:" << usdaPath;

    // Create USDZ archive
    QString usdzPath = filename;
    if (!usdzPath.endsWith(".usdz", Qt::CaseInsensitive)) {
        usdzPath += ".usdz";
    }

    bool success = createUSDZArchive(usdaPath, usdzPath, textureFiles);

    // Clean up temporary files if USDZ creation succeeded
    if (success) {
        QFile::remove(usdaPath);
        for (const QString& texFile : textureFiles) {
            QFile::remove(texFile);
        }
    }

    return success;
}

QString USDZExporter::generateUSDA(const StructureLine3DList& structureLines,
                                    const QVector<DoseIsosurface>& isosurfaces,
                                    const DicomVolume* ctVolume,
                                    double ctWindow,
                                    double ctLevel,
                                    int axialSliceIndex,
                                    int sagittalSliceIndex,
                                    int coronalSliceIndex,
                                    const QString& outputDir,
                                    QStringList& textureFiles)
{
    QString usda;
    QTextStream stream(&usda);

    // USDA header
    stream << "#usda 1.0\n";
    stream << "(\n";
    stream << "    defaultPrim = \"Scene\"\n";
    stream << "    upAxis = \"Y\"\n";
    stream << "    metersPerUnit = 1.0\n"; // Normalized coordinates scaled to ~1m scene for better visibility
    stream << ")\n\n";

    // Materials scope (must be defined at root level)
    stream << "def Scope \"Materials\" {\n";

    // Add materials for CT textures if needed
    if (ctVolume != nullptr) {
        // Axial texture material
        if (axialSliceIndex >= 0 && axialSliceIndex < ctVolume->depth()) {
            stream << "    def Material \"Material_CTSlice_Axial\" {\n";
            stream << "        token outputs:surface.connect = </Materials/Material_CTSlice_Axial/PreviewSurface.outputs:surface>\n";
            stream << "        def Shader \"PreviewSurface\" {\n";
            stream << "            uniform token info:id = \"UsdPreviewSurface\"\n";
            stream << "            color3f inputs:diffuseColor.connect = </Materials/Material_CTSlice_Axial/Texture.outputs:rgb>\n";
            stream << "            float inputs:opacity = 0.8\n";
            stream << "            token outputs:surface\n";
            stream << "        }\n";
            stream << "        def Shader \"Texture\" {\n";
            stream << "            uniform token info:id = \"UsdUVTexture\"\n";
            stream << "            asset inputs:file = @ct_axial.png@\n";
            stream << "            float2 inputs:st.connect = </Materials/Material_CTSlice_Axial/TexCoordReader.outputs:result>\n";
            stream << "            token inputs:wrapS = \"clamp\"\n";
            stream << "            token inputs:wrapT = \"clamp\"\n";
            stream << "            color3f outputs:rgb\n";
            stream << "        }\n";
            stream << "        def Shader \"TexCoordReader\" {\n";
            stream << "            uniform token info:id = \"UsdPrimvarReader_float2\"\n";
            stream << "            string inputs:varname = \"st\"\n";
            stream << "            float2 outputs:result\n";
            stream << "        }\n";
            stream << "    }\n\n";
        }

        // Sagittal texture material
        if (sagittalSliceIndex >= 0 && sagittalSliceIndex < ctVolume->width()) {
            stream << "    def Material \"Material_CTSlice_Sagittal\" {\n";
            stream << "        token outputs:surface.connect = </Materials/Material_CTSlice_Sagittal/PreviewSurface.outputs:surface>\n";
            stream << "        def Shader \"PreviewSurface\" {\n";
            stream << "            uniform token info:id = \"UsdPreviewSurface\"\n";
            stream << "            color3f inputs:diffuseColor.connect = </Materials/Material_CTSlice_Sagittal/Texture.outputs:rgb>\n";
            stream << "            float inputs:opacity = 0.8\n";
            stream << "            token outputs:surface\n";
            stream << "        }\n";
            stream << "        def Shader \"Texture\" {\n";
            stream << "            uniform token info:id = \"UsdUVTexture\"\n";
            stream << "            asset inputs:file = @ct_sagittal.png@\n";
            stream << "            float2 inputs:st.connect = </Materials/Material_CTSlice_Sagittal/TexCoordReader.outputs:result>\n";
            stream << "            token inputs:wrapS = \"clamp\"\n";
            stream << "            token inputs:wrapT = \"clamp\"\n";
            stream << "            color3f outputs:rgb\n";
            stream << "        }\n";
            stream << "        def Shader \"TexCoordReader\" {\n";
            stream << "            uniform token info:id = \"UsdPrimvarReader_float2\"\n";
            stream << "            string inputs:varname = \"st\"\n";
            stream << "            float2 outputs:result\n";
            stream << "        }\n";
            stream << "    }\n\n";
        }

        // Coronal texture material
        if (coronalSliceIndex >= 0 && coronalSliceIndex < ctVolume->height()) {
            stream << "    def Material \"Material_CTSlice_Coronal\" {\n";
            stream << "        token outputs:surface.connect = </Materials/Material_CTSlice_Coronal/PreviewSurface.outputs:surface>\n";
            stream << "        def Shader \"PreviewSurface\" {\n";
            stream << "            uniform token info:id = \"UsdPreviewSurface\"\n";
            stream << "            color3f inputs:diffuseColor.connect = </Materials/Material_CTSlice_Coronal/Texture.outputs:rgb>\n";
            stream << "            float inputs:opacity = 0.8\n";
            stream << "            token outputs:surface\n";
            stream << "        }\n";
            stream << "        def Shader \"Texture\" {\n";
            stream << "            uniform token info:id = \"UsdUVTexture\"\n";
            stream << "            asset inputs:file = @ct_coronal.png@\n";
            stream << "            float2 inputs:st.connect = </Materials/Material_CTSlice_Coronal/TexCoordReader.outputs:result>\n";
            stream << "            token inputs:wrapS = \"clamp\"\n";
            stream << "            token inputs:wrapT = \"clamp\"\n";
            stream << "            color3f outputs:rgb\n";
            stream << "        }\n";
            stream << "        def Shader \"TexCoordReader\" {\n";
            stream << "            uniform token info:id = \"UsdPrimvarReader_float2\"\n";
            stream << "            string inputs:varname = \"st\"\n";
            stream << "            float2 outputs:result\n";
            stream << "        }\n";
            stream << "    }\n\n";
        }
    }

    stream << "}\n\n";

    // Root transform
    stream << "def Xform \"Scene\" {\n";

    // Calculate coordinate transformation parameters (for Structure lines)
    double px_mm = 1.0, py_mm = 1.0, pz_mm = 1.0, sx = 1.0, sy = 1.0, sz = 1.0;
    if (ctVolume != nullptr) {
        int width = ctVolume->width();
        int height = ctVolume->height();
        int depth = ctVolume->depth();
        px_mm = width * ctVolume->spacingX();
        py_mm = height * ctVolume->spacingY();
        pz_mm = depth * ctVolume->spacingZ();
        double maxDim = qMax(qMax(px_mm, py_mm), pz_mm);
        sx = px_mm / maxDim;
        sy = py_mm / maxDim;
        sz = pz_mm / maxDim;
        qDebug() << "CT Volume dimensions:" << width << "x" << height << "x" << depth;
        qDebug() << "CT Spacing:" << ctVolume->spacingX() << ctVolume->spacingY() << ctVolume->spacingZ();
        qDebug() << "Physical dimensions (mm):" << px_mm << py_mm << pz_mm;
        qDebug() << "Normalization factors:" << sx << sy << sz;
    }

    // Export CT slices if available
    if (ctVolume != nullptr) {
        qDebug() << "Exporting CT slices - Axial:" << axialSliceIndex << "Sagittal:" << sagittalSliceIndex << "Coronal:" << coronalSliceIndex;
        stream << generateCTSliceMeshes(ctVolume, ctWindow, ctLevel,
                                        axialSliceIndex, sagittalSliceIndex, coronalSliceIndex,
                                        outputDir, textureFiles);
        qDebug() << "CT texture files created:" << textureFiles.size();
    }

    // Export dose isosurfaces as meshes
    qDebug() << "Exporting" << isosurfaces.size() << "dose isosurfaces";
    int exportedIsosurfaces = 0;
    for (int i = 0; i < isosurfaces.size(); ++i) {
        const DoseIsosurface& iso = isosurfaces[i];

        if (iso.isEmpty()) {
            continue;
        }

        // Get triangles from isosurface
        const QVector<DoseTriangle>& triangles = iso.triangles();

        if (triangles.isEmpty()) {
            continue;
        }
        exportedIsosurfaces++;

        // Convert triangles to vertices, indices, and normals
        // Apply mmToGL transformation if ctVolume is available
        // Use vertex deduplication to reduce file size
        QVector<QVector3D> vertices;
        QVector<int> indices;
        QVector<QVector3D> normals;
        QMap<QString, int> vertexMap; // Map vertex coordinate string to index
        QVector<int> normalCount; // Count of normals accumulated for each vertex

        // Helper function to create vertex key for deduplication
        auto makeVertexKey = [](const QVector3D& v) -> QString {
            // Use limited precision to allow nearby vertices to be merged
            return QString("%1,%2,%3")
                .arg(v.x(), 0, 'f', 6)
                .arg(v.y(), 0, 'f', 6)
                .arg(v.z(), 0, 'f', 6);
        };

        // Helper function to add or find vertex
        auto addVertex = [&](const QVector3D& v, const QVector3D& n) -> int {
            QString key = makeVertexKey(v);
            if (vertexMap.contains(key)) {
                int idx = vertexMap[key];
                // Accumulate normals for smooth shading
                normals[idx] += n;
                normalCount[idx]++;
                return idx;
            } else {
                int idx = vertices.size();
                vertices.append(v);
                normals.append(n);
                normalCount.append(1);
                vertexMap[key] = idx;
                return idx;
            }
        };

        for (int t = 0; t < triangles.size(); ++t) {
            const DoseTriangle& tri = triangles[t];

            QVector3D transformedVertices[3];
            QVector3D transformedNormal;

            if (ctVolume != nullptr) {
                // Apply mmToGL transformation
                auto mmToGL = [&](const QVector3D& pt) {
                    float x = (-pt.x() / px_mm) * sx;
                    float y = (pt.y() / py_mm) * sy;
                    float z = (pt.z() / pz_mm) * sz;
                    return QVector3D(x, y, z);
                };

                transformedVertices[0] = mmToGL(tri.vertices[0]);
                transformedVertices[1] = mmToGL(tri.vertices[1]);
                transformedVertices[2] = mmToGL(tri.vertices[2]);

                // Transform normal as well
                transformedNormal = QVector3D(
                    -tri.normal.x(),
                    tri.normal.y(),
                    tri.normal.z()
                );
                transformedNormal.normalize();
            } else {
                transformedVertices[0] = tri.vertices[0];
                transformedVertices[1] = tri.vertices[1];
                transformedVertices[2] = tri.vertices[2];
                transformedNormal = tri.normal;
            }

            // Add vertices with deduplication
            int idx0 = addVertex(transformedVertices[0], transformedNormal);
            int idx1 = addVertex(transformedVertices[1], transformedNormal);
            int idx2 = addVertex(transformedVertices[2], transformedNormal);

            // Add indices
            indices.append(idx0);
            indices.append(idx1);
            indices.append(idx2);
        }

        // Normalize accumulated normals
        for (int i = 0; i < normals.size(); ++i) {
            if (normalCount[i] > 1) {
                normals[i] /= normalCount[i];
            }
            normals[i].normalize();
        }

        QString meshName = QString("DoseIsosurface_%1").arg(i);

        // Convert color from QColor to normalized RGB
        QColor qcolor = iso.color();
        QVector3D color(qcolor.redF(), qcolor.greenF(), qcolor.blueF());

        stream << generateMeshGeometry(meshName, vertices, indices,
                                       normals, color, iso.opacity());
    }
    qDebug() << "Exported" << exportedIsosurfaces << "dose isosurfaces";

    // Export RT structure contours as tube meshes (more visible than lines)
    qDebug() << "Exporting" << structureLines.size() << "structure lines";
    int exportedStructures = 0;
    for (int i = 0; i < structureLines.size(); ++i) {
        const StructureLine3D& line = structureLines[i];

        if (line.points.size() < 2) {
            continue;
        }
        exportedStructures++;

        // Transform structure points using mmToGL
        StructureLine3D transformedLine = line;
        if (ctVolume != nullptr) {
            auto mmToGL = [&](const QVector3D& pt) {
                float x = (-pt.x() / px_mm) * sx;
                float y = (pt.y() / py_mm) * sy;
                float z = (pt.z() / pz_mm) * sz;
                return QVector3D(x, y, z);
            };

            for (int j = 0; j < transformedLine.points.size(); ++j) {
                transformedLine.points[j] = mmToGL(line.points[j]);
            }
        }

        // Decimate points (use every other point) to reduce file size
        StructureLine3D decimatedLine = transformedLine;
        QVector<QVector3D> decimatedPoints;
        for (int j = 0; j < transformedLine.points.size(); j += 2) {
            decimatedPoints.append(transformedLine.points[j]);
        }
        // Always include the last point if not already included
        if (transformedLine.points.size() > 1 && transformedLine.points.size() % 2 == 0) {
            decimatedPoints.append(transformedLine.points.last());
        }
        decimatedLine.points = decimatedPoints;

        QString lineName = QString("Structure_%1").arg(i);
        // Use very thin square tube (4 segments, no spheres, decimated points) for minimal file size
        stream << generateTubeMeshGeometry(lineName, decimatedLine, 0.001f, 4);
    }
    qDebug() << "Exported" << exportedStructures << "structure lines";

    stream << "}\n"; // Close Scene

    qDebug() << "USDA generation complete. Size:" << usda.length() << "bytes";
    return usda;
}

QString USDZExporter::generateMeshGeometry(const QString& meshName,
                                            const QVector<QVector3D>& vertices,
                                            const QVector<int>& indices,
                                            const QVector<QVector3D>& normals,
                                            const QVector3D& color,
                                            float opacity)
{
    QString mesh;
    QTextStream stream(&mesh);

    stream << "    def Mesh \"" << meshName << "\" {\n";

    // Vertices
    stream << "        float3[] points = [\n";
    for (int i = 0; i < vertices.size(); ++i) {
        const QVector3D& v = vertices[i];
        stream << "            (" << v.x() << ", " << v.y() << ", " << v.z() << ")";
        if (i < vertices.size() - 1) {
            stream << ",";
        }
        stream << "\n";
    }
    stream << "        ]\n";

    // Face vertex counts (all triangles = 3 vertices per face)
    int faceCount = indices.size() / 3;
    stream << "        int[] faceVertexCounts = [";
    for (int i = 0; i < faceCount; ++i) {
        stream << "3";
        if (i < faceCount - 1) {
            stream << ", ";
        }
    }
    stream << "]\n";

    // Face vertex indices
    stream << "        int[] faceVertexIndices = [\n";
    for (int i = 0; i < indices.size(); i += 3) {
        stream << "            " << indices[i] << ", " << indices[i+1] << ", " << indices[i+2];
        if (i < indices.size() - 3) {
            stream << ",";
        }
        stream << "\n";
    }
    stream << "        ]\n";

    // Normals (with vertex interpolation)
    if (!normals.isEmpty()) {
        stream << "        normal3f[] normals = [\n";
        for (int i = 0; i < normals.size(); ++i) {
            const QVector3D& n = normals[i];
            stream << "            (" << n.x() << ", " << n.y() << ", " << n.z() << ")";
            if (i < normals.size() - 1) {
                stream << ",";
            }
            stream << "\n";
        }
        stream << "        ] (\n";
        stream << "            interpolation = \"vertex\"\n";
        stream << "        )\n";
    }

    // Color and opacity
    stream << "        color3f[] primvars:displayColor = [("
           << color.x() << ", " << color.y() << ", " << color.z() << ")]\n";
    stream << "        float[] primvars:displayOpacity = [" << opacity << "]\n";

    // Double-sided rendering
    stream << "        uniform token subdivisionScheme = \"none\"\n";
    stream << "        uniform bool doubleSided = true\n";

    stream << "    }\n\n";

    return mesh;
}

QString USDZExporter::generateTexturedMeshGeometry(const QString& meshName,
                                                     const QVector<QVector3D>& vertices,
                                                     const QVector<int>& indices,
                                                     const QVector<QVector3D>& normals,
                                                     const QVector<QVector2D>& uvs,
                                                     const QString& texturePath,
                                                     float opacity)
{
    QString mesh;
    QTextStream stream(&mesh);

    stream << "    def Mesh \"" << meshName << "\" {\n";

    // Vertices
    stream << "        float3[] points = [\n";
    for (int i = 0; i < vertices.size(); ++i) {
        const QVector3D& v = vertices[i];
        stream << "            (" << v.x() << ", " << v.y() << ", " << v.z() << ")";
        if (i < vertices.size() - 1) {
            stream << ",";
        }
        stream << "\n";
    }
    stream << "        ]\n";

    // Face vertex counts (all triangles = 3 vertices per face)
    int faceCount = indices.size() / 3;
    stream << "        int[] faceVertexCounts = [";
    for (int i = 0; i < faceCount; ++i) {
        stream << "3";
        if (i < faceCount - 1) {
            stream << ", ";
        }
    }
    stream << "]\n";

    // Face vertex indices
    stream << "        int[] faceVertexIndices = [\n";
    for (int i = 0; i < indices.size(); i += 3) {
        stream << "            " << indices[i] << ", " << indices[i+1] << ", " << indices[i+2];
        if (i < indices.size() - 3) {
            stream << ",";
        }
        stream << "\n";
    }
    stream << "        ]\n";

    // Normals (with vertex interpolation)
    if (!normals.isEmpty()) {
        stream << "        normal3f[] normals = [\n";
        for (int i = 0; i < normals.size(); ++i) {
            const QVector3D& n = normals[i];
            stream << "            (" << n.x() << ", " << n.y() << ", " << n.z() << ")";
            if (i < normals.size() - 1) {
                stream << ",";
            }
            stream << "\n";
        }
        stream << "        ] (\n";
        stream << "            interpolation = \"vertex\"\n";
        stream << "        )\n";
    }

    // UV coordinates
    if (!uvs.isEmpty()) {
        stream << "        texCoord2f[] primvars:st = [\n";
        for (int i = 0; i < uvs.size(); ++i) {
            const QVector2D& uv = uvs[i];
            stream << "            (" << uv.x() << ", " << uv.y() << ")";
            if (i < uvs.size() - 1) {
                stream << ",";
            }
            stream << "\n";
        }
        stream << "        ] (\n";
        stream << "            interpolation = \"vertex\"\n";
        stream << "        )\n";
    }

    // Material binding
    stream << "        rel material:binding = </Materials/Material_" << meshName << ">\n";

    // Double-sided rendering
    stream << "        uniform token subdivisionScheme = \"none\"\n";
    stream << "        uniform bool doubleSided = true\n";

    stream << "    }\n\n";

    return mesh;
}

QString USDZExporter::generateLineGeometry(const QString& lineName,
                                            const StructureLine3D& line)
{
    QString lineGeom;
    QTextStream stream(&lineGeom);

    stream << "    def BasisCurves \"" << lineName << "\" {\n";
    stream << "        uniform token type = \"linear\"\n";
    stream << "        uniform token wrap = \"nonperiodic\"\n";
    stream << "        uniform token widthsInterpolation = \"constant\"\n";

    // Points
    stream << "        float3[] points = [\n";
    for (int i = 0; i < line.points.size(); ++i) {
        const QVector3D& p = line.points[i];
        stream << "            (" << p.x() << ", " << p.y() << ", " << p.z() << ")";
        if (i < line.points.size() - 1) {
            stream << ",";
        }
        stream << "\n";
    }
    stream << "        ]\n";

    // Curve vertex counts (one continuous line)
    stream << "        int[] curveVertexCounts = [" << line.points.size() << "]\n";

    // Color from line color
    QVector3D color(line.color.redF(), line.color.greenF(), line.color.blueF());
    stream << "        color3f[] primvars:displayColor = [("
           << color.x() << ", " << color.y() << ", " << color.z() << ")]\n";
    stream << "        float[] primvars:displayOpacity = [1.0]\n";

    // Line width (constant width for the entire curve - larger for visibility)
    stream << "        float[] widths = [0.01]\n";

    stream << "    }\n\n";

    return lineGeom;
}

QString USDZExporter::generateTubeMeshGeometry(const QString& tubeName,
                                                const StructureLine3D& line,
                                                float radius,
                                                int segments)
{
    // Generate ball-and-stick model: spheres at points, cylinders between them
    QVector<QVector3D> vertices;
    QVector<int> indices;
    QVector<QVector3D> normals;

    // Lambda to generate sphere mesh
    auto addSphere = [&](const QVector3D& center, float sphereRadius, int rings = 8, int sectors = 8) {
        int baseVertex = vertices.size();

        // Generate sphere vertices
        for (int r = 0; r <= rings; ++r) {
            float phi = M_PI * r / rings;
            float y = sphereRadius * cos(phi);
            float radiusAtHeight = sphereRadius * sin(phi);

            for (int s = 0; s <= sectors; ++s) {
                float theta = 2.0f * M_PI * s / sectors;
                float x = radiusAtHeight * cos(theta);
                float z = radiusAtHeight * sin(theta);

                QVector3D position = center + QVector3D(x, y, z);
                QVector3D normal = QVector3D(x, y, z).normalized();

                vertices.append(position);
                normals.append(normal);
            }
        }

        // Generate sphere indices
        for (int r = 0; r < rings; ++r) {
            for (int s = 0; s < sectors; ++s) {
                int current = baseVertex + r * (sectors + 1) + s;
                int next = current + sectors + 1;

                // First triangle
                indices.append(current);
                indices.append(next);
                indices.append(current + 1);

                // Second triangle
                indices.append(current + 1);
                indices.append(next);
                indices.append(next + 1);
            }
        }
    };

    // Lambda to generate cylinder mesh
    auto addCylinder = [&](const QVector3D& p1, const QVector3D& p2, float cylRadius, int cylSegments = 8) {
        QVector3D direction = p2 - p1;
        float length = direction.length();

        if (length < 0.0001f) {
            return; // Skip degenerate segments
        }

        direction.normalize();

        // Find perpendicular vectors
        QVector3D up = QVector3D(0, 1, 0);
        if (qAbs(QVector3D::dotProduct(direction, up)) > 0.99f) {
            up = QVector3D(1, 0, 0);
        }

        QVector3D right = QVector3D::crossProduct(direction, up).normalized();
        QVector3D forward = QVector3D::crossProduct(right, direction).normalized();

        int baseVertex = vertices.size();

        // Generate circle vertices at both ends
        for (int s = 0; s <= cylSegments; ++s) {
            float angle = 2.0f * M_PI * s / cylSegments;
            float x = cylRadius * cos(angle);
            float z = cylRadius * sin(angle);

            QVector3D offset = right * x + forward * z;
            QVector3D normal = offset.normalized();

            // Start circle
            vertices.append(p1 + offset);
            normals.append(normal);

            // End circle
            vertices.append(p2 + offset);
            normals.append(normal);
        }

        // Generate triangle indices for cylinder surface
        for (int s = 0; s < cylSegments; ++s) {
            int v1 = baseVertex + s * 2;
            int v2 = baseVertex + s * 2 + 1;
            int v3 = baseVertex + (s + 1) * 2;
            int v4 = baseVertex + (s + 1) * 2 + 1;

            // First triangle
            indices.append(v1);
            indices.append(v2);
            indices.append(v3);

            // Second triangle
            indices.append(v3);
            indices.append(v2);
            indices.append(v4);
        }
    };

    // Add cylinders between consecutive points (spheres removed for smaller file size)
    for (int i = 0; i < line.points.size() - 1; ++i) {
        addCylinder(line.points[i], line.points[i + 1], radius, segments);
    }

    if (vertices.isEmpty()) {
        return QString(); // No valid geometry
    }

    // Convert color from QColor to normalized RGB
    QVector3D color(line.color.redF(), line.color.greenF(), line.color.blueF());

    // Generate mesh using standard mesh geometry function
    return generateMeshGeometry(tubeName, vertices, indices, normals, color, 1.0f);
}

QString USDZExporter::generateCTSliceMeshes(const DicomVolume* ctVolume,
                                             double window,
                                             double level,
                                             int axialSliceIndex,
                                             int sagittalSliceIndex,
                                             int coronalSliceIndex,
                                             const QString& outputDir,
                                             QStringList& textureFiles)
{
    if (ctVolume == nullptr) {
        return QString();
    }

    int depth = ctVolume->depth();
    int width = ctVolume->width();
    int height = ctVolume->height();

    if (depth == 0 || width == 0 || height == 0) {
        return QString();
    }

    double spacingX = ctVolume->spacingX();
    double spacingY = ctVolume->spacingY();
    double spacingZ = ctVolume->spacingZ();

    // Calculate physical dimensions (same as OpenGL3DWidget)
    double px_mm = width * spacingX;
    double py_mm = height * spacingY;
    double pz_mm = depth * spacingZ;

    // Normalization factor (same as OpenGL3DWidget)
    double maxDim = qMax(qMax(px_mm, py_mm), pz_mm);
    double sx = px_mm / maxDim;
    double sy = py_mm / maxDim;
    double sz = pz_mm / maxDim;

    // Helper lambda: mmToGL conversion (SAME AS OpenGL3DWidget)
    // Note: X axis is negated!
    auto mmToGL = [&](double x_mm, double y_mm, double z_mm) -> QVector3D {
        double x = (-x_mm / px_mm) * sx;
        double y = (y_mm / py_mm) * sy;
        double z = (z_mm / pz_mm) * sz;
        return QVector3D(x, y, z);
    };

    QString result;
    QTextStream stream(&result);

    stream << "    def Xform \"CTSlices\" {\n";

    // Maximum texture size to reduce file size
    const int MAX_TEXTURE_SIZE = 512;

    // Export Axial slice (XY plane, constant Z)
    if (axialSliceIndex >= 0 && axialSliceIndex < depth) {
        QImage slice = ctVolume->getSlice(axialSliceIndex, DicomVolume::Orientation::Axial, window, level);

        // Resize to reduce file size
        if (slice.width() > MAX_TEXTURE_SIZE || slice.height() > MAX_TEXTURE_SIZE) {
            slice = slice.scaled(MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        }

        QString texturePath = outputDir + "/ct_axial.png";
        if (slice.save(texturePath)) {
            textureFiles.append(texturePath);

            // Calculate Z position in mm (same as OpenGL3DWidget)
            double axZmm = (depth > 1) ? ((double)axialSliceIndex / (depth - 1) - 0.5) * pz_mm : 0.0;

            // Vertices in SAME order and position as OpenGL3DWidget
            QVector3D v0 = mmToGL(px_mm / 2, py_mm / 2, axZmm);      // +X, +Y
            QVector3D v1 = mmToGL(-px_mm / 2, py_mm / 2, axZmm);     // -X, +Y
            QVector3D v2 = mmToGL(-px_mm / 2, -py_mm / 2, axZmm);    // -X, -Y
            QVector3D v3 = mmToGL(px_mm / 2, -py_mm / 2, axZmm);     // +X, -Y

            QVector<QVector3D> vertices;
            vertices << v0 << v1 << v2 << v3;

            QVector<int> indices;
            indices << 0 << 1 << 2 << 0 << 2 << 3;

            QVector3D edge1 = v1 - v0;
            QVector3D edge2 = v3 - v0;
            QVector3D normal = QVector3D::crossProduct(edge1, edge2).normalized();

            QVector<QVector3D> normals;
            normals << normal << normal << normal << normal;

            // UV coordinates (Y-flipped, X-flipped for proper L-R orientation)
            QVector<QVector2D> uvs;
            uvs << QVector2D(1, 1)   // v0 (+X, +Y) -> right top
                << QVector2D(0, 1)   // v1 (-X, +Y) -> left top
                << QVector2D(0, 0)   // v2 (-X, -Y) -> left bottom
                << QVector2D(1, 0);  // v3 (+X, -Y) -> right bottom

            stream << generateTexturedMeshGeometry("CTSlice_Axial", vertices, indices, normals, uvs, "ct_axial.png", 0.8f);
        }
    }

    // Export Sagittal slice (YZ plane, constant X)
    if (sagittalSliceIndex >= 0 && sagittalSliceIndex < width) {
        QImage slice = ctVolume->getSlice(sagittalSliceIndex, DicomVolume::Orientation::Sagittal, window, level);

        if (slice.width() > MAX_TEXTURE_SIZE || slice.height() > MAX_TEXTURE_SIZE) {
            slice = slice.scaled(MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        }

        QString texturePath = outputDir + "/ct_sagittal.png";
        if (slice.save(texturePath)) {
            textureFiles.append(texturePath);

            // Calculate X position in mm (same as OpenGL3DWidget)
            double sagXmm = (width > 1) ? ((double)sagittalSliceIndex / (width - 1) - 0.5) * px_mm : 0.0;

            // Vertices in SAME order and position as OpenGL3DWidget
            QVector3D v0 = mmToGL(sagXmm, py_mm / 2, pz_mm / 2);     // X, +Y, +Z
            QVector3D v1 = mmToGL(sagXmm, -py_mm / 2, pz_mm / 2);    // X, -Y, +Z
            QVector3D v2 = mmToGL(sagXmm, -py_mm / 2, -pz_mm / 2);   // X, -Y, -Z
            QVector3D v3 = mmToGL(sagXmm, py_mm / 2, -pz_mm / 2);    // X, +Y, -Z

            QVector<QVector3D> vertices;
            vertices << v0 << v1 << v2 << v3;

            QVector<int> indices;
            indices << 0 << 1 << 2 << 0 << 2 << 3;

            QVector3D edge1 = v1 - v0;
            QVector3D edge2 = v3 - v0;
            QVector3D normal = QVector3D::crossProduct(edge1, edge2).normalized();

            QVector<QVector3D> normals;
            normals << normal << normal << normal << normal;

            // UV coordinates (Y-flipped for USD texture coordinate system)
            QVector<QVector2D> uvs;
            uvs << QVector2D(0, 1)   // v0
                << QVector2D(1, 1)   // v1
                << QVector2D(1, 0)   // v2
                << QVector2D(0, 0);  // v3

            stream << generateTexturedMeshGeometry("CTSlice_Sagittal", vertices, indices, normals, uvs, "ct_sagittal.png", 0.8f);
        }
    }

    // Export Coronal slice (XZ plane, constant Y)
    if (coronalSliceIndex >= 0 && coronalSliceIndex < height) {
        QImage slice = ctVolume->getSlice(coronalSliceIndex, DicomVolume::Orientation::Coronal, window, level);

        if (slice.width() > MAX_TEXTURE_SIZE || slice.height() > MAX_TEXTURE_SIZE) {
            slice = slice.scaled(MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        }

        QString texturePath = outputDir + "/ct_coronal.png";
        if (slice.save(texturePath)) {
            textureFiles.append(texturePath);

            // Calculate Y position in mm (same as OpenGL3DWidget)
            double corYmm = (height > 1)
                ? ((double)(height - 1 - coronalSliceIndex) / (height - 1) - 0.5) * py_mm
                : 0.0;

            // Vertices in SAME order and position as OpenGL3DWidget
            QVector3D v0 = mmToGL(-px_mm / 2, corYmm, -pz_mm / 2);   // -X, Y, -Z
            QVector3D v1 = mmToGL(px_mm / 2, corYmm, -pz_mm / 2);    // +X, Y, -Z
            QVector3D v2 = mmToGL(px_mm / 2, corYmm, pz_mm / 2);     // +X, Y, +Z
            QVector3D v3 = mmToGL(-px_mm / 2, corYmm, pz_mm / 2);    // -X, Y, +Z

            QVector<QVector3D> vertices;
            vertices << v0 << v1 << v2 << v3;

            QVector<int> indices;
            indices << 0 << 1 << 2 << 0 << 2 << 3;

            QVector3D edge1 = v1 - v0;
            QVector3D edge2 = v3 - v0;
            QVector3D normal = QVector3D::crossProduct(edge1, edge2).normalized();

            QVector<QVector3D> normals;
            normals << normal << normal << normal << normal;

            // UV coordinates (NOT flipped for Coronal - Z axis orientation is correct)
            QVector<QVector2D> uvs;
            uvs << QVector2D(0, 0)   // v0
                << QVector2D(1, 0)   // v1
                << QVector2D(1, 1)   // v2
                << QVector2D(0, 1);  // v3

            stream << generateTexturedMeshGeometry("CTSlice_Coronal", vertices, indices, normals, uvs, "ct_coronal.png", 0.8f);
        }
    }

    stream << "    }\n\n";

    return result;
}

bool USDZExporter::createUSDZArchive(const QString& usdaPath, const QString& usdzPath, const QStringList& textureFiles)
{
    qDebug() << "Creating USDZ archive...";
    qDebug() << "USDA path:" << usdaPath;
    qDebug() << "USDZ path:" << usdzPath;
    qDebug() << "Texture files:" << textureFiles;

    // USDZ is essentially a ZIP archive with specific requirements
    // We'll use Qt's built-in compression or call system zip command

    // Method 1: Try using system 'zip' command (most reliable)
    QProcess zipProcess;
    QFileInfo usdaInfo(usdaPath);
    QString workingDir = usdaInfo.absolutePath();
    QString usdaFileName = usdaInfo.fileName();

    zipProcess.setWorkingDirectory(workingDir);
    qDebug() << "Working directory:" << workingDir;

    // Remove existing USDZ file if it exists
    QFile::remove(usdzPath);

    // Create zip archive (USDZ must be uncompressed zip for best compatibility)
    QStringList args;
    args << "-0" << "-r" << usdzPath << usdaFileName;

    // Add texture files to archive
    for (const QString& texFile : textureFiles) {
        QFileInfo texInfo(texFile);
        args << texInfo.fileName();
    }

    qDebug() << "Zip command:" << "zip" << args.join(" ");
    zipProcess.start("zip", args);

    if (!zipProcess.waitForStarted()) {
        qWarning() << "Failed to start zip process. Trying alternative method...";

        // Method 2: Just rename .usda to .usdz (Vision Pro may accept it)
        // This is not ideal but works as a fallback
        if (QFile::copy(usdaPath, usdzPath)) {
            qInfo() << "Created USDZ (fallback method - renamed USDA):" << usdzPath;
            qInfo() << "Note: For best compatibility, install 'zip' command or convert with Apple's Reality Converter";
            return true;
        }
        return false;
    }

    if (!zipProcess.waitForFinished(30000)) {
        qWarning() << "Zip process timeout";
        zipProcess.kill();
        return false;
    }

    QString zipStdout = zipProcess.readAllStandardOutput();
    QString zipStderr = zipProcess.readAllStandardError();

    qDebug() << "Zip stdout:" << zipStdout;
    if (!zipStderr.isEmpty()) {
        qDebug() << "Zip stderr:" << zipStderr;
    }
    qDebug() << "Zip exit code:" << zipProcess.exitCode();

    if (zipProcess.exitCode() != 0) {
        qWarning() << "Zip process failed with exit code:" << zipProcess.exitCode();

        // Try fallback method
        if (QFile::copy(usdaPath, usdzPath)) {
            qWarning() << "Created USDZ (fallback method - no textures!):" << usdzPath;
            QFileInfo usdzInfo(usdzPath);
            qDebug() << "USDZ file size:" << usdzInfo.size() << "bytes";
            return true;
        }
        return false;
    }

    qInfo() << "USDZ archive created successfully:" << usdzPath;
    QFileInfo usdzInfo(usdzPath);
    qDebug() << "USDZ file size:" << usdzInfo.size() << "bytes";
    return true;
}
