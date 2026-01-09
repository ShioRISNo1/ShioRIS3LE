#ifndef OPENGL_3D_WIDGET_H
#define OPENGL_3D_WIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QImage>
#include <QPointF>
#include <QPair>
#include "dicom/rtstruct.h"
#include "dicom/dose_isosurface.h"
#include "dicom/structure_surface.h"

class OpenGL3DWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT
public:
    explicit OpenGL3DWidget(QWidget* parent = nullptr);
    ~OpenGL3DWidget();

    void setSlices(const QImage& axial, int axIndex,
                   const QImage& sagittal, int sagIndex,
                   const QImage& coronal, int corIndex,
                   int width, int height, int depth,
                   double spacingX, double spacingY, double spacingZ);
    void addRotation(float dx, float dy);
      void setPan(const QPointF& pan);
      void setZoom(double zoom);
      void setStructureLines(const StructureLine3DList& lines);
      void setStructureSurfaces(const QVector<StructureSurface>& surfaces);
      void setStructureLineWidth(int width);
      void setSourcePoints(const QVector<QVector3D>& pts);
      void setSourcePointsColored(const StructurePoint3DList& pts);
      void setActiveSourcePoints(const QVector<QVector3D>& pts);
      void setInactiveSourcePoints(const QVector<QVector3D>& pts);
      void setActiveSourceSegments(const QVector<QPair<QVector3D,QVector3D>>& segs);
      void setInactiveSourceSegments(const QVector<QPair<QVector3D,QVector3D>>& segs);
      void setDoseIsosurfaces(const QVector<DoseIsosurface>& surfaces);
    void setShowImages(bool show);
    void setShowLines(bool show);
    void setShowSurfaces(bool show);

    // Getters for export functionality
    const StructureLine3DList& getStructureLines() const { return m_structureLines; }
    const QVector<StructureSurface>& getStructureSurfaces() const { return m_structureSurfaces; }
    const QVector<DoseIsosurface>& getDoseIsosurfaces() const { return m_doseIsosurfaces; }

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;

private:
    void updateTexture(int i, const QImage& img);

    QImage m_axialImg;
    QImage m_sagittalImg;
    QImage m_coronalImg;
    GLuint m_textures[3];

    int m_axIndex{0};
    int m_sagIndex{0};
    int m_corIndex{0};
    int m_width{1};
    int m_height{1};
    int m_depth{1};
    double m_spacingX{1.0};
    double m_spacingY{1.0};
    double m_spacingZ{1.0};

    float m_rotX{0.0f};
    float m_rotY{0.0f};
    QPointF m_pan{0.0, 0.0};
    double m_zoom{2.25};

      StructureLine3DList m_structureLines;
      QVector<StructureSurface> m_structureSurfaces;
      int m_structureLineWidth{1};
      QVector<QVector3D> m_sourcePoints;
      StructurePoint3DList m_sourcePointsColored;
      QVector<QVector3D> m_activeSourcePoints;
      QVector<QVector3D> m_inactiveSourcePoints;
      QVector<QPair<QVector3D,QVector3D>> m_activeSourceSegments;
      QVector<QPair<QVector3D,QVector3D>> m_inactiveSourceSegments;
      QVector<DoseIsosurface> m_doseIsosurfaces;
      bool m_showImages{true};
      bool m_showLines{true};
      bool m_showSurfaces{true};
};

#endif // OPENGL_3D_WIDGET_H
