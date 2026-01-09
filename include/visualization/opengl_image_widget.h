#ifndef OPENGL_IMAGE_WIDGET_H
#define OPENGL_IMAGE_WIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QImage>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QMatrix4x4>
#include <QMouseEvent>  // ★新規追加: QMouseEventの完全な定義をインクルード
#include "dicom/rtstruct.h"

class OpenGLImageWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT
public:
    explicit OpenGLImageWidget(QWidget* parent = nullptr);
    ~OpenGLImageWidget();

    void setImage(const QImage& image);
    void setWindowLevel(float window, float level);
    void setZoom(float zoom);
    void setPan(const QPointF& pan);
    float zoom() const;
    QPointF pan() const;
    void setPixelSpacing(float spacingX, float spacingY);
    void setStructureLines(const StructureLineList& lines);
    void setStructurePoints(const StructurePointList& points);
    void setDoseLines(const StructureLineList& lines);
    void setStructureLineWidth(int width);
    void setSlicePositionLines(const StructureLineList& lines);
    void setCursorCross(const QPointF& pos);
    void clearCursorCross();

signals:
    void doubleClicked();  // ★新規追加: ダブルクリックシグナル

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;  // ★新規追加

private:
    QOpenGLShaderProgram* m_program{nullptr};
    QOpenGLTexture* m_texture{nullptr};
    int m_matrixLoc{-1};
    QOpenGLShaderProgram* m_lineProgram{nullptr};
    int m_lineMatrixLoc{-1};
    int m_lineColorLoc{-1};
    float m_zoom{1.0f};
    QPointF m_pan{0.0f, 0.0f};
    float m_window{256.0f};
    float m_level{128.0f};
    QImage m_image;
    float m_spacingX{1.0f};
    float m_spacingY{1.0f};
    StructureLineList m_structureLines;
    StructurePointList m_structurePoints;
    StructureLineList m_doseLines;
    StructureLineList m_slicePositionLines;
    int m_structureLineWidth{1};
    bool m_showCursorCross{false};
    QPointF m_cursorCross;
};

#endif // OPENGL_IMAGE_WIDGET_H
