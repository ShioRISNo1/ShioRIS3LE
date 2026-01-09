#include "visualization/opengl_image_widget.h"
#include <QOpenGLBuffer>
#include <QOpenGLShader>
#include <QVector>
#include <QtGlobal>
#include <QColor>

#include "theme_manager.h"
#include <algorithm>
#include <cmath>

OpenGLImageWidget::OpenGLImageWidget(QWidget *parent) : QOpenGLWidget(parent) {
  // 5分割表示などでも全ウィンドウが収まるよう最小サイズを縮小
  setMinimumSize(50, 50);
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

OpenGLImageWidget::~OpenGLImageWidget() {
  if (isValid()) {
    makeCurrent();
    delete m_texture;
    delete m_program;
    delete m_lineProgram;
    doneCurrent();
  } else {
    delete m_texture;
    delete m_program;
    delete m_lineProgram;
  }
}

void OpenGLImageWidget::setImage(const QImage &image) {
  m_image = image.convertToFormat(QImage::Format_RGBA8888);

  if (isValid()) {
    makeCurrent();
    if (m_texture) {
      m_texture->destroy();
      delete m_texture;
      m_texture = nullptr;
    }
    if (!m_image.isNull()) {
      // 修正: mirrored() を flipped() に変更（Qt6.13+対応）
      m_texture = new QOpenGLTexture(m_image.mirrored(false, true));
      m_texture->setMinMagFilters(QOpenGLTexture::Linear,
                                  QOpenGLTexture::Linear);
    }
    doneCurrent();
  }

  update();
}

void OpenGLImageWidget::setWindowLevel(float window, float level) {
  m_window = window;
  m_level = level;
  update();
}

void OpenGLImageWidget::setZoom(float zoom) {
  m_zoom = zoom;
  update();
}

void OpenGLImageWidget::setPan(const QPointF &pan) {
  m_pan = pan;
  update();
}

float OpenGLImageWidget::zoom() const { return m_zoom; }

QPointF OpenGLImageWidget::pan() const { return m_pan; }

void OpenGLImageWidget::setPixelSpacing(float spacingX, float spacingY) {
  m_spacingX = spacingX;
  m_spacingY = spacingY;
  update();
}

void OpenGLImageWidget::setStructureLines(const StructureLineList &lines) {
  m_structureLines = lines;
  update();
}

void OpenGLImageWidget::setStructurePoints(const StructurePointList &points) {
  m_structurePoints = points;
  update();
}

void OpenGLImageWidget::setDoseLines(const StructureLineList &lines) {
  m_doseLines = lines;
  update();
}

void OpenGLImageWidget::setStructureLineWidth(int width) {
  m_structureLineWidth = width;
  update();
}

void OpenGLImageWidget::setSlicePositionLines(const StructureLineList &lines) {
  m_slicePositionLines = lines;
  update();
}

void OpenGLImageWidget::setCursorCross(const QPointF &pos) {
  m_cursorCross = pos;
  m_showCursorCross = true;
  update();
}

void OpenGLImageWidget::clearCursorCross() {
  m_showCursorCross = false;
  update();
}

void OpenGLImageWidget::initializeGL() {
  initializeOpenGLFunctions();

  const char *vsrc = R"(
        attribute vec2 position;
        attribute vec2 texCoord;
        uniform mat4 transform;
        varying vec2 vTexCoord;
        void main() {
            gl_Position = transform * vec4(position, 0.0, 1.0);
            vTexCoord = texCoord;
        }
    )";

  const char *fsrc = R"(
        varying vec2 vTexCoord;
        uniform sampler2D tex;
        void main() {
            gl_FragColor = texture2D(tex, vTexCoord);
        }
    )";

  m_program = new QOpenGLShaderProgram();
  m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vsrc);
  m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fsrc);
  m_program->bindAttributeLocation("position", 0);
  m_program->bindAttributeLocation("texCoord", 1);
  m_program->link();
  m_matrixLoc = m_program->uniformLocation("transform");

  const char *vsrcLine = R"(
        attribute vec2 position;
        uniform mat4 transform;
        void main() {
            gl_Position = transform * vec4(position, 0.0, 1.0);
        }
    )";

  const char *fsrcLine = R"(
        uniform vec4 color;
        void main() {
            gl_FragColor = color;
        }
    )";

  m_lineProgram = new QOpenGLShaderProgram();
  m_lineProgram->addShaderFromSourceCode(QOpenGLShader::Vertex, vsrcLine);
  m_lineProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, fsrcLine);
  m_lineProgram->bindAttributeLocation("position", 0);
  m_lineProgram->link();
  m_lineMatrixLoc = m_lineProgram->uniformLocation("transform");
  m_lineColorLoc = m_lineProgram->uniformLocation("color");
}

void OpenGLImageWidget::paintGL() {
  glClear(GL_COLOR_BUFFER_BIT);

  if (!m_program)
    return;

  if (!m_texture && !m_image.isNull()) {
    // 修正: mirrored() を flipped() に変更（Qt6.13+対応）
    m_texture = new QOpenGLTexture(m_image.mirrored(false, true));
    m_texture->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
  }

  if (!m_texture)
    return;

  QMatrix4x4 mat;
  mat.translate(2.0f * m_pan.x() / width(), -2.0f * m_pan.y() / height());
  mat.scale(m_zoom, m_zoom);

  // ウィジェットのアスペクト比による歪みを補正
  float aspect = static_cast<float>(width()) / static_cast<float>(height());
  if (aspect > 1.0f) {
    mat.scale(1.0f / aspect, 1.0f);
  } else {
    mat.scale(1.0f, aspect);
  }

  // DICOM のピクセル間隔を反映した物理サイズを設定
  float w_mm = m_image.width() * m_spacingX;
  float h_mm = m_image.height() * m_spacingY;
  float maxDim = std::max(w_mm, h_mm);
  if (maxDim <= 0.0f) {
    maxDim = 1.0f;
  }
  mat.scale(2.0f / maxDim, 2.0f / maxDim);

  m_program->bind();
  m_program->setUniformValue(m_matrixLoc, mat);
  m_texture->bind();

  GLfloat hw = w_mm * 0.5f;
  GLfloat hh = h_mm * 0.5f;
  const GLfloat vertices[] = {-hw, -hh, 0.0f, 0.0f, hw,  -hh, 1.0f, 0.0f,
                              hw,  hh,  1.0f, 1.0f, -hw, hh,  0.0f, 1.0f};

  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat),
                        vertices);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat),
                        vertices + 2);
  glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);

  m_texture->release();
  m_program->release();

  if (m_lineProgram && (!m_slicePositionLines.isEmpty() || !m_structureLines.isEmpty() ||
                        !m_doseLines.isEmpty() || !m_structurePoints.isEmpty() ||
                        m_showCursorCross)) {
    m_lineProgram->bind();
    m_lineProgram->setUniformValue(m_lineMatrixLoc, mat);

    // draw slice position lines in gray with thin width
    for (const auto &line : m_slicePositionLines) {
      if (line.points.size() < 2)
        continue;
      m_lineProgram->setUniformValue(m_lineColorLoc, line.color);
      glLineWidth(1.0f);
      QVector<GLfloat> verts;
      verts.reserve(line.points.size() * 2);
      for (const QPointF &pt : line.points) {
        verts.append(pt.x());
        verts.append(pt.y());
      }
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, verts.constData());
      // Use LINE_STRIP so rectangles and polylines render as expected
      glDrawArrays(GL_LINE_STRIP, 0, line.points.size());
      glDisableVertexAttribArray(0);
    }

    // draw structure lines with user-specified width
    glLineWidth(static_cast<GLfloat>(m_structureLineWidth));
    for (const auto &line : m_structureLines) {
      if (line.points.size() < 2)
        continue;

      m_lineProgram->setUniformValue(m_lineColorLoc, line.color);

      QVector<GLfloat> verts;
      verts.reserve(line.points.size() * 2);
      for (const QPointF &pt : line.points) {
        verts.append(pt.x());
        verts.append(pt.y());
      }
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, verts.constData());
      GLenum mode = (line.points.size() == 2) ? GL_LINES : GL_LINE_STRIP;
      glDrawArrays(mode, 0, line.points.size());
      glDisableVertexAttribArray(0);
    }

    // draw isodose lines
    if (!m_doseLines.isEmpty()) {
      glLineWidth(2.0f);
      for (const auto &line : m_doseLines) {
        if (line.points.size() < 2)
          continue;

        m_lineProgram->setUniformValue(m_lineColorLoc, line.color);

        QVector<GLfloat> verts;
        verts.reserve(line.points.size() * 2);
        for (const QPointF &pt : line.points) {
          verts.append(pt.x());
          verts.append(pt.y());
        }
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, verts.constData());
        GLenum mode = (line.points.size() == 2) ? GL_LINES : GL_LINE_STRIP;
        glDrawArrays(mode, 0, line.points.size());
        glDisableVertexAttribArray(0);
      }
    }

    // draw structure points
    for (const auto &ptItem : m_structurePoints) {
      GLfloat vert[2] = {static_cast<GLfloat>(ptItem.point.x()),
                         static_cast<GLfloat>(ptItem.point.y())};
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, vert);
      m_lineProgram->setUniformValue(m_lineColorLoc, ptItem.color);
      glPointSize(5.0f);
      glDrawArrays(GL_POINTS, 0, 1);
      glDisableVertexAttribArray(0);
    }

    if (m_showCursorCross) {
      glLineWidth(1.0f);
      m_lineProgram->setUniformValue(
          m_lineColorLoc, ThemeManager::instance().textColor());
      const float s = 3.0f; // half length of cross in mm
      GLfloat verts[8] = {static_cast<GLfloat>(m_cursorCross.x() - s),
                          static_cast<GLfloat>(m_cursorCross.y() - s),
                          static_cast<GLfloat>(m_cursorCross.x() + s),
                          static_cast<GLfloat>(m_cursorCross.y() + s),
                          static_cast<GLfloat>(m_cursorCross.x() - s),
                          static_cast<GLfloat>(m_cursorCross.y() + s),
                          static_cast<GLfloat>(m_cursorCross.x() + s),
                          static_cast<GLfloat>(m_cursorCross.y() - s)};
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, verts);
      glDrawArrays(GL_LINES, 0, 4);
      glDisableVertexAttribArray(0);
    }

    glLineWidth(1.0f);
    m_lineProgram->release();
  }
}

void OpenGLImageWidget::resizeGL(int w, int h) {
  Q_UNUSED(w);
  Q_UNUSED(h);
}

void OpenGLImageWidget::mouseDoubleClickEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    emit doubleClicked();
    event->accept();
  } else {
    QOpenGLWidget::mouseDoubleClickEvent(event);
  }
}
