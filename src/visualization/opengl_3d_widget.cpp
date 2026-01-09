#include "visualization/opengl_3d_widget.h"
#include <QMatrix4x4>
#include <QVector3D>
#include <algorithm>

OpenGL3DWidget::OpenGL3DWidget(QWidget *parent) : QOpenGLWidget(parent) {
  m_textures[0] = m_textures[1] = m_textures[2] = 0;
  setFocusPolicy(Qt::ClickFocus);
  // 多ウィンドウ構成でも十分に縮小できるよう最小サイズを設定
  setMinimumSize(50, 50);
}

OpenGL3DWidget::~OpenGL3DWidget() {
  if (context()) {
    makeCurrent();
    glDeleteTextures(3, m_textures);
    doneCurrent();
  }
}

void OpenGL3DWidget::initializeGL() {
  initializeOpenGLFunctions();
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_TEXTURE_2D);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_ALPHA_TEST);
  glAlphaFunc(GL_GREATER, 0.0f);
}

void OpenGL3DWidget::resizeGL(int w, int h) { glViewport(0, 0, w, h); }

void OpenGL3DWidget::setSlices(const QImage &axial, int axIndex,
                               const QImage &sagittal, int sagIndex,
                               const QImage &coronal, int corIndex, int width,
                               int height, int depth, double spacingX,
                               double spacingY, double spacingZ) {
  m_axialImg = axial;
  m_sagittalImg = sagittal;
  m_coronalImg = coronal;
  m_axIndex = axIndex;
  m_sagIndex = sagIndex;
  m_corIndex = corIndex;
  m_width = width;
  m_height = height;
  m_depth = depth;
  m_spacingX = spacingX;
  m_spacingY = spacingY;
  m_spacingZ = spacingZ;

  makeCurrent();
  updateTexture(0, m_axialImg);
  updateTexture(1, m_sagittalImg);
  updateTexture(2, m_coronalImg);
  doneCurrent();
  update();
}

void OpenGL3DWidget::addRotation(float dx, float dy) {
  m_rotY += dx;
  m_rotX += dy;
  update();
}

void OpenGL3DWidget::setPan(const QPointF &pan) {
  m_pan = pan;
  update();
}

void OpenGL3DWidget::setZoom(double zoom) {
  m_zoom = zoom;
  update();
}

void OpenGL3DWidget::setStructureLines(const StructureLine3DList &lines) {
  m_structureLines = lines;
  update();
}

  void OpenGL3DWidget::setStructureLineWidth(int width) {
    m_structureLineWidth = width;
    update();
  }

void OpenGL3DWidget::setSourcePoints(const QVector<QVector3D> &pts) {
  m_sourcePoints = pts;
  m_sourcePointsColored.clear();
  m_activeSourcePoints.clear();
  m_inactiveSourcePoints.clear();
  m_activeSourceSegments.clear();
  m_inactiveSourceSegments.clear();
  update();
}

void OpenGL3DWidget::setSourcePointsColored(const StructurePoint3DList &pts) {
  m_sourcePointsColored = pts;
  m_sourcePoints.clear();
  m_activeSourcePoints.clear();
  m_inactiveSourcePoints.clear();
  m_activeSourceSegments.clear();
  m_inactiveSourceSegments.clear();
  update();
}

void OpenGL3DWidget::setActiveSourcePoints(const QVector<QVector3D> &pts) {
  m_activeSourcePoints = pts;
  // Clear others to avoid double draw
  m_sourcePointsColored.clear();
  m_sourcePoints.clear();
  m_activeSourceSegments.clear();
  m_inactiveSourceSegments.clear();
  update();
}

void OpenGL3DWidget::setInactiveSourcePoints(const QVector<QVector3D> &pts) {
  m_inactiveSourcePoints = pts;
  // Keep active list; clear legacy lists
  m_sourcePointsColored.clear();
  m_sourcePoints.clear();
  m_activeSourceSegments.clear();
  m_inactiveSourceSegments.clear();
  update();
}

void OpenGL3DWidget::setActiveSourceSegments(const QVector<QPair<QVector3D,QVector3D>> &segs) {
  m_activeSourceSegments = segs;
  update();
}

void OpenGL3DWidget::setInactiveSourceSegments(const QVector<QPair<QVector3D,QVector3D>> &segs) {
  m_inactiveSourceSegments = segs;
  update();
}

void OpenGL3DWidget::setDoseIsosurfaces(const QVector<DoseIsosurface> &surfaces) {
  m_doseIsosurfaces = surfaces;
  update();
}

void OpenGL3DWidget::setStructureSurfaces(const QVector<StructureSurface> &surfaces) {
  m_structureSurfaces = surfaces;
  update();
}

void OpenGL3DWidget::setShowImages(bool show) {
  m_showImages = show;
  update();
}

void OpenGL3DWidget::setShowLines(bool show) {
  m_showLines = show;
  update();
}

void OpenGL3DWidget::setShowSurfaces(bool show) {
  m_showSurfaces = show;
  update();
}

void OpenGL3DWidget::updateTexture(int i, const QImage &img) {
  if (img.isNull())
    return;
  if (m_textures[i]) {
    glDeleteTextures(1, &m_textures[i]);
    m_textures[i] = 0;
  }
  glGenTextures(1, &m_textures[i]);
  glBindTexture(GL_TEXTURE_2D, m_textures[i]);
  QImage tex = img.convertToFormat(QImage::Format_RGBA8888);
  for (int y = 0; y < tex.height(); ++y) {
    QRgb *line = reinterpret_cast<QRgb *>(tex.scanLine(y));
    for (int x = 0; x < tex.width(); ++x) {
      QRgb px = line[x];
      int r = qRed(px);
      int g = qGreen(px);
      int b = qBlue(px);

      // Calculate brightness (0-255)
      int brightness = (r + g + b) / 3;

      // Map brightness to alpha: darker pixels become more transparent
      // brightness 0-64: fully transparent to semi-transparent (alpha 0-127)
      // brightness 65+: semi-transparent to opaque (alpha 128-255)
      int alpha = brightness;
      if (brightness < 65) {
        alpha = brightness * 2;  // 0-64 -> 0-128
      } else {
        alpha = 128 + (brightness - 65) * 127 / 190;  // 65-255 -> 128-255
      }

      line[x] = qRgba(r, g, b, alpha);
    }
  }
  if (i == 0) {
    tex = tex.mirrored(true, false); // Axial: flip horizontally
  } else if (i == 2) {
    tex = tex.mirrored(false, true); // Coronal: flip vertically
  }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex.width(), tex.height(), 0, GL_RGBA,
               GL_UNSIGNED_BYTE, tex.bits());
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void OpenGL3DWidget::paintGL() {
  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glEnable(GL_TEXTURE_2D);
  glColor4f(1.f, 1.f, 1.f, 0.8f);

  QMatrix4x4 proj;
  proj.perspective(45.0f, float(width()) / float(height()), 0.1f, 10.0f);
  glMatrixMode(GL_PROJECTION);
  glLoadMatrixf(proj.constData());

  QMatrix4x4 mv;
  mv.translate(2.0f * m_pan.x() / width(), -2.0f * m_pan.y() / height(), -2.5f);
  mv.scale(m_zoom);
  mv.rotate(m_rotX, 1, 0, 0);
  mv.rotate(m_rotY, 0, 1, 0);
  glMatrixMode(GL_MODELVIEW);
  glLoadMatrixf(mv.constData());

  float px_mm = m_width * m_spacingX;
  float py_mm = m_height * m_spacingY;
  float pz_mm = m_depth * m_spacingZ;
  float maxDim = qMax(qMax(px_mm, py_mm), pz_mm);
  float sx = px_mm / maxDim;
  float sy = py_mm / maxDim;
  float sz = pz_mm / maxDim;

  auto mmToGL = [&](float x_mm, float y_mm, float z_mm) {
    float x = (-x_mm / px_mm) * sx;
    float y = (y_mm / py_mm) * sy;
    float z = (z_mm / pz_mm) * sz;
    return QVector3D(x, y, z);
  };

  float axZmm =
      (m_depth > 1) ? ((float)m_axIndex / (m_depth - 1) - 0.5f) * pz_mm : 0.0f;
  float sagXmm =
      (m_width > 1) ? ((float)m_sagIndex / (m_width - 1) - 0.5f) * px_mm : 0.0f;
  float corYmm =
      (m_height > 1)
          ? ((float)(m_height - 1 - m_corIndex) / (m_height - 1) - 0.5f) * py_mm
          : 0.0f;

  float axZ = mmToGL(0, 0, axZmm).z();
  float sagX = mmToGL(sagXmm, 0, 0).x();
  float corY = mmToGL(0, corYmm, 0).y();

  // Draw CT image slices (only if m_showImages is true)
  // Use two-pass rendering for proper transparency from all viewing angles:
  // Pass 1: Draw back faces, Pass 2: Draw front faces
  if (m_showImages && (m_textures[0] || m_textures[1] || m_textures[2])) {
    glDepthMask(GL_FALSE);  // Disable depth buffer writes for transparency
    glEnable(GL_CULL_FACE);

    // Two-pass rendering: first back faces, then front faces
    for (int pass = 0; pass < 2; ++pass) {
      // Pass 0: render back faces (cull front)
      // Pass 1: render front faces (cull back)
      glCullFace(pass == 0 ? GL_FRONT : GL_BACK);

      // Axial slice
      if (m_textures[0]) {
        QVector3D v0 = mmToGL(px_mm / 2, py_mm / 2, axZmm);
        QVector3D v1 = mmToGL(-px_mm / 2, py_mm / 2, axZmm);
        QVector3D v2 = mmToGL(-px_mm / 2, -py_mm / 2, axZmm);
        QVector3D v3 = mmToGL(px_mm / 2, -py_mm / 2, axZmm);
        glBindTexture(GL_TEXTURE_2D, m_textures[0]);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex3f(v0.x(), v0.y(), v0.z());
        glTexCoord2f(1, 0);
        glVertex3f(v1.x(), v1.y(), v1.z());
        glTexCoord2f(1, 1);
        glVertex3f(v2.x(), v2.y(), v2.z());
        glTexCoord2f(0, 1);
        glVertex3f(v3.x(), v3.y(), v3.z());
        glEnd();
      }

      // Sagittal slice
      if (m_textures[1]) {
        QVector3D v0 = mmToGL(sagXmm, py_mm / 2, pz_mm / 2);
        QVector3D v1 = mmToGL(sagXmm, -py_mm / 2, pz_mm / 2);
        QVector3D v2 = mmToGL(sagXmm, -py_mm / 2, -pz_mm / 2);
        QVector3D v3 = mmToGL(sagXmm, py_mm / 2, -pz_mm / 2);
        glBindTexture(GL_TEXTURE_2D, m_textures[1]);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex3f(v0.x(), v0.y(), v0.z());
        glTexCoord2f(1, 0);
        glVertex3f(v1.x(), v1.y(), v1.z());
        glTexCoord2f(1, 1);
        glVertex3f(v2.x(), v2.y(), v2.z());
        glTexCoord2f(0, 1);
        glVertex3f(v3.x(), v3.y(), v3.z());
        glEnd();
      }

      // Coronal slice
      if (m_textures[2]) {
        QVector3D v0 = mmToGL(-px_mm / 2, corYmm, -pz_mm / 2);
        QVector3D v1 = mmToGL(px_mm / 2, corYmm, -pz_mm / 2);
        QVector3D v2 = mmToGL(px_mm / 2, corYmm, pz_mm / 2);
        QVector3D v3 = mmToGL(-px_mm / 2, corYmm, pz_mm / 2);
        glBindTexture(GL_TEXTURE_2D, m_textures[2]);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex3f(v0.x(), v0.y(), v0.z());
        glTexCoord2f(1, 0);
        glVertex3f(v1.x(), v1.y(), v1.z());
        glTexCoord2f(1, 1);
        glVertex3f(v2.x(), v2.y(), v2.z());
        glTexCoord2f(0, 1);
        glVertex3f(v3.x(), v3.y(), v3.z());
        glEnd();
      }
    }

    glDisable(GL_CULL_FACE);
    glDepthMask(GL_TRUE);  // Re-enable depth buffer writes
  }

  glDisable(GL_TEXTURE_2D);

  // Draw intersection lines between planes (only if m_showLines is true)
  if (m_showLines) {
    glDisable(GL_DEPTH_TEST);
    glColor3f(1.f, 1.f, 1.f);
    glBegin(GL_LINES);
    // Axial & Sagittal intersection (line along Y)
    QVector3D p1 = mmToGL(sagXmm, py_mm / 2, axZmm);
    QVector3D p2 = mmToGL(sagXmm, -py_mm / 2, axZmm);
    glVertex3f(p1.x(), p1.y(), p1.z());
    glVertex3f(p2.x(), p2.y(), p2.z());
    // Axial & Coronal intersection (line along X)
    QVector3D p3 = mmToGL(px_mm / 2, corYmm, axZmm);
    QVector3D p4 = mmToGL(-px_mm / 2, corYmm, axZmm);
    glVertex3f(p3.x(), p3.y(), p3.z());
    glVertex3f(p4.x(), p4.y(), p4.z());
    // Sagittal & Coronal intersection (line along Z)
    QVector3D p5 = mmToGL(sagXmm, corYmm, pz_mm / 2);
    QVector3D p6 = mmToGL(sagXmm, corYmm, -pz_mm / 2);
    glVertex3f(p5.x(), p5.y(), p5.z());
    glVertex3f(p6.x(), p6.y(), p6.z());
    glEnd();
    glEnable(GL_DEPTH_TEST);
  }

  // Draw Structure Lines (only if m_showLines is true)
  if (m_showLines) {
    glLineWidth(static_cast<GLfloat>(m_structureLineWidth));
    for (const auto &line : m_structureLines) {
        if (line.points.size() < 2)
          continue;
      glColor4f(line.color.redF(), line.color.greenF(), line.color.blueF(),
                line.color.alphaF());
      GLenum mode = (line.points.size() == 2) ? GL_LINES : GL_LINE_STRIP;
      glBegin(mode);
      for (const QVector3D &pt : line.points) {
        QVector3D v = mmToGL(pt.x(), pt.y(), pt.z());
        glVertex3f(v.x(), v.y(), v.z());
      }
      glEnd();
    }
    glLineWidth(1.0f);
  }

  // Draw colored points if provided; otherwise draw active/inactive; otherwise legacy yellow points
  if (!m_sourcePointsColored.isEmpty()) {
    glPointSize(5.0f);
    // Draw each point individually to ensure per-point colors on all drivers
    for (const auto &item : m_sourcePointsColored) {
      glColor4f(item.color.redF(), item.color.greenF(), item.color.blueF(), item.color.alphaF());
      QVector3D v = mmToGL(item.point.x(), -item.point.y(), item.point.z());
      glBegin(GL_POINTS);
      glVertex3f(v.x(), v.y(), v.z());
      glEnd();
    }
    glPointSize(1.0f);
  } else if (!m_activeSourcePoints.isEmpty() || !m_inactiveSourcePoints.isEmpty()) {
    glPointSize(5.0f);
    // Active: red
    if (!m_activeSourcePoints.isEmpty()) {
      glColor3f(1.f, 0.f, 0.f);
      glBegin(GL_POINTS);
      for (const auto &pt : m_activeSourcePoints) {
        QVector3D v = mmToGL(pt.x(), -pt.y(), pt.z());
        glVertex3f(v.x(), v.y(), v.z());
      }
      glEnd();
    }
    // Inactive: yellow
    if (!m_inactiveSourcePoints.isEmpty()) {
      glColor3f(1.f, 1.f, 0.f);
      glBegin(GL_POINTS);
      for (const auto &pt : m_inactiveSourcePoints) {
        QVector3D v = mmToGL(pt.x(), -pt.y(), pt.z());
        glVertex3f(v.x(), v.y(), v.z());
      }
      glEnd();
    }
    glPointSize(1.0f);
  } else if (!m_sourcePoints.isEmpty()) {
    glPointSize(5.0f);
    glColor3f(1.f, 1.f, 0.f);
    glBegin(GL_POINTS);
    for (const QVector3D &pt : m_sourcePoints) {
      QVector3D v = mmToGL(pt.x(), -pt.y(), pt.z());
      glVertex3f(v.x(), v.y(), v.z());
    }
    glEnd();
    glPointSize(1.0f);
  }

  // Draw source segments (length representation) if provided
  if (!m_activeSourceSegments.isEmpty() || !m_inactiveSourceSegments.isEmpty()) {
    glLineWidth(2.0f);
    if (!m_activeSourceSegments.isEmpty()) {
      glColor3f(1.f, 0.f, 0.f);
      glBegin(GL_LINES);
      for (const auto &seg : m_activeSourceSegments) {
        QVector3D a = mmToGL(seg.first.x(), -seg.first.y(), seg.first.z());
        QVector3D b = mmToGL(seg.second.x(), -seg.second.y(), seg.second.z());
        glVertex3f(a.x(), a.y(), a.z());
        glVertex3f(b.x(), b.y(), b.z());
      }
      glEnd();
    }
    if (!m_inactiveSourceSegments.isEmpty()) {
      glColor3f(1.f, 1.f, 0.f);
      glBegin(GL_LINES);
      for (const auto &seg : m_inactiveSourceSegments) {
        QVector3D a = mmToGL(seg.first.x(), -seg.first.y(), seg.first.z());
        QVector3D b = mmToGL(seg.second.x(), -seg.second.y(), seg.second.z());
        glVertex3f(a.x(), a.y(), a.z());
        glVertex3f(b.x(), b.y(), b.z());
      }
      glEnd();
    }
    glLineWidth(1.0f);
  }

  // Render dose isosurfaces with transparency
  if (!m_doseIsosurfaces.isEmpty()) {
    // Save current OpenGL state
    GLboolean lightingWasEnabled = glIsEnabled(GL_LIGHTING);
    GLboolean textureWasEnabled = glIsEnabled(GL_TEXTURE_2D);

    // Disable texture and enable lighting for better 3D appearance
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    GLfloat lightPos[] = {1.0f, 1.0f, 1.0f, 0.0f};
    GLfloat lightAmb[] = {0.2f, 0.2f, 0.2f, 1.0f};
    GLfloat lightDiff[] = {0.8f, 0.8f, 0.8f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmb);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiff);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    // Disable depth buffer writes for proper transparency blending
    // (depth testing remains enabled so surfaces are still occluded by opaque objects)
    glDepthMask(GL_FALSE);

    // Sort isosurfaces by distance from camera (back to front)
    // Get current modelview matrix to transform centers to camera space
    QMatrix4x4 mvMatrix;
    mvMatrix.translate(2.0f * m_pan.x() / width(), -2.0f * m_pan.y() / height(), -2.5f);
    mvMatrix.scale(m_zoom);
    mvMatrix.rotate(m_rotX, 1, 0, 0);
    mvMatrix.rotate(m_rotY, 0, 1, 0);

    // Create a list of surface indices with their camera-space depth
    QVector<QPair<float, int>> depthIndexPairs;
    for (int i = 0; i < m_doseIsosurfaces.size(); ++i) {
      if (m_doseIsosurfaces[i].isEmpty())
        continue;

      // Transform center to camera space and get Z coordinate (depth)
      QVector3D center = m_doseIsosurfaces[i].center();
      QVector3D centerGL = mmToGL(center.x(), center.y(), center.z());
      QVector3D centerCameraSpace = mvMatrix.map(centerGL);

      // Store negative Z because we want to sort from far to near (larger -Z = farther)
      depthIndexPairs.append(qMakePair(-centerCameraSpace.z(), i));
    }

    // Sort by depth (farthest first)
    std::sort(depthIndexPairs.begin(), depthIndexPairs.end(),
              [](const QPair<float, int>& a, const QPair<float, int>& b) {
                return a.first > b.first;  // Sort descending (farthest to nearest)
              });

    // Render each isosurface in sorted order (back to front)
    for (const auto& pair : depthIndexPairs) {
      const auto &surface = m_doseIsosurfaces[pair.second];

      QColor col = surface.color();
      float opacity = surface.opacity();
      glColor4f(col.redF(), col.greenF(), col.blueF(), opacity);

      // Render triangles
      glBegin(GL_TRIANGLES);
      for (const auto &tri : surface.triangles()) {
        // Set normal for lighting (coordinates already in 3D widget space, same as structure lines)
        glNormal3f(tri.normal.x(), tri.normal.y(), tri.normal.z());

        // Render three vertices (coordinates already in 3D widget space with Y-axis inverted, same as structure lines)
        for (int i = 0; i < 3; ++i) {
          QVector3D v = mmToGL(tri.vertices[i].x(), tri.vertices[i].y(), tri.vertices[i].z());
          glVertex3f(v.x(), v.y(), v.z());
        }
      }
      glEnd();
    }

    // Re-enable depth buffer writes
    glDepthMask(GL_TRUE);

    // Restore OpenGL state
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_LIGHTING);
    glDisable(GL_LIGHT0);

    // Restore texture state
    if (textureWasEnabled) {
      glEnable(GL_TEXTURE_2D);
    }

    // Reset color to default (white) for proper texture rendering in next frame
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
  }

  // Render structure surfaces with transparency
  if (!m_structureSurfaces.isEmpty() && m_showSurfaces) {
    // Save current OpenGL state
    GLboolean lightingWasEnabled = glIsEnabled(GL_LIGHTING);
    GLboolean textureWasEnabled = glIsEnabled(GL_TEXTURE_2D);

    // Disable texture and enable lighting for better 3D appearance
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    GLfloat lightPos[] = {1.0f, 1.0f, 1.0f, 0.0f};
    GLfloat lightAmb[] = {0.2f, 0.2f, 0.2f, 1.0f};
    GLfloat lightDiff[] = {0.8f, 0.8f, 0.8f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmb);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiff);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    // Disable depth buffer writes for proper transparency blending
    glDepthMask(GL_FALSE);

    // Sort structure surfaces by distance from camera (back to front)
    QMatrix4x4 mvMatrix;
    mvMatrix.translate(2.0f * m_pan.x() / width(), -2.0f * m_pan.y() / height(), -2.5f);
    mvMatrix.scale(m_zoom);
    mvMatrix.rotate(m_rotX, 1, 0, 0);
    mvMatrix.rotate(m_rotY, 0, 1, 0);

    // Create a list of surface indices with their camera-space depth
    QVector<QPair<float, int>> depthIndexPairs;
    for (int i = 0; i < m_structureSurfaces.size(); ++i) {
      if (m_structureSurfaces[i].isEmpty())
        continue;

      // Transform center to camera space and get Z coordinate (depth)
      QVector3D center = m_structureSurfaces[i].center();
      QVector3D centerGL = mmToGL(center.x(), center.y(), center.z());
      QVector3D centerCameraSpace = mvMatrix.map(centerGL);

      // Store negative Z because we want to sort from far to near (larger -Z = farther)
      depthIndexPairs.append(qMakePair(-centerCameraSpace.z(), i));
    }

    // Sort by depth (farthest first)
    std::sort(depthIndexPairs.begin(), depthIndexPairs.end(),
              [](const QPair<float, int>& a, const QPair<float, int>& b) {
                return a.first > b.first;  // Sort descending (farthest to nearest)
              });

    // Render each structure surface in sorted order (back to front)
    for (const auto& pair : depthIndexPairs) {
      const auto &surface = m_structureSurfaces[pair.second];

      QColor col = surface.color();
      float opacity = surface.opacity();
      glColor4f(col.redF(), col.greenF(), col.blueF(), opacity);

      // Render triangles
      glBegin(GL_TRIANGLES);
      for (const auto &tri : surface.triangles()) {
        // Set normal for lighting
        glNormal3f(tri.normal.x(), tri.normal.y(), tri.normal.z());

        // Render three vertices
        for (int i = 0; i < 3; ++i) {
          QVector3D v = mmToGL(tri.vertices[i].x(), tri.vertices[i].y(), tri.vertices[i].z());
          glVertex3f(v.x(), v.y(), v.z());
        }
      }
      glEnd();
    }

    // Re-enable depth buffer writes
    glDepthMask(GL_TRUE);

    // Restore OpenGL state
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_LIGHTING);
    glDisable(GL_LIGHT0);

    // Restore texture state
    if (textureWasEnabled) {
      glEnable(GL_TEXTURE_2D);
    }

    // Reset color to default (white) for proper texture rendering in next frame
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
  }
}
