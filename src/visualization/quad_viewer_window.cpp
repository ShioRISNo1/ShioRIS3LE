#include "visualization/quad_viewer_window.h"

QuadViewerWindow::QuadViewerWindow(QWidget* parent)
    : QMainWindow(parent)
{
    m_quadViewer = new QuadViewer(this);
    setCentralWidget(m_quadViewer);
    setWindowTitle("Quad View Mode");
    resize(1200, 800);
}

bool QuadViewerWindow::loadDicomFile(const QString& filename)
{
    return m_quadViewer->loadDicomFile(filename);
}

bool QuadViewerWindow::loadDicomDirectory(const QString& directory)
{
    return m_quadViewer->loadDicomDirectory(directory);
}
