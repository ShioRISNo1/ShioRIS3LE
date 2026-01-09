#include "visualization/volume_viewer_window.h"

VolumeViewerWindow::VolumeViewerWindow(QWidget* parent)
    : QMainWindow(parent)
{
    m_viewer = new VolumeViewer(this);
    setCentralWidget(m_viewer);
    setWindowTitle("Volume Viewer");
    resize(1000,800);
}

bool VolumeViewerWindow::loadVolume(const QString& directory)
{
    return m_viewer->loadVolume(directory);
}

