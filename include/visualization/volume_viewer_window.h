#ifndef VOLUME_VIEWER_WINDOW_H
#define VOLUME_VIEWER_WINDOW_H

#include <QMainWindow>
#include "visualization/volume_viewer.h"

class VolumeViewerWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit VolumeViewerWindow(QWidget* parent = nullptr);
    bool loadVolume(const QString& directory);
private:
    VolumeViewer* m_viewer;
};

#endif // VOLUME_VIEWER_WINDOW_H
