#ifndef QUAD_VIEWER_WINDOW_H
#define QUAD_VIEWER_WINDOW_H

#include <QMainWindow>

#include "visualization/quad_viewer.h"

class QuadViewerWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit QuadViewerWindow(QWidget* parent = nullptr);

    bool loadDicomFile(const QString& filename);
    bool loadDicomDirectory(const QString& directory);

private:
    QuadViewer* m_quadViewer;
};

#endif // QUAD_VIEWER_WINDOW_H
