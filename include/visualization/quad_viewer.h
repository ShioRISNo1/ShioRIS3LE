#ifndef QUAD_VIEWER_H
#define QUAD_VIEWER_H

#include <QWidget>
#include <QGridLayout>

#include "visualization/dicom_viewer.h"

class QuadViewer : public QWidget
{
    Q_OBJECT
public:
    explicit QuadViewer(QWidget* parent = nullptr);

    bool loadDicomFile(const QString& filename);
    bool loadDicomDirectory(const QString& directory);

private:
    QGridLayout* m_layout;
    DicomViewer* m_viewers[4];
};

#endif // QUAD_VIEWER_H
